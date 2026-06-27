"""
test_predictor.py
-----------------
Unit tests for ``app.predictor``.

The real TF-IDF vectorizer and Logistic Regression model are replaced with
lightweight MagicMock objects so these tests are completely isolated from the
filesystem and run instantly without requiring trained artifacts.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.predictor import predict
from app.schemas import PredictResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def positive_mocks() -> tuple[MagicMock, MagicMock]:
    """Return (vectorizer, model) mocks configured for a Positive prediction."""
    vec = MagicMock()
    vec.transform.return_value = np.zeros((1, 10))
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.05, 0.95]])
    return vec, model


@pytest.fixture()
def negative_mocks() -> tuple[MagicMock, MagicMock]:
    """Return (vectorizer, model) mocks configured for a Negative prediction."""
    vec = MagicMock()
    vec.transform.return_value = np.zeros((1, 10))
    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.88, 0.12]])
    return vec, model


@pytest.fixture()
def borderline_mocks() -> tuple[MagicMock, MagicMock]:
    """Return mocks for a borderline 50/50 prediction (Positive wins tie)."""
    vec = MagicMock()
    vec.transform.return_value = np.zeros((1, 10))
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.50, 0.50]])
    return vec, model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredict:
    """Unit tests for the ``predict`` function."""

    def test_returns_predict_response_type(self, positive_mocks):
        """``predict`` must return a ``PredictResponse`` instance."""
        vec, model = positive_mocks
        with (
            patch("app.predictor.get_vectorizer", return_value=vec),
            patch("app.predictor.get_model", return_value=model),
        ):
            result = predict("Great movie!")
        assert isinstance(result, PredictResponse)

    def test_positive_label(self, positive_mocks):
        """When model predicts class 1, sentiment must be 'Positive'."""
        vec, model = positive_mocks
        with (
            patch("app.predictor.get_vectorizer", return_value=vec),
            patch("app.predictor.get_model", return_value=model),
        ):
            result = predict("Great movie!")
        assert result.sentiment == "Positive"

    def test_negative_label(self, negative_mocks):
        """When model predicts class 0, sentiment must be 'Negative'."""
        vec, model = negative_mocks
        with (
            patch("app.predictor.get_vectorizer", return_value=vec),
            patch("app.predictor.get_model", return_value=model),
        ):
            result = predict("Boring and terrible.")
        assert result.sentiment == "Negative"

    def test_confidence_is_float(self, positive_mocks):
        """confidence must be a Python float."""
        vec, model = positive_mocks
        with (
            patch("app.predictor.get_vectorizer", return_value=vec),
            patch("app.predictor.get_model", return_value=model),
        ):
            result = predict("Loved it!")
        assert isinstance(result.confidence, float)

    def test_confidence_within_bounds(self, positive_mocks):
        """confidence must be within [0.0, 1.0]."""
        vec, model = positive_mocks
        with (
            patch("app.predictor.get_vectorizer", return_value=vec),
            patch("app.predictor.get_model", return_value=model),
        ):
            result = predict("Loved it!")
        assert 0.0 <= result.confidence <= 1.0

    def test_positive_confidence_value(self, positive_mocks):
        """Positive mock (proba=[0.05, 0.95]) → confidence must be 0.95."""
        vec, model = positive_mocks
        with (
            patch("app.predictor.get_vectorizer", return_value=vec),
            patch("app.predictor.get_model", return_value=model),
        ):
            result = predict("Loved it!")
        assert result.confidence == pytest.approx(0.95, abs=1e-4)

    def test_negative_confidence_value(self, negative_mocks):
        """Negative mock (proba=[0.88, 0.12]) → confidence must be 0.88."""
        vec, model = negative_mocks
        with (
            patch("app.predictor.get_vectorizer", return_value=vec),
            patch("app.predictor.get_model", return_value=model),
        ):
            result = predict("Hated it.")
        assert result.confidence == pytest.approx(0.88, abs=1e-4)

    def test_borderline_confidence(self, borderline_mocks):
        """Borderline 50/50 prediction → confidence must be 0.50."""
        vec, model = borderline_mocks
        with (
            patch("app.predictor.get_vectorizer", return_value=vec),
            patch("app.predictor.get_model", return_value=model),
        ):
            result = predict("It was okay I guess.")
        assert result.confidence == pytest.approx(0.50, abs=1e-4)

    def test_vectorizer_called_with_list(self, positive_mocks):
        """``vectorizer.transform`` must be called with a list wrapping the text."""
        vec, model = positive_mocks
        with (
            patch("app.predictor.get_vectorizer", return_value=vec),
            patch("app.predictor.get_model", return_value=model),
        ):
            predict("Some review text.")
        vec.transform.assert_called_once_with(["Some review text."])

    def test_propagates_file_not_found(self):
        """``predict`` must propagate ``FileNotFoundError`` if loader raises it."""
        with (
            patch(
                "app.predictor.get_vectorizer",
                side_effect=FileNotFoundError("Vectorizer not found"),
            ),
            pytest.raises(FileNotFoundError),
        ):
            predict("Some text.")

    def test_confidence_is_rounded_to_4dp(self, positive_mocks):
        """Confidence must be rounded to at most 4 decimal places."""
        vec, model = positive_mocks
        model.predict_proba.return_value = np.array([[0.123456789, 0.876543211]])
        with (
            patch("app.predictor.get_vectorizer", return_value=vec),
            patch("app.predictor.get_model", return_value=model),
        ):
            result = predict("Some text.")
        assert result.confidence == round(result.confidence, 4)
