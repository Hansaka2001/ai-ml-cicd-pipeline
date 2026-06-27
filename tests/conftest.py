"""
conftest.py
-----------
Shared pytest fixtures for the entire test suite.

Fixtures here are automatically available to every test module without
needing an explicit import.  Scope is set per-fixture to minimise
redundant object creation:

  - ``mock_vectorizer`` / ``mock_model`` — lightweight fakes that avoid
    touching the filesystem or loading real joblib artifacts.
  - ``patch_model_loader`` — patches the module-level singletons in
    ``app.model_loader`` so unit tests are fully isolated from disk state.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

# ---------------------------------------------------------------------------
# Fake ML objects
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_vectorizer() -> MagicMock:
    """Return a MagicMock that mimics a fitted TfidfVectorizer.

    ``transform`` returns a (1, 1) sparse-like matrix so the model mock
    can call ``predict`` / ``predict_proba`` without raising shape errors.
    """
    vec = MagicMock()
    vec.transform.return_value = np.zeros((1, 1))
    return vec


@pytest.fixture()
def mock_model_positive(mock_vectorizer: MagicMock) -> MagicMock:
    """Return a mock classifier that always predicts *Positive* (class 1)."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.05, 0.95]])
    return model


@pytest.fixture()
def mock_model_negative(mock_vectorizer: MagicMock) -> MagicMock:
    """Return a mock classifier that always predicts *Negative* (class 0)."""
    model = MagicMock()
    model.predict.return_value = np.array([0])
    model.predict_proba.return_value = np.array([[0.88, 0.12]])
    return model


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_client_positive(mock_vectorizer: MagicMock, mock_model_positive: MagicMock):
    """TestClient whose model always returns Positive — no real artifacts needed."""
    with (
        patch("app.model_loader.get_vectorizer", return_value=mock_vectorizer),
        patch("app.model_loader.get_model", return_value=mock_model_positive),
        patch("app.predictor.get_vectorizer", return_value=mock_vectorizer),
        patch("app.predictor.get_model", return_value=mock_model_positive),
    ):
        yield TestClient(app)


@pytest.fixture()
def patched_client_negative(mock_vectorizer: MagicMock, mock_model_negative: MagicMock):
    """TestClient whose model always returns Negative — no real artifacts needed."""
    with (
        patch("app.model_loader.get_vectorizer", return_value=mock_vectorizer),
        patch("app.model_loader.get_model", return_value=mock_model_negative),
        patch("app.predictor.get_vectorizer", return_value=mock_vectorizer),
        patch("app.predictor.get_model", return_value=mock_model_negative),
    ):
        yield TestClient(app)
