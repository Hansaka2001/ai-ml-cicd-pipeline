"""
test_schemas.py
---------------
Unit tests for ``app.schemas`` Pydantic models.

These tests exercise validation rules, field constraints, and the
``text_must_not_be_blank`` validator — completely independent of any HTTP or
ML layer.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import PredictRequest, PredictResponse, WelcomeResponse

# ---------------------------------------------------------------------------
# PredictRequest
# ---------------------------------------------------------------------------


class TestPredictRequest:
    """Tests for the ``PredictRequest`` schema."""

    def test_valid_text_is_accepted(self) -> None:
        """A normal review string must be accepted without errors."""
        req = PredictRequest(text="This movie was great!")
        assert req.text == "This movie was great!"

    def test_text_is_stripped(self) -> None:
        """Leading and trailing whitespace must be stripped from text."""
        req = PredictRequest(text="  Great film.  ")
        assert req.text == "Great film."

    def test_empty_string_raises_validation_error(self) -> None:
        """An empty string must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            PredictRequest(text="")

    def test_whitespace_only_raises_validation_error(self) -> None:
        """A whitespace-only string must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            PredictRequest(text="   \t\n")

    def test_text_at_max_length_is_accepted(self) -> None:
        """A string of exactly 10,000 characters must be accepted."""
        req = PredictRequest(text="a" * 10_000)
        assert len(req.text) == 10_000

    def test_text_exceeding_max_length_raises_validation_error(self) -> None:
        """A string longer than 10,000 characters must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            PredictRequest(text="a" * 10_001)

    def test_missing_text_field_raises_validation_error(self) -> None:
        """Omitting the ``text`` field entirely must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            PredictRequest()  # type: ignore[call-arg]

    def test_none_text_raises_validation_error(self) -> None:
        """Passing ``None`` as text must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            PredictRequest(text=None)  # type: ignore[arg-type]

    def test_single_character_is_accepted(self) -> None:
        """A single non-whitespace character must be accepted."""
        req = PredictRequest(text="A")
        assert req.text == "A"

    def test_text_with_internal_whitespace_preserved(self) -> None:
        """Internal whitespace must not be collapsed — only edges are stripped."""
        req = PredictRequest(text="  great   movie  ")
        assert req.text == "great   movie"


# ---------------------------------------------------------------------------
# PredictResponse
# ---------------------------------------------------------------------------


class TestPredictResponse:
    """Tests for the ``PredictResponse`` schema."""

    def test_positive_sentiment_is_valid(self) -> None:
        """'Positive' sentiment must be stored without modification."""
        resp = PredictResponse(sentiment="Positive", confidence=0.95)
        assert resp.sentiment == "Positive"

    def test_negative_sentiment_is_valid(self) -> None:
        """'Negative' sentiment must be stored without modification."""
        resp = PredictResponse(sentiment="Negative", confidence=0.88)
        assert resp.sentiment == "Negative"

    def test_confidence_at_zero_is_valid(self) -> None:
        """confidence=0.0 must be accepted (lower boundary)."""
        resp = PredictResponse(sentiment="Negative", confidence=0.0)
        assert resp.confidence == 0.0

    def test_confidence_at_one_is_valid(self) -> None:
        """confidence=1.0 must be accepted (upper boundary)."""
        resp = PredictResponse(sentiment="Positive", confidence=1.0)
        assert resp.confidence == 1.0

    def test_confidence_below_zero_raises_validation_error(self) -> None:
        """confidence < 0.0 must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            PredictResponse(sentiment="Negative", confidence=-0.1)

    def test_confidence_above_one_raises_validation_error(self) -> None:
        """confidence > 1.0 must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            PredictResponse(sentiment="Positive", confidence=1.01)

    def test_missing_sentiment_raises_validation_error(self) -> None:
        """Omitting the ``sentiment`` field must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            PredictResponse(confidence=0.9)  # type: ignore[call-arg]

    def test_missing_confidence_raises_validation_error(self) -> None:
        """Omitting the ``confidence`` field must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            PredictResponse(sentiment="Positive")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# WelcomeResponse
# ---------------------------------------------------------------------------


class TestWelcomeResponse:
    """Tests for the ``WelcomeResponse`` schema."""

    def test_valid_response_is_accepted(self) -> None:
        """A fully populated WelcomeResponse must be created without errors."""
        resp = WelcomeResponse(message="Welcome!", version="1.0.0", docs_url="/docs")
        assert resp.message == "Welcome!"
        assert resp.version == "1.0.0"
        assert resp.docs_url == "/docs"

    def test_missing_message_raises_validation_error(self) -> None:
        """Omitting ``message`` must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            WelcomeResponse(version="1.0.0", docs_url="/docs")  # type: ignore[call-arg]

    def test_missing_version_raises_validation_error(self) -> None:
        """Omitting ``version`` must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            WelcomeResponse(message="Hi", docs_url="/docs")  # type: ignore[call-arg]

    def test_missing_docs_url_raises_validation_error(self) -> None:
        """Omitting ``docs_url`` must raise a ``ValidationError``."""
        with pytest.raises(ValidationError):
            WelcomeResponse(message="Hi", version="1.0.0")  # type: ignore[call-arg]
