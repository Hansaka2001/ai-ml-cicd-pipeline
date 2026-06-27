"""
test_api.py
-----------
Integration tests for the Sentiment Analysis REST API.

These tests exercise the full HTTP request/response cycle using FastAPI's
``TestClient``.  They are split into two groups:

* **Unit-style** — use mocked ML objects (no real artifacts required).
  These always run and cover the contract of every endpoint.

* **Live** — load the real trained model from disk.  Automatically skipped
  when the model artifacts are not present so CI does not fail during the
  scaffold phase before training has been run.

Run all tests:
    pytest tests/test_api.py -v

Run only the always-on mock-backed tests:
    pytest tests/test_api.py -v -m "not live"
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_DIR = Path(__file__).resolve().parent.parent / "training" / "model"
_MODELS_EXIST = (_MODEL_DIR / "tfidf_vectorizer.joblib").exists() and (
    _MODEL_DIR / "logistic_regression.joblib"
).exists()

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

# Tests decorated with @pytest.mark.live are skipped unless model is trained.
_live = pytest.mark.skipif(
    not _MODELS_EXIST,
    reason="Model artifacts not found. Run `python training/train.py` first.",
)


# ===========================================================================
# Section 1: GET / — mock-backed (always runs)
# ===========================================================================


class TestRootEndpoint:
    """Tests for the GET / welcome endpoint (no model required)."""

    def test_status_200(self) -> None:
        """Root endpoint must return HTTP 200."""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200

    def test_response_has_required_keys(self) -> None:
        """Response body must contain message, version, and docs_url."""
        client = TestClient(app)
        body = client.get("/").json()
        assert "message" in body
        assert "version" in body
        assert "docs_url" in body

    def test_message_is_non_empty_string(self) -> None:
        """message field must be a non-empty string."""
        client = TestClient(app)
        message = client.get("/").json()["message"]
        assert isinstance(message, str)
        assert len(message) > 0

    def test_version_is_semver_format(self) -> None:
        """version field must follow major.minor.patch format."""
        client = TestClient(app)
        version = client.get("/").json()["version"]
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_docs_url_starts_with_slash(self) -> None:
        """docs_url must be a relative path starting with /."""
        client = TestClient(app)
        docs_url = client.get("/").json()["docs_url"]
        assert docs_url.startswith("/")

    def test_content_type_is_json(self) -> None:
        """Response must declare application/json content-type."""
        client = TestClient(app)
        response = client.get("/")
        assert "application/json" in response.headers["content-type"]


# ===========================================================================
# Section 2: POST /predict — mock-backed happy path (always runs)
# ===========================================================================


class TestPredictMocked:
    """POST /predict tests using lightweight mock models — no artifacts needed."""

    def test_positive_prediction_returns_200(self, patched_client_positive: TestClient) -> None:
        """A mock Positive prediction must return HTTP 200."""
        response = patched_client_positive.post("/predict", json={"text": "Outstanding film!"})
        assert response.status_code == 200

    def test_positive_sentiment_label(self, patched_client_positive: TestClient) -> None:
        """Mock Positive model must return 'Positive' label."""
        body = patched_client_positive.post("/predict", json={"text": "Outstanding film!"}).json()
        assert body["sentiment"] == "Positive"

    def test_positive_confidence_range(self, patched_client_positive: TestClient) -> None:
        """Confidence must be a float in [0.0, 1.0]."""
        body = patched_client_positive.post("/predict", json={"text": "Outstanding film!"}).json()
        assert isinstance(body["confidence"], float)
        assert 0.0 <= body["confidence"] <= 1.0

    def test_negative_prediction_returns_200(self, patched_client_negative: TestClient) -> None:
        """A mock Negative prediction must return HTTP 200."""
        response = patched_client_negative.post("/predict", json={"text": "Awful and boring."})
        assert response.status_code == 200

    def test_negative_sentiment_label(self, patched_client_negative: TestClient) -> None:
        """Mock Negative model must return 'Negative' label."""
        body = patched_client_negative.post("/predict", json={"text": "Awful and boring."}).json()
        assert body["sentiment"] == "Negative"

    def test_response_schema_has_exactly_two_keys(
        self, patched_client_positive: TestClient
    ) -> None:
        """Response must contain exactly 'sentiment' and 'confidence'."""
        body = patched_client_positive.post("/predict", json={"text": "Some text."}).json()
        assert set(body.keys()) == {"sentiment", "confidence"}

    def test_sentiment_is_one_of_allowed_values(self, patched_client_positive: TestClient) -> None:
        """sentiment must be either 'Positive' or 'Negative', never anything else."""
        body = patched_client_positive.post("/predict", json={"text": "Some text."}).json()
        assert body["sentiment"] in {"Positive", "Negative"}

    def test_model_not_available_returns_503(self) -> None:
        """When model artifacts are missing, POST /predict must return 503."""
        with patch(
            "app.predictor.get_vectorizer",
            side_effect=FileNotFoundError("Vectorizer not found"),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/predict", json={"text": "Any text."})
        assert response.status_code == 503

    def test_503_response_contains_detail_key(self) -> None:
        """503 response body must include a 'detail' key."""
        with patch(
            "app.predictor.get_vectorizer",
            side_effect=FileNotFoundError("Vectorizer not found"),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            body = client.post("/predict", json={"text": "Any text."}).json()
        assert "detail" in body


# ===========================================================================
# Section 3: POST /predict — input validation (always runs)
# ===========================================================================


class TestPredictValidation:
    """Request-body validation tests — no model required."""

    def test_missing_text_field_returns_422(self) -> None:
        """Omitting the text field must produce a 422 validation error."""
        client = TestClient(app)
        assert client.post("/predict", json={}).status_code == 422

    def test_empty_string_returns_422(self) -> None:
        """An empty string must be rejected with 422."""
        client = TestClient(app)
        assert client.post("/predict", json={"text": ""}).status_code == 422

    def test_whitespace_only_returns_422(self) -> None:
        """A whitespace-only string must be rejected with 422."""
        client = TestClient(app)
        assert client.post("/predict", json={"text": "   \t\n"}).status_code == 422

    def test_text_exceeding_max_length_returns_422(self) -> None:
        """Text longer than 10,000 characters must be rejected with 422."""
        client = TestClient(app)
        long_text = "a" * 10_001
        assert client.post("/predict", json={"text": long_text}).status_code == 422

    def test_null_text_returns_422(self) -> None:
        """null text value must produce a 422 validation error."""
        client = TestClient(app)
        assert client.post("/predict", json={"text": None}).status_code == 422

    def test_422_response_has_detail_key(self) -> None:
        """422 error response must include a 'detail' array."""
        client = TestClient(app)
        body = client.post("/predict", json={}).json()
        assert "detail" in body

    def test_non_string_numeric_text(self) -> None:
        """Integer text is coerced to string by Pydantic; must not crash."""
        client = TestClient(app)
        # Pydantic v2 coerces int→str, so this may succeed or fail validation;
        # it must never return 500.
        response = client.post("/predict", json={"text": 12345})
        assert response.status_code in {200, 422}


# ===========================================================================
# Section 4: Live tests — real model from disk (skipped if not trained)
# ===========================================================================


@_live
class TestPredictLive:
    """End-to-end tests that load the real trained model from disk."""

    client: TestClient = TestClient(app)

    def test_clearly_positive_review(self) -> None:
        """A strongly positive review must be classified as Positive."""
        payload = {"text": "This movie was absolutely incredible! A true masterpiece."}
        body = self.client.post("/predict", json=payload).json()
        assert body["sentiment"] == "Positive"
        assert body["confidence"] >= 0.5

    def test_clearly_negative_review(self) -> None:
        """A strongly negative review must be classified as Negative."""
        payload = {"text": "Terrible film. Worst movie I have ever seen. Boring and dull."}
        body = self.client.post("/predict", json=payload).json()
        assert body["sentiment"] == "Negative"
        assert body["confidence"] >= 0.5

    def test_confidence_never_exceeds_1(self) -> None:
        """Model confidence must always be ≤ 1.0."""
        payload = {"text": "An average film. Not great, not terrible."}
        body = self.client.post("/predict", json=payload).json()
        assert body["confidence"] <= 1.0

    def test_confidence_never_below_0(self) -> None:
        """Model confidence must always be ≥ 0.0."""
        payload = {"text": "An average film. Not great, not terrible."}
        body = self.client.post("/predict", json=payload).json()
        assert body["confidence"] >= 0.0
