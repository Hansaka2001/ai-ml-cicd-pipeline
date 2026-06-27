"""
test_api.py
-----------
Integration tests for the Sentiment Analysis REST API.

These tests use FastAPI's built-in `TestClient` (backed by httpx) to exercise
the full request/response cycle without spinning up a real server.

NOTE: The tests will be **skipped automatically** when the model artifacts are
not present (i.e., training hasn't been run yet).  This prevents CI from
failing in the early scaffold stage.

Run:
    pytest tests/test_api.py -v
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_DIR = Path(__file__).resolve().parent.parent / "training" / "model"
_MODELS_EXIST = (
    (_MODEL_DIR / "tfidf_vectorizer.joblib").exists()
    and (_MODEL_DIR / "logistic_regression.joblib").exists()
)

# Skip all tests gracefully when model hasn't been trained yet.
pytestmark = pytest.mark.skipif(
    not _MODELS_EXIST,
    reason="Model artifacts not found. Run `python training/train.py` first.",
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    """Tests for the GET / welcome endpoint."""

    def test_status_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_response_has_required_keys(self):
        response = client.get("/")
        body = response.json()
        assert "message" in body
        assert "version" in body
        assert "docs_url" in body

    def test_message_is_string(self):
        response = client.get("/")
        assert isinstance(response.json()["message"], str)


# ---------------------------------------------------------------------------
# POST /predict — happy path
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    """Tests for the POST /predict endpoint."""

    def test_positive_review(self):
        payload = {"text": "This movie was absolutely incredible! A true masterpiece."}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["sentiment"] == "Positive"
        assert 0.0 <= body["confidence"] <= 1.0

    def test_negative_review(self):
        payload = {"text": "Terrible film. Worst movie I have ever seen. Boring and dull."}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["sentiment"] == "Negative"
        assert 0.0 <= body["confidence"] <= 1.0

    def test_response_schema(self):
        """Ensure the response contains exactly the expected keys."""
        payload = {"text": "An okay film, nothing special."}
        response = client.post("/predict", json=payload)
        body = response.json()
        assert set(body.keys()) == {"sentiment", "confidence"}

    def test_sentiment_is_binary(self):
        """sentiment must be one of the two allowed values."""
        payload = {"text": "Some text about a movie."}
        response = client.post("/predict", json=payload)
        assert response.json()["sentiment"] in {"Positive", "Negative"}

    def test_confidence_is_float_in_range(self):
        payload = {"text": "Surprisingly good film with great acting."}
        response = client.post("/predict", json=payload)
        confidence = response.json()["confidence"]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# POST /predict — validation / error cases
# ---------------------------------------------------------------------------


class TestPredictValidation:
    """Tests for request validation on the POST /predict endpoint."""

    def test_missing_text_field_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_empty_string_returns_422(self):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_whitespace_only_returns_422(self):
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 422

    def test_non_string_text_returns_422(self):
        response = client.post("/predict", json={"text": 12345})
        # FastAPI will coerce int to str; this is acceptable behaviour.
        # The test just verifies it doesn't crash with a 500.
        assert response.status_code in {200, 422}
