"""
predictor.py
------------
Pure prediction logic, decoupled from the web framework.

The `predict` function acts as the single entry point for inference:
  1. Vectorises the incoming text with the loaded TF-IDF vectorizer.
  2. Runs the text through the trained Logistic Regression model.
  3. Returns a human-readable label and the model's confidence score.

Keeping inference logic here (rather than inside a route handler) makes it
easy to unit-test independently of HTTP concerns.
"""

import logging

import numpy as np

from app.model_loader import get_model, get_vectorizer
from app.schemas import PredictResponse

log = logging.getLogger(__name__)

# Map the numeric class index → human-readable sentiment label.
_LABEL_MAP: dict[int, str] = {
    0: "Negative",
    1: "Positive",
}


def predict(text: str) -> PredictResponse:
    """Classify a piece of text as Positive or Negative sentiment.

    Args:
        text: Pre-validated, stripped input text from the API request.

    Returns:
        A :class:`~app.schemas.PredictResponse` containing the predicted
        sentiment label and the model's probability confidence.

    Raises:
        RuntimeError: Propagates any unexpected error from the model or
            vectorizer so the API layer can return an appropriate HTTP 500.
    """
    log.debug("Running prediction for text (first 80 chars): %.80s …", text)

    # Vectorise — `transform` expects an iterable, so we pass a list of one.
    vectorizer = get_vectorizer()
    X = vectorizer.transform([text])

    # Predict class label and class probabilities.
    model = get_model()
    predicted_class: int = int(model.predict(X)[0])

    # predict_proba returns [[prob_class_0, prob_class_1]]; we want the
    # probability of whichever class was predicted.
    probabilities: np.ndarray = model.predict_proba(X)[0]
    confidence: float = round(float(probabilities[predicted_class]), 4)

    sentiment: str = _LABEL_MAP[predicted_class]

    log.info("Prediction → %s (confidence: %.4f)", sentiment, confidence)

    return PredictResponse(sentiment=sentiment, confidence=confidence)
