"""
model_loader.py
---------------
Responsible for loading the trained TF-IDF vectorizer and Logistic Regression
model from disk exactly once (module-level singleton pattern).

The singleton approach avoids reloading heavy joblib artifacts on every
prediction request, keeping latency low under production load.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to the project root)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _PROJECT_ROOT / "training" / "model"
_VECTORIZER_PATH = _MODEL_DIR / "tfidf_vectorizer.joblib"
_MODEL_PATH = _MODEL_DIR / "logistic_regression.joblib"

# ---------------------------------------------------------------------------
# Module-level singletons (populated on first call to `get_model`)
# ---------------------------------------------------------------------------

_vectorizer: Optional[TfidfVectorizer] = None
_model: Optional[LogisticRegression] = None


def _load_artifacts() -> tuple[TfidfVectorizer, LogisticRegression]:
    """Load vectorizer and model from disk.

    Raises:
        FileNotFoundError: If the model artifacts are missing (training has
            not been run yet).

    Returns:
        A tuple of (vectorizer, model).
    """
    if not _VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer not found at '{_VECTORIZER_PATH}'. "
            "Please run `python training/train.py` first."
        )
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{_MODEL_PATH}'. "
            "Please run `python training/train.py` first."
        )

    log.info("Loading vectorizer from %s …", _VECTORIZER_PATH)
    vectorizer: TfidfVectorizer = joblib.load(_VECTORIZER_PATH)

    log.info("Loading model from %s …", _MODEL_PATH)
    model: LogisticRegression = joblib.load(_MODEL_PATH)

    log.info("Model artifacts loaded successfully.")
    return vectorizer, model


def get_vectorizer() -> TfidfVectorizer:
    """Return the singleton TfidfVectorizer, loading it on first access.

    Returns:
        Fitted TfidfVectorizer instance.
    """
    global _vectorizer, _model
    if _vectorizer is None:
        _vectorizer, _model = _load_artifacts()
    return _vectorizer


def get_model() -> LogisticRegression:
    """Return the singleton LogisticRegression model, loading it on first access.

    Returns:
        Trained LogisticRegression instance.
    """
    global _vectorizer, _model
    if _model is None:
        _vectorizer, _model = _load_artifacts()
    return _model
