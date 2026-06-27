"""
train.py
--------
Trains a binary sentiment classifier on the Hugging Face IMDb dataset.

Pipeline:
    1. Load the IMDb dataset via HuggingFace `datasets`.
    2. Vectorise text with TfidfVectorizer (character + word n-grams).
    3. Train a LogisticRegression classifier.
    4. Evaluate on the test split and report metrics.
    5. Persist the vectorizer and model to `training/model/` using joblib.

Usage:
    python training/train.py
"""

import os
import time
import logging
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Paths
MODEL_DIR = Path(__file__).parent / "model"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"

# TF-IDF hyper-parameters
TFIDF_PARAMS: dict = {
    "max_features": 50_000,
    "ngram_range": (1, 2),      # unigrams + bigrams
    "sublinear_tf": True,       # apply log(1 + tf) scaling
    "min_df": 3,                # ignore very rare terms
    "strip_accents": "unicode",
    "analyzer": "word",
}

# Logistic Regression hyper-parameters
LR_PARAMS: dict = {
    "C": 5.0,                   # inverse regularisation strength
    "max_iter": 1_000,
    "solver": "lbfgs",          # lbfgs is single-threaded; n_jobs removed in sk-learn ≥1.8
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_imdb_data() -> tuple[list[str], list[int], list[str], list[int]]:
    """Download and return the IMDb train/test splits.

    Returns:
        train_texts: list of training review strings.
        train_labels: list of binary labels (0 = negative, 1 = positive).
        test_texts:  list of test review strings.
        test_labels: list of binary labels.
    """
    log.info("Loading IMDb dataset from Hugging Face …")
    # Use the fully-qualified dataset name; newer datasets library versions
    # require "namespace/name" format (legacy shorthand "imdb" is deprecated).
    dataset = load_dataset("stanfordnlp/imdb")

    train_texts: list[str] = dataset["train"]["text"]
    train_labels: list[int] = dataset["train"]["label"]
    test_texts: list[str] = dataset["test"]["text"]
    test_labels: list[int] = dataset["test"]["label"]

    log.info(
        "Dataset loaded — train: %d samples, test: %d samples",
        len(train_texts),
        len(test_texts),
    )
    return train_texts, train_labels, test_texts, test_labels


def build_vectorizer() -> TfidfVectorizer:
    """Instantiate and return a configured TfidfVectorizer."""
    return TfidfVectorizer(**TFIDF_PARAMS)


def build_classifier() -> LogisticRegression:
    """Instantiate and return a configured LogisticRegression model."""
    return LogisticRegression(**LR_PARAMS)


def evaluate(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: list[int],
) -> None:
    """Print a detailed evaluation report to stdout.

    Args:
        model:   Trained classifier.
        X_test:  TF-IDF feature matrix for test set.
        y_test:  Ground-truth labels for test set.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    log.info("─" * 50)
    log.info("Evaluation Results")
    log.info("─" * 50)
    log.info("Accuracy : %.4f", accuracy)
    log.info("ROC-AUC  : %.4f", roc_auc)
    log.info("\n%s", classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))


def save_artifacts(vectorizer: TfidfVectorizer, model: LogisticRegression) -> None:
    """Persist the vectorizer and classifier to disk.

    Args:
        vectorizer: Fitted TfidfVectorizer instance.
        model:      Trained LogisticRegression instance.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    log.info("Vectorizer saved → %s", VECTORIZER_PATH)
    log.info("Model saved      → %s", MODEL_PATH)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """End-to-end training pipeline."""
    start = time.perf_counter()

    # 1. Load data
    train_texts, train_labels, test_texts, test_labels = load_imdb_data()

    # 2. Fit vectorizer on training data only (no data leakage)
    log.info("Fitting TF-IDF vectorizer …")
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    log.info("Vocabulary size: %d", len(vectorizer.vocabulary_))

    # 3. Train classifier
    log.info("Training Logistic Regression classifier …")
    model = build_classifier()
    model.fit(X_train, train_labels)

    # 4. Evaluate
    evaluate(model, X_test, test_labels)

    # 5. Save artifacts
    save_artifacts(vectorizer, model)

    elapsed = time.perf_counter() - start
    log.info("Training completed in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
