"""
test_model_loader.py
--------------------
Unit tests for ``app.model_loader``.

All tests patch the filesystem so they run without real joblib artifacts,
verifying only the loader's logic and error-handling behaviour.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import app.model_loader as loader_module

# ---------------------------------------------------------------------------
# Helpers — reset module-level singletons between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singletons() -> None:
    """Ensure the module-level _vectorizer and _model are None before each test.

    This prevents state leaking between tests that mock the loader differently.
    """
    loader_module._vectorizer = None
    loader_module._model = None
    yield
    loader_module._vectorizer = None
    loader_module._model = None


# ---------------------------------------------------------------------------
# Tests for _load_artifacts
# ---------------------------------------------------------------------------


class TestLoadArtifacts:
    """Unit tests for the internal ``_load_artifacts`` function."""

    def test_raises_file_not_found_when_vectorizer_missing(self, tmp_path: Path) -> None:
        """FileNotFoundError must be raised when the vectorizer file is absent."""
        with (
            patch.object(loader_module, "_VECTORIZER_PATH", tmp_path / "missing.joblib"),
            pytest.raises(FileNotFoundError, match="Vectorizer not found"),
        ):
            loader_module._load_artifacts()

    def test_raises_file_not_found_when_model_missing(self, tmp_path: Path) -> None:
        """FileNotFoundError must be raised when the model file is absent."""
        # Vectorizer exists but model does not.
        vec_path = tmp_path / "tfidf_vectorizer.joblib"
        vec_path.touch()
        with (
            patch.object(loader_module, "_VECTORIZER_PATH", vec_path),
            patch.object(loader_module, "_MODEL_PATH", tmp_path / "missing_model.joblib"),
            pytest.raises(FileNotFoundError, match="Model not found"),
        ):
            loader_module._load_artifacts()

    def test_error_message_contains_path(self, tmp_path: Path) -> None:
        """The FileNotFoundError message must mention the missing file's path."""
        with (
            patch.object(loader_module, "_VECTORIZER_PATH", tmp_path / "missing.joblib"),
            pytest.raises(FileNotFoundError) as exc_info,
        ):
            loader_module._load_artifacts()
        assert str(tmp_path) in str(exc_info.value)

    def test_returns_vectorizer_and_model_on_success(self, tmp_path: Path) -> None:
        """``_load_artifacts`` must return (vectorizer, model) when both files exist."""
        vec_path = tmp_path / "tfidf_vectorizer.joblib"
        model_path = tmp_path / "logistic_regression.joblib"
        vec_path.touch()
        model_path.touch()

        fake_vec = MagicMock(name="vectorizer")
        fake_model = MagicMock(name="model")

        with (
            patch.object(loader_module, "_VECTORIZER_PATH", vec_path),
            patch.object(loader_module, "_MODEL_PATH", model_path),
            patch("app.model_loader.joblib.load", side_effect=[fake_vec, fake_model]),
        ):
            vectorizer, model = loader_module._load_artifacts()

        assert vectorizer is fake_vec
        assert model is fake_model


# ---------------------------------------------------------------------------
# Tests for get_vectorizer
# ---------------------------------------------------------------------------


class TestGetVectorizer:
    """Unit tests for the ``get_vectorizer`` public accessor."""

    def test_returns_vectorizer(self) -> None:
        """``get_vectorizer`` must return the singleton vectorizer."""
        fake_vec = MagicMock(name="vectorizer")
        fake_model = MagicMock(name="model")
        with patch.object(loader_module, "_load_artifacts", return_value=(fake_vec, fake_model)):
            result = loader_module.get_vectorizer()
        assert result is fake_vec

    def test_loads_only_once(self) -> None:
        """``_load_artifacts`` must be called exactly once even across repeated calls."""
        fake_vec = MagicMock(name="vectorizer")
        fake_model = MagicMock(name="model")
        with patch.object(
            loader_module, "_load_artifacts", return_value=(fake_vec, fake_model)
        ) as mock_load:
            loader_module.get_vectorizer()
            loader_module.get_vectorizer()
            loader_module.get_vectorizer()
        mock_load.assert_called_once()

    def test_propagates_file_not_found(self) -> None:
        """``get_vectorizer`` must propagate ``FileNotFoundError`` from ``_load_artifacts``."""
        with (
            patch.object(
                loader_module,
                "_load_artifacts",
                side_effect=FileNotFoundError("missing"),
            ),
            pytest.raises(FileNotFoundError),
        ):
            loader_module.get_vectorizer()


# ---------------------------------------------------------------------------
# Tests for get_model
# ---------------------------------------------------------------------------


class TestGetModel:
    """Unit tests for the ``get_model`` public accessor."""

    def test_returns_model(self) -> None:
        """``get_model`` must return the singleton model."""
        fake_vec = MagicMock(name="vectorizer")
        fake_model = MagicMock(name="model")
        with patch.object(loader_module, "_load_artifacts", return_value=(fake_vec, fake_model)):
            result = loader_module.get_model()
        assert result is fake_model

    def test_loads_only_once(self) -> None:
        """``_load_artifacts`` must be called exactly once across repeated calls."""
        fake_vec = MagicMock(name="vectorizer")
        fake_model = MagicMock(name="model")
        with patch.object(
            loader_module, "_load_artifacts", return_value=(fake_vec, fake_model)
        ) as mock_load:
            loader_module.get_model()
            loader_module.get_model()
        mock_load.assert_called_once()

    def test_propagates_file_not_found(self) -> None:
        """``get_model`` must propagate ``FileNotFoundError`` from ``_load_artifacts``."""
        with (
            patch.object(
                loader_module,
                "_load_artifacts",
                side_effect=FileNotFoundError("missing"),
            ),
            pytest.raises(FileNotFoundError),
        ):
            loader_module.get_model()
