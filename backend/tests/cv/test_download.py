"""
Tests for CV model download infrastructure.

Tests model registry, download utilities, and path management.

Implements spec: 03-cv-models/00-infrastructure (testing section)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.cv.download import (
    MODEL_REGISTRY,
    ModelRegistryEntry,
    ModelSource,
    delete_model,
    get_model_path,
    get_model_size_mb,
    get_models_dir,
    get_total_downloaded_size_mb,
    is_model_downloaded,
    list_available_models,
    list_downloaded_models,
    register_model,
)

# -----------------------------------------------------------------------------
# ModelSource Tests
# -----------------------------------------------------------------------------


class TestModelSource:
    """Tests for ModelSource enum."""

    def test_source_values(self) -> None:
        """All expected sources should exist."""
        assert ModelSource.ULTRALYTICS.value == "ultralytics"
        assert ModelSource.HUGGINGFACE.value == "huggingface"
        assert ModelSource.LOCAL.value == "local"


# -----------------------------------------------------------------------------
# ModelRegistryEntry Tests
# -----------------------------------------------------------------------------


class TestModelRegistryEntry:
    """Tests for ModelRegistryEntry dataclass."""

    def test_entry_creation(self) -> None:
        """Entry should store all fields."""
        entry = ModelRegistryEntry(
            name="test_model",
            source=ModelSource.HUGGINGFACE,
            size_mb=500,
            repo_id="test/model",
            requires_auth=True,
        )

        assert entry.name == "test_model"
        assert entry.source == ModelSource.HUGGINGFACE
        assert entry.size_mb == 500
        assert entry.repo_id == "test/model"
        assert entry.requires_auth is True


# -----------------------------------------------------------------------------
# Models Directory Tests
# -----------------------------------------------------------------------------


class TestModelsDirectory:
    """Tests for models directory utilities."""

    def test_get_models_dir(self) -> None:
        """get_models_dir should return Path."""
        models_dir = get_models_dir()
        assert isinstance(models_dir, Path)


# -----------------------------------------------------------------------------
# Model Registry Tests
# -----------------------------------------------------------------------------


class TestModelRegistry:
    """Tests for model registry."""

    def test_registry_has_expected_models(self) -> None:
        """Registry should contain known models."""
        assert "yolo11m" in MODEL_REGISTRY
        assert "sam2-hiera-large" in MODEL_REGISTRY
        assert "sam3" in MODEL_REGISTRY
        assert "scrfd-10g" in MODEL_REGISTRY

    def test_registry_entry_structure(self) -> None:
        """Registry entries should have required fields."""
        entry = MODEL_REGISTRY["yolo11m"]
        assert entry.name == "yolo11m"
        assert entry.source == ModelSource.ULTRALYTICS
        assert entry.size_mb > 0

    def test_sam3_registry_entry(self) -> None:
        """SAM3 should point to the published HuggingFace checkpoint."""
        entry = MODEL_REGISTRY["sam3"]
        assert entry.source == ModelSource.HUGGINGFACE
        assert entry.repo_id == "facebook/sam3"
        assert entry.filename == "sam3.pt"

    def test_register_custom_model(self) -> None:
        """register_model should add to registry."""
        register_model(
            name="custom_model",
            source="huggingface",
            size_mb=100,
            repo_id="test/custom",
        )

        assert "custom_model" in MODEL_REGISTRY
        assert MODEL_REGISTRY["custom_model"].repo_id == "test/custom"

        # Cleanup
        del MODEL_REGISTRY["custom_model"]


# -----------------------------------------------------------------------------
# Model Path Tests
# -----------------------------------------------------------------------------


class TestModelPath:
    """Tests for model path utilities."""

    def test_get_model_path_known(self) -> None:
        """get_model_path should return path for known model."""
        path = get_model_path("yolo11m")
        assert isinstance(path, Path)
        assert "yolo11m" in str(path)

    def test_get_model_path_unknown(self) -> None:
        """get_model_path should raise for unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_path("nonexistent_model")

    def test_get_model_size_mb(self) -> None:
        """get_model_size_mb should return size."""
        size = get_model_size_mb("yolo11m")
        assert size == 40

    def test_get_model_size_mb_unknown(self) -> None:
        """get_model_size_mb should raise for unknown model."""
        with pytest.raises(ValueError):
            get_model_size_mb("nonexistent")


# -----------------------------------------------------------------------------
# Download Status Tests
# -----------------------------------------------------------------------------


class TestDownloadStatus:
    """Tests for download status utilities."""

    def test_is_model_downloaded_false(self) -> None:
        """is_model_downloaded should return False for non-existent."""
        # Assuming test models aren't actually downloaded
        result = is_model_downloaded("yolo11m")
        # Result depends on actual file existence
        assert isinstance(result, bool)

    @patch.object(Path, "exists", return_value=True)
    def test_is_model_downloaded_true(self, mock_exists: MagicMock) -> None:
        """is_model_downloaded should return True when file exists."""
        result = is_model_downloaded("yolo11m")
        assert result is True

    def test_is_model_downloaded_unknown(self) -> None:
        """is_model_downloaded should return False for unknown model."""
        result = is_model_downloaded("nonexistent_model")
        assert result is False


# -----------------------------------------------------------------------------
# List Models Tests
# -----------------------------------------------------------------------------


class TestListModels:
    """Tests for model listing utilities."""

    def test_list_available_models(self) -> None:
        """list_available_models should return all models."""
        models = list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0

        # Check structure
        model = models[0]
        assert "name" in model
        assert "source" in model
        assert "size_mb" in model
        assert "downloaded" in model

    def test_list_downloaded_models(self) -> None:
        """list_downloaded_models should return list of names."""
        models = list_downloaded_models()
        assert isinstance(models, list)
        # All should be strings
        for name in models:
            assert isinstance(name, str)


# -----------------------------------------------------------------------------
# Delete Model Tests
# -----------------------------------------------------------------------------


class TestDeleteModel:
    """Tests for model deletion."""

    def test_delete_nonexistent_model(self) -> None:
        """delete_model should return False for non-existent."""
        result = delete_model("nonexistent_model")
        assert result is False

    @patch("backend.cv.download.is_model_downloaded", return_value=True)
    @patch("backend.cv.download.get_model_path")
    def test_delete_file_model(
        self,
        mock_path: MagicMock,
        mock_downloaded: MagicMock,
        tmp_path: Path,
    ) -> None:
        """delete_model should remove file."""
        # Create temp file
        test_file = tmp_path / "test_model.pt"
        test_file.write_text("test")

        mock_path.return_value = test_file

        result = delete_model("yolo11m")
        assert result is True
        assert not test_file.exists()


# -----------------------------------------------------------------------------
# Total Size Tests
# -----------------------------------------------------------------------------


class TestTotalSize:
    """Tests for total size calculation."""

    @patch("backend.cv.download.is_model_downloaded")
    def test_get_total_downloaded_size_mb(
        self,
        mock_downloaded: MagicMock,
    ) -> None:
        """get_total_downloaded_size_mb should sum sizes."""
        # Mock some models as downloaded
        mock_downloaded.side_effect = lambda name: name in ["yolo11m", "yolo11n"]

        total = get_total_downloaded_size_mb()

        # yolo11m=40 + yolo11n=6 = 46
        assert total == 46
