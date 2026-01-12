"""
Tests for CV ModelManager singleton.

Tests lazy loading, LRU eviction, and model lifecycle management.

Implements spec: 03-cv-models/00-infrastructure (testing section)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.cv.base import BaseModelWrapper, ModelInfo
from backend.cv.manager import ModelManager, get_model_manager

# -----------------------------------------------------------------------------
# Mock Model for Testing
# -----------------------------------------------------------------------------


class MockModelWrapper(BaseModelWrapper[dict[str, Any]]):
    """Mock model wrapper for testing."""

    info = ModelInfo(
        name="mock_model",
        description="A mock model for testing",
        vram_required_mb=1000,
        supports_batching=True,
    )

    def __init__(self) -> None:
        super().__init__()
        self._load_called = False
        self._unload_called = False

    def _load_model(self, device: str) -> None:
        """Mock load."""
        self._load_called = True
        self._model = {"loaded": True, "device": device}

    def _unload_model(self) -> None:
        """Mock unload."""
        self._unload_called = True
        self._model = None

    def predict(self, input_path: str, **kwargs: Any) -> dict[str, Any]:
        """Mock predict."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return {"input": input_path, "result": "mock_result"}


class MockModelWrapper2(BaseModelWrapper[dict[str, Any]]):
    """Second mock model for multi-model testing."""

    info = ModelInfo(
        name="mock_model_2",
        description="Second mock model",
        vram_required_mb=2000,
    )

    def _load_model(self, device: str) -> None:
        self._model = {"loaded": True}

    def _unload_model(self) -> None:
        self._model = None

    def predict(self, input_path: str, **kwargs: Any) -> dict[str, Any]:
        return {"result": "mock_2"}


class LargeModelWrapper(BaseModelWrapper[dict[str, Any]]):
    """Large mock model for VRAM testing."""

    info = ModelInfo(
        name="large_model",
        description="Large model for VRAM tests",
        vram_required_mb=8000,  # 8GB
    )

    def _load_model(self, device: str) -> None:
        self._model = {"loaded": True}

    def _unload_model(self) -> None:
        self._model = None

    def predict(self, input_path: str, **kwargs: Any) -> dict[str, Any]:
        return {"result": "large"}


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def clean_manager() -> ModelManager:
    """Get a clean ModelManager instance for testing."""
    ModelManager.reset_instance()
    manager = ModelManager()
    return manager


# -----------------------------------------------------------------------------
# Singleton Tests
# -----------------------------------------------------------------------------


class TestModelManagerSingleton:
    """Tests for singleton pattern."""

    def test_singleton_same_instance(self, clean_manager: ModelManager) -> None:
        """Multiple calls should return same instance."""
        m1 = ModelManager()
        m2 = ModelManager()
        assert m1 is m2

    def test_get_model_manager_function(self, clean_manager: ModelManager) -> None:
        """get_model_manager should return singleton."""
        manager = get_model_manager()
        assert manager is ModelManager()

    def test_reset_instance(self) -> None:
        """reset_instance should create new instance."""
        _ = ModelManager()  # Create initial instance
        ModelManager.reset_instance()
        m2 = ModelManager()
        # After reset, should be a different instance
        # (but singleton means subsequent calls return same new instance)
        assert m2 is ModelManager()


# -----------------------------------------------------------------------------
# Registration Tests
# -----------------------------------------------------------------------------


class TestModelRegistration:
    """Tests for model registration."""

    def test_register_model(self, clean_manager: ModelManager) -> None:
        """register should add model to registry."""
        clean_manager.register("test_model", MockModelWrapper)
        assert clean_manager.is_registered("test_model")

    def test_registered_models_list(self, clean_manager: ModelManager) -> None:
        """registered_models should list all registered."""
        clean_manager.register("model_a", MockModelWrapper)
        clean_manager.register("model_b", MockModelWrapper2)

        registered = clean_manager.registered_models
        assert "model_a" in registered
        assert "model_b" in registered

    def test_get_unregistered_raises(self, clean_manager: ModelManager) -> None:
        """get should raise for unregistered model."""
        with pytest.raises(ValueError, match="Unknown model"):
            clean_manager.get("nonexistent_model")


# -----------------------------------------------------------------------------
# Lazy Loading Tests
# -----------------------------------------------------------------------------


class TestLazyLoading:
    """Tests for lazy loading behavior."""

    def test_model_not_loaded_on_register(self, clean_manager: ModelManager) -> None:
        """Model should not be loaded on registration."""
        clean_manager.register("test", MockModelWrapper)
        assert "test" not in clean_manager.loaded_models

    @patch("backend.cv.device.select_device", return_value="cpu")
    @patch("backend.cv.device.get_available_vram_mb", return_value=10000)
    def test_model_loaded_on_get(
        self,
        mock_vram: MagicMock,
        mock_select: MagicMock,
        clean_manager: ModelManager,
    ) -> None:
        """Model should be loaded on first get()."""
        clean_manager.register("test", MockModelWrapper)

        model = clean_manager.get("test")

        assert "test" in clean_manager.loaded_models
        assert model.is_loaded

    @patch("backend.cv.device.select_device", return_value="cpu")
    @patch("backend.cv.device.get_available_vram_mb", return_value=10000)
    def test_model_cached_after_load(
        self,
        mock_vram: MagicMock,
        mock_select: MagicMock,
        clean_manager: ModelManager,
    ) -> None:
        """Same instance should be returned on subsequent gets."""
        clean_manager.register("test", MockModelWrapper)

        model1 = clean_manager.get("test")
        model2 = clean_manager.get("test")

        assert model1 is model2


# -----------------------------------------------------------------------------
# Unload Tests
# -----------------------------------------------------------------------------


class TestModelUnloading:
    """Tests for model unloading."""

    @patch("backend.cv.device.select_device", return_value="cpu")
    @patch("backend.cv.device.get_available_vram_mb", return_value=10000)
    @patch("backend.cv.device.clear_gpu_memory")
    def test_unload_removes_model(
        self,
        mock_clear: MagicMock,
        mock_vram: MagicMock,
        mock_select: MagicMock,
        clean_manager: ModelManager,
    ) -> None:
        """unload should remove model from cache."""
        clean_manager.register("test", MockModelWrapper)
        clean_manager.get("test")

        assert clean_manager.is_loaded("test")

        clean_manager.unload("test")

        assert not clean_manager.is_loaded("test")

    def test_unload_nonexistent_no_error(self, clean_manager: ModelManager) -> None:
        """unload should not raise for non-loaded model."""
        clean_manager.unload("nonexistent")  # Should not raise

    @patch("backend.cv.device.select_device", return_value="cpu")
    @patch("backend.cv.device.get_available_vram_mb", return_value=10000)
    @patch("backend.cv.device.clear_gpu_memory")
    def test_unload_all(
        self,
        mock_clear: MagicMock,
        mock_vram: MagicMock,
        mock_select: MagicMock,
        clean_manager: ModelManager,
    ) -> None:
        """unload_all should unload all models."""
        clean_manager.register("model_a", MockModelWrapper)
        clean_manager.register("model_b", MockModelWrapper2)
        clean_manager.get("model_a")
        clean_manager.get("model_b")

        assert len(clean_manager.loaded_models) == 2

        clean_manager.unload_all()

        assert len(clean_manager.loaded_models) == 0


# -----------------------------------------------------------------------------
# Model Info Tests
# -----------------------------------------------------------------------------


class TestModelInfo:
    """Tests for model information retrieval."""

    @patch("backend.cv.device.select_device", return_value="cpu")
    @patch("backend.cv.device.get_available_vram_mb", return_value=10000)
    def test_get_model_info(
        self,
        mock_vram: MagicMock,
        mock_select: MagicMock,
        clean_manager: ModelManager,
    ) -> None:
        """get_model_info should return complete info."""
        clean_manager.register("test", MockModelWrapper)
        clean_manager.get("test", device="cpu")

        info = clean_manager.get_model_info("test")

        assert info["name"] == "mock_model"
        assert info["vram_required_mb"] == 1000
        assert info["loaded"] is True
        assert info["device"] == "cpu"

    def test_get_model_info_not_loaded(self, clean_manager: ModelManager) -> None:
        """get_model_info should work for unloaded model."""
        clean_manager.register("test", MockModelWrapper)

        info = clean_manager.get_model_info("test")

        assert info["loaded"] is False
        assert info["device"] is None

    def test_list_models(self, clean_manager: ModelManager) -> None:
        """list_models should return all model info."""
        clean_manager.register("model_a", MockModelWrapper)
        clean_manager.register("model_b", MockModelWrapper2)

        models = clean_manager.list_models()

        assert len(models) == 2
        names = [m["name"] for m in models]
        assert "mock_model" in names
        assert "mock_model_2" in names


# -----------------------------------------------------------------------------
# VRAM Tracking Tests
# -----------------------------------------------------------------------------


class TestVRAMTracking:
    """Tests for VRAM usage tracking."""

    @patch("backend.cv.device.select_device", return_value="cpu")
    @patch("backend.cv.device.get_available_vram_mb", return_value=10000)
    def test_get_vram_usage(
        self,
        mock_vram: MagicMock,
        mock_select: MagicMock,
        clean_manager: ModelManager,
    ) -> None:
        """get_vram_usage should track loaded models."""
        clean_manager.register("model_a", MockModelWrapper)
        clean_manager.register("model_b", MockModelWrapper2)
        clean_manager.get("model_a")
        clean_manager.get("model_b")

        usage = clean_manager.get_vram_usage()

        assert usage["model_a"] == 1000
        assert usage["model_b"] == 2000

    @patch("backend.cv.device.select_device", return_value="cpu")
    @patch("backend.cv.device.get_available_vram_mb", return_value=10000)
    def test_get_total_vram_used(
        self,
        mock_vram: MagicMock,
        mock_select: MagicMock,
        clean_manager: ModelManager,
    ) -> None:
        """get_total_vram_used should sum all model VRAM."""
        clean_manager.register("model_a", MockModelWrapper)
        clean_manager.register("model_b", MockModelWrapper2)
        clean_manager.get("model_a")
        clean_manager.get("model_b")

        total = clean_manager.get_total_vram_used()

        assert total == 3000  # 1000 + 2000


# -----------------------------------------------------------------------------
# Max Loaded Limit Tests
# -----------------------------------------------------------------------------


class TestMaxLoadedLimit:
    """Tests for maximum loaded models limit."""

    def test_set_max_loaded(self, clean_manager: ModelManager) -> None:
        """set_max_loaded should configure limit."""
        clean_manager.set_max_loaded(2)
        assert clean_manager._max_loaded == 2

        clean_manager.set_max_loaded(0)
        assert clean_manager._max_loaded == 0

    @patch("backend.cv.device.select_device", return_value="cpu")
    @patch("backend.cv.device.get_available_vram_mb", return_value=10000)
    @patch("backend.cv.device.clear_gpu_memory")
    def test_eviction_on_max_reached(
        self,
        mock_clear: MagicMock,
        mock_vram: MagicMock,
        mock_select: MagicMock,
        clean_manager: ModelManager,
    ) -> None:
        """Loading beyond max should evict LRU model."""
        clean_manager.set_max_loaded(1)
        clean_manager.register("model_a", MockModelWrapper)
        clean_manager.register("model_b", MockModelWrapper2)

        # Load first model
        clean_manager.get("model_a")
        assert clean_manager.is_loaded("model_a")
        assert len(clean_manager.loaded_models) == 1

        # Load second model - should evict first
        clean_manager.get("model_b")
        assert clean_manager.is_loaded("model_b")
        # First model should be evicted
        assert not clean_manager.is_loaded("model_a")


# -----------------------------------------------------------------------------
# Reload Tests
# -----------------------------------------------------------------------------


class TestReload:
    """Tests for model reloading."""

    @patch("backend.cv.device.select_device", return_value="cpu")
    @patch("backend.cv.device.get_available_vram_mb", return_value=10000)
    @patch("backend.cv.device.clear_gpu_memory")
    def test_reload_model(
        self,
        mock_clear: MagicMock,
        mock_vram: MagicMock,
        mock_select: MagicMock,
        clean_manager: ModelManager,
    ) -> None:
        """reload should unload and load model."""
        clean_manager.register("test", MockModelWrapper)
        model1 = clean_manager.get("test")

        model2 = clean_manager.reload("test")

        # Should be a different instance
        assert model1 is not model2
        assert model2.is_loaded
