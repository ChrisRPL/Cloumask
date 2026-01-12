"""
Tests for CV BaseModelWrapper abstract class.

Tests model loading, unloading, prediction, and state management.

Implements spec: 03-cv-models/00-infrastructure (testing section)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.cv.base import BaseModelWrapper, ModelInfo, ModelState

# -----------------------------------------------------------------------------
# Concrete Mock Implementation
# -----------------------------------------------------------------------------


class ConcreteModelWrapper(BaseModelWrapper[dict[str, Any]]):
    """Concrete implementation for testing abstract base."""

    info = ModelInfo(
        name="concrete_model",
        description="A concrete model for testing",
        vram_required_mb=500,
        supports_batching=True,
        supports_gpu=True,
    )

    def __init__(self) -> None:
        super().__init__()
        self.load_device: str | None = None

    def _load_model(self, device: str) -> None:
        """Load model to device."""
        self.load_device = device
        self._model = {"type": "concrete", "device": device}

    def _unload_model(self) -> None:
        """Unload model."""
        self._model = None
        self.load_device = None

    def predict(self, input_path: str, **kwargs: Any) -> dict[str, Any]:
        """Run prediction."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return {
            "input": input_path,
            "device": self.load_device,
            **kwargs,
        }


class FailingModelWrapper(BaseModelWrapper[dict[str, Any]]):
    """Model that fails to load for error testing."""

    info = ModelInfo(name="failing_model", vram_required_mb=100)

    def _load_model(self, device: str) -> None:
        raise RuntimeError("Simulated load failure")

    def _unload_model(self) -> None:
        pass

    def predict(self, input_path: str, **kwargs: Any) -> dict[str, Any]:
        return {}


class SlowPredictWrapper(BaseModelWrapper[dict[str, Any]]):
    """Model with slow predictions for batch testing."""

    info = ModelInfo(name="slow_model", vram_required_mb=100, supports_batching=True)

    def _load_model(self, device: str) -> None:
        self._model = True

    def _unload_model(self) -> None:
        self._model = None

    def predict(self, input_path: str, **kwargs: Any) -> dict[str, Any]:
        return {"path": input_path}


# -----------------------------------------------------------------------------
# ModelInfo Tests
# -----------------------------------------------------------------------------


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_defaults(self) -> None:
        """ModelInfo should have sensible defaults."""
        info = ModelInfo(name="test")
        assert info.name == "test"
        assert info.description == ""
        assert info.vram_required_mb == 0
        assert info.supports_batching is False
        assert info.supports_gpu is True
        assert info.default_device == "auto"

    def test_model_info_custom(self) -> None:
        """ModelInfo should accept custom values."""
        info = ModelInfo(
            name="custom",
            description="Custom model",
            vram_required_mb=4000,
            supports_batching=True,
            version="2.0.0",
            source="huggingface",
        )
        assert info.name == "custom"
        assert info.vram_required_mb == 4000
        assert info.version == "2.0.0"


# -----------------------------------------------------------------------------
# ModelState Tests
# -----------------------------------------------------------------------------


class TestModelState:
    """Tests for ModelState enum."""

    def test_state_values(self) -> None:
        """All expected states should exist."""
        assert ModelState.UNLOADED.value == "unloaded"
        assert ModelState.LOADING.value == "loading"
        assert ModelState.LOADED.value == "loaded"
        assert ModelState.ERROR.value == "error"


# -----------------------------------------------------------------------------
# BaseModelWrapper Init Tests
# -----------------------------------------------------------------------------


class TestModelWrapperInit:
    """Tests for model wrapper initialization."""

    def test_initial_state(self) -> None:
        """Model should start unloaded."""
        model = ConcreteModelWrapper()
        assert model.state == ModelState.UNLOADED
        assert model.is_loaded is False
        assert model.device == "cpu"
        assert model.error is None

    def test_info_accessible(self) -> None:
        """Model info should be accessible."""
        model = ConcreteModelWrapper()
        assert model.info.name == "concrete_model"
        assert model.info.vram_required_mb == 500


# -----------------------------------------------------------------------------
# Load Tests
# -----------------------------------------------------------------------------


class TestModelLoading:
    """Tests for model loading."""

    @patch("backend.cv.device.select_device", return_value="cuda")
    def test_load_auto_device(self, mock_select: MagicMock) -> None:
        """load should auto-select device."""
        model = ConcreteModelWrapper()
        model.load(device="auto")

        assert model.is_loaded
        assert model.device == "cuda"
        mock_select.assert_called_once()

    def test_load_specific_device(self) -> None:
        """load should use specified device."""
        model = ConcreteModelWrapper()
        model.load(device="cpu")

        assert model.is_loaded
        assert model.device == "cpu"
        assert model.load_device == "cpu"

    def test_load_already_loaded_same_device(self) -> None:
        """load should skip if already loaded on same device."""
        model = ConcreteModelWrapper()
        model.load(device="cpu")

        # Should not reload
        model.load(device="cpu")
        assert model.is_loaded

    @patch("backend.cv.device.clear_gpu_memory")
    def test_load_different_device_reloads(self, mock_clear: MagicMock) -> None:
        """load on different device should reload."""
        model = ConcreteModelWrapper()
        model.load(device="cpu")

        model.load(device="cuda")
        assert model.device == "cuda"

    def test_load_failure_sets_error_state(self) -> None:
        """Failed load should set error state."""
        model = FailingModelWrapper()

        with pytest.raises(RuntimeError, match="Failed to load"):
            model.load(device="cpu")

        assert model.state == ModelState.ERROR
        assert model.error is not None
        assert "Simulated load failure" in model.error

    def test_load_time_recorded(self) -> None:
        """load_time_ms should be recorded."""
        model = ConcreteModelWrapper()
        model.load(device="cpu")

        assert model.load_time_ms > 0


# -----------------------------------------------------------------------------
# Unload Tests
# -----------------------------------------------------------------------------


class TestModelUnloading:
    """Tests for model unloading."""

    @patch("backend.cv.device.clear_gpu_memory")
    def test_unload_clears_model(self, mock_clear: MagicMock) -> None:
        """unload should clear model and state."""
        model = ConcreteModelWrapper()
        model.load(device="cpu")

        model.unload()

        assert model.state == ModelState.UNLOADED
        assert model.is_loaded is False
        mock_clear.assert_called()

    def test_unload_when_not_loaded(self) -> None:
        """unload should be safe when not loaded."""
        model = ConcreteModelWrapper()
        model.unload()  # Should not raise
        assert model.state == ModelState.UNLOADED


# -----------------------------------------------------------------------------
# Predict Tests
# -----------------------------------------------------------------------------


class TestModelPredict:
    """Tests for model prediction."""

    def test_predict_success(self) -> None:
        """predict should return results when loaded."""
        model = ConcreteModelWrapper()
        model.load(device="cpu")

        result = model.predict("test.jpg", threshold=0.5)

        assert result["input"] == "test.jpg"
        assert result["device"] == "cpu"
        assert result["threshold"] == 0.5

    def test_predict_not_loaded_raises(self) -> None:
        """predict should raise when not loaded."""
        model = ConcreteModelWrapper()

        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict("test.jpg")


# -----------------------------------------------------------------------------
# Batch Predict Tests
# -----------------------------------------------------------------------------


class TestBatchPredict:
    """Tests for batch prediction."""

    def test_predict_batch_success(self) -> None:
        """predict_batch should process all inputs."""
        model = SlowPredictWrapper()
        model.load(device="cpu")

        inputs = ["a.jpg", "b.jpg", "c.jpg"]
        results = model.predict_batch(inputs)

        assert len(results) == 3
        assert results[0]["path"] == "a.jpg"
        assert results[2]["path"] == "c.jpg"

    def test_predict_batch_with_progress(self) -> None:
        """predict_batch should call progress callback."""
        model = SlowPredictWrapper()
        model.load(device="cpu")

        progress_calls: list[tuple[int, int]] = []

        def callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        inputs = ["a.jpg", "b.jpg", "c.jpg"]
        model.predict_batch(inputs, progress_callback=callback)

        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3)
        assert progress_calls[1] == (2, 3)
        assert progress_calls[2] == (3, 3)

    def test_predict_batch_not_loaded_raises(self) -> None:
        """predict_batch should raise when not loaded."""
        model = SlowPredictWrapper()

        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict_batch(["a.jpg"])


# -----------------------------------------------------------------------------
# Ensure Loaded Tests
# -----------------------------------------------------------------------------


class TestEnsureLoaded:
    """Tests for ensure_loaded helper."""

    def test_ensure_loaded_loads_if_needed(self) -> None:
        """ensure_loaded should load if not loaded."""
        model = ConcreteModelWrapper()
        model.ensure_loaded(device="cpu")

        assert model.is_loaded

    def test_ensure_loaded_skips_if_loaded(self) -> None:
        """ensure_loaded should skip if already loaded."""
        model = ConcreteModelWrapper()
        model.load(device="cpu")

        model.ensure_loaded(device="cuda")  # Should not change device
        assert model.device == "cpu"


# -----------------------------------------------------------------------------
# String Representation Tests
# -----------------------------------------------------------------------------


class TestStringRepresentation:
    """Tests for __repr__."""

    def test_repr_unloaded(self) -> None:
        """repr should show unloaded state."""
        model = ConcreteModelWrapper()
        repr_str = repr(model)

        assert "ConcreteModelWrapper" in repr_str
        assert "concrete_model" in repr_str
        assert "unloaded" in repr_str

    def test_repr_loaded(self) -> None:
        """repr should show loaded state and device."""
        model = ConcreteModelWrapper()
        model.load(device="cuda")
        repr_str = repr(model)

        assert "loaded" in repr_str
        assert "cuda" in repr_str
