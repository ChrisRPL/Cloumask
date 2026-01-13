"""
Base model wrapper abstract class for CV models.

This module defines the abstract interface that all model wrappers must
implement. It provides common functionality for loading, unloading,
and running inference.

Implements spec: 03-cv-models/00-infrastructure
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)


# Type variable for result type
T = TypeVar("T")


class ModelState(str, Enum):
    """Model loading state."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


# Type for progress callback: (current, total) -> None
ProgressCallback = Callable[[int, int], None]


@dataclass
class ModelInfo:
    """Model metadata and resource requirements."""

    name: str
    description: str = ""
    vram_required_mb: int = 0
    supports_batching: bool = False
    supports_gpu: bool = True
    default_device: str = "auto"
    version: str = "1.0.0"
    source: str = ""  # "ultralytics", "huggingface", etc.
    extra: dict[str, Any] = field(default_factory=dict)


class BaseModelWrapper(ABC, Generic[T]):
    """
    Abstract base class for all model wrappers.

    Provides the standard interface for loading, unloading, and running
    inference with CV models. Subclasses must implement the abstract
    methods and define model info.

    Type Parameters:
        T: The result type returned by predict() method.

    Attributes:
        info: Model metadata and resource requirements.

    Example:
        class YOLOWrapper(BaseModelWrapper[DetectionResult]):
            info = ModelInfo(
                name="yolo11m",
                vram_required_mb=2000,
                supports_batching=True,
            )

            def _load_model(self, device: str) -> None:
                self._model = YOLO("yolo11m.pt")
                self._model.to(device)

            def _unload_model(self) -> None:
                del self._model
                self._model = None

            def predict(self, input_path: str, **kwargs) -> DetectionResult:
                results = self._model(input_path)
                return self._convert_results(results)
    """

    # Class-level model info, must be overridden by subclasses
    info: ModelInfo = ModelInfo(name="base_model")

    def __init__(self) -> None:
        """Initialize the model wrapper."""
        self._state: ModelState = ModelState.UNLOADED
        self._device: str = "cpu"
        self._model: Any = None
        self._error: str | None = None
        self._load_time_ms: float = 0.0

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._state == ModelState.LOADED

    @property
    def state(self) -> ModelState:
        """Get current model state."""
        return self._state

    @property
    def device(self) -> str:
        """Get current device model is loaded on."""
        return self._device

    @property
    def load_time_ms(self) -> float:
        """Get time taken to load model in milliseconds."""
        return self._load_time_ms

    @property
    def error(self) -> str | None:
        """Get error message if model is in error state."""
        return self._error

    def load(self, device: str = "auto") -> None:
        """
        Load model weights to device.

        Args:
            device: Target device ("cuda", "cpu", "mps", or "auto").

        Raises:
            RuntimeError: If model fails to load.
        """
        if self._state == ModelState.LOADED:
            if device == self._device or device == "auto":
                logger.debug("Model %s already loaded on %s", self.info.name, self._device)
                return
            # Need to reload on different device
            self.unload()

        self._state = ModelState.LOADING
        self._error = None

        # Resolve auto device
        if device == "auto":
            from backend.cv.device import select_device

            device = select_device(required_mb=self.info.vram_required_mb)

        logger.info("Loading model %s to %s", self.info.name, device)
        start_time = time.perf_counter()

        try:
            self._load_model(device)
            self._device = device
            self._state = ModelState.LOADED
            self._load_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Model %s loaded in %.1fms on %s",
                self.info.name,
                self._load_time_ms,
                device,
            )
        except Exception as e:
            self._state = ModelState.ERROR
            self._error = str(e)
            logger.error("Failed to load model %s: %s", self.info.name, e)
            raise RuntimeError(f"Failed to load model {self.info.name}: {e}") from e

    def unload(self) -> None:
        """
        Unload model and free memory.

        Safe to call even if model is not loaded.
        """
        if self._state == ModelState.UNLOADED:
            return

        logger.info("Unloading model %s from %s", self.info.name, self._device)

        try:
            self._unload_model()
        except Exception as e:
            logger.warning("Error during model unload: %s", e)
        finally:
            self._model = None
            self._state = ModelState.UNLOADED
            self._device = "cpu"

            # Clear GPU memory
            from backend.cv.device import clear_gpu_memory

            clear_gpu_memory()

    @abstractmethod
    def _load_model(self, device: str) -> None:
        """
        Internal method to load model weights.

        Must be implemented by subclasses to perform actual model loading.
        Should set self._model to the loaded model.

        Args:
            device: Target device string.

        Raises:
            Exception: If loading fails.
        """
        ...

    @abstractmethod
    def _unload_model(self) -> None:
        """
        Internal method to unload model.

        Must be implemented by subclasses to perform cleanup.
        Should handle cleanup of self._model.
        """
        ...

    @abstractmethod
    def predict(self, input_path: str, **kwargs: Any) -> T:
        """
        Run inference on single input.

        Must be implemented by subclasses to perform actual inference.

        Args:
            input_path: Path to input file (image, video, point cloud).
            **kwargs: Model-specific parameters.

        Returns:
            Model-specific result type.

        Raises:
            RuntimeError: If model not loaded or inference fails.
        """
        ...

    def predict_batch(
        self,
        input_paths: list[str],
        progress_callback: ProgressCallback | None = None,
        **kwargs: Any,
    ) -> list[T]:
        """
        Run inference on batch of inputs.

        Default implementation iterates over inputs. Subclasses with
        native batch support can override for better performance.

        Args:
            input_paths: List of paths to input files.
            progress_callback: Optional callback for progress updates.
            **kwargs: Model-specific parameters.

        Returns:
            List of results, one per input.

        Raises:
            RuntimeError: If model not loaded or inference fails.
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.info.name} not loaded")

        results: list[T] = []
        total = len(input_paths)

        for i, path in enumerate(input_paths):
            result = self.predict(path, **kwargs)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def ensure_loaded(self, device: str = "auto") -> None:
        """
        Ensure model is loaded, loading if necessary.

        Args:
            device: Target device if loading is needed.

        Raises:
            RuntimeError: If model fails to load.
        """
        if not self.is_loaded:
            self.load(device)

    def __repr__(self) -> str:
        """String representation of model wrapper."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.info.name!r}, "
            f"state={self._state.value!r}, "
            f"device={self._device!r})"
        )


def register_model(cls: type[BaseModelWrapper[Any]]) -> type[BaseModelWrapper[Any]]:
    """
    Decorator to register a model wrapper with the ModelManager.

    Usage:
        @register_model
        class YOLOWrapper(BaseModelWrapper[DetectionResult]):
            info = ModelInfo(name="yolo11m", ...)
            ...

    This registers the wrapper class so it can be retrieved by name
    via ModelManager.get("yolo11m").

    Args:
        cls: Model wrapper class to register.

    Returns:
        The same class (unchanged).
    """
    # Import here to avoid circular imports
    from backend.cv.manager import ModelManager

    manager = ModelManager()
    manager.register(cls.info.name, cls)
    logger.debug("Registered model wrapper: %s", cls.info.name)
    return cls
