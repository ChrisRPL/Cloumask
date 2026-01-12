"""
Model Manager singleton for lazy-loading and managing CV models.

This module provides centralized management of CV model lifecycle,
including registration, lazy loading, LRU-based eviction, and
memory monitoring.

Implements spec: 03-cv-models/00-infrastructure
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.cv.base import BaseModelWrapper

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager for lazy-loading CV models.

    Provides centralized model lifecycle management with:
    - Lazy loading on first access
    - LRU-based eviction when VRAM is full
    - Thread-safe operations
    - Device auto-selection based on VRAM requirements

    Usage:
        manager = ModelManager()
        manager.register("yolo11m", YOLOWrapper)
        model = manager.get("yolo11m")  # Loads on first access
        manager.unload("yolo11m")       # Frees VRAM
    """

    _instance: ModelManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelManager:
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._init_instance()
                    cls._instance = instance
        return cls._instance

    def _init_instance(self) -> None:
        """Initialize instance attributes."""
        # Loaded model instances, ordered by access time (LRU)
        self._models: OrderedDict[str, BaseModelWrapper[Any]] = OrderedDict()
        # Registered model wrapper classes
        self._registry: dict[str, type[BaseModelWrapper[Any]]] = {}
        # Lock for thread-safe operations
        self._models_lock = threading.Lock()
        # Maximum loaded models (0 = unlimited)
        self._max_loaded: int = 0

    def register(
        self,
        name: str,
        wrapper_class: type[BaseModelWrapper[Any]],
    ) -> None:
        """
        Register a model wrapper class.

        Args:
            name: Unique model name for retrieval.
            wrapper_class: The model wrapper class (not instance).

        Raises:
            ValueError: If name already registered with different class.
        """
        if name in self._registry and self._registry[name] is not wrapper_class:
            logger.warning(
                "Overwriting registration for %s: %s -> %s",
                name,
                self._registry[name].__name__,
                wrapper_class.__name__,
            )
        self._registry[name] = wrapper_class
        logger.debug("Registered model: %s (%s)", name, wrapper_class.__name__)

    def get(self, name: str, device: str = "auto") -> BaseModelWrapper[Any]:
        """
        Get or load a model by name.

        Lazy-loads the model if not already loaded. Moves model to
        front of LRU list on access.

        Args:
            name: Registered model name.
            device: Target device ("cuda", "cpu", "mps", or "auto").

        Returns:
            Loaded model wrapper instance.

        Raises:
            ValueError: If model name not registered.
            RuntimeError: If model fails to load.
        """
        with self._models_lock:
            # Return cached model if loaded
            if name in self._models:
                # Move to end for LRU tracking
                self._models.move_to_end(name)
                logger.debug("Returning cached model: %s", name)
                return self._models[name]

            # Check registration
            if name not in self._registry:
                raise ValueError(f"Unknown model: {name}. Register with manager.register() first.")

            # Check VRAM and potentially evict
            wrapper_class = self._registry[name]
            if device == "auto":
                device = self._select_device(wrapper_class)

            # Evict if necessary
            self._ensure_capacity(wrapper_class.info.vram_required_mb)

            # Create and load model
            wrapper = wrapper_class()
            wrapper.load(device)
            self._models[name] = wrapper

            logger.info(
                "Loaded model: %s (device=%s, vram=%dMB, loaded=%d)",
                name,
                device,
                wrapper_class.info.vram_required_mb,
                len(self._models),
            )

            return wrapper

    def unload(self, name: str) -> None:
        """
        Unload a model and free memory.

        Args:
            name: Model name to unload.
        """
        with self._models_lock:
            if name not in self._models:
                logger.debug("Model %s not loaded, nothing to unload", name)
                return

            model = self._models.pop(name)
            model.unload()
            logger.info("Unloaded model: %s (remaining=%d)", name, len(self._models))

    def unload_all(self) -> None:
        """Unload all models and free all GPU memory."""
        with self._models_lock:
            names = list(self._models.keys())

        # Unload outside lock to avoid deadlock
        for name in names:
            self.unload(name)

        logger.info("Unloaded all models")

    def reload(self, name: str, device: str = "auto") -> BaseModelWrapper[Any]:
        """
        Reload a model (unload then load).

        Args:
            name: Model name to reload.
            device: Target device for reloading.

        Returns:
            Reloaded model wrapper.
        """
        self.unload(name)
        return self.get(name, device)

    def is_loaded(self, name: str) -> bool:
        """
        Check if a model is currently loaded.

        Args:
            name: Model name to check.

        Returns:
            True if model is loaded, False otherwise.
        """
        with self._models_lock:
            return name in self._models

    def is_registered(self, name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            name: Model name to check.

        Returns:
            True if model is registered, False otherwise.
        """
        return name in self._registry

    @property
    def loaded_models(self) -> list[str]:
        """List of currently loaded model names (LRU order)."""
        with self._models_lock:
            return list(self._models.keys())

    @property
    def registered_models(self) -> list[str]:
        """List of all registered model names."""
        return list(self._registry.keys())

    def get_model_info(self, name: str) -> dict[str, Any]:
        """
        Get information about a registered model.

        Args:
            name: Model name.

        Returns:
            Dict with model info and status.

        Raises:
            ValueError: If model not registered.
        """
        if name not in self._registry:
            raise ValueError(f"Unknown model: {name}")

        wrapper_class = self._registry[name]
        info = wrapper_class.info

        with self._models_lock:
            loaded = name in self._models
            device = self._models[name].device if loaded else None

        return {
            "name": info.name,
            "description": info.description,
            "vram_required_mb": info.vram_required_mb,
            "supports_batching": info.supports_batching,
            "supports_gpu": info.supports_gpu,
            "source": info.source,
            "version": info.version,
            "loaded": loaded,
            "device": device,
        }

    def list_models(self) -> list[dict[str, Any]]:
        """
        List all registered models with their status.

        Returns:
            List of model info dicts.
        """
        return [self.get_model_info(name) for name in self._registry]

    def set_max_loaded(self, max_models: int) -> None:
        """
        Set maximum number of simultaneously loaded models.

        Args:
            max_models: Maximum models (0 = unlimited).
        """
        self._max_loaded = max_models
        logger.debug("Set max loaded models to %d", max_models)

    def get_vram_usage(self) -> dict[str, int]:
        """
        Get VRAM usage by loaded models.

        Returns:
            Dict mapping model name to VRAM usage in MB.
        """
        with self._models_lock:
            return {
                name: self._registry[name].info.vram_required_mb
                for name in self._models
                if name in self._registry
            }

    def get_total_vram_used(self) -> int:
        """
        Get total VRAM used by all loaded models.

        Returns:
            Total VRAM in MB.
        """
        return sum(self.get_vram_usage().values())

    def _select_device(
        self,
        wrapper_class: type[BaseModelWrapper[Any]],
    ) -> str:
        """
        Select best available device based on VRAM requirements.

        Args:
            wrapper_class: The model wrapper class.

        Returns:
            Device string ("cuda", "mps", or "cpu").
        """
        from backend.cv.device import select_device

        required_mb = wrapper_class.info.vram_required_mb
        return select_device(required_mb=required_mb)

    def _ensure_capacity(self, required_mb: int) -> None:
        """
        Ensure there's capacity for a new model, evicting LRU if needed.

        Args:
            required_mb: VRAM required by the new model.
        """
        from backend.cv.device import get_available_vram_mb

        available = get_available_vram_mb()

        # Check model count limit
        if self._max_loaded > 0 and len(self._models) >= self._max_loaded:
            self._evict_lru(1)

        # Check VRAM if GPU is being used
        if available > 0 and required_mb > available:
            self._evict_for_vram(required_mb)

    def _evict_lru(self, count: int = 1) -> None:
        """
        Evict least recently used models.

        Args:
            count: Number of models to evict.
        """
        with self._models_lock:
            # Get LRU models (first in OrderedDict)
            to_evict = list(self._models.keys())[:count]

        for name in to_evict:
            logger.info("Evicting LRU model: %s", name)
            self.unload(name)

    def _evict_for_vram(self, required_mb: int) -> None:
        """
        Evict models until there's enough VRAM.

        Args:
            required_mb: VRAM needed in MB.
        """
        from backend.cv.device import get_available_vram_mb

        max_attempts = len(self._models)
        attempts = 0

        while get_available_vram_mb() < required_mb and attempts < max_attempts:
            with self._models_lock:
                if not self._models:
                    break
                # Get LRU model (first in OrderedDict)
                lru_name = next(iter(self._models))

            logger.info(
                "Evicting model %s to free VRAM (need %dMB, have %dMB)",
                lru_name,
                required_mb,
                get_available_vram_mb(),
            )
            self.unload(lru_name)
            attempts += 1

            # Small delay to allow GPU memory to be freed
            time.sleep(0.1)

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (for testing).

        Warning: This will lose all registered models and loaded state.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.unload_all()
                cls._instance = None
        logger.debug("ModelManager instance reset")


# Convenience function for module-level access
def get_model_manager() -> ModelManager:
    """Get the ModelManager singleton instance."""
    return ModelManager()
