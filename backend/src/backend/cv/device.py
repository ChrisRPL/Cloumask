"""
Device management for GPU/CPU detection and VRAM monitoring.

This module provides utilities for detecting available compute devices,
monitoring VRAM usage, and handling CUDA out-of-memory errors gracefully.

Implements spec: 03-cv-models/00-infrastructure
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VRAMInfo:
    """VRAM usage information."""

    used_mb: int
    total_mb: int

    @property
    def available_mb(self) -> int:
        """Available VRAM in megabytes."""
        return self.total_mb - self.used_mb

    @property
    def usage_percent(self) -> float:
        """VRAM usage as percentage [0-100]."""
        if self.total_mb == 0:
            return 0.0
        return (self.used_mb / self.total_mb) * 100


@dataclass
class DeviceInfo:
    """Comprehensive device information."""

    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    vram_total_mb: int
    vram_used_mb: int
    vram_available_mb: int
    cpu_count: int
    mps_available: bool  # Apple Metal Performance Shaders

    @property
    def best_device(self) -> str:
        """Get the best available device string."""
        if self.cuda_available:
            return "cuda"
        if self.mps_available:
            return "mps"
        return "cpu"


def get_vram_usage() -> tuple[int, int]:
    """
    Get current and total VRAM in MB.

    Attempts to use pynvml for accurate readings, falls back to nvidia-smi
    subprocess if pynvml is unavailable.

    Returns:
        Tuple of (used_mb, total_mb). Returns (0, 0) if no GPU detected.
    """
    try:
        return _get_vram_pynvml()
    except (ImportError, ModuleNotFoundError) as e:
        logger.debug("pynvml not installed (%s), trying nvidia-smi", e)
        return _get_vram_nvidia_smi()
    except Exception as e:
        logger.warning("pynvml failed unexpectedly (%s), trying nvidia-smi", e)
        return _get_vram_nvidia_smi()


def _get_vram_pynvml() -> tuple[int, int]:
    """Get VRAM usage via pynvml library."""
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used / 1024 / 1024), int(info.total / 1024 / 1024)
    finally:
        pynvml.nvmlShutdown()


def _get_vram_nvidia_smi() -> tuple[int, int]:
    """
    Fallback VRAM detection via nvidia-smi subprocess.

    Returns:
        Tuple of (used_mb, total_mb). Returns (0, 0) on error.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return 0, 0

        lines = result.stdout.strip().split("\n")
        if not lines:
            return 0, 0

        # Take first GPU
        parts = lines[0].split(",")
        if len(parts) != 2:
            return 0, 0

        used = int(parts[0].strip())
        total = int(parts[1].strip())
        return used, total
    except FileNotFoundError:
        logger.debug("nvidia-smi not found, no GPU detected")
        return 0, 0
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
        return 0, 0
    except (ValueError, IndexError) as e:
        logger.warning("Failed to parse nvidia-smi output: %s", e)
        return 0, 0


def get_vram_info() -> VRAMInfo:
    """
    Get VRAM usage as a structured object.

    Returns:
        VRAMInfo with used, total, and available VRAM.
    """
    used, total = get_vram_usage()
    return VRAMInfo(used_mb=used, total_mb=total)


def get_available_vram_mb() -> int:
    """
    Get available VRAM in MB.

    Returns:
        Available VRAM in megabytes, 0 if no GPU.
    """
    used, total = get_vram_usage()
    return max(0, total - used)


def get_device_info() -> DeviceInfo:
    """
    Get comprehensive device information.

    Detects CUDA availability, GPU name, VRAM, CPU count, and MPS
    (Apple Metal) availability.

    Returns:
        DeviceInfo with all device details.
    """
    # Import torch here to avoid import errors when torch not installed
    cuda_available = False
    cuda_device_count = 0
    cuda_device_name = None
    mps_available = False

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_count = torch.cuda.device_count()
            cuda_device_name = torch.cuda.get_device_name(0)

        # Check for Apple Metal
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        logger.warning("PyTorch not installed, assuming CPU-only")

    vram_used, vram_total = get_vram_usage()
    cpu_count = os.cpu_count() or 1

    return DeviceInfo(
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        vram_total_mb=vram_total,
        vram_used_mb=vram_used,
        vram_available_mb=max(0, vram_total - vram_used),
        cpu_count=cpu_count,
        mps_available=mps_available,
    )


def select_device(required_mb: int = 0, preferred: str = "auto") -> str:
    """
    Select the best available device based on requirements.

    Args:
        required_mb: Minimum VRAM required in MB (0 for any GPU).
        preferred: Preferred device ("auto", "cuda", "mps", "cpu").

    Returns:
        Device string suitable for PyTorch ("cuda", "mps", or "cpu").
    """
    if preferred == "cpu":
        return "cpu"

    info = get_device_info()

    if (preferred == "cuda" or preferred == "auto") and info.cuda_available:
        if required_mb == 0 or info.vram_available_mb >= required_mb:
            return "cuda"
        logger.warning(
            "Insufficient VRAM: need %dMB, have %dMB available",
            required_mb,
            info.vram_available_mb,
        )

    if (preferred == "mps" or preferred == "auto") and info.mps_available:
        return "mps"

    return "cpu"


class CUDAOOMHandler:
    """
    Context manager for handling CUDA out-of-memory errors.

    Catches CUDA OOM errors, clears GPU cache, and optionally
    calls a callback to retry on CPU.

    Example:
        handler = CUDAOOMHandler(callback=lambda dev: model.to(dev))
        with handler:
            result = model(input)
        if handler.used_fallback:
            print("Fell back to CPU due to OOM")
    """

    def __init__(
        self,
        fallback_device: str = "cpu",
        callback: Callable[[str], Any] | None = None,
    ) -> None:
        """
        Initialize OOM handler.

        Args:
            fallback_device: Device to fall back to on OOM.
            callback: Optional callback with fallback device on OOM.
        """
        self.fallback_device = fallback_device
        self.callback = callback
        self.used_fallback = False
        self.error: Exception | None = None

    def __enter__(self) -> CUDAOOMHandler:
        """Enter context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """
        Handle exceptions, catching CUDA OOM.

        Returns:
            True if OOM was handled, False otherwise.
        """
        if exc_type is None:
            return False

        # Check for CUDA OOM error
        is_oom = False

        # First, check for RuntimeError with OOM message (works without torch)
        if isinstance(exc_val, RuntimeError) and "out of memory" in str(exc_val).lower():
            is_oom = True

        # Also check for torch.cuda.OutOfMemoryError if torch is available
        if not is_oom:
            try:
                import torch

                if exc_type is torch.cuda.OutOfMemoryError:
                    is_oom = True
            except ImportError:
                pass

        if is_oom:
            logger.warning("CUDA OOM detected, falling back to %s", self.fallback_device)
            self.used_fallback = True
            self.error = exc_val  # type: ignore[assignment]

            # Clear GPU cache
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

            # Call callback if provided
            if self.callback:
                try:
                    self.callback(self.fallback_device)
                except Exception as e:
                    logger.error("Fallback callback failed: %s", e)

            return True  # Suppress the OOM exception

        return False


def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache.

    Useful for freeing memory after unloading models.
    Safe to call even if no GPU is available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Failed to clear GPU memory: %s", e)


def get_gpu_memory_summary() -> str:
    """
    Get a human-readable GPU memory summary.

    Returns:
        Formatted string with memory usage, or "No GPU" message.
    """
    info = get_device_info()

    if not info.cuda_available:
        return "No CUDA GPU available"

    return (
        f"GPU: {info.cuda_device_name}\n"
        f"VRAM: {info.vram_used_mb}MB / {info.vram_total_mb}MB "
        f"({info.vram_available_mb}MB available)"
    )
