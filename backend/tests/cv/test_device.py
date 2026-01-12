"""
Tests for CV device management module.

Tests GPU detection, VRAM monitoring, and CUDA OOM handling.

Implements spec: 03-cv-models/00-infrastructure (testing section)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.cv.device import (
    CUDAOOMHandler,
    DeviceInfo,
    VRAMInfo,
    clear_gpu_memory,
    get_available_vram_mb,
    get_device_info,
    get_gpu_memory_summary,
    get_vram_info,
    get_vram_usage,
    select_device,
)

# -----------------------------------------------------------------------------
# VRAMInfo Tests
# -----------------------------------------------------------------------------


class TestVRAMInfo:
    """Tests for VRAMInfo dataclass."""

    def test_vram_info_available(self) -> None:
        """available_mb should compute correctly."""
        info = VRAMInfo(used_mb=4000, total_mb=8000)
        assert info.available_mb == 4000

    def test_vram_info_usage_percent(self) -> None:
        """usage_percent should compute correctly."""
        info = VRAMInfo(used_mb=4000, total_mb=8000)
        assert info.usage_percent == 50.0

    def test_vram_info_zero_total(self) -> None:
        """usage_percent should handle zero total."""
        info = VRAMInfo(used_mb=0, total_mb=0)
        assert info.usage_percent == 0.0


# -----------------------------------------------------------------------------
# DeviceInfo Tests
# -----------------------------------------------------------------------------


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass."""

    def test_device_info_best_device_cuda(self) -> None:
        """best_device should return cuda when available."""
        info = DeviceInfo(
            cuda_available=True,
            cuda_device_count=1,
            cuda_device_name="Test GPU",
            vram_total_mb=8000,
            vram_used_mb=1000,
            vram_available_mb=7000,
            cpu_count=8,
            mps_available=False,
        )
        assert info.best_device == "cuda"

    def test_device_info_best_device_mps(self) -> None:
        """best_device should return mps when cuda not available."""
        info = DeviceInfo(
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_name=None,
            vram_total_mb=0,
            vram_used_mb=0,
            vram_available_mb=0,
            cpu_count=8,
            mps_available=True,
        )
        assert info.best_device == "mps"

    def test_device_info_best_device_cpu(self) -> None:
        """best_device should return cpu as fallback."""
        info = DeviceInfo(
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_name=None,
            vram_total_mb=0,
            vram_used_mb=0,
            vram_available_mb=0,
            cpu_count=8,
            mps_available=False,
        )
        assert info.best_device == "cpu"


# -----------------------------------------------------------------------------
# VRAM Usage Tests
# -----------------------------------------------------------------------------


class TestVRAMUsage:
    """Tests for VRAM usage functions."""

    def test_get_vram_usage_returns_tuple(self) -> None:
        """get_vram_usage should return (used, total) tuple."""
        used, total = get_vram_usage()
        assert isinstance(used, int)
        assert isinstance(total, int)
        assert total >= used

    def test_get_vram_info_returns_object(self) -> None:
        """get_vram_info should return VRAMInfo object."""
        info = get_vram_info()
        assert isinstance(info, VRAMInfo)
        assert info.total_mb >= info.used_mb

    def test_get_available_vram_mb(self) -> None:
        """get_available_vram_mb should return non-negative."""
        available = get_available_vram_mb()
        assert isinstance(available, int)
        assert available >= 0


# -----------------------------------------------------------------------------
# Device Selection Tests
# -----------------------------------------------------------------------------


class TestDeviceSelection:
    """Tests for device selection logic."""

    def test_select_device_cpu_preferred(self) -> None:
        """select_device should return cpu when preferred."""
        device = select_device(preferred="cpu")
        assert device == "cpu"

    @patch("backend.cv.device.get_device_info")
    def test_select_device_auto_with_cuda(self, mock_get_info: MagicMock) -> None:
        """select_device should return cuda when available with auto."""
        mock_get_info.return_value = DeviceInfo(
            cuda_available=True,
            cuda_device_count=1,
            cuda_device_name="Test GPU",
            vram_total_mb=8000,
            vram_used_mb=1000,
            vram_available_mb=7000,
            cpu_count=8,
            mps_available=False,
        )

        device = select_device(required_mb=2000, preferred="auto")
        assert device == "cuda"

    @patch("backend.cv.device.get_device_info")
    def test_select_device_insufficient_vram(self, mock_get_info: MagicMock) -> None:
        """select_device should fall back when insufficient VRAM."""
        mock_get_info.return_value = DeviceInfo(
            cuda_available=True,
            cuda_device_count=1,
            cuda_device_name="Test GPU",
            vram_total_mb=4000,
            vram_used_mb=3500,
            vram_available_mb=500,  # Only 500MB available
            cpu_count=8,
            mps_available=False,
        )

        device = select_device(required_mb=2000, preferred="auto")
        assert device == "cpu"


# -----------------------------------------------------------------------------
# CUDAOOMHandler Tests
# -----------------------------------------------------------------------------


class TestCUDAOOMHandler:
    """Tests for CUDA OOM handler context manager."""

    def test_oom_handler_no_error(self) -> None:
        """Handler should not interfere with normal execution."""
        handler = CUDAOOMHandler()

        with handler:
            result = 1 + 1

        assert result == 2
        assert handler.used_fallback is False

    def test_oom_handler_non_oom_error(self) -> None:
        """Handler should not catch non-OOM errors."""
        handler = CUDAOOMHandler()

        with pytest.raises(ValueError), handler:
            raise ValueError("Not an OOM error")

        assert handler.used_fallback is False

    def test_oom_handler_catches_oom(self) -> None:
        """Handler should catch OOM and call callback."""
        callback_called = False
        callback_device = None

        def callback(device: str) -> None:
            nonlocal callback_called, callback_device
            callback_called = True
            callback_device = device

        handler = CUDAOOMHandler(fallback_device="cpu", callback=callback)

        # Simulate OOM by raising RuntimeError with "out of memory" message
        # The handler checks for this string in the error message
        with handler:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1024 MiB")

        assert handler.used_fallback is True
        assert callback_called is True
        assert callback_device == "cpu"


# -----------------------------------------------------------------------------
# GPU Memory Utilities
# -----------------------------------------------------------------------------


class TestGPUMemoryUtilities:
    """Tests for GPU memory utility functions."""

    def test_clear_gpu_memory_no_error(self) -> None:
        """clear_gpu_memory should not raise without GPU."""
        # Should not raise even without GPU
        clear_gpu_memory()

    def test_get_gpu_memory_summary(self) -> None:
        """get_gpu_memory_summary should return string."""
        summary = get_gpu_memory_summary()
        assert isinstance(summary, str)
        # Should contain either GPU info or "No CUDA GPU"
        assert "GPU" in summary or "CUDA" in summary


# -----------------------------------------------------------------------------
# Device Info Tests
# -----------------------------------------------------------------------------


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_get_device_info_structure(self) -> None:
        """get_device_info should return complete structure."""
        info = get_device_info()

        assert isinstance(info, DeviceInfo)
        assert isinstance(info.cuda_available, bool)
        assert isinstance(info.cuda_device_count, int)
        assert isinstance(info.cpu_count, int)
        assert info.cpu_count > 0
        assert isinstance(info.mps_available, bool)
