"""
Tests for anonymization pipeline.

Unit tests use mocked models to avoid requiring real model downloads.
Integration tests (marked) require actual models and optionally GPU.

Implements spec: 03-cv-models/06-anonymization (testing section)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from backend.cv.anonymization import (
    AnonymizationConfig,
    AnonymizationPipeline,
    anonymize,
)
from backend.cv.types import (
    BBox,
    FaceDetection,
    FaceDetectionResult,
    Mask,
    PlateDetection,
    PlateDetectionResult,
    SegmentationResult,
)

# -----------------------------------------------------------------------------
# Mock Classes for Unit Testing (without real models)
# -----------------------------------------------------------------------------


class MockFaceDetector:
    """Mock face detector for anonymization tests."""

    info = MagicMock()
    info.vram_required_mb = 1500

    def __init__(self) -> None:
        self._is_loaded = False
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self, device: str = "cpu") -> None:
        self._is_loaded = True
        self._device = device

    def unload(self) -> None:
        self._is_loaded = False

    def predict(
        self,
        input_path: str,
        *,
        confidence: float = 0.5,
        **kwargs: Any,
    ) -> FaceDetectionResult:
        """Return mock face detections."""
        return FaceDetectionResult(
            faces=[
                FaceDetection(
                    bbox=BBox(x=0.25, y=0.2, width=0.15, height=0.2),
                    confidence=0.95,
                    landmarks=None,
                ),
                FaceDetection(
                    bbox=BBox(x=0.6, y=0.3, width=0.12, height=0.18),
                    confidence=0.78,
                    landmarks=None,
                ),
            ],
            image_path=input_path,
            processing_time_ms=5.0,
            model_name="mock-face",
        )


class MockPlateDetector:
    """Mock plate detector for anonymization tests."""

    info = MagicMock()
    info.vram_required_mb = 4500

    def __init__(self) -> None:
        self._is_loaded = False
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self, device: str = "cpu") -> None:
        self._is_loaded = True
        self._device = device

    def unload(self) -> None:
        self._is_loaded = False

    def predict(
        self,
        input_path: str,
        *,
        confidence: float = 0.3,
        **kwargs: Any,
    ) -> PlateDetectionResult:
        """Return mock plate detections."""
        return PlateDetectionResult(
            plates=[
                PlateDetection(
                    bbox=BBox(x=0.5, y=0.7, width=0.2, height=0.08),
                    confidence=0.88,
                    text=None,
                ),
            ],
            image_path=input_path,
            processing_time_ms=10.0,
            model_name="mock-plate",
        )


class MockSAM3:
    """Mock SAM3 for anonymization mask mode tests."""

    info = MagicMock()
    info.vram_required_mb = 8000

    def __init__(self) -> None:
        self._is_loaded = False
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self, device: str = "cpu") -> None:
        self._is_loaded = True
        self._device = device

    def unload(self) -> None:
        self._is_loaded = False

    def predict(
        self,
        input_path: str,
        *,
        box: tuple[int, int, int, int] | None = None,
        **kwargs: Any,
    ) -> SegmentationResult:
        """Return mock segmentation mask for the given box."""
        if box is None:
            return SegmentationResult(
                masks=[],
                image_path=input_path,
                processing_time_ms=300.0,
                model_name="mock-sam3",
            )

        x1, y1, x2, y2 = box
        h, w = y2 - y1, x2 - x1

        # Create a binary mask (ellipse inside the box)
        mask_data = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (w // 2 - 5, h // 2 - 5)
        if axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(mask_data, center, axes, 0, 0, 360, 1, -1)

        return SegmentationResult(
            masks=[Mask.from_numpy(mask_data, confidence=0.92)],
            image_path=input_path,
            processing_time_ms=300.0,
            model_name="mock-sam3",
        )


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a sample test image (640x480 white)."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    # Add some color variations for pixelate/blur testing
    cv2.rectangle(img, (100, 80), (200, 180), (0, 0, 255), -1)  # Red box (face area)
    cv2.rectangle(img, (350, 130), (430, 220), (0, 255, 0), -1)  # Green box (face area)
    cv2.rectangle(img, (256, 310), (384, 350), (255, 0, 0), -1)  # Blue box (plate area)
    path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(path), img)
    return path


# -----------------------------------------------------------------------------
# Configuration Tests
# -----------------------------------------------------------------------------


class TestAnonymizationConfig:
    """Test AnonymizationConfig validation and defaults."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AnonymizationConfig()
        assert config.faces is True
        assert config.plates is True
        assert config.mode == "blur"
        assert config.blur_kernel_size == 51
        assert config.blackbox_color == (0, 0, 0)
        assert config.pixelate_block_size == 10
        assert config.mask_feather_radius == 3
        assert config.face_confidence == 0.5
        assert config.plate_confidence == 0.3

    def test_blur_kernel_must_be_odd(self) -> None:
        """Blur kernel size must be odd."""
        with pytest.raises(ValueError, match="blur_kernel_size must be odd"):
            AnonymizationConfig(blur_kernel_size=50)

    def test_blur_kernel_must_be_positive(self) -> None:
        """Blur kernel size must be >= 1 (and odd, so 0 triggers odd check first)."""
        with pytest.raises(ValueError, match="blur_kernel_size must be odd"):
            AnonymizationConfig(blur_kernel_size=0)

    def test_pixelate_block_must_be_positive(self) -> None:
        """Pixelate block size must be >= 1."""
        with pytest.raises(ValueError, match="pixelate_block_size must be >= 1"):
            AnonymizationConfig(pixelate_block_size=0)

    def test_feather_radius_must_be_non_negative(self) -> None:
        """Mask feather radius must be >= 0."""
        with pytest.raises(ValueError, match="mask_feather_radius must be >= 0"):
            AnonymizationConfig(mask_feather_radius=-1)

    def test_confidence_range(self) -> None:
        """Confidence values must be between 0 and 1."""
        with pytest.raises(ValueError, match="face_confidence must be between"):
            AnonymizationConfig(face_confidence=1.5)

        with pytest.raises(ValueError, match="plate_confidence must be between"):
            AnonymizationConfig(plate_confidence=-0.1)

    def test_valid_custom_config(self) -> None:
        """Test valid custom configuration."""
        config = AnonymizationConfig(
            faces=True,
            plates=False,
            mode="pixelate",
            pixelate_block_size=20,
            face_confidence=0.7,
        )
        assert config.plates is False
        assert config.mode == "pixelate"
        assert config.pixelate_block_size == 20


# -----------------------------------------------------------------------------
# Effect Function Tests
# -----------------------------------------------------------------------------


class TestEffectFunctions:
    """Test individual anonymization effect functions."""

    def test_apply_blur(self, sample_image: Path, tmp_path: Path) -> None:
        """Test Gaussian blur effect."""
        # Create a gradient image for better blur testing
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        roi[:50, :, :] = 255  # Top half white, bottom half black

        pipeline = AnonymizationPipeline(AnonymizationConfig(blur_kernel_size=31))
        blurred = pipeline._apply_blur(roi.copy())

        # Verify shape preserved
        assert blurred.shape == roi.shape
        # Blurred image should have smooth transition (pixels near edge should be gray)
        # Check middle row (row 50) - should be gray after blur
        middle_row_avg = np.mean(blurred[50, :, :])
        assert 50 < middle_row_avg < 200, "Blur should create smooth transition"

    def test_apply_blackbox(self, sample_image: Path) -> None:
        """Test blackbox (solid fill) effect."""
        img = cv2.imread(str(sample_image))
        roi = img[80:180, 100:200].copy()

        config = AnonymizationConfig(blackbox_color=(255, 0, 0))  # RGB red
        pipeline = AnonymizationPipeline(config)
        filled = pipeline._apply_blackbox(roi)

        # Should be solid blue in BGR (OpenCV format)
        assert np.all(filled == [0, 0, 255])  # BGR blue = RGB red

    def test_apply_blackbox_black(self, sample_image: Path) -> None:
        """Test blackbox with default black color."""
        img = cv2.imread(str(sample_image))
        roi = img[80:180, 100:200].copy()

        config = AnonymizationConfig(blackbox_color=(0, 0, 0))
        pipeline = AnonymizationPipeline(config)
        filled = pipeline._apply_blackbox(roi)

        assert np.all(filled == [0, 0, 0])

    def test_apply_pixelate(self, sample_image: Path) -> None:
        """Test pixelate (mosaic) effect."""
        img = cv2.imread(str(sample_image))
        roi = img[80:180, 100:200].copy()

        config = AnonymizationConfig(pixelate_block_size=10)
        pipeline = AnonymizationPipeline(config)
        pixelated = pipeline._apply_pixelate(roi)

        # Shape should remain the same
        assert pixelated.shape == roi.shape

        # Pixelated image should have block structure (less unique colors)
        unique_original = len(np.unique(roi.reshape(-1, 3), axis=0))
        unique_pixelated = len(np.unique(pixelated.reshape(-1, 3), axis=0))
        assert unique_pixelated < unique_original

    def test_apply_effect_dispatch(self, sample_image: Path) -> None:
        """Test _apply_effect dispatches to correct method."""
        img = cv2.imread(str(sample_image))
        roi = img[80:180, 100:200].copy()

        # Test blur dispatch
        pipeline_blur = AnonymizationPipeline(AnonymizationConfig(mode="blur"))
        result_blur = pipeline_blur._apply_effect(roi.copy())
        assert result_blur.shape == roi.shape

        # Test blackbox dispatch
        pipeline_bb = AnonymizationPipeline(AnonymizationConfig(mode="blackbox"))
        result_bb = pipeline_bb._apply_effect(roi.copy())
        assert np.all(result_bb == [0, 0, 0])  # Default black

        # Test pixelate dispatch
        pipeline_px = AnonymizationPipeline(AnonymizationConfig(mode="pixelate"))
        result_px = pipeline_px._apply_effect(roi.copy())
        assert result_px.shape == roi.shape


# -----------------------------------------------------------------------------
# Pipeline Unit Tests (Mocked Models)
# -----------------------------------------------------------------------------


class TestAnonymizationPipelineUnit:
    """Unit tests for AnonymizationPipeline with mocked models."""

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    def test_load_unload_cycle(
        self, mock_plate_factory: MagicMock, mock_face_factory: MagicMock
    ) -> None:
        """Test pipeline load/unload lifecycle."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()

        pipeline = AnonymizationPipeline()
        assert not pipeline.is_loaded

        pipeline.load("cpu")
        assert pipeline.is_loaded
        assert pipeline.device == "cpu"

        pipeline.unload()
        assert not pipeline.is_loaded

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    def test_process_not_loaded_raises(
        self,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        sample_image: Path,
    ) -> None:
        """Process should raise if pipeline not loaded."""
        pipeline = AnonymizationPipeline()

        with pytest.raises(RuntimeError, match="Pipeline not loaded"):
            pipeline.process(str(sample_image))

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    def test_process_blur_mode(
        self,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        sample_image: Path,
        tmp_path: Path,
    ) -> None:
        """Test processing with blur mode."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()

        config = AnonymizationConfig(mode="blur")
        pipeline = AnonymizationPipeline(config)
        pipeline.load("cpu")

        output_path = tmp_path / "output.jpg"
        result = pipeline.process(str(sample_image), str(output_path))

        assert result.faces_anonymized == 2
        assert result.plates_anonymized == 1
        assert result.mode_used == "blur"
        assert output_path.exists()

        pipeline.unload()

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    def test_output_path_auto_generation(
        self,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        sample_image: Path,
    ) -> None:
        """Test automatic output path generation with _anon suffix."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()

        pipeline = AnonymizationPipeline()
        pipeline.load("cpu")

        result = pipeline.process(str(sample_image))  # No output_path

        expected_suffix = "_anon"
        assert expected_suffix in result.output_path
        assert Path(result.output_path).exists()

        pipeline.unload()

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    def test_return_detections_flag(
        self,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        sample_image: Path,
        tmp_path: Path,
    ) -> None:
        """Test return_detections includes detection lists."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()

        pipeline = AnonymizationPipeline()
        pipeline.load("cpu")

        result = pipeline.process(
            str(sample_image),
            str(tmp_path / "out.jpg"),
            return_detections=True,
        )

        assert result.face_detections is not None
        assert len(result.face_detections) == 2
        assert result.plate_detections is not None
        assert len(result.plate_detections) == 1

        pipeline.unload()

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    def test_faces_only_config(
        self,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        sample_image: Path,
        tmp_path: Path,
    ) -> None:
        """Test config with only faces enabled."""
        mock_face_factory.return_value = MockFaceDetector()

        config = AnonymizationConfig(faces=True, plates=False)
        pipeline = AnonymizationPipeline(config)
        pipeline.load("cpu")

        result = pipeline.process(str(sample_image), str(tmp_path / "out.jpg"))

        assert result.faces_anonymized == 2
        assert result.plates_anonymized == 0
        mock_plate_factory.assert_not_called()

        pipeline.unload()

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    def test_plates_only_config(
        self,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        sample_image: Path,
        tmp_path: Path,
    ) -> None:
        """Test config with only plates enabled."""
        mock_plate_factory.return_value = MockPlateDetector()

        config = AnonymizationConfig(faces=False, plates=True)
        pipeline = AnonymizationPipeline(config)
        pipeline.load("cpu")

        result = pipeline.process(str(sample_image), str(tmp_path / "out.jpg"))

        assert result.faces_anonymized == 0
        assert result.plates_anonymized == 1
        mock_face_factory.assert_not_called()

        pipeline.unload()

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    def test_no_modification_outside_detections(
        self,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        sample_image: Path,
        tmp_path: Path,
    ) -> None:
        """Verify regions outside detections are unchanged."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()

        original = cv2.imread(str(sample_image))
        output_path = tmp_path / "output.jpg"

        pipeline = AnonymizationPipeline(AnonymizationConfig(mode="blur"))
        pipeline.load("cpu")
        pipeline.process(str(sample_image), str(output_path))
        pipeline.unload()

        modified = cv2.imread(str(output_path))

        # Check corners (should be unchanged - white background)
        # Top-left 10x10 corner
        assert np.allclose(original[0:10, 0:10], modified[0:10, 0:10], atol=5)


# -----------------------------------------------------------------------------
# Batch Processing Tests
# -----------------------------------------------------------------------------


class TestBatchProcessing:
    """Test batch processing functionality."""

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    def test_batch_progress_callback(
        self,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test progress callback is called correctly."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()

        # Create multiple test images
        images = []
        for i in range(3):
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            path = tmp_path / f"img_{i}.jpg"
            cv2.imwrite(str(path), img)
            images.append(str(path))

        progress_calls: list[tuple[int, int]] = []

        def progress_cb(current: int, total: int) -> None:
            progress_calls.append((current, total))

        pipeline = AnonymizationPipeline()
        pipeline.load("cpu")
        results = pipeline.process_batch(images, progress_callback=progress_cb)
        pipeline.unload()

        assert len(results) == 3
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    def test_batch_error_callback(
        self,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test error callback is called on failures."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()

        # Create one valid image and one invalid path
        valid_img = tmp_path / "valid.jpg"
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(valid_img), img)

        images = [str(valid_img), "/nonexistent/image.jpg"]

        error_calls: list[tuple[str, Exception]] = []

        def error_cb(path: str, err: Exception) -> None:
            error_calls.append((path, err))

        pipeline = AnonymizationPipeline()
        pipeline.load("cpu")
        results = pipeline.process_batch(images, error_callback=error_cb)
        pipeline.unload()

        assert len(results) == 1  # Only valid image succeeded
        assert len(error_calls) == 1
        assert "/nonexistent/image.jpg" in error_calls[0][0]

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    def test_batch_output_directory(
        self,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test batch processing with output directory."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()

        # Create test images in one directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for i in range(2):
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            cv2.imwrite(str(input_dir / f"img_{i}.jpg"), img)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        images = [str(p) for p in input_dir.glob("*.jpg")]

        pipeline = AnonymizationPipeline()
        pipeline.load("cpu")
        results = pipeline.process_batch(images, output_dir=str(output_dir))
        pipeline.unload()

        assert len(results) == 2
        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) == 2


# -----------------------------------------------------------------------------
# Quick API Tests
# -----------------------------------------------------------------------------


class TestQuickAPI:
    """Test the quick anonymize() function."""

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    @patch("backend.cv.anonymization.get_device_info")
    def test_anonymize_quick_api(
        self,
        mock_device_info: MagicMock,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        sample_image: Path,
        tmp_path: Path,
    ) -> None:
        """Test quick API creates and cleans up pipeline."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()
        mock_device_info.return_value = MagicMock(best_device="cpu")

        output_path = tmp_path / "quick_output.jpg"
        result = anonymize(str(sample_image), str(output_path))

        assert result.faces_anonymized == 2
        assert result.plates_anonymized == 1
        assert output_path.exists()

    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    @patch("backend.cv.anonymization.get_device_info")
    def test_anonymize_with_kwargs(
        self,
        mock_device_info: MagicMock,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        sample_image: Path,
        tmp_path: Path,
    ) -> None:
        """Test quick API passes kwargs to config."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()
        mock_device_info.return_value = MagicMock(best_device="cpu")

        output_path = tmp_path / "kwargs_output.jpg"
        result = anonymize(
            str(sample_image),
            str(output_path),
            mode="pixelate",
            pixelate_block_size=15,
        )

        assert result.mode_used == "pixelate"


# -----------------------------------------------------------------------------
# Mask Mode Tests
# -----------------------------------------------------------------------------


class TestMaskMode:
    """Tests for mask mode with SAM3 integration."""

    @patch("backend.cv.segmentation.SAM3Wrapper", MockSAM3)
    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    @patch("backend.cv.anonymization.clear_gpu_memory")
    def test_mask_mode_loads_sam3(
        self,
        mock_clear: MagicMock,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        sample_image: Path,
        tmp_path: Path,
    ) -> None:
        """Test mask mode loads SAM3 for precise boundaries."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()

        config = AnonymizationConfig(mode="mask")
        pipeline = AnonymizationPipeline(config)
        pipeline.load("cpu")

        output_path = tmp_path / "mask_output.jpg"
        result = pipeline.process(str(sample_image), str(output_path))

        assert result.mode_used == "mask"
        assert result.faces_anonymized == 2
        assert result.plates_anonymized == 1
        assert output_path.exists()

        pipeline.unload()

    @patch("backend.cv.segmentation.SAM3Wrapper", MockSAM3)
    @patch("backend.cv.anonymization.get_face_detector")
    @patch("backend.cv.anonymization.get_plate_detector")
    @patch("backend.cv.anonymization.clear_gpu_memory")
    def test_mask_feathering(
        self,
        mock_clear: MagicMock,
        mock_plate_factory: MagicMock,
        mock_face_factory: MagicMock,
        sample_image: Path,
        tmp_path: Path,
    ) -> None:
        """Test mask feathering produces smooth edges."""
        mock_face_factory.return_value = MockFaceDetector()
        mock_plate_factory.return_value = MockPlateDetector()

        config = AnonymizationConfig(mode="mask", mask_feather_radius=5)
        pipeline = AnonymizationPipeline(config)
        pipeline.load("cpu")

        result = pipeline.process(str(sample_image), str(tmp_path / "out.jpg"))
        assert result.mode_used == "mask"

        pipeline.unload()


# -----------------------------------------------------------------------------
# Integration Tests (Real Models Required)
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestAnonymizationIntegration:
    """Integration tests requiring real models."""

    @pytest.fixture
    def real_pipeline(self) -> AnonymizationPipeline:
        """Create real pipeline for integration tests."""
        try:
            config = AnonymizationConfig(faces=True, plates=True, mode="blur")
            pipeline = AnonymizationPipeline(config)
            pipeline.load("cpu")
            yield pipeline
            pipeline.unload()
        except Exception as e:
            pytest.skip(f"Models not available: {e}")

    def test_blur_mode_real(
        self, real_pipeline: AnonymizationPipeline, sample_image: Path, tmp_path: Path
    ) -> None:
        """Integration test with real models - blur mode."""
        output = tmp_path / "real_blur.jpg"
        result = real_pipeline.process(str(sample_image), str(output))

        assert output.exists()
        assert result.processing_time_ms > 0
        assert result.mode_used == "blur"


# -----------------------------------------------------------------------------
# GPU Benchmark Tests
# -----------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.benchmark
class TestAnonymizationBenchmarks:
    """Performance benchmarks (GPU required)."""

    @pytest.fixture
    def gpu_pipeline(self) -> AnonymizationPipeline:
        """Create GPU pipeline for benchmarks."""
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")

            config = AnonymizationConfig(faces=True, plates=True, mode="blur")
            pipeline = AnonymizationPipeline(config)
            pipeline.load("cuda")
            yield pipeline
            pipeline.unload()
        except Exception as e:
            pytest.skip(f"GPU pipeline not available: {e}")

    def test_blur_performance(
        self, gpu_pipeline: AnonymizationPipeline, sample_image: Path, tmp_path: Path
    ) -> None:
        """Blur mode should complete in <50ms per image."""
        output = tmp_path / "perf_blur.jpg"
        result = gpu_pipeline.process(str(sample_image), str(output))

        assert result.processing_time_ms < 50, (
            f"Blur took {result.processing_time_ms:.1f}ms, expected <50ms"
        )

    def test_vram_budget_detection_phase(self) -> None:
        """Detection phase (faces + plates) should use <6GB VRAM."""
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")

            from backend.cv.device import get_vram_usage

            config = AnonymizationConfig(faces=True, plates=True, mode="blur")
            pipeline = AnonymizationPipeline(config)
            pipeline.load("cuda")

            used_mb, _ = get_vram_usage()
            assert used_mb < 7000, f"Detection phase uses {used_mb}MB, expected <7GB"

            pipeline.unload()
        except Exception as e:
            pytest.skip(f"VRAM test not available: {e}")
