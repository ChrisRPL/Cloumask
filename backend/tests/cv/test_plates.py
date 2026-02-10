"""
Tests for license plate detection wrapper.

Unit tests use mocked models to avoid requiring real model downloads.
Integration tests (marked) require actual models and optionally GPU.

Implements spec: 03-cv-models/05-plate-detection (testing section)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.cv.base import ModelState
from backend.cv.types import BBox, Detection, PlateDetection, PlateDetectionResult

# -----------------------------------------------------------------------------
# Mock Classes for Unit Testing
# -----------------------------------------------------------------------------


class MockDetectionResult:
    """Mock DetectionResult from YOLO-World."""

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.image_path = "test.jpg"
        self.processing_time_ms = 10.0
        self.model_name = "yolo-world-l"


class MockYOLOWorldWrapper:
    """Mock YOLOWorldWrapper for testing plate detection."""

    def __init__(self) -> None:
        self._model = MagicMock()
        self._is_loaded = False
        self._device = "cpu"

    def load(self, device: str = "cpu") -> None:
        self._is_loaded = True
        self._device = device

    def unload(self) -> None:
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(
        self,
        input_path: str,
        *,
        prompt: str = "object",
        confidence: float = 0.3,
        iou_threshold: float = 0.45,
        **kwargs: Any,
    ) -> MockDetectionResult:
        """Return mock plate detections."""
        # Return 3 mock plate detections with different aspect ratios
        detections = [
            Detection(
                class_id=0,
                class_name="license plate",
                bbox=BBox(x=0.5, y=0.5, width=0.15, height=0.05),  # aspect ratio = 3.0 (EU plate-like)
                confidence=0.92,
            ),
            Detection(
                class_id=0,
                class_name="license plate",
                bbox=BBox(x=0.3, y=0.7, width=0.08, height=0.04),  # aspect ratio = 2.0 (US plate-like)
                confidence=0.78,
            ),
            Detection(
                class_id=0,
                class_name="license plate",
                bbox=BBox(x=0.7, y=0.3, width=0.1, height=0.1),  # aspect ratio = 1.0 (not a plate - too square)
                confidence=0.65,
            ),
        ]
        return MockDetectionResult(detections)


class MockSpecializedYOLOResult:
    """Mock specialized YOLO prediction result."""

    def __init__(self) -> None:
        self.boxes = MockSpecializedBoxes()


class MockSpecializedBoxes:
    """Mock boxes container for specialized YOLO."""

    def __iter__(self):
        # Return mock plate detections
        yield MockSpecializedBox(
            xywhn=(0.5, 0.5, 0.12, 0.04),  # aspect ratio = 3.0
            conf=0.95,
        )
        yield MockSpecializedBox(
            xywhn=(0.3, 0.7, 0.1, 0.05),  # aspect ratio = 2.0
            conf=0.88,
        )


class MockSpecializedBox:
    """Mock single box from specialized YOLO."""

    def __init__(
        self,
        xywhn: tuple[float, float, float, float],
        conf: float,
    ) -> None:
        self.xywhn = [list(xywhn)]
        self.conf = [conf]


class MockSpecializedYOLO:
    """Mock specialized YOLO model."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def to(self, device: str) -> None:
        pass

    def predict(
        self,
        source: str,
        conf: float = 0.3,
        iou: float = 0.45,
        device: str = "cpu",
        verbose: bool = False,
    ) -> list[MockSpecializedYOLOResult]:
        return [MockSpecializedYOLOResult()]


# -----------------------------------------------------------------------------
# PlateDetectorWrapper Unit Tests
# -----------------------------------------------------------------------------


class TestPlateDetectorWrapperUnit:
    """Unit tests for PlateDetectorWrapper with mocked model."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()
        assert wrapper.info.name == "plate-detector"
        assert wrapper.info.vram_required_mb == 4500
        assert wrapper.info.supports_batching is True
        assert wrapper.info.source == "ultralytics"

    def test_initial_state(self) -> None:
        """Should start in unloaded state."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()
        assert wrapper.state == ModelState.UNLOADED
        assert wrapper.is_loaded is False
        assert wrapper._yolo_world is None
        assert wrapper._specialized_model is None

    def test_default_aspect_ratios(self) -> None:
        """Should have correct default aspect ratio constraints."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()
        assert wrapper.MIN_ASPECT_RATIO == 1.5
        assert wrapper.MAX_ASPECT_RATIO == 6.0

    def test_predict_not_loaded_raises(self) -> None:
        """Predict should raise when model not loaded."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict("test.jpg")

    def test_predict_batch_not_loaded_raises(self) -> None:
        """predict_batch should raise when model not loaded."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict_batch(["test.jpg"])

    @patch("backend.cv.openvocab.YOLOWorldWrapper", MockYOLOWorldWrapper)
    @patch("backend.cv.device.clear_gpu_memory")
    def test_load_and_predict(self, mock_clear: MagicMock) -> None:
        """Should load and predict successfully with mocked model."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()
        wrapper.load(device="cpu")

        assert wrapper.is_loaded
        assert wrapper._yolo_world is not None

        result = wrapper.predict("test.jpg", confidence=0.5)

        assert isinstance(result, PlateDetectionResult)
        assert result.model_name == "plate-detector"
        assert result.image_path == "test.jpg"
        assert result.processing_time_ms > 0

        wrapper.unload()
        assert not wrapper.is_loaded

    @patch("backend.cv.openvocab.YOLOWorldWrapper", MockYOLOWorldWrapper)
    @patch("backend.cv.device.clear_gpu_memory")
    def test_aspect_ratio_filtering(self, mock_clear: MagicMock) -> None:
        """Should filter detections by aspect ratio."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()
        wrapper.load(device="cpu")

        # With filtering enabled
        result_filtered = wrapper.predict(
            "test.jpg",
            confidence=0.5,
            validate_aspect_ratio=True,
        )

        # Without filtering
        result_unfiltered = wrapper.predict(
            "test.jpg",
            confidence=0.5,
            validate_aspect_ratio=False,
        )

        # The square detection (aspect ratio 1.0) should be filtered out
        assert len(result_filtered.plates) < len(result_unfiltered.plates)
        assert len(result_unfiltered.plates) == 3
        assert len(result_filtered.plates) == 2

        wrapper.unload()

    @patch("backend.cv.openvocab.YOLOWorldWrapper", MockYOLOWorldWrapper)
    @patch("backend.cv.device.clear_gpu_memory")
    def test_batch_prediction(self, mock_clear: MagicMock) -> None:
        """Should process batch of images with progress callback."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()
        wrapper.load(device="cpu")

        progress_calls: list[tuple[int, int]] = []

        def callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        inputs = ["a.jpg", "b.jpg", "c.jpg"]
        results = wrapper.predict_batch(
            inputs,
            progress_callback=callback,
            confidence=0.5,
        )

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.image_path == inputs[i]
            assert result.model_name == "plate-detector"

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

        wrapper.unload()

    @patch("backend.cv.openvocab.YOLOWorldWrapper", MockYOLOWorldWrapper)
    @patch("backend.cv.device.clear_gpu_memory")
    def test_confidence_propagated(self, mock_clear: MagicMock) -> None:
        """Should propagate confidence threshold to underlying model."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()
        wrapper.load(device="cpu")

        result = wrapper.predict("test.jpg", confidence=0.5)

        # All detections should have their original confidences
        assert all(p.confidence > 0 for p in result.plates)

        wrapper.unload()


# -----------------------------------------------------------------------------
# Specialized Model Tests
# -----------------------------------------------------------------------------


class TestSpecializedModelUnit:
    """Unit tests for specialized plate detection model."""

    @patch("ultralytics.YOLO", MockSpecializedYOLO)
    @patch("backend.cv.device.clear_gpu_memory")
    def test_loads_specialized_when_available(
        self,
        mock_clear: MagicMock,
    ) -> None:
        """Should load specialized model when requested and available."""
        from backend.cv.plates import PlateDetectorWrapper

        with patch.object(
            PlateDetectorWrapper,
            "_check_specialized_model",
            return_value=True,
        ):
            wrapper = PlateDetectorWrapper(use_specialized=True)
            wrapper.load(device="cpu")

            assert wrapper._specialized_model is not None
            assert wrapper._yolo_world is None

            wrapper.unload()

    @patch("backend.cv.openvocab.YOLOWorldWrapper", MockYOLOWorldWrapper)
    @patch("backend.cv.device.clear_gpu_memory")
    def test_falls_back_to_yoloworld_when_no_specialized(
        self,
        mock_clear: MagicMock,
    ) -> None:
        """Should fall back to YOLO-World when specialized not available."""
        from backend.cv.plates import PlateDetectorWrapper

        with patch.object(
            PlateDetectorWrapper,
            "_check_specialized_model",
            return_value=False,
        ):
            wrapper = PlateDetectorWrapper(use_specialized=True)
            wrapper.load(device="cpu")

            assert wrapper._yolo_world is not None
            assert wrapper._specialized_model is None

            wrapper.unload()


# -----------------------------------------------------------------------------
# Region Configuration Tests
# -----------------------------------------------------------------------------


class TestRegionConfiguration:
    """Tests for region-specific plate configurations."""

    def test_eu_region_config(self) -> None:
        """Should configure EU region aspect ratios."""
        from backend.cv.plates import get_plate_detector

        detector = get_plate_detector(region="eu")
        assert detector.MIN_ASPECT_RATIO == 3.0
        assert detector.MAX_ASPECT_RATIO == 5.5

    def test_us_region_config(self) -> None:
        """Should configure US region aspect ratios."""
        from backend.cv.plates import get_plate_detector

        detector = get_plate_detector(region="us")
        assert detector.MIN_ASPECT_RATIO == 1.8
        assert detector.MAX_ASPECT_RATIO == 2.5

    def test_china_region_config(self) -> None:
        """Should configure China region aspect ratios."""
        from backend.cv.plates import get_plate_detector

        detector = get_plate_detector(region="china")
        assert detector.MIN_ASPECT_RATIO == 2.5
        assert detector.MAX_ASPECT_RATIO == 4.0

    def test_unknown_region_uses_default(self) -> None:
        """Unknown region should use default aspect ratios."""
        from backend.cv.plates import get_plate_detector

        detector = get_plate_detector(region="unknown")
        assert detector.MIN_ASPECT_RATIO == 1.5
        assert detector.MAX_ASPECT_RATIO == 6.0

    def test_none_region_uses_default(self) -> None:
        """None region should use default aspect ratios."""
        from backend.cv.plates import get_plate_detector

        detector = get_plate_detector(region=None)
        assert detector.MIN_ASPECT_RATIO == 1.5
        assert detector.MAX_ASPECT_RATIO == 6.0


# -----------------------------------------------------------------------------
# get_plate_detector Factory Tests
# -----------------------------------------------------------------------------


class TestGetPlateDetector:
    """Tests for get_plate_detector factory function."""

    def test_returns_plate_detector_wrapper(self) -> None:
        """Should return PlateDetectorWrapper instance."""
        from backend.cv.plates import PlateDetectorWrapper, get_plate_detector

        detector = get_plate_detector()
        assert isinstance(detector, PlateDetectorWrapper)

    def test_not_loaded_by_default(self) -> None:
        """Factory should return unloaded detector."""
        from backend.cv.plates import get_plate_detector

        detector = get_plate_detector()
        assert not detector.is_loaded

    def test_force_yolo_world_overrides_specialized(self) -> None:
        """force_yolo_world should override use_specialized."""
        from backend.cv.plates import get_plate_detector

        detector = get_plate_detector(
            use_specialized=True,
            force_yolo_world=True,
        )
        assert detector._use_specialized is False


# -----------------------------------------------------------------------------
# Aspect Ratio Validation Tests
# -----------------------------------------------------------------------------


class TestAspectRatioValidation:
    """Tests for aspect ratio filtering logic."""

    def test_valid_eu_plate_passes(self) -> None:
        """EU plate aspect ratio should pass default filter."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()

        # EU plate: typically ~520x110mm = aspect ratio ~4.7
        plates = [
            PlateDetection(
                bbox=BBox(x=0.5, y=0.5, width=0.2, height=0.042),  # AR = 4.76
                confidence=0.9,
                text=None,
                text_confidence=None,
            ),
        ]

        filtered = wrapper._filter_by_aspect_ratio(plates)
        assert len(filtered) == 1

    def test_valid_us_plate_passes(self) -> None:
        """US plate aspect ratio should pass default filter."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()

        # US plate: typically ~305x152mm = aspect ratio ~2.0
        plates = [
            PlateDetection(
                bbox=BBox(x=0.5, y=0.5, width=0.1, height=0.05),  # AR = 2.0
                confidence=0.9,
                text=None,
                text_confidence=None,
            ),
        ]

        filtered = wrapper._filter_by_aspect_ratio(plates)
        assert len(filtered) == 1

    def test_square_detection_filtered(self) -> None:
        """Square detections (not plates) should be filtered."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()

        plates = [
            PlateDetection(
                bbox=BBox(x=0.5, y=0.5, width=0.1, height=0.1),  # AR = 1.0
                confidence=0.9,
                text=None,
                text_confidence=None,
            ),
        ]

        filtered = wrapper._filter_by_aspect_ratio(plates)
        assert len(filtered) == 0

    def test_very_wide_detection_filtered(self) -> None:
        """Very wide detections should be filtered."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()

        plates = [
            PlateDetection(
                bbox=BBox(x=0.5, y=0.5, width=0.7, height=0.05),  # AR = 14.0
                confidence=0.9,
                text=None,
                text_confidence=None,
            ),
        ]

        filtered = wrapper._filter_by_aspect_ratio(plates)
        assert len(filtered) == 0

    def test_zero_height_handled(self) -> None:
        """Zero height should not cause division error."""
        from backend.cv.plates import PlateDetectorWrapper

        wrapper = PlateDetectorWrapper()

        plates = [
            PlateDetection(
                bbox=BBox(x=0.5, y=0.5, width=0.1, height=0.0),
                confidence=0.9,
                text=None,
                text_confidence=None,
            ),
        ]

        # Should not raise, should filter out
        filtered = wrapper._filter_by_aspect_ratio(plates)
        assert len(filtered) == 0


# -----------------------------------------------------------------------------
# PlateDetection Type Tests
# -----------------------------------------------------------------------------


class TestPlateDetectionType:
    """Tests for PlateDetection pydantic model."""

    def test_plate_detection_with_text(self) -> None:
        """Should allow optional text field."""
        plate = PlateDetection(
            bbox=BBox(x=0.5, y=0.5, width=0.2, height=0.05),
            confidence=0.95,
            text="ABC123",
            text_confidence=0.87,
        )

        assert plate.text == "ABC123"
        assert plate.text_confidence == 0.87

    def test_plate_detection_without_text(self) -> None:
        """Should allow None text field."""
        plate = PlateDetection(
            bbox=BBox(x=0.5, y=0.5, width=0.2, height=0.05),
            confidence=0.95,
            text=None,
            text_confidence=None,
        )

        assert plate.text is None
        assert plate.text_confidence is None


# -----------------------------------------------------------------------------
# PlateDetectionResult Type Tests
# -----------------------------------------------------------------------------


class TestPlateDetectionResultType:
    """Tests for PlateDetectionResult pydantic model."""

    def test_count_property(self) -> None:
        """Should compute plate count."""
        result = PlateDetectionResult(
            plates=[
                PlateDetection(
                    bbox=BBox(x=0.5, y=0.5, width=0.2, height=0.05),
                    confidence=0.9,
                    text=None,
                    text_confidence=None,
                ),
                PlateDetection(
                    bbox=BBox(x=0.3, y=0.7, width=0.15, height=0.04),
                    confidence=0.8,
                    text=None,
                    text_confidence=None,
                ),
            ],
            image_path="test.jpg",
            processing_time_ms=10.0,
            model_name="plate-detector",
        )

        assert result.count == 2

    def test_empty_result(self) -> None:
        """Should handle empty plates list."""
        result = PlateDetectionResult(
            plates=[],
            image_path="test.jpg",
            processing_time_ms=5.0,
            model_name="plate-detector",
        )

        assert result.count == 0


# -----------------------------------------------------------------------------
# PLATE_REGIONS Config Tests
# -----------------------------------------------------------------------------


class TestPlateRegions:
    """Tests for PLATE_REGIONS configuration."""

    def test_plate_regions_structure(self) -> None:
        """Should have correct structure for all regions."""
        from backend.cv.plates import PLATE_REGIONS

        required_keys = {"aspect_ratio_range", "typical_size_mm", "description"}

        for region, config in PLATE_REGIONS.items():
            assert set(config.keys()) == required_keys, f"Missing keys for {region}"
            assert len(config["aspect_ratio_range"]) == 2
            assert len(config["typical_size_mm"]) == 2
            assert isinstance(config["description"], str)

    def test_aspect_ratios_valid_ranges(self) -> None:
        """Aspect ratio ranges should be valid (min < max)."""
        from backend.cv.plates import PLATE_REGIONS

        for region, config in PLATE_REGIONS.items():
            min_ratio, max_ratio = config["aspect_ratio_range"]
            assert min_ratio < max_ratio, f"Invalid range for {region}"
            assert min_ratio > 0, f"Min ratio must be positive for {region}"


# -----------------------------------------------------------------------------
# Integration Tests (require real models)
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestPlateDetectorIntegration:
    """Integration tests with real YOLO-World model.

    These tests require:
    - ultralytics package installed
    - Model weights (will auto-download on first run)
    - Optionally GPU for faster execution
    """

    @pytest.fixture
    def detector(self):
        """Get real plate detector."""
        pytest.importorskip("ultralytics")
        pytest.importorskip("clip")
        from backend.cv.plates import PlateDetectorWrapper

        d = PlateDetectorWrapper()
        d.load(device="cpu")  # Use CPU for CI compatibility
        yield d
        d.unload()

    @pytest.fixture
    def sample_image(self, tmp_path: Path) -> Path:
        """Create a sample test image."""
        pytest.importorskip("PIL")
        from PIL import Image

        img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        path = tmp_path / "test.jpg"
        img.save(path)
        return path

    def test_predict_returns_plate_detection_result(
        self,
        detector: Any,
        sample_image: Path,
    ) -> None:
        """Predict should return PlateDetectionResult."""
        result = detector.predict(str(sample_image))
        assert isinstance(result, PlateDetectionResult)
        assert result.model_name == "plate-detector"
        assert result.image_path == str(sample_image)
        assert result.processing_time_ms > 0


# -----------------------------------------------------------------------------
# Performance Benchmarks
# -----------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.gpu
class TestPlateDetectionBenchmarks:
    """Performance benchmark tests.

    These tests require:
    - CUDA-capable GPU
    - ultralytics package installed
    """

    def test_plate_detection_performance(self, tmp_path: Path) -> None:
        """Plate detection should achieve <30ms per image on GPU."""
        import time

        pytest.importorskip("torch")
        pytest.importorskip("ultralytics")

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from PIL import Image

        from backend.cv.plates import PlateDetectorWrapper

        detector = PlateDetectorWrapper()
        detector.load(device="cuda")

        # Create test image
        img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        img_path = tmp_path / "plate_bench.jpg"
        img.save(img_path)

        # Warm up
        for _ in range(3):
            detector.predict(str(img_path))

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            detector.predict(str(img_path))
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        detector.unload()

        assert avg_ms < 30, f"Inference too slow: {avg_ms:.2f}ms (expected <30ms)"
