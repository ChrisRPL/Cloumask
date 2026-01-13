"""
Tests for YOLO11 and RT-DETR detection wrappers.

Unit tests use mocked models to avoid requiring real model downloads.
Integration tests (marked) require actual models and optionally GPU.

Implements spec: 03-cv-models/01-yolo11-detection (testing section)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.cv.base import ModelState
from backend.cv.types import DetectionResult

# -----------------------------------------------------------------------------
# Mock Classes for Unit Testing (without real models)
# -----------------------------------------------------------------------------


class MockUltralyticsResult:
    """Mock ultralytics prediction result object."""

    def __init__(self, boxes_data: list[dict[str, Any]]) -> None:
        self.boxes = MockBoxes(boxes_data) if boxes_data else None


class MockBoxes:
    """Mock ultralytics boxes container."""

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data

    def __iter__(self):
        for d in self._data:
            yield MockBox(d)


class MockBox:
    """Mock single box detection."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.cls = [data["cls"]]
        self.conf = [data["conf"]]
        self.xywhn = [[data["x"], data["y"], data["w"], data["h"]]]


class MockYOLO:
    """Mock YOLO model for testing."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def to(self, device: str) -> None:
        pass

    def predict(
        self,
        source: str | list[str],
        conf: float = 0.5,
        iou: float = 0.45,
        classes: list[int] | None = None,
        device: str = "cpu",
        verbose: bool = False,
    ) -> list[MockUltralyticsResult]:
        # Return mock detections
        mock_boxes = [
            {"cls": 0, "conf": 0.95, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3},  # person
            {"cls": 2, "conf": 0.87, "x": 0.3, "y": 0.4, "w": 0.1, "h": 0.15},  # car
        ]

        # Filter by classes if provided
        if classes is not None:
            mock_boxes = [b for b in mock_boxes if b["cls"] in classes]

        # Filter by confidence
        mock_boxes = [b for b in mock_boxes if b["conf"] >= conf]

        if isinstance(source, list):
            return [MockUltralyticsResult(mock_boxes) for _ in source]
        return [MockUltralyticsResult(mock_boxes)]


class MockRTDETR:
    """Mock RT-DETR model for testing."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def to(self, device: str) -> None:
        pass

    def predict(
        self,
        source: str | list[str],
        conf: float = 0.5,
        iou: float = 0.45,
        classes: list[int] | None = None,
        device: str = "cpu",
        verbose: bool = False,
    ) -> list[MockUltralyticsResult]:
        mock_boxes = [
            {"cls": 0, "conf": 0.98, "x": 0.5, "y": 0.5, "w": 0.22, "h": 0.32},
        ]
        if classes is not None:
            mock_boxes = [b for b in mock_boxes if b["cls"] in classes]
        mock_boxes = [b for b in mock_boxes if b["conf"] >= conf]

        if isinstance(source, list):
            return [MockUltralyticsResult(mock_boxes) for _ in source]
        return [MockUltralyticsResult(mock_boxes)]


# -----------------------------------------------------------------------------
# COCO Classes Tests
# -----------------------------------------------------------------------------


class TestCOCOClasses:
    """Tests for COCO class constants and utilities."""

    def test_coco_classes_count(self) -> None:
        """Should have exactly 80 COCO classes."""
        from backend.cv.detection import COCO_CLASSES

        assert len(COCO_CLASSES) == 80

    def test_coco_classes_common(self) -> None:
        """Should include common detection classes."""
        from backend.cv.detection import COCO_CLASSES

        common_classes = ["person", "car", "dog", "cat", "chair", "bicycle"]
        for cls in common_classes:
            assert cls in COCO_CLASSES, f"Missing common class: {cls}"

    def test_coco_classes_no_duplicates(self) -> None:
        """Should have no duplicate classes."""
        from backend.cv.detection import COCO_CLASSES

        assert len(COCO_CLASSES) == len(set(COCO_CLASSES))


class TestGetClassIndices:
    """Tests for get_class_indices utility."""

    def test_valid_classes(self) -> None:
        """Should convert valid class names to indices."""
        from backend.cv.detection import COCO_CLASSES, get_class_indices

        indices = get_class_indices(["person", "car"])
        assert indices is not None
        assert COCO_CLASSES.index("person") in indices
        assert COCO_CLASSES.index("car") in indices

    def test_case_insensitive(self) -> None:
        """Should match class names case-insensitively."""
        from backend.cv.detection import get_class_indices

        indices1 = get_class_indices(["Person", "CAR"])
        indices2 = get_class_indices(["person", "car"])
        assert indices1 == indices2

    def test_invalid_class_ignored(self) -> None:
        """Should ignore invalid class names with warning."""
        from backend.cv.detection import get_class_indices

        indices = get_class_indices(["person", "invalid_class_xyz"])
        assert indices is not None
        assert len(indices) == 1  # Only 'person' should be in result

    def test_none_input(self) -> None:
        """Should return None for None input."""
        from backend.cv.detection import get_class_indices

        assert get_class_indices(None) is None

    def test_empty_list(self) -> None:
        """Should return None for empty list."""
        from backend.cv.detection import get_class_indices

        assert get_class_indices([]) is None

    def test_all_invalid(self) -> None:
        """Should return None when all classes are invalid."""
        from backend.cv.detection import get_class_indices

        assert get_class_indices(["invalid1", "invalid2"]) is None


# -----------------------------------------------------------------------------
# YOLO11Wrapper Unit Tests
# -----------------------------------------------------------------------------


class TestYOLO11WrapperUnit:
    """Unit tests for YOLO11Wrapper with mocked model."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.detection import YOLO11Wrapper

        wrapper = YOLO11Wrapper()
        assert wrapper.info.name == "yolo11m"
        assert wrapper.info.vram_required_mb == 2500
        assert wrapper.info.supports_batching is True
        assert wrapper.info.source == "ultralytics"

    def test_initial_state(self) -> None:
        """Should start in unloaded state."""
        from backend.cv.detection import YOLO11Wrapper

        wrapper = YOLO11Wrapper()
        assert wrapper.state == ModelState.UNLOADED
        assert wrapper.is_loaded is False
        assert wrapper._yolo is None

    def test_predict_not_loaded_raises(self) -> None:
        """Predict should raise when model not loaded."""
        from backend.cv.detection import YOLO11Wrapper

        wrapper = YOLO11Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict("test.jpg")

    def test_predict_batch_not_loaded_raises(self) -> None:
        """predict_batch should raise when model not loaded."""
        from backend.cv.detection import YOLO11Wrapper

        wrapper = YOLO11Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict_batch(["test.jpg"])

    @patch("ultralytics.YOLO", MockYOLO)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_load_and_predict(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should load and predict successfully with mocked model."""
        from backend.cv.detection import YOLO11Wrapper

        # Use MagicMock as path with exists() returning True
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/path/yolo11m.pt"
        mock_path.return_value = fake_path

        wrapper = YOLO11Wrapper()
        wrapper.load(device="cpu")

        assert wrapper.is_loaded
        assert wrapper._yolo is not None

        result = wrapper.predict("test.jpg", confidence=0.5)

        assert isinstance(result, DetectionResult)
        assert result.model_name == "yolo11m"
        assert len(result.detections) == 2  # person and car from mock
        assert result.detections[0].class_name == "person"
        assert result.detections[1].class_name == "car"

        wrapper.unload()
        assert not wrapper.is_loaded

    @patch("ultralytics.YOLO", MockYOLO)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_predict_with_class_filter(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should filter by class names."""
        from backend.cv.detection import YOLO11Wrapper

        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/path/yolo11m.pt"
        mock_path.return_value = fake_path

        wrapper = YOLO11Wrapper()
        wrapper.load(device="cpu")

        result = wrapper.predict("test.jpg", classes=["person"])

        # Only person should be returned
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "person"

        wrapper.unload()

    @patch("ultralytics.YOLO", MockYOLO)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_predict_with_confidence_filter(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should filter by confidence threshold."""
        from backend.cv.detection import YOLO11Wrapper

        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/path/yolo11m.pt"
        mock_path.return_value = fake_path

        wrapper = YOLO11Wrapper()
        wrapper.load(device="cpu")

        result = wrapper.predict("test.jpg", confidence=0.9)

        # Only person (0.95) should pass 0.9 threshold
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "person"

        wrapper.unload()

    @patch("ultralytics.YOLO", MockYOLO)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_predict_batch(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should process batch of images."""
        from backend.cv.detection import YOLO11Wrapper

        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/path/yolo11m.pt"
        mock_path.return_value = fake_path

        wrapper = YOLO11Wrapper()
        wrapper.load(device="cpu")

        inputs = ["a.jpg", "b.jpg", "c.jpg"]
        results = wrapper.predict_batch(inputs, confidence=0.5)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.image_path == inputs[i]
            assert result.model_name == "yolo11m"

        wrapper.unload()

    @patch("ultralytics.YOLO", MockYOLO)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_predict_batch_with_progress(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should call progress callback during batch processing."""
        from backend.cv.detection import YOLO11Wrapper

        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/path/yolo11m.pt"
        mock_path.return_value = fake_path

        wrapper = YOLO11Wrapper()
        wrapper.load(device="cpu")

        progress_calls: list[tuple[int, int]] = []

        def callback(current: int, total: int) -> None:
            progress_calls.append((current, total))

        inputs = ["a.jpg", "b.jpg", "c.jpg"]
        wrapper.predict_batch(inputs, progress_callback=callback, batch_size=2)

        # With batch_size=2, should have 2 progress calls (batch 1: 2 images, batch 2: 1 image)
        assert len(progress_calls) >= 1
        # Last call should indicate completion
        assert progress_calls[-1][0] == 3  # All 3 processed

        wrapper.unload()


# -----------------------------------------------------------------------------
# RTDETRWrapper Unit Tests
# -----------------------------------------------------------------------------


class TestRTDETRWrapperUnit:
    """Unit tests for RTDETRWrapper."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.detection import RTDETRWrapper

        wrapper = RTDETRWrapper()
        assert wrapper.info.name == "rtdetr-l"
        assert wrapper.info.vram_required_mb == 3500
        assert wrapper.info.supports_batching is True

    def test_initial_state(self) -> None:
        """Should start in unloaded state."""
        from backend.cv.detection import RTDETRWrapper

        wrapper = RTDETRWrapper()
        assert wrapper.is_loaded is False
        assert wrapper._rtdetr is None

    @patch("ultralytics.RTDETR", MockRTDETR)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_load_and_predict(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should load and predict successfully."""
        from backend.cv.detection import RTDETRWrapper

        mock_path.return_value = Path("/fake/path/rtdetr-l.pt")

        wrapper = RTDETRWrapper()
        wrapper.load(device="cpu")

        assert wrapper.is_loaded

        result = wrapper.predict("test.jpg")
        assert isinstance(result, DetectionResult)
        assert result.model_name == "rtdetr-l"

        wrapper.unload()


# -----------------------------------------------------------------------------
# get_detector Factory Tests
# -----------------------------------------------------------------------------


class TestGetDetector:
    """Tests for get_detector factory function."""

    def test_default_returns_yolo(self) -> None:
        """Default should return YOLO11Wrapper."""
        from backend.cv.detection import YOLO11Wrapper, get_detector

        detector = get_detector()
        assert isinstance(detector, YOLO11Wrapper)

    def test_force_yolo(self) -> None:
        """force_model='yolo11m' should return YOLO11Wrapper."""
        from backend.cv.detection import YOLO11Wrapper, get_detector

        detector = get_detector(force_model="yolo11m")
        assert isinstance(detector, YOLO11Wrapper)

    def test_force_rtdetr(self) -> None:
        """force_model='rtdetr-l' should return RTDETRWrapper."""
        from backend.cv.detection import RTDETRWrapper, get_detector

        detector = get_detector(force_model="rtdetr-l")
        assert isinstance(detector, RTDETRWrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=4000)
    def test_prefer_accuracy_with_vram(self, mock_vram: MagicMock) -> None:
        """prefer_accuracy with enough VRAM should return RTDETRWrapper."""
        from backend.cv.detection import RTDETRWrapper, get_detector

        detector = get_detector(prefer_accuracy=True)
        assert isinstance(detector, RTDETRWrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=2000)
    def test_prefer_accuracy_low_vram(self, mock_vram: MagicMock) -> None:
        """prefer_accuracy with low VRAM should fall back to YOLO11Wrapper."""
        from backend.cv.detection import YOLO11Wrapper, get_detector

        detector = get_detector(prefer_accuracy=True)
        assert isinstance(detector, YOLO11Wrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=3500)
    def test_prefer_accuracy_exact_vram(self, mock_vram: MagicMock) -> None:
        """prefer_accuracy with exactly enough VRAM should return RTDETRWrapper."""
        from backend.cv.detection import RTDETRWrapper, get_detector

        detector = get_detector(prefer_accuracy=True)
        assert isinstance(detector, RTDETRWrapper)


# -----------------------------------------------------------------------------
# Result Conversion Tests
# -----------------------------------------------------------------------------


class TestResultConversion:
    """Tests for ultralytics result conversion."""

    def test_convert_empty_result(self) -> None:
        """Should handle empty results."""
        from backend.cv.detection import _convert_ultralytics_result

        mock_result = MockUltralyticsResult([])
        detections = _convert_ultralytics_result(mock_result)
        assert detections == []

    def test_convert_none_boxes(self) -> None:
        """Should handle None boxes."""
        from backend.cv.detection import _convert_ultralytics_result

        class NoneBoxesResult:
            boxes = None

        detections = _convert_ultralytics_result(NoneBoxesResult())
        assert detections == []

    def test_convert_valid_result(self) -> None:
        """Should correctly convert ultralytics result."""
        from backend.cv.detection import _convert_ultralytics_result

        mock_result = MockUltralyticsResult([
            {"cls": 0, "conf": 0.95, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3},
        ])
        detections = _convert_ultralytics_result(mock_result)

        assert len(detections) == 1
        det = detections[0]
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.confidence == 0.95
        assert det.bbox.x == 0.5
        assert det.bbox.y == 0.5
        assert det.bbox.width == 0.2
        assert det.bbox.height == 0.3

    def test_convert_invalid_class_id_skipped(self) -> None:
        """Should skip detections with invalid class IDs."""
        from backend.cv.detection import _convert_ultralytics_result

        mock_result = MockUltralyticsResult([
            {"cls": 0, "conf": 0.95, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3},
            {"cls": 999, "conf": 0.90, "x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1},  # Invalid
        ])
        detections = _convert_ultralytics_result(mock_result)

        assert len(detections) == 1  # Only valid detection
        assert detections[0].class_id == 0


# -----------------------------------------------------------------------------
# Integration Tests (require real models)
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestYOLO11Integration:
    """Integration tests with real YOLO11 model.

    These tests require:
    - ultralytics package installed
    - Model weights (will auto-download on first run)
    - Optionally GPU for faster execution
    """

    @pytest.fixture
    def detector(self):
        """Get real YOLO11 detector."""
        pytest.importorskip("ultralytics")
        from backend.cv.detection import YOLO11Wrapper

        d = YOLO11Wrapper()
        d.load(device="cpu")  # Use CPU for CI compatibility
        yield d
        d.unload()

    @pytest.fixture
    def sample_image(self, tmp_path: Path) -> Path:
        """Create a sample test image."""
        pytest.importorskip("PIL")
        import numpy as np
        from PIL import Image

        img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        path = tmp_path / "test.jpg"
        img.save(path)
        return path

    def test_predict_returns_detection_result(
        self, detector, sample_image: Path
    ) -> None:
        """Predict should return DetectionResult."""
        result = detector.predict(str(sample_image))
        assert isinstance(result, DetectionResult)
        assert result.model_name == "yolo11m"
        assert result.image_path == str(sample_image)
        assert result.processing_time_ms > 0

    def test_class_filtering_returns_only_requested(
        self, detector, sample_image: Path
    ) -> None:
        """Class filtering should only return specified classes."""
        result = detector.predict(str(sample_image), classes=["person"])
        for det in result.detections:
            assert det.class_name == "person"

    def test_high_confidence_returns_fewer(
        self, detector, sample_image: Path
    ) -> None:
        """Higher confidence threshold should return fewer or equal detections."""
        low_conf = detector.predict(str(sample_image), confidence=0.1)
        high_conf = detector.predict(str(sample_image), confidence=0.9)
        assert len(low_conf.detections) >= len(high_conf.detections)


# -----------------------------------------------------------------------------
# Performance Tests (GPU required)
# -----------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.gpu
class TestPerformanceBenchmarks:
    """Performance benchmark tests.

    These tests require:
    - CUDA-capable GPU
    - ultralytics package installed
    - Model weights downloaded
    """

    def test_yolo11_inference_speed(self) -> None:
        """YOLO11m should achieve <5ms per image on GPU."""
        import time

        pytest.importorskip("torch")
        pytest.importorskip("ultralytics")

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        import numpy as np
        from PIL import Image

        from backend.cv.detection import YOLO11Wrapper

        detector = YOLO11Wrapper()
        detector.load(device="cuda")

        # Create test image
        img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        img.save("/tmp/bench.jpg")

        # Warm up
        for _ in range(3):
            detector.predict("/tmp/bench.jpg")

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            detector.predict("/tmp/bench.jpg")
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        detector.unload()

        assert avg_ms < 5, f"Inference too slow: {avg_ms:.2f}ms (expected <5ms)"

    def test_yolo11_vram_budget(self) -> None:
        """YOLO11m should use <2.5GB VRAM."""
        pytest.importorskip("torch")
        pytest.importorskip("ultralytics")

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from backend.cv.detection import YOLO11Wrapper
        from backend.cv.device import get_vram_usage

        initial = get_vram_usage()[0]

        detector = YOLO11Wrapper()
        detector.load(device="cuda")

        loaded = get_vram_usage()[0]
        vram_used = loaded - initial

        detector.unload()

        assert vram_used < 2500, f"VRAM budget exceeded: {vram_used}MB (expected <2500MB)"
