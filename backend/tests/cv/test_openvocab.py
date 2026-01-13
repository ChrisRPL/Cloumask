"""
Tests for YOLO-World and GroundingDINO open-vocabulary detection wrappers.

Unit tests use mocked models to avoid requiring real model downloads.
Integration tests (marked) require actual models and optionally GPU.

Implements spec: 03-cv-models/04-yolo-world-openvocab (testing section)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from backend.cv.base import ModelState
from backend.cv.types import DetectionResult

# -----------------------------------------------------------------------------
# Mock Classes for YOLO-World Unit Testing
# -----------------------------------------------------------------------------


class MockYOLOWorldResult:
    """Mock YOLO-World prediction result object."""

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


class MockYOLOWorld:
    """Mock YOLOWorld model for testing."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._classes: list[str] = []
        self.set_classes_call_count = 0

    def to(self, device: str) -> None:
        pass

    def set_classes(self, classes: list[str]) -> None:
        """Track set_classes calls for caching tests."""
        self._classes = classes
        self.set_classes_call_count += 1

    def predict(
        self,
        source: str | list[str],
        conf: float = 0.3,
        iou: float = 0.45,
        device: str = "cpu",
        verbose: bool = False,
    ) -> list[MockYOLOWorldResult]:
        # Return mock detections based on current classes
        mock_boxes: list[dict[str, Any]] = []

        # Generate detections based on set classes
        for i, _cls in enumerate(self._classes):
            mock_boxes.append({
                "cls": i,
                "conf": 0.9 - (i * 0.1),  # Decreasing confidence
                "x": 0.5,
                "y": 0.5,
                "w": 0.2,
                "h": 0.3,
            })

        # Filter by confidence
        mock_boxes = [b for b in mock_boxes if b["conf"] >= conf]

        if isinstance(source, list):
            return [MockYOLOWorldResult(mock_boxes) for _ in source]
        return [MockYOLOWorldResult(mock_boxes)]


# -----------------------------------------------------------------------------
# Mock Classes for GroundingDINO Unit Testing
# -----------------------------------------------------------------------------


class MockGroundingDINOOutputs:
    """Mock GroundingDINO model outputs."""

    pass


class MockGroundingDINOModel:
    """Mock GroundingDINO model."""

    def __init__(self) -> None:
        self._device = "cpu"

    def to(self, device: str) -> None:
        self._device = device

    def __call__(self, **inputs: Any) -> MockGroundingDINOOutputs:
        return MockGroundingDINOOutputs()


class MockGroundingDINOProcessor:
    """Mock GroundingDINO processor."""

    def __init__(self) -> None:
        pass

    @classmethod
    def from_pretrained(cls, model_id: str) -> MockGroundingDINOProcessor:
        return cls()

    def __call__(
        self,
        images: Any,
        text: str,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.zeros(1, 3, 800, 1200),
        }

    def post_process_grounded_object_detection(
        self,
        outputs: Any,
        input_ids: torch.Tensor,
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return mock detections."""
        h, w = target_sizes[0] if target_sizes else (480, 640)

        # Return mock boxes in xyxy format (absolute coordinates)
        return [{
            "boxes": torch.tensor([
                [100.0, 100.0, 300.0, 400.0],  # x1, y1, x2, y2
                [400.0, 200.0, 600.0, 450.0],
            ]),
            "scores": torch.tensor([0.95, 0.87]),
            "labels": ["car", "person"],
        }]


# -----------------------------------------------------------------------------
# Prompt Parsing Tests
# -----------------------------------------------------------------------------


class TestPromptParsing:
    """Tests for prompt parsing utility."""

    def test_parse_simple_prompt(self) -> None:
        """Should parse comma-separated prompt."""
        from backend.cv.openvocab import _parse_prompt

        result = _parse_prompt("car, person, dog")
        assert result == ["car", "person", "dog"]

    def test_parse_prompt_with_spaces(self) -> None:
        """Should handle extra spaces."""
        from backend.cv.openvocab import _parse_prompt

        result = _parse_prompt("  red car  ,  person walking  ,  dog  ")
        assert result == ["red car", "person walking", "dog"]

    def test_parse_empty_prompt(self) -> None:
        """Should return empty list for empty prompt."""
        from backend.cv.openvocab import _parse_prompt

        result = _parse_prompt("")
        assert result == []

    def test_parse_single_class(self) -> None:
        """Should handle single class."""
        from backend.cv.openvocab import _parse_prompt

        result = _parse_prompt("car")
        assert result == ["car"]


# -----------------------------------------------------------------------------
# YOLOWorldWrapper Unit Tests
# -----------------------------------------------------------------------------


class TestYOLOWorldWrapperUnit:
    """Unit tests for YOLOWorldWrapper with mocked model."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.openvocab import YOLOWorldWrapper

        wrapper = YOLOWorldWrapper()
        assert wrapper.info.name == "yolo-world-l"
        assert wrapper.info.vram_required_mb == 4000
        assert wrapper.info.supports_batching is True
        assert wrapper.info.source == "ultralytics"

    def test_initial_state(self) -> None:
        """Should start in unloaded state."""
        from backend.cv.openvocab import YOLOWorldWrapper

        wrapper = YOLOWorldWrapper()
        assert wrapper.state == ModelState.UNLOADED
        assert wrapper.is_loaded is False
        assert wrapper._yoloworld is None
        assert wrapper._current_classes == []

    def test_predict_not_loaded_raises(self) -> None:
        """Predict should raise when model not loaded."""
        from backend.cv.openvocab import YOLOWorldWrapper

        wrapper = YOLOWorldWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict("test.jpg", prompt="car")

    def test_predict_batch_not_loaded_raises(self) -> None:
        """predict_batch should raise when model not loaded."""
        from backend.cv.openvocab import YOLOWorldWrapper

        wrapper = YOLOWorldWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict_batch(["test.jpg"], prompt="car")

    @patch("ultralytics.YOLOWorld", MockYOLOWorld)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_load_and_predict(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should load and predict successfully with mocked model."""
        from backend.cv.openvocab import YOLOWorldWrapper

        # Mock model path
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/path/yolov8l-worldv2.pt"
        mock_path.return_value = fake_path

        wrapper = YOLOWorldWrapper()
        wrapper.load(device="cpu")

        assert wrapper.is_loaded
        assert wrapper._yoloworld is not None

        result = wrapper.predict("test.jpg", prompt="red car, person")

        assert isinstance(result, DetectionResult)
        assert result.model_name == "yolo-world-l"
        assert len(result.detections) == 2
        assert result.detections[0].class_name == "red car"
        assert result.detections[1].class_name == "person"

        wrapper.unload()
        assert not wrapper.is_loaded

    @patch("ultralytics.YOLOWorld", MockYOLOWorld)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_class_embedding_cache(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should cache class embeddings - set_classes only called on change."""
        from backend.cv.openvocab import YOLOWorldWrapper

        # Mock model path
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/path/yolov8l-worldv2.pt"
        mock_path.return_value = fake_path

        wrapper = YOLOWorldWrapper()
        wrapper.load(device="cpu")

        # First predict with prompt
        wrapper.predict("test1.jpg", prompt="car, person")
        initial_count = wrapper._yoloworld.set_classes_call_count  # type: ignore

        # Same prompt should not call set_classes again
        wrapper.predict("test2.jpg", prompt="car, person")
        assert wrapper._yoloworld.set_classes_call_count == initial_count  # type: ignore

        # Different prompt should call set_classes
        wrapper.predict("test3.jpg", prompt="dog, cat")
        assert wrapper._yoloworld.set_classes_call_count == initial_count + 1  # type: ignore

        wrapper.unload()

    @patch("ultralytics.YOLOWorld", MockYOLOWorld)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_batch_prediction(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should handle batch predictions with progress callback."""
        from backend.cv.openvocab import YOLOWorldWrapper

        # Mock model path
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/path/yolov8l-worldv2.pt"
        mock_path.return_value = fake_path

        wrapper = YOLOWorldWrapper()
        wrapper.load(device="cpu")

        progress_calls: list[tuple[int, int]] = []

        def track_progress(current: int, total: int) -> None:
            progress_calls.append((current, total))

        images = ["img1.jpg", "img2.jpg", "img3.jpg"]
        results = wrapper.predict_batch(
            images,
            progress_callback=track_progress,
            prompt="car, person",
            batch_size=2,
        )

        assert len(results) == 3
        assert all(isinstance(r, DetectionResult) for r in results)
        assert len(progress_calls) == 2  # Two batches (2 + 1)

        wrapper.unload()

    @patch("ultralytics.YOLOWorld", MockYOLOWorld)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_empty_prompt_defaults_to_object(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Empty prompt should default to 'object'."""
        from backend.cv.openvocab import YOLOWorldWrapper

        # Mock model path
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/path/yolov8l-worldv2.pt"
        mock_path.return_value = fake_path

        wrapper = YOLOWorldWrapper()
        wrapper.load(device="cpu")

        result = wrapper.predict("test.jpg", prompt="")

        assert len(result.detections) == 1
        assert result.detections[0].class_name == "object"

        wrapper.unload()


# -----------------------------------------------------------------------------
# GroundingDINOWrapper Unit Tests
# -----------------------------------------------------------------------------


class TestGroundingDINOWrapperUnit:
    """Unit tests for GroundingDINOWrapper with mocked model."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.openvocab import GroundingDINOWrapper

        wrapper = GroundingDINOWrapper()
        assert wrapper.info.name == "groundingdino"
        assert wrapper.info.vram_required_mb == 5000
        assert wrapper.info.supports_batching is False
        assert wrapper.info.source == "huggingface"

    def test_initial_state(self) -> None:
        """Should start in unloaded state."""
        from backend.cv.openvocab import GroundingDINOWrapper

        wrapper = GroundingDINOWrapper()
        assert wrapper.state == ModelState.UNLOADED
        assert wrapper.is_loaded is False
        assert wrapper._gdino is None
        assert wrapper._processor is None

    def test_predict_not_loaded_raises(self) -> None:
        """Predict should raise when model not loaded."""
        from backend.cv.openvocab import GroundingDINOWrapper

        wrapper = GroundingDINOWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict("test.jpg", prompt="car")

    def test_format_prompt(self) -> None:
        """Should format prompt with periods for GroundingDINO."""
        from backend.cv.openvocab import GroundingDINOWrapper

        assert GroundingDINOWrapper._format_prompt("car, person") == "car . person ."
        assert GroundingDINOWrapper._format_prompt("red car") == "red car ."
        assert GroundingDINOWrapper._format_prompt("") == "object ."

    @patch("transformers.AutoModelForZeroShotObjectDetection.from_pretrained")
    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("backend.cv.device.clear_gpu_memory")
    @patch("PIL.Image.open")
    def test_load_and_predict(
        self,
        mock_image_open: MagicMock,
        mock_clear: MagicMock,
        mock_processor: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Should load and predict with mocked model."""
        from backend.cv.openvocab import GroundingDINOWrapper

        # Setup mocks
        mock_model.return_value = MockGroundingDINOModel()
        mock_processor.return_value = MockGroundingDINOProcessor()

        # Mock image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.size = (640, 480)
        mock_image_open.return_value = mock_img

        wrapper = GroundingDINOWrapper()
        wrapper.load(device="cpu")

        assert wrapper.is_loaded
        assert wrapper._gdino is not None
        assert wrapper._processor is not None

        result = wrapper.predict("test.jpg", prompt="car, person")

        assert isinstance(result, DetectionResult)
        assert result.model_name == "groundingdino"
        assert len(result.detections) == 2
        assert result.detections[0].class_name == "car"
        assert result.detections[1].class_name == "person"

        # Verify bbox is normalized
        assert 0 <= result.detections[0].bbox.x <= 1
        assert 0 <= result.detections[0].bbox.y <= 1

        wrapper.unload()
        assert not wrapper.is_loaded

    def test_no_batch_support(self) -> None:
        """Should indicate no batch support in model info."""
        from backend.cv.openvocab import GroundingDINOWrapper

        wrapper = GroundingDINOWrapper()
        assert wrapper.info.supports_batching is False

    @patch("transformers.AutoModelForZeroShotObjectDetection.from_pretrained")
    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("backend.cv.device.clear_gpu_memory")
    @patch("PIL.Image.open")
    def test_batch_uses_sequential_predict(
        self,
        mock_image_open: MagicMock,
        mock_clear: MagicMock,
        mock_processor: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """predict_batch should iterate since supports_batching=False."""
        from backend.cv.openvocab import GroundingDINOWrapper

        # Setup mocks
        mock_model.return_value = MockGroundingDINOModel()
        mock_processor.return_value = MockGroundingDINOProcessor()

        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.size = (640, 480)
        mock_image_open.return_value = mock_img

        wrapper = GroundingDINOWrapper()
        wrapper.load(device="cpu")

        results = wrapper.predict_batch(["img1.jpg", "img2.jpg"], prompt="car")

        assert len(results) == 2
        assert all(isinstance(r, DetectionResult) for r in results)

        wrapper.unload()


# -----------------------------------------------------------------------------
# Factory Function Tests
# -----------------------------------------------------------------------------


class TestGetOpenvocabDetector:
    """Tests for get_openvocab_detector factory function."""

    def test_default_returns_yoloworld(self) -> None:
        """Default should return YOLOWorldWrapper."""
        from backend.cv.openvocab import YOLOWorldWrapper, get_openvocab_detector

        detector = get_openvocab_detector()
        assert isinstance(detector, YOLOWorldWrapper)

    def test_force_yoloworld(self) -> None:
        """force_model='yoloworld' should return YOLOWorldWrapper."""
        from backend.cv.openvocab import YOLOWorldWrapper, get_openvocab_detector

        detector = get_openvocab_detector(force_model="yoloworld")
        assert isinstance(detector, YOLOWorldWrapper)

    def test_force_groundingdino(self) -> None:
        """force_model='groundingdino' should return GroundingDINOWrapper."""
        from backend.cv.openvocab import GroundingDINOWrapper, get_openvocab_detector

        detector = get_openvocab_detector(force_model="groundingdino")
        assert isinstance(detector, GroundingDINOWrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=6000)
    def test_prefer_accuracy_with_vram(self, mock_vram: MagicMock) -> None:
        """prefer_accuracy with enough VRAM should return GroundingDINOWrapper."""
        from backend.cv.openvocab import GroundingDINOWrapper, get_openvocab_detector

        detector = get_openvocab_detector(prefer_accuracy=True)
        assert isinstance(detector, GroundingDINOWrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=3000)
    def test_prefer_accuracy_insufficient_vram(self, mock_vram: MagicMock) -> None:
        """prefer_accuracy with insufficient VRAM should fallback to YOLOWorld."""
        from backend.cv.openvocab import YOLOWorldWrapper, get_openvocab_detector

        detector = get_openvocab_detector(prefer_accuracy=True)
        assert isinstance(detector, YOLOWorldWrapper)


# -----------------------------------------------------------------------------
# Result Conversion Tests
# -----------------------------------------------------------------------------


class TestResultConversion:
    """Tests for result conversion utilities."""

    def test_convert_yoloworld_result(self) -> None:
        """Should correctly convert YOLO-World result."""
        from backend.cv.openvocab import _convert_yoloworld_result

        mock_result = MockYOLOWorldResult([
            {"cls": 0, "conf": 0.95, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3},
            {"cls": 1, "conf": 0.87, "x": 0.3, "y": 0.4, "w": 0.1, "h": 0.15},
        ])

        classes = ["red car", "person"]
        detections = _convert_yoloworld_result(mock_result, classes)

        assert len(detections) == 2
        assert detections[0].class_name == "red car"
        assert detections[0].class_id == 0
        assert detections[0].confidence == 0.95
        assert detections[1].class_name == "person"
        assert detections[1].class_id == 1

    def test_convert_yoloworld_empty_result(self) -> None:
        """Should handle empty results."""
        from backend.cv.openvocab import _convert_yoloworld_result

        mock_result = MockYOLOWorldResult([])
        detections = _convert_yoloworld_result(mock_result, ["car"])
        assert detections == []

    def test_convert_yoloworld_invalid_class_id(self) -> None:
        """Should skip detections with invalid class IDs."""
        from backend.cv.openvocab import _convert_yoloworld_result

        mock_result = MockYOLOWorldResult([
            {"cls": 0, "conf": 0.95, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3},
            {"cls": 5, "conf": 0.87, "x": 0.3, "y": 0.4, "w": 0.1, "h": 0.15},  # Invalid
        ])

        classes = ["car"]  # Only one class
        detections = _convert_yoloworld_result(mock_result, classes)

        assert len(detections) == 1
        assert detections[0].class_name == "car"


# -----------------------------------------------------------------------------
# Integration Tests (require real models)
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestYOLOWorldIntegration:
    """Integration tests with real YOLO-World model."""

    @pytest.fixture
    def detector(self):
        """Get YOLO-World detector."""
        pytest.importorskip("ultralytics")
        from backend.cv.openvocab import YOLOWorldWrapper

        d = YOLOWorldWrapper()
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

    def test_custom_class_detection(self, detector, sample_image: Path) -> None:
        """Should detect with custom class prompts."""
        result = detector.predict(str(sample_image), prompt="red car, person")

        assert isinstance(result, DetectionResult)
        assert result.model_name == "yolo-world-l"
        # Note: empty image may have no detections, which is fine

    def test_multiple_classes(self, detector, sample_image: Path) -> None:
        """Should handle multiple classes."""
        result = detector.predict(str(sample_image), prompt="car, person, bicycle")

        # Class names should match input (if detected)
        for det in result.detections:
            assert det.class_name in ["car", "person", "bicycle"]


# -----------------------------------------------------------------------------
# GPU/Performance Tests
# -----------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.gpu
class TestPerformanceBenchmarks:
    """Performance benchmark tests.

    These tests require:
    - CUDA-capable GPU
    - ultralytics package installed
    """

    def test_yoloworld_vram_budget(self) -> None:
        """YOLO-World should use <4.5GB VRAM."""
        pytest.importorskip("torch")
        pytest.importorskip("ultralytics")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from backend.cv.device import get_vram_usage
        from backend.cv.openvocab import YOLOWorldWrapper

        initial = get_vram_usage()[0]

        detector = YOLOWorldWrapper()
        detector.load(device="cuda")

        loaded = get_vram_usage()[0]
        vram_used = loaded - initial

        assert vram_used < 4500, f"YOLO-World used {vram_used}MB VRAM (budget: 4500MB)"

        detector.unload()

    def test_embedding_cache_speedup(self) -> None:
        """Cached prompts should be significantly faster."""
        pytest.importorskip("torch")
        pytest.importorskip("ultralytics")
        import numpy as np
        from PIL import Image

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from backend.cv.openvocab import YOLOWorldWrapper

        detector = YOLOWorldWrapper()
        detector.load(device="cuda")

        # Create test image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
            img.save(f.name)
            img_path = f.name

        prompt = "car, person, bicycle"

        # First call - creates embeddings
        import time
        start = time.perf_counter()
        detector.predict(img_path, prompt=prompt)
        first_time = time.perf_counter() - start

        # Second call - uses cached embeddings
        start = time.perf_counter()
        detector.predict(img_path, prompt=prompt)
        second_time = time.perf_counter() - start

        # Cached should be faster (allow some tolerance)
        assert second_time < first_time, (
            f"Cached call ({second_time:.3f}s) should be faster than "
            f"first call ({first_time:.3f}s)"
        )

        detector.unload()
