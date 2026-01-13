"""
Tests for SAM3, SAM2, and MobileSAM segmentation wrappers.

Unit tests use mocked models to avoid requiring real model downloads.
Integration tests (marked) require actual models and optionally GPU.

Implements spec: 03-cv-models/02-sam3-segmentation (testing section)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.cv.base import ModelState
from backend.cv.types import SegmentationResult

# -----------------------------------------------------------------------------
# Mock Classes for Unit Testing (without real models)
# -----------------------------------------------------------------------------


class MockMasks:
    """Mock ultralytics masks container."""

    def __init__(self, masks_data: np.ndarray, conf: np.ndarray | None = None) -> None:
        self.data = masks_data
        self.conf = conf


class MockBoxes:
    """Mock ultralytics boxes container."""

    def __init__(self, conf: np.ndarray | None = None) -> None:
        self.conf = conf


class MockSAMResult:
    """Mock SAM prediction result object."""

    def __init__(
        self,
        masks_data: np.ndarray | None,
        scores: np.ndarray | None = None,
    ) -> None:
        if masks_data is not None:
            self.masks = MockMasks(masks_data, scores)
            self.boxes = MockBoxes(scores)
        else:
            self.masks = None
            self.boxes = None


class MockSAM:
    """Mock SAM model for testing."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def to(self, device: str) -> None:
        pass

    def predict(
        self,
        source: str,
        points: list[list[int]] | None = None,
        labels: list[int] | None = None,
        bboxes: list[list[int]] | None = None,
        device: str = "cpu",
        verbose: bool = False,
    ) -> list[MockSAMResult]:
        # Return mock segmentation mask (64x64 for testing)
        mask = np.zeros((1, 64, 64), dtype=np.uint8)
        mask[0, 16:48, 16:48] = 1  # Central square mask
        scores = np.array([0.95])
        return [MockSAMResult(mask, scores)]


class MockSAM3SemanticPredictor:
    """Mock SAM3SemanticPredictor for testing text prompts."""

    def __init__(self, overrides: dict[str, Any]) -> None:
        self.overrides = overrides
        self._image_set = False

    def set_image(self, source: str) -> None:
        self._image_set = True

    def __call__(
        self,
        text: list[str] | None = None,
        bboxes: list[list[float]] | None = None,
        points: list[list[int]] | None = None,
        labels: list[int] | None = None,
    ) -> list[MockSAMResult]:
        # Return mock segmentation mask
        mask = np.zeros((1, 64, 64), dtype=np.uint8)
        mask[0, 16:48, 16:48] = 1  # Central square mask
        scores = np.array([0.92])
        return [MockSAMResult(mask, scores)]


def _sam3_available() -> bool:
    """Check if SAM3SemanticPredictor is available in ultralytics."""
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: F401
        return True
    except (ImportError, AttributeError):
        return False


# Mark SAM3 tests to skip if SAM3 not available
requires_sam3 = pytest.mark.skipif(
    not _sam3_available(),
    reason="SAM3SemanticPredictor not available in this ultralytics version"
)


# -----------------------------------------------------------------------------
# SAM3Wrapper Unit Tests
# -----------------------------------------------------------------------------


class TestSAM3WrapperUnit:
    """Unit tests for SAM3Wrapper with mocked model."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.segmentation import SAM3Wrapper

        wrapper = SAM3Wrapper()
        assert wrapper.info.name == "sam3"
        assert wrapper.info.vram_required_mb == 8000
        assert wrapper.info.supports_batching is False
        assert wrapper.info.source == "ultralytics"
        assert wrapper.info.extra.get("supports_text_prompts") is True

    def test_initial_state(self) -> None:
        """Should start in unloaded state."""
        from backend.cv.segmentation import SAM3Wrapper

        wrapper = SAM3Wrapper()
        assert wrapper.state == ModelState.UNLOADED
        assert wrapper.is_loaded is False
        assert wrapper._predictor is None

    def test_predict_not_loaded_raises(self) -> None:
        """Predict should raise when model not loaded."""
        from backend.cv.segmentation import SAM3Wrapper

        wrapper = SAM3Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict("test.jpg", prompt="car")

    def test_predict_no_prompt_raises(self) -> None:
        """Predict should raise when no prompt provided."""
        from backend.cv.segmentation import SAM3Wrapper

        wrapper = SAM3Wrapper()
        wrapper._state = ModelState.LOADED
        wrapper._predictor = MagicMock()

        with pytest.raises(ValueError, match="Must provide prompt"):
            wrapper.predict("test.jpg")

    @requires_sam3
    @patch(
        "ultralytics.models.sam.SAM3SemanticPredictor",
        MockSAM3SemanticPredictor,
    )
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_load_and_predict_text(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should load and predict with text prompt successfully."""
        from backend.cv.segmentation import SAM3Wrapper

        mock_path.return_value = Path("/fake/path/sam3.pt")

        wrapper = SAM3Wrapper()
        wrapper.load(device="cpu")

        assert wrapper.is_loaded
        assert wrapper._predictor is not None

        result = wrapper.predict("test.jpg", prompt="red car")

        assert isinstance(result, SegmentationResult)
        assert result.model_name == "sam3"
        assert result.prompt == "red car"
        assert len(result.masks) == 1
        assert result.masks[0].confidence > 0.9

        wrapper.unload()
        assert not wrapper.is_loaded

    @requires_sam3
    @patch(
        "ultralytics.models.sam.SAM3SemanticPredictor",
        MockSAM3SemanticPredictor,
    )
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_predict_with_box(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should predict with box prompt."""
        from backend.cv.segmentation import SAM3Wrapper

        mock_path.return_value = Path("/fake/path/sam3.pt")

        wrapper = SAM3Wrapper()
        wrapper.load(device="cpu")

        result = wrapper.predict("test.jpg", box=(10, 20, 100, 200))

        assert isinstance(result, SegmentationResult)
        assert len(result.masks) >= 1

        wrapper.unload()

    @requires_sam3
    @patch(
        "ultralytics.models.sam.SAM3SemanticPredictor",
        MockSAM3SemanticPredictor,
    )
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_predict_with_point(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should predict with point prompt."""
        from backend.cv.segmentation import SAM3Wrapper

        mock_path.return_value = Path("/fake/path/sam3.pt")

        wrapper = SAM3Wrapper()
        wrapper.load(device="cpu")

        result = wrapper.predict("test.jpg", point=(50, 75))

        assert isinstance(result, SegmentationResult)
        assert len(result.masks) >= 1

        wrapper.unload()


# -----------------------------------------------------------------------------
# SAM2Wrapper Unit Tests
# -----------------------------------------------------------------------------


class TestSAM2WrapperUnit:
    """Unit tests for SAM2Wrapper with mocked model."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.segmentation import SAM2Wrapper

        wrapper = SAM2Wrapper()
        assert wrapper.info.name == "sam2"
        assert wrapper.info.vram_required_mb == 6000
        assert wrapper.info.supports_batching is False
        assert wrapper.info.extra.get("supports_text_prompts") is False

    def test_initial_state(self) -> None:
        """Should start in unloaded state."""
        from backend.cv.segmentation import SAM2Wrapper

        wrapper = SAM2Wrapper()
        assert wrapper.state == ModelState.UNLOADED
        assert wrapper.is_loaded is False

    def test_predict_not_loaded_raises(self) -> None:
        """Predict should raise when model not loaded."""
        from backend.cv.segmentation import SAM2Wrapper

        wrapper = SAM2Wrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict("test.jpg", point=(100, 100))

    def test_predict_text_prompt_raises(self) -> None:
        """SAM2 should raise NotImplementedError for text prompts."""
        from backend.cv.segmentation import SAM2Wrapper

        wrapper = SAM2Wrapper()
        wrapper._state = ModelState.LOADED
        wrapper._sam = MagicMock()

        with pytest.raises(NotImplementedError, match="text prompts"):
            wrapper.predict("test.jpg", prompt="car")

    def test_predict_no_prompt_raises(self) -> None:
        """Predict should raise when no prompt provided."""
        from backend.cv.segmentation import SAM2Wrapper

        wrapper = SAM2Wrapper()
        wrapper._state = ModelState.LOADED
        wrapper._sam = MagicMock()

        with pytest.raises(ValueError, match="requires point or box"):
            wrapper.predict("test.jpg")

    @patch("ultralytics.SAM", MockSAM)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_load_and_predict_point(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should load and predict with point prompt."""
        from backend.cv.segmentation import SAM2Wrapper

        mock_path.return_value = Path("/fake/path/sam2.pt")

        wrapper = SAM2Wrapper()
        wrapper.load(device="cpu")

        assert wrapper.is_loaded

        result = wrapper.predict("test.jpg", point=(100, 150))

        assert isinstance(result, SegmentationResult)
        assert result.model_name == "sam2"
        assert len(result.masks) >= 1

        wrapper.unload()

    @patch("ultralytics.SAM", MockSAM)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_load_and_predict_box(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should load and predict with box prompt."""
        from backend.cv.segmentation import SAM2Wrapper

        mock_path.return_value = Path("/fake/path/sam2.pt")

        wrapper = SAM2Wrapper()
        wrapper.load(device="cpu")

        result = wrapper.predict("test.jpg", box=(10, 20, 100, 200))

        assert isinstance(result, SegmentationResult)
        assert result.model_name == "sam2"
        assert len(result.masks) >= 1

        wrapper.unload()


# -----------------------------------------------------------------------------
# MobileSAMWrapper Unit Tests
# -----------------------------------------------------------------------------


class TestMobileSAMWrapperUnit:
    """Unit tests for MobileSAMWrapper with mocked model."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.segmentation import MobileSAMWrapper

        wrapper = MobileSAMWrapper()
        assert wrapper.info.name == "mobilesam"
        assert wrapper.info.vram_required_mb == 1500
        assert wrapper.info.supports_batching is False
        assert wrapper.info.extra.get("supports_text_prompts") is False

    def test_initial_state(self) -> None:
        """Should start in unloaded state."""
        from backend.cv.segmentation import MobileSAMWrapper

        wrapper = MobileSAMWrapper()
        assert wrapper.state == ModelState.UNLOADED
        assert wrapper.is_loaded is False

    def test_predict_text_prompt_raises(self) -> None:
        """MobileSAM should raise NotImplementedError for text prompts."""
        from backend.cv.segmentation import MobileSAMWrapper

        wrapper = MobileSAMWrapper()
        wrapper._state = ModelState.LOADED
        wrapper._sam = MagicMock()

        with pytest.raises(NotImplementedError, match="text prompts"):
            wrapper.predict("test.jpg", prompt="car")

    def test_predict_box_prompt_raises(self) -> None:
        """MobileSAM should raise NotImplementedError for box prompts."""
        from backend.cv.segmentation import MobileSAMWrapper

        wrapper = MobileSAMWrapper()
        wrapper._state = ModelState.LOADED
        wrapper._sam = MagicMock()

        with pytest.raises(NotImplementedError, match="box prompts"):
            wrapper.predict("test.jpg", box=(10, 20, 100, 200))

    def test_predict_no_point_raises(self) -> None:
        """Predict should raise when no point provided."""
        from backend.cv.segmentation import MobileSAMWrapper

        wrapper = MobileSAMWrapper()
        wrapper._state = ModelState.LOADED
        wrapper._sam = MagicMock()

        with pytest.raises(ValueError, match="requires point"):
            wrapper.predict("test.jpg")

    @patch("ultralytics.SAM", MockSAM)
    @patch("backend.cv.download.get_model_path")
    @patch("backend.cv.device.clear_gpu_memory")
    def test_load_and_predict_point(
        self,
        mock_clear: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Should load and predict with point prompt."""
        from backend.cv.segmentation import MobileSAMWrapper

        mock_path.return_value = Path("/fake/path/mobile_sam.pt")

        wrapper = MobileSAMWrapper()
        wrapper.load(device="cpu")

        assert wrapper.is_loaded

        result = wrapper.predict("test.jpg", point=(50, 75))

        assert isinstance(result, SegmentationResult)
        assert result.model_name == "mobilesam"
        assert len(result.masks) >= 1

        wrapper.unload()


# -----------------------------------------------------------------------------
# get_segmenter Factory Tests
# -----------------------------------------------------------------------------


class TestGetSegmenter:
    """Tests for get_segmenter factory function."""

    def test_force_sam3(self) -> None:
        """force_model='sam3' should return SAM3Wrapper."""
        from backend.cv.segmentation import SAM3Wrapper, get_segmenter

        segmenter = get_segmenter(force_model="sam3")
        assert isinstance(segmenter, SAM3Wrapper)

    def test_force_sam2(self) -> None:
        """force_model='sam2' should return SAM2Wrapper."""
        from backend.cv.segmentation import SAM2Wrapper, get_segmenter

        segmenter = get_segmenter(force_model="sam2")
        assert isinstance(segmenter, SAM2Wrapper)

    def test_force_mobilesam(self) -> None:
        """force_model='mobilesam' should return MobileSAMWrapper."""
        from backend.cv.segmentation import MobileSAMWrapper, get_segmenter

        segmenter = get_segmenter(force_model="mobilesam")
        assert isinstance(segmenter, MobileSAMWrapper)

    def test_force_unknown_raises(self) -> None:
        """force_model with unknown value should raise ValueError."""
        from backend.cv.segmentation import get_segmenter

        with pytest.raises(ValueError, match="Unknown model"):
            get_segmenter(force_model="unknown_model")

    @patch("backend.cv.device.get_available_vram_mb", return_value=10000)
    def test_text_prompt_with_vram(self, mock_vram: MagicMock) -> None:
        """Text prompt with enough VRAM should return SAM3Wrapper."""
        from backend.cv.segmentation import SAM3Wrapper, get_segmenter

        segmenter = get_segmenter(prompt_type="text")
        assert isinstance(segmenter, SAM3Wrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=4000)
    def test_text_prompt_low_vram_raises(self, mock_vram: MagicMock) -> None:
        """Text prompt with insufficient VRAM should raise RuntimeError."""
        from backend.cv.segmentation import get_segmenter

        with pytest.raises(RuntimeError, match="VRAM"):
            get_segmenter(prompt_type="text")

    @patch("backend.cv.device.get_available_vram_mb", return_value=8000)
    def test_point_prompt_default(self, mock_vram: MagicMock) -> None:
        """Point prompt by default should return SAM2Wrapper with enough VRAM."""
        from backend.cv.segmentation import SAM2Wrapper, get_segmenter

        segmenter = get_segmenter(prompt_type="point")
        assert isinstance(segmenter, SAM2Wrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=8000)
    def test_point_prompt_prefer_speed(self, mock_vram: MagicMock) -> None:
        """Point prompt with prefer_speed should return MobileSAMWrapper."""
        from backend.cv.segmentation import MobileSAMWrapper, get_segmenter

        segmenter = get_segmenter(prompt_type="point", prefer_speed=True)
        assert isinstance(segmenter, MobileSAMWrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=2000)
    def test_point_prompt_low_vram(self, mock_vram: MagicMock) -> None:
        """Point prompt with low VRAM should return MobileSAMWrapper."""
        from backend.cv.segmentation import MobileSAMWrapper, get_segmenter

        segmenter = get_segmenter(prompt_type="point")
        assert isinstance(segmenter, MobileSAMWrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=8000)
    def test_box_prompt(self, mock_vram: MagicMock) -> None:
        """Box prompt should return SAM2Wrapper with enough VRAM."""
        from backend.cv.segmentation import SAM2Wrapper, get_segmenter

        segmenter = get_segmenter(prompt_type="box")
        assert isinstance(segmenter, SAM2Wrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=1000)
    def test_box_prompt_low_vram_raises(self, mock_vram: MagicMock) -> None:
        """Box prompt with insufficient VRAM should raise RuntimeError."""
        from backend.cv.segmentation import get_segmenter

        with pytest.raises(RuntimeError, match="SAM2 requires"):
            get_segmenter(prompt_type="box")


# -----------------------------------------------------------------------------
# Mask Conversion Tests
# -----------------------------------------------------------------------------


class TestMaskConversion:
    """Tests for mask conversion utilities."""

    def test_convert_empty_masks(self) -> None:
        """Should handle empty mask results."""
        from backend.cv.segmentation import _convert_sam_masks_to_result

        result = _convert_sam_masks_to_result(
            masks=np.array([]),
            scores=np.array([]),
            image_path="test.jpg",
            elapsed_ms=100.0,
            model_name="test",
        )
        assert len(result.masks) == 0

    def test_convert_single_mask(self) -> None:
        """Should convert single mask correctly."""
        from backend.cv.segmentation import _convert_sam_masks_to_result

        mask_data = np.ones((1, 64, 64), dtype=np.uint8)
        scores = np.array([0.95])

        result = _convert_sam_masks_to_result(
            masks=mask_data,
            scores=scores,
            image_path="test.jpg",
            elapsed_ms=150.0,
            model_name="sam3",
            prompt="car",
        )

        assert len(result.masks) == 1
        assert result.masks[0].confidence == 0.95
        assert result.masks[0].width == 64
        assert result.masks[0].height == 64
        assert result.prompt == "car"
        assert result.model_name == "sam3"

    def test_convert_multiple_masks_sorted(self) -> None:
        """Should sort masks by confidence (highest first)."""
        from backend.cv.segmentation import _convert_sam_masks_to_result

        masks_data = np.ones((3, 32, 32), dtype=np.uint8)
        scores = np.array([0.70, 0.95, 0.80])

        result = _convert_sam_masks_to_result(
            masks=masks_data,
            scores=scores,
            image_path="test.jpg",
            elapsed_ms=200.0,
            model_name="sam2",
        )

        assert len(result.masks) == 3
        # Should be sorted by confidence descending
        assert result.masks[0].confidence == 0.95
        assert result.masks[1].confidence == 0.80
        assert result.masks[2].confidence == 0.70

    def test_mask_to_numpy_roundtrip(self) -> None:
        """Mask should survive compression/decompression roundtrip."""
        from backend.cv.types import Mask

        original = np.zeros((100, 100), dtype=np.uint8)
        original[25:75, 25:75] = 1  # Central square

        mask = Mask.from_numpy(original, confidence=0.9)
        restored = mask.to_numpy()

        np.testing.assert_array_equal(original, restored)


# -----------------------------------------------------------------------------
# Integration Tests (require real models)
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestSAM2Integration:
    """Integration tests with real SAM2 model.

    These tests require:
    - ultralytics package installed
    - Model weights (will auto-download on first run)
    - Optionally GPU for faster execution
    """

    @pytest.fixture
    def segmenter(self):
        """Get real SAM2 segmenter."""
        pytest.importorskip("ultralytics")
        from backend.cv.segmentation import SAM2Wrapper

        s = SAM2Wrapper()
        s.load(device="cpu")  # Use CPU for CI compatibility
        yield s
        s.unload()

    @pytest.fixture
    def sample_image(self, tmp_path: Path) -> Path:
        """Create a sample test image."""
        pytest.importorskip("PIL")
        from PIL import Image

        img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        path = tmp_path / "test.jpg"
        img.save(path)
        return path

    def test_predict_returns_segmentation_result(
        self, segmenter, sample_image: Path
    ) -> None:
        """Predict should return SegmentationResult."""
        result = segmenter.predict(str(sample_image), point=(128, 128))
        assert isinstance(result, SegmentationResult)
        assert result.model_name == "sam2"
        assert result.image_path == str(sample_image)
        assert result.processing_time_ms > 0


@pytest.mark.integration
@pytest.mark.slow
class TestMobileSAMIntegration:
    """Integration tests with real MobileSAM model."""

    @pytest.fixture
    def segmenter(self):
        """Get real MobileSAM segmenter."""
        pytest.importorskip("ultralytics")
        from backend.cv.segmentation import MobileSAMWrapper

        s = MobileSAMWrapper()
        s.load(device="cpu")
        yield s
        s.unload()

    @pytest.fixture
    def sample_image(self, tmp_path: Path) -> Path:
        """Create a sample test image."""
        pytest.importorskip("PIL")
        from PIL import Image

        img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        path = tmp_path / "test.jpg"
        img.save(path)
        return path

    def test_predict_returns_segmentation_result(
        self, segmenter, sample_image: Path
    ) -> None:
        """Predict should return SegmentationResult."""
        result = segmenter.predict(str(sample_image), point=(128, 128))
        assert isinstance(result, SegmentationResult)
        assert result.model_name == "mobilesam"


# -----------------------------------------------------------------------------
# Performance Tests (GPU required)
# -----------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.gpu
class TestSegmentationPerformance:
    """Performance benchmark tests.

    These tests require:
    - CUDA-capable GPU
    - ultralytics package installed
    - Model weights downloaded
    """

    def test_mobilesam_inference_speed(self) -> None:
        """MobileSAM should achieve <100ms per image on GPU."""
        import time

        pytest.importorskip("torch")
        pytest.importorskip("ultralytics")

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from PIL import Image

        from backend.cv.segmentation import MobileSAMWrapper

        segmenter = MobileSAMWrapper()
        segmenter.load(device="cuda")

        # Create test image
        img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        img.save("/tmp/seg_bench.jpg")

        # Warm up
        for _ in range(3):
            segmenter.predict("/tmp/seg_bench.jpg", point=(256, 256))

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            segmenter.predict("/tmp/seg_bench.jpg", point=(256, 256))
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        segmenter.unload()

        assert avg_ms < 100, f"Inference too slow: {avg_ms:.2f}ms (expected <100ms)"

    def test_mobilesam_vram_budget(self) -> None:
        """MobileSAM should use <1.5GB VRAM."""
        pytest.importorskip("torch")
        pytest.importorskip("ultralytics")

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from backend.cv.device import get_vram_usage
        from backend.cv.segmentation import MobileSAMWrapper

        initial = get_vram_usage()[0]

        segmenter = MobileSAMWrapper()
        segmenter.load(device="cuda")

        loaded = get_vram_usage()[0]
        vram_used = loaded - initial

        segmenter.unload()

        assert vram_used < 1500, f"VRAM budget exceeded: {vram_used}MB (expected <1500MB)"
