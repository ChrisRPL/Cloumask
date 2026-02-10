"""
Tests for SCRFD and YuNet face detection wrappers.

Unit tests use mocked models to avoid requiring real model downloads.
Integration tests (marked) require actual models and optionally GPU.

Implements spec: 03-cv-models/03-scrfd-faces (testing section)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.cv.base import ModelState
from backend.cv.types import FaceDetectionResult

# -----------------------------------------------------------------------------
# Mock Classes for Unit Testing (without real models)
# -----------------------------------------------------------------------------


class MockFace:
    """Mock InsightFace face detection result."""

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        det_score: float,
        kps: list[list[float]] | None = None,
    ) -> None:
        self.bbox = bbox
        self.det_score = det_score
        self.kps = np.array(kps) if kps else None


class MockFaceAnalysis:
    """Mock InsightFace FaceAnalysis class."""

    def __init__(
        self,
        name: str = "buffalo_sc",
        providers: list[str] | None = None,
        allowed_modules: list[str] | None = None,
    ) -> None:
        self.name = name
        self.providers = providers or ["CPUExecutionProvider"]
        self.allowed_modules = allowed_modules
        self._prepared = False

    def prepare(
        self, ctx_id: int = -1, det_size: tuple[int, int] = (640, 640)
    ) -> None:
        self._prepared = True
        self.ctx_id = ctx_id
        self.det_size = det_size

    def get(self, image: np.ndarray) -> list[MockFace]:
        """Return mock face detections."""
        # Return 2 mock faces
        return [
            MockFace(
                bbox=(100.0, 100.0, 200.0, 250.0),  # x1, y1, x2, y2
                det_score=0.98,
                kps=[
                    [120.0, 130.0],  # left eye
                    [180.0, 130.0],  # right eye
                    [150.0, 170.0],  # nose
                    [125.0, 210.0],  # left mouth
                    [175.0, 210.0],  # right mouth
                ],
            ),
            MockFace(
                bbox=(300.0, 150.0, 380.0, 280.0),
                det_score=0.75,
                kps=[
                    [320.0, 180.0],
                    [360.0, 180.0],
                    [340.0, 210.0],
                    [325.0, 250.0],
                    [355.0, 250.0],
                ],
            ),
        ]


class MockFaceDetectorYN:
    """Mock OpenCV FaceDetectorYN class."""

    def __init__(
        self,
        model: str,
        config: str,
        input_size: tuple[int, int],
        score_threshold: float,
        nms_threshold: float,
        top_k: int,
    ) -> None:
        self.model = model
        self.input_size = input_size
        self.score_threshold = score_threshold

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.input_size = size

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        """Return mock face detections in YuNet format."""
        # YuNet returns 15 values per face:
        # x, y, w, h, 5 landmarks (x,y pairs), confidence
        faces = np.array(
            [
                [
                    100.0,
                    100.0,
                    100.0,
                    150.0,  # x, y, w, h
                    120.0,
                    130.0,  # left eye
                    180.0,
                    130.0,  # right eye
                    150.0,
                    170.0,  # nose
                    125.0,
                    210.0,  # left mouth
                    175.0,
                    210.0,  # right mouth
                    0.95,  # confidence
                ],
                [
                    300.0,
                    150.0,
                    80.0,
                    130.0,
                    320.0,
                    180.0,
                    360.0,
                    180.0,
                    340.0,
                    210.0,
                    325.0,
                    250.0,
                    355.0,
                    250.0,
                    0.65,
                ],
            ],
            dtype=np.float32,
        )
        return 2, faces


# -----------------------------------------------------------------------------
# Fixtures for Mocking InsightFace
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_insightface():
    """Mock the insightface module before importing the faces module."""
    # Create mock module structure
    mock_app_module = MagicMock()
    mock_app_module.FaceAnalysis = MockFaceAnalysis

    mock_insightface_module = MagicMock()
    mock_insightface_module.app = mock_app_module

    # Patch sys.modules
    with patch.dict(
        sys.modules,
        {
            "insightface": mock_insightface_module,
            "insightface.app": mock_app_module,
        },
    ):
        yield mock_insightface_module


@pytest.fixture
def mock_cv2_imread():
    """Mock cv2.imread to return a valid image."""
    mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("cv2.imread", return_value=mock_image):
        yield mock_image


@pytest.fixture
def mock_cv2_cvtColor():
    """Mock cv2.cvtColor to return the same image."""
    mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("cv2.cvtColor", return_value=mock_image):
        yield


# -----------------------------------------------------------------------------
# SCRFDWrapper Unit Tests
# -----------------------------------------------------------------------------


class TestSCRFDWrapperUnit:
    """Unit tests for SCRFDWrapper with mocked model."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.faces import SCRFDWrapper

        wrapper = SCRFDWrapper()
        assert wrapper.info.name == "scrfd-10g"
        assert wrapper.info.vram_required_mb == 1500
        assert wrapper.info.supports_batching is True
        assert wrapper.info.source == "insightface"

    def test_initial_state(self) -> None:
        """Should start in unloaded state."""
        from backend.cv.faces import SCRFDWrapper

        wrapper = SCRFDWrapper()
        assert wrapper.state == ModelState.UNLOADED
        assert wrapper.is_loaded is False
        assert wrapper._face_app is None

    def test_predict_not_loaded_raises(self) -> None:
        """Predict should raise when model not loaded."""
        from backend.cv.faces import SCRFDWrapper

        wrapper = SCRFDWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict("test.jpg")

    def test_predict_batch_not_loaded_raises(self) -> None:
        """predict_batch should raise when model not loaded."""
        from backend.cv.faces import SCRFDWrapper

        wrapper = SCRFDWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict_batch(["test.jpg"])

    def test_load_and_predict(
        self,
        mock_insightface: Any,
        mock_cv2_imread: np.ndarray,
        mock_cv2_cvtColor: None,
    ) -> None:
        """Should load and predict successfully with mocked model."""
        from backend.cv.faces import SCRFDWrapper

        with patch("backend.cv.device.clear_gpu_memory"):
            wrapper = SCRFDWrapper()
            wrapper.load(device="cpu")

            assert wrapper.is_loaded
            assert wrapper._face_app is not None

            result = wrapper.predict("test.jpg", confidence=0.5)

            assert isinstance(result, FaceDetectionResult)
            assert result.model_name == "scrfd-10g"
            assert len(result.faces) == 2
            assert result.faces[0].confidence == 0.98
            assert result.faces[1].confidence == 0.75

            wrapper.unload()
            assert not wrapper.is_loaded

    def test_predict_with_confidence_filter(
        self,
        mock_insightface: Any,
        mock_cv2_imread: np.ndarray,
        mock_cv2_cvtColor: None,
    ) -> None:
        """Should filter by confidence threshold."""
        from backend.cv.faces import SCRFDWrapper

        with patch("backend.cv.device.clear_gpu_memory"):
            wrapper = SCRFDWrapper()
            wrapper.load(device="cpu")

            result = wrapper.predict("test.jpg", confidence=0.9)

            # Only face with 0.98 confidence should pass 0.9 threshold
            assert len(result.faces) == 1
            assert result.faces[0].confidence == 0.98

            wrapper.unload()

    def test_landmarks_included(
        self,
        mock_insightface: Any,
        mock_cv2_imread: np.ndarray,
        mock_cv2_cvtColor: None,
    ) -> None:
        """Should include 5-point landmarks when requested."""
        from backend.cv.faces import SCRFDWrapper

        with patch("backend.cv.device.clear_gpu_memory"):
            wrapper = SCRFDWrapper()
            wrapper.load(device="cpu")

            result = wrapper.predict("test.jpg", include_landmarks=True)

            assert result.faces[0].landmarks is not None
            assert len(result.faces[0].landmarks) == 5  # 5-point landmarks

            wrapper.unload()

    def test_landmarks_excluded(
        self,
        mock_insightface: Any,
        mock_cv2_imread: np.ndarray,
        mock_cv2_cvtColor: None,
    ) -> None:
        """Should exclude landmarks when not requested."""
        from backend.cv.faces import SCRFDWrapper

        with patch("backend.cv.device.clear_gpu_memory"):
            wrapper = SCRFDWrapper()
            wrapper.load(device="cpu")

            result = wrapper.predict("test.jpg", include_landmarks=False)

            assert result.faces[0].landmarks is None

            wrapper.unload()

    def test_predict_batch(
        self,
        mock_insightface: Any,
        mock_cv2_imread: np.ndarray,
        mock_cv2_cvtColor: None,
    ) -> None:
        """Should process batch of images."""
        from backend.cv.faces import SCRFDWrapper

        with patch("backend.cv.device.clear_gpu_memory"):
            wrapper = SCRFDWrapper()
            wrapper.load(device="cpu")

            inputs = ["a.jpg", "b.jpg", "c.jpg"]
            results = wrapper.predict_batch(inputs, confidence=0.5)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.image_path == inputs[i]
                assert result.model_name == "scrfd-10g"

            wrapper.unload()

    def test_predict_batch_with_progress(
        self,
        mock_insightface: Any,
        mock_cv2_imread: np.ndarray,
        mock_cv2_cvtColor: None,
    ) -> None:
        """Should call progress callback during batch processing."""
        from backend.cv.faces import SCRFDWrapper

        with patch("backend.cv.device.clear_gpu_memory"):
            wrapper = SCRFDWrapper()
            wrapper.load(device="cpu")

            progress_calls: list[tuple[int, int]] = []

            def callback(current: int, total: int) -> None:
                progress_calls.append((current, total))

            inputs = ["a.jpg", "b.jpg", "c.jpg"]
            wrapper.predict_batch(inputs, progress_callback=callback)

            assert len(progress_calls) == 3
            assert progress_calls[-1] == (3, 3)  # Final call

            wrapper.unload()

    def test_invalid_image_raises(self, mock_insightface: Any) -> None:
        """Should raise ValueError for invalid image."""
        from backend.cv.faces import SCRFDWrapper

        with (
            patch("cv2.imread", return_value=None),
            patch("backend.cv.device.clear_gpu_memory"),
        ):
            wrapper = SCRFDWrapper()
            wrapper.load(device="cpu")

            with pytest.raises(ValueError, match="Could not read image"):
                wrapper.predict("nonexistent.jpg")

            wrapper.unload()


# -----------------------------------------------------------------------------
# YuNetWrapper Unit Tests
# -----------------------------------------------------------------------------


class TestYuNetWrapperUnit:
    """Unit tests for YuNetWrapper."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.faces import YuNetWrapper

        wrapper = YuNetWrapper()
        assert wrapper.info.name == "yunet"
        assert wrapper.info.vram_required_mb == 200
        assert wrapper.info.supports_batching is False
        assert wrapper.info.supports_gpu is False

    def test_initial_state(self) -> None:
        """Should start in unloaded state."""
        from backend.cv.faces import YuNetWrapper

        wrapper = YuNetWrapper()
        assert wrapper.is_loaded is False
        assert wrapper._detector is None

    def test_predict_not_loaded_raises(self) -> None:
        """Predict should raise when model not loaded."""
        from backend.cv.faces import YuNetWrapper

        wrapper = YuNetWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict("test.jpg")

    def test_load_and_predict(self) -> None:
        """Should load and predict successfully with mocked model."""
        from backend.cv.faces import YuNetWrapper

        # Mock detector
        mock_detector = MockFaceDetectorYN(
            model="fake.onnx",
            config="",
            input_size=(320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )

        # Mock image reading
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)

        with (
            patch("cv2.FaceDetectorYN.create", return_value=mock_detector),
            patch("cv2.imread", return_value=mock_image),
            patch.object(YuNetWrapper, "_get_model_path", return_value=Path("/fake.onnx")),
            patch("backend.cv.device.clear_gpu_memory"),
        ):
            wrapper = YuNetWrapper()
            wrapper.load(device="cpu")

            assert wrapper.is_loaded
            assert wrapper._detector is not None

            result = wrapper.predict("test.jpg", confidence=0.5)

            assert isinstance(result, FaceDetectionResult)
            assert result.model_name == "yunet"
            assert len(result.faces) == 2
            assert result.faces[0].confidence == pytest.approx(0.95, abs=0.001)
            assert result.faces[1].confidence == pytest.approx(0.65, abs=0.001)

            wrapper.unload()
            assert not wrapper.is_loaded

    def test_predict_with_confidence_filter(self) -> None:
        """Should filter by confidence threshold."""
        from backend.cv.faces import YuNetWrapper

        mock_detector = MockFaceDetectorYN(
            model="fake.onnx",
            config="",
            input_size=(320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )

        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)

        with (
            patch("cv2.FaceDetectorYN.create", return_value=mock_detector),
            patch("cv2.imread", return_value=mock_image),
            patch.object(YuNetWrapper, "_get_model_path", return_value=Path("/fake.onnx")),
            patch("backend.cv.device.clear_gpu_memory"),
        ):
            wrapper = YuNetWrapper()
            wrapper.load(device="cpu")

            result = wrapper.predict("test.jpg", confidence=0.9)

            # Only face with 0.95 confidence should pass 0.9 threshold
            assert len(result.faces) == 1
            assert result.faces[0].confidence == pytest.approx(0.95, abs=0.001)

            wrapper.unload()

    def test_yunet_landmarks(self) -> None:
        """Should include 5-point landmarks."""
        from backend.cv.faces import YuNetWrapper

        mock_detector = MockFaceDetectorYN(
            model="fake.onnx",
            config="",
            input_size=(320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )

        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)

        with (
            patch("cv2.FaceDetectorYN.create", return_value=mock_detector),
            patch("cv2.imread", return_value=mock_image),
            patch.object(YuNetWrapper, "_get_model_path", return_value=Path("/fake.onnx")),
            patch("backend.cv.device.clear_gpu_memory"),
        ):
            wrapper = YuNetWrapper()
            wrapper.load(device="cpu")

            result = wrapper.predict("test.jpg", include_landmarks=True)

            assert result.faces[0].landmarks is not None
            assert len(result.faces[0].landmarks) == 5

            wrapper.unload()

    def test_no_faces_detected(self) -> None:
        """Should handle images with no faces."""
        from backend.cv.faces import YuNetWrapper

        # Create mock detector that returns no faces
        mock_detector = MagicMock()
        mock_detector.detect.return_value = (0, None)
        mock_detector.setInputSize = MagicMock()

        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)

        with (
            patch("cv2.FaceDetectorYN.create", return_value=mock_detector),
            patch("cv2.imread", return_value=mock_image),
            patch.object(YuNetWrapper, "_get_model_path", return_value=Path("/fake.onnx")),
            patch("backend.cv.device.clear_gpu_memory"),
        ):
            wrapper = YuNetWrapper()
            wrapper.load(device="cpu")

            result = wrapper.predict("test.jpg")

            assert len(result.faces) == 0
            assert result.count == 0

            wrapper.unload()


# -----------------------------------------------------------------------------
# get_face_detector Factory Tests
# -----------------------------------------------------------------------------


class TestGetFaceDetector:
    """Tests for get_face_detector factory function."""

    def test_default_returns_scrfd(self) -> None:
        """Default should return SCRFDWrapper when VRAM is available."""
        from backend.cv.faces import SCRFDWrapper, get_face_detector

        with patch("backend.cv.device.get_available_vram_mb", return_value=2000):
            detector = get_face_detector()
            assert isinstance(detector, SCRFDWrapper)

    def test_force_yunet(self) -> None:
        """force_model='yunet' should return YuNetWrapper."""
        from backend.cv.faces import YuNetWrapper, get_face_detector

        detector = get_face_detector(force_model="yunet")
        assert isinstance(detector, YuNetWrapper)

    def test_force_scrfd(self) -> None:
        """force_model='scrfd-10g' should return SCRFDWrapper."""
        from backend.cv.faces import SCRFDWrapper, get_face_detector

        detector = get_face_detector(force_model="scrfd-10g")
        assert isinstance(detector, SCRFDWrapper)

    def test_realtime_returns_yunet(self) -> None:
        """realtime=True should return YuNetWrapper."""
        from backend.cv.faces import YuNetWrapper, get_face_detector

        detector = get_face_detector(realtime=True)
        assert isinstance(detector, YuNetWrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=2000)
    def test_enough_vram_returns_scrfd(self, mock_vram: MagicMock) -> None:
        """With enough VRAM should return SCRFDWrapper."""
        from backend.cv.faces import SCRFDWrapper, get_face_detector

        detector = get_face_detector()
        assert isinstance(detector, SCRFDWrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=500)
    def test_low_vram_returns_yunet(self, mock_vram: MagicMock) -> None:
        """With low VRAM should fall back to YuNetWrapper."""
        from backend.cv.faces import YuNetWrapper, get_face_detector

        detector = get_face_detector()
        assert isinstance(detector, YuNetWrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=1500)
    def test_exact_vram_returns_scrfd(self, mock_vram: MagicMock) -> None:
        """With exactly enough VRAM should return SCRFDWrapper."""
        from backend.cv.faces import SCRFDWrapper, get_face_detector

        detector = get_face_detector()
        assert isinstance(detector, SCRFDWrapper)


# -----------------------------------------------------------------------------
# BBox Conversion Tests
# -----------------------------------------------------------------------------


class TestBBoxConversion:
    """Tests for bbox coordinate conversion in face detection."""

    def test_bbox_normalized_correctly(
        self,
        mock_insightface: Any,
        mock_cv2_imread: np.ndarray,
        mock_cv2_cvtColor: None,
    ) -> None:
        """Should convert bbox to normalized center coordinates."""
        from backend.cv.faces import SCRFDWrapper

        with patch("backend.cv.device.clear_gpu_memory"):
            wrapper = SCRFDWrapper()
            wrapper.load(device="cpu")

            result = wrapper.predict("test.jpg")

            # First mock face has bbox (100, 100, 200, 250) in 640x480 image
            # Center x = (100 + 200) / 2 / 640 = 150/640 = 0.234375
            # Center y = (100 + 250) / 2 / 480 = 175/480 = 0.364583...
            # Width = (200 - 100) / 640 = 100/640 = 0.15625
            # Height = (250 - 100) / 480 = 150/480 = 0.3125

            bbox = result.faces[0].bbox
            assert abs(bbox.x - 0.234375) < 0.0001
            assert abs(bbox.y - 175 / 480) < 0.0001
            assert abs(bbox.width - 0.15625) < 0.0001
            assert abs(bbox.height - 0.3125) < 0.0001

            wrapper.unload()


# -----------------------------------------------------------------------------
# FaceDetection Type Tests
# -----------------------------------------------------------------------------


class TestFaceDetectionType:
    """Tests for FaceDetection pydantic model."""

    def test_eye_distance_calculation(self) -> None:
        """Should calculate inter-eye distance correctly."""
        from backend.cv.types import BBox, FaceDetection

        face = FaceDetection(
            bbox=BBox(x=0.5, y=0.5, width=0.2, height=0.3),
            confidence=0.95,
            landmarks=[
                (0.4, 0.4),  # left eye
                (0.6, 0.4),  # right eye
                (0.5, 0.5),  # nose
                (0.45, 0.6),  # left mouth
                (0.55, 0.6),  # right mouth
            ],
        )

        # Image 640x480
        distance = face.get_eye_distance(640, 480)
        assert distance is not None

        # (0.6 - 0.4) * 640 = 128 pixels, no vertical difference
        assert abs(distance - 128.0) < 0.001

    def test_eye_distance_no_landmarks(self) -> None:
        """Should return None when no landmarks."""
        from backend.cv.types import BBox, FaceDetection

        face = FaceDetection(
            bbox=BBox(x=0.5, y=0.5, width=0.2, height=0.3),
            confidence=0.95,
            landmarks=None,
        )

        assert face.get_eye_distance(640, 480) is None


# -----------------------------------------------------------------------------
# Integration Tests (require real models)
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestSCRFDIntegration:
    """Integration tests with real SCRFD model.

    These tests require:
    - insightface package installed
    - Model weights (will auto-download on first run)
    - Optionally GPU for faster execution
    """

    @pytest.fixture
    def detector(self):
        """Get real SCRFD detector."""
        pytest.importorskip("insightface")
        from backend.cv.faces import SCRFDWrapper

        d = SCRFDWrapper()
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

    def test_predict_returns_face_detection_result(
        self, detector: Any, sample_image: Path
    ) -> None:
        """Predict should return FaceDetectionResult."""
        result = detector.predict(str(sample_image))
        assert isinstance(result, FaceDetectionResult)
        assert result.model_name == "scrfd-10g"
        assert result.image_path == str(sample_image)
        assert result.processing_time_ms > 0


@pytest.mark.integration
@pytest.mark.slow
class TestYuNetIntegration:
    """Integration tests with real YuNet model."""

    @pytest.fixture
    def detector(self):
        """Get real YuNet detector."""
        pytest.importorskip("cv2")
        from backend.cv.faces import YuNetWrapper

        d = YuNetWrapper()
        d.load(device="cpu")
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

    def test_predict_returns_face_detection_result(
        self, detector: Any, sample_image: Path
    ) -> None:
        """Predict should return FaceDetectionResult."""
        result = detector.predict(str(sample_image))
        assert isinstance(result, FaceDetectionResult)
        assert result.model_name == "yunet"


# -----------------------------------------------------------------------------
# Performance Tests (GPU required)
# -----------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.gpu
class TestFaceDetectionBenchmarks:
    """Performance benchmark tests.

    These tests require:
    - CUDA-capable GPU
    - insightface package installed
    - Model weights downloaded
    """

    def test_scrfd_inference_speed(self) -> None:
        """SCRFD should achieve <10ms per image on GPU."""
        import time

        pytest.importorskip("torch")
        pytest.importorskip("insightface")

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from PIL import Image

        from backend.cv.faces import SCRFDWrapper

        detector = SCRFDWrapper()
        detector.load(device="cuda")

        # Create test image
        img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        img.save("/tmp/face_bench.jpg")

        # Warm up
        for _ in range(3):
            detector.predict("/tmp/face_bench.jpg")

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            detector.predict("/tmp/face_bench.jpg")
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        detector.unload()

        assert avg_ms < 10, f"Inference too slow: {avg_ms:.2f}ms (expected <10ms)"

    def test_scrfd_vram_budget(self) -> None:
        """SCRFD should use <1.5GB VRAM."""
        pytest.importorskip("torch")
        pytest.importorskip("insightface")

        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from backend.cv.device import get_vram_usage
        from backend.cv.faces import SCRFDWrapper

        initial = get_vram_usage()[0]

        detector = SCRFDWrapper()
        detector.load(device="cuda")

        loaded = get_vram_usage()[0]
        vram_used = loaded - initial

        detector.unload()

        assert vram_used < 1500, f"VRAM budget exceeded: {vram_used}MB (expected <1500MB)"

    def test_yunet_cpu_speed(self) -> None:
        """YuNet should achieve <5ms per image on CPU."""
        import os
        import time

        if os.getenv("RUN_BENCHMARK_TESTS") != "1":
            pytest.skip("CPU benchmark tests are disabled by default")

        pytest.importorskip("cv2")

        from PIL import Image

        from backend.cv.faces import YuNetWrapper

        detector = YuNetWrapper()
        detector.load(device="cpu")

        # Create test image
        img = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        img.save("/tmp/yunet_bench.jpg")

        # Warm up
        for _ in range(3):
            detector.predict("/tmp/yunet_bench.jpg")

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            detector.predict("/tmp/yunet_bench.jpg")
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        detector.unload()

        assert avg_ms < 5, f"Inference too slow: {avg_ms:.2f}ms (expected <5ms)"
