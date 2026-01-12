"""
SCRFD and YuNet face detection wrappers.

This module provides SCRFD-10G as the primary face detector (via InsightFace)
and YuNet as a lightweight CPU fallback (via OpenCV DNN). Both support
5-point facial landmark detection for blur alignment.

Implements spec: 03-cv-models/03-scrfd-faces
"""

from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.cv.base import BaseModelWrapper, ModelInfo, ProgressCallback, register_model
from backend.cv.types import BBox, FaceDetection, FaceDetectionResult

if TYPE_CHECKING:
    import cv2
    from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SCRFD Wrapper (Primary - GPU)
# -----------------------------------------------------------------------------


@register_model
class SCRFDWrapper(BaseModelWrapper[FaceDetectionResult]):
    """
    SCRFD-10G face detection wrapper using InsightFace.

    Primary face detector offering state-of-the-art accuracy on WIDER FACE
    benchmark (95%+) while maintaining fast inference (5-10ms on GPU).
    Supports 5-point facial landmark detection.

    Attributes:
        info: Model metadata with VRAM requirements and capabilities.
    """

    info = ModelInfo(
        name="scrfd-10g",
        description="SCRFD-10G face detector via InsightFace (5-point landmarks)",
        vram_required_mb=1500,
        supports_batching=True,
        supports_gpu=True,
        source="insightface",
        version="1.0",
        extra={"landmarks": 5, "benchmark": "WIDER FACE 95%+"},
    )

    def __init__(self) -> None:
        """Initialize SCRFD wrapper."""
        super().__init__()
        self._face_app: FaceAnalysis | None = None

    def _load_model(self, device: str) -> None:
        """
        Load SCRFD model via InsightFace FaceAnalysis.

        Args:
            device: Target device ("cuda", "cpu", "mps").
        """
        from insightface.app import FaceAnalysis

        # Determine ONNX execution provider
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        elif device == "mps":
            # MPS not directly supported by ONNX Runtime, fallback to CPU
            logger.warning("MPS not supported by InsightFace, using CPU")
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        logger.info("Loading InsightFace FaceAnalysis with providers: %s", providers)

        # FaceAnalysis with buffalo_sc (SCRFD-based lightweight pack)
        # name="buffalo_sc" uses SCRFD for detection
        self._face_app = FaceAnalysis(
            name="buffalo_sc",
            providers=providers,
            allowed_modules=["detection"],  # Only load detection, not recognition
        )

        # Prepare with detection size (affects accuracy/speed tradeoff)
        self._face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # Store reference for base class
        self._model = self._face_app

        logger.info("SCRFD model loaded via InsightFace on %s", device)

    def _unload_model(self) -> None:
        """Unload model and free memory."""
        if self._face_app is not None:
            del self._face_app
            self._face_app = None

    def predict(
        self,
        input_path: str,
        *,
        confidence: float = 0.5,
        include_landmarks: bool = True,
        **kwargs: Any,
    ) -> FaceDetectionResult:
        """
        Detect faces in an image.

        Args:
            input_path: Path to input image.
            confidence: Minimum confidence threshold (0-1).
            include_landmarks: Include 5-point facial landmarks.
            **kwargs: Additional arguments (ignored).

        Returns:
            FaceDetectionResult with list of detected faces.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If image cannot be read.
        """
        import cv2

        if not self.is_loaded or self._face_app is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Read image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")

        # Convert BGR to RGB (InsightFace expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        start = time.perf_counter()
        faces = self._face_app.get(image_rgb)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Convert to FaceDetection results
        results: list[FaceDetection] = []
        for face in faces:
            # Check confidence threshold
            det_score = float(face.det_score)
            if det_score < confidence:
                continue

            # Convert bbox to normalized coordinates (center format)
            x1, y1, x2, y2 = face.bbox
            bbox = BBox(
                x=(x1 + x2) / 2 / w,
                y=(y1 + y2) / 2 / h,
                width=(x2 - x1) / w,
                height=(y2 - y1) / h,
            )

            # Extract 5-point landmarks (normalized)
            landmarks: list[tuple[float, float]] | None = None
            if include_landmarks and face.kps is not None:
                landmarks = [
                    (float(kp[0]) / w, float(kp[1]) / h) for kp in face.kps
                ]

            results.append(
                FaceDetection(
                    bbox=bbox,
                    confidence=det_score,
                    landmarks=landmarks,
                )
            )

        # Sort by confidence descending
        results.sort(key=lambda f: f.confidence, reverse=True)

        return FaceDetectionResult(
            faces=results,
            image_path=input_path,
            processing_time_ms=elapsed_ms,
            model_name=self.info.name,
        )

    def predict_batch(
        self,
        input_paths: list[str],
        progress_callback: ProgressCallback | None = None,
        *,
        confidence: float = 0.5,
        include_landmarks: bool = True,
        **kwargs: Any,
    ) -> list[FaceDetectionResult]:
        """
        Batch face detection with progress tracking.

        Note: InsightFace doesn't have native batch support, so this
        iterates over inputs. For true batch processing, consider
        using multiple workers.

        Args:
            input_paths: List of paths to input images.
            progress_callback: Optional callback with (current, total) progress.
            confidence: Minimum confidence threshold (0-1).
            include_landmarks: Include 5-point facial landmarks.
            **kwargs: Additional arguments (ignored).

        Returns:
            List of FaceDetectionResult, one per input image.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        results: list[FaceDetectionResult] = []
        total = len(input_paths)

        for i, path in enumerate(input_paths):
            result = self.predict(
                path,
                confidence=confidence,
                include_landmarks=include_landmarks,
            )
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results


# -----------------------------------------------------------------------------
# YuNet Wrapper (Fallback - CPU)
# -----------------------------------------------------------------------------


# YuNet model download URL from OpenCV Zoo
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"


@register_model
class YuNetWrapper(BaseModelWrapper[FaceDetectionResult]):
    """
    YuNet face detection wrapper using OpenCV DNN.

    Lightweight CPU-based face detector for real-time applications.
    Lower accuracy than SCRFD but runs efficiently without GPU.

    Attributes:
        info: Model metadata with VRAM requirements and capabilities.
    """

    info = ModelInfo(
        name="yunet",
        description="YuNet face detector via OpenCV DNN (CPU-optimized)",
        vram_required_mb=200,  # CPU memory estimate
        supports_batching=False,
        supports_gpu=False,  # Always runs on CPU
        source="opencv",
        version="2023mar",
        extra={"landmarks": 5, "benchmark": "WIDER FACE ~90%"},
    )

    def __init__(self) -> None:
        """Initialize YuNet wrapper."""
        super().__init__()
        self._detector: cv2.FaceDetectorYN | None = None

    def _get_model_path(self) -> Path:
        """
        Get path to YuNet model, downloading if needed.

        Returns:
            Path to YuNet ONNX model.
        """
        from backend.cv.download import get_models_dir

        models_dir = get_models_dir()
        model_path = models_dir / "yunet" / YUNET_MODEL_FILENAME

        if not model_path.exists():
            logger.info("Downloading YuNet model from OpenCV Zoo...")
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress reporting
            try:
                urllib.request.urlretrieve(YUNET_MODEL_URL, model_path)
                logger.info("YuNet model downloaded to %s", model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download YuNet model: {e}") from e

        return model_path

    def _load_model(self, device: str) -> None:
        """
        Load YuNet model via OpenCV DNN.

        Note: YuNet always runs on CPU via OpenCV DNN backend.

        Args:
            device: Target device (ignored - always CPU).
        """
        import cv2

        if device != "cpu":
            logger.info("YuNet always runs on CPU, ignoring device=%s", device)

        model_path = self._get_model_path()

        logger.info("Loading YuNet from %s", model_path)

        self._detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(320, 320),  # Default input size
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )

        # Store reference for base class
        self._model = self._detector

        logger.info("YuNet model loaded (CPU)")

    def _unload_model(self) -> None:
        """Unload model."""
        if self._detector is not None:
            del self._detector
            self._detector = None

    def predict(
        self,
        input_path: str,
        *,
        confidence: float = 0.5,
        include_landmarks: bool = True,
        **kwargs: Any,
    ) -> FaceDetectionResult:
        """
        Detect faces using YuNet.

        YuNet returns 15 values per face:
        - bbox: x, y, w, h (4 values)
        - landmarks: 5 points (10 values)
        - confidence: 1 value

        Args:
            input_path: Path to input image.
            confidence: Minimum confidence threshold (0-1).
            include_landmarks: Include 5-point facial landmarks.
            **kwargs: Additional arguments (ignored).

        Returns:
            FaceDetectionResult with list of detected faces.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If image cannot be read.
        """
        import cv2

        if not self.is_loaded or self._detector is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Read image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")

        h, w = image.shape[:2]

        # Set input size to match image dimensions
        self._detector.setInputSize((w, h))

        start = time.perf_counter()
        _, faces = self._detector.detect(image)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results: list[FaceDetection] = []

        if faces is None:
            return FaceDetectionResult(
                faces=results,
                image_path=input_path,
                processing_time_ms=elapsed_ms,
                model_name=self.info.name,
            )

        for i in range(len(faces)):
            face = faces[i]  # type: ignore[index]
            # Confidence is at index 14
            det_score = float(face[14])
            if det_score < confidence:
                continue

            # Bbox: x, y, w, h (top-left corner format)
            fx, fy, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])

            # Convert to normalized center format
            bbox = BBox(
                x=(fx + fw / 2) / w,
                y=(fy + fh / 2) / h,
                width=fw / w,
                height=fh / h,
            )

            # Extract 5-point landmarks (normalized)
            # Landmarks start at index 4: (x0,y0), (x1,y1), ..., (x4,y4)
            landmarks: list[tuple[float, float]] | None = None
            if include_landmarks:
                landmarks = [
                    (float(face[4 + j * 2]) / w, float(face[5 + j * 2]) / h)
                    for j in range(5)
                ]

            results.append(
                FaceDetection(
                    bbox=bbox,
                    confidence=det_score,
                    landmarks=landmarks,
                )
            )

        # Sort by confidence descending
        results.sort(key=lambda f: f.confidence, reverse=True)

        return FaceDetectionResult(
            faces=results,
            image_path=input_path,
            processing_time_ms=elapsed_ms,
            model_name=self.info.name,
        )


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------


def get_face_detector(
    realtime: bool = False,
    force_model: str | None = None,
) -> BaseModelWrapper[FaceDetectionResult]:
    """
    Get appropriate face detector based on requirements and available resources.

    Factory function that selects the best face detector for the given
    requirements. By default returns SCRFD for accuracy. With realtime=True,
    returns YuNet for speed.

    Args:
        realtime: If True, prefer YuNet for speed over accuracy.
        force_model: Force specific model ("scrfd-10g" or "yunet").

    Returns:
        Appropriate face detector wrapper (unloaded - call load() before use).

    Example:
        detector = get_face_detector(realtime=True)
        detector.load()
        result = detector.predict("photo.jpg", confidence=0.5)
        print(f"Found {result.count} faces")
        detector.unload()
    """
    from backend.cv.device import get_available_vram_mb

    # Force specific model if requested
    if force_model == "yunet":
        logger.info("Returning YuNet face detector (forced)")
        return YuNetWrapper()
    elif force_model == "scrfd-10g":
        logger.info("Returning SCRFD-10G face detector (forced)")
        return SCRFDWrapper()

    # Prefer YuNet for realtime mode
    if realtime:
        logger.info("Returning YuNet face detector (realtime mode)")
        return YuNetWrapper()

    # Check if we have enough VRAM for SCRFD
    available = get_available_vram_mb()
    if available >= SCRFDWrapper.info.vram_required_mb:
        logger.info(
            "Selecting SCRFD-10G face detector (VRAM available: %dMB)",
            available,
        )
        return SCRFDWrapper()

    # Fallback to YuNet if insufficient VRAM
    logger.info(
        "SCRFD needs %dMB VRAM, only %dMB available, using YuNet",
        SCRFDWrapper.info.vram_required_mb,
        available,
    )
    return YuNetWrapper()
