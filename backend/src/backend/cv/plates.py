"""
License plate detection wrapper using YOLO-World with "license plate" prompt.

This module provides PlateDetectorWrapper which leverages YOLO-World's open-vocabulary
capabilities for license plate detection. Supports specialized fine-tuned models
as fallback when available.

Implements spec: 03-cv-models/05-plate-detection
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.cv.base import BaseModelWrapper, ModelInfo, ProgressCallback, register_model
from backend.cv.types import BBox, PlateDetection, PlateDetectionResult

if TYPE_CHECKING:
    from ultralytics import YOLO

    from backend.cv.openvocab import YOLOWorldWrapper

logger = logging.getLogger(__name__)


# Region-specific plate configurations
PLATE_REGIONS: dict[str, dict[str, Any]] = {
    "eu": {
        "aspect_ratio_range": (3.0, 5.5),
        "typical_size_mm": (520, 110),
        "description": "European standard plates",
    },
    "us": {
        "aspect_ratio_range": (1.8, 2.5),
        "typical_size_mm": (305, 152),
        "description": "US standard plates",
    },
    "china": {
        "aspect_ratio_range": (2.5, 4.0),
        "typical_size_mm": (440, 140),
        "description": "Chinese standard plates",
    },
}


@register_model
class PlateDetectorWrapper(BaseModelWrapper[PlateDetectionResult]):
    """
    License plate detection wrapper.

    Uses YOLO-World with "license plate" prompt as the primary approach.
    Supports specialized fine-tuned YOLO models as fallback when available.
    Includes aspect ratio validation to filter false positives.

    Attributes:
        info: Model metadata with VRAM requirements and capabilities.
        MIN_ASPECT_RATIO: Minimum valid plate width/height ratio.
        MAX_ASPECT_RATIO: Maximum valid plate width/height ratio.
    """

    info = ModelInfo(
        name="plate-detector",
        description="License plate detector (via YOLO-World or specialized model)",
        vram_required_mb=4500,  # Uses YOLO-World
        supports_batching=True,
        supports_gpu=True,
        source="ultralytics",
        version="1.0",
        extra={"prompt": "license plate"},
    )

    # Default aspect ratio constraints (can be overridden per-region)
    MIN_ASPECT_RATIO: float = 1.5
    MAX_ASPECT_RATIO: float = 6.0

    def __init__(self, use_specialized: bool = False) -> None:
        """
        Initialize plate detector.

        Args:
            use_specialized: If True, use specialized model if available.
        """
        super().__init__()
        self._use_specialized = use_specialized
        self._yolo_world: YOLOWorldWrapper | None = None
        self._specialized_model: YOLO | None = None

    def _check_specialized_model(self) -> bool:
        """
        Check if specialized plate detection model is available.

        Returns:
            True if specialized model exists, False otherwise.
        """
        from backend.cv.download import get_models_dir

        models_dir = get_models_dir()
        specialized_path = models_dir / "plate_detector" / "best.pt"
        return specialized_path.exists()

    def _get_specialized_model_path(self) -> Path:
        """Get path to specialized plate detection model."""
        from backend.cv.download import get_models_dir

        return get_models_dir() / "plate_detector" / "best.pt"

    def _load_model(self, device: str) -> None:
        """
        Load plate detection model.

        If use_specialized=True and specialized model exists, load it.
        Otherwise, load YOLO-World for open-vocabulary detection.

        Args:
            device: Target device ("cuda", "cpu", "mps").
        """
        if self._use_specialized and self._check_specialized_model():
            self._load_specialized(device)
        else:
            self._load_yolo_world(device)

    def _load_yolo_world(self, device: str) -> None:
        """
        Load YOLO-World for license plate detection.

        Args:
            device: Target device.
        """
        from backend.cv.openvocab import YOLOWorldWrapper

        logger.info("Loading YOLO-World for plate detection")
        self._yolo_world = YOLOWorldWrapper()
        self._yolo_world.load(device)

        # Sync device in case YOLO-World changed it internally
        self._device = self._yolo_world._device

        # Store reference for base class compatibility
        self._model = self._yolo_world._model

    def _load_specialized(self, device: str) -> None:
        """
        Load specialized plate detection model.

        Args:
            device: Target device.
        """
        import torch
        from ultralytics import YOLO

        model_path = self._get_specialized_model_path()
        logger.info("Loading specialized plate detector from %s", model_path)

        self._specialized_model = YOLO(str(model_path))

        try:
            if device == "cuda":
                if torch.cuda.is_available():
                    self._specialized_model.to(device)
                else:
                    logger.warning("CUDA requested but not available, using CPU")
                    device = "cpu"
            elif device == "mps":
                if torch.backends.mps.is_available():
                    self._specialized_model.to(device)
                else:
                    logger.warning("MPS requested but not available, using CPU")
                    device = "cpu"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM loading specialized model, falling back to CPU")
                self._specialized_model.to("cpu")
                device = "cpu"
            else:
                raise

        self._device = device

        # Store reference for base class compatibility
        self._model = self._specialized_model

    def _unload_model(self) -> None:
        """Unload model and free memory."""
        if self._yolo_world is not None:
            self._yolo_world.unload()
            self._yolo_world = None
        if self._specialized_model is not None:
            del self._specialized_model
            self._specialized_model = None

    def _filter_by_aspect_ratio(
        self,
        plates: list[PlateDetection],
    ) -> list[PlateDetection]:
        """
        Filter detections by typical license plate aspect ratio.

        Args:
            plates: List of plate detections.

        Returns:
            Filtered list with only valid aspect ratios.
        """
        filtered: list[PlateDetection] = []

        for plate in plates:
            if plate.bbox.height > 0:
                aspect_ratio = plate.bbox.width / plate.bbox.height
                if self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                    filtered.append(plate)
                else:
                    logger.debug(
                        "Filtered plate with aspect ratio %.2f (valid: %.1f-%.1f)",
                        aspect_ratio,
                        self.MIN_ASPECT_RATIO,
                        self.MAX_ASPECT_RATIO,
                    )

        return filtered

    def predict(
        self,
        input_path: str,
        *,
        confidence: float = 0.3,
        validate_aspect_ratio: bool = True,
        iou_threshold: float = 0.45,
        **kwargs: Any,
    ) -> PlateDetectionResult:
        """
        Detect license plates in an image.

        Args:
            input_path: Path to input image.
            confidence: Minimum confidence threshold (0-1).
            validate_aspect_ratio: If True, filter by plate aspect ratio.
            iou_threshold: IoU threshold for NMS.
            **kwargs: Additional arguments (ignored).

        Returns:
            PlateDetectionResult with list of detected plates.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start = time.perf_counter()

        if self._specialized_model is not None:
            plates = self._predict_specialized(input_path, confidence, iou_threshold)
        else:
            plates = self._predict_yoloworld(input_path, confidence, iou_threshold)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Apply aspect ratio filter
        if validate_aspect_ratio:
            plates = self._filter_by_aspect_ratio(plates)

        return PlateDetectionResult(
            plates=plates,
            image_path=input_path,
            processing_time_ms=elapsed_ms,
            model_name=self.info.name,
        )

    def _predict_yoloworld(
        self,
        input_path: str,
        confidence: float,
        iou_threshold: float,
    ) -> list[PlateDetection]:
        """
        Predict using YOLO-World with "license plate" prompt.

        Args:
            input_path: Path to input image.
            confidence: Minimum confidence threshold.
            iou_threshold: IoU threshold for NMS.

        Returns:
            List of PlateDetection objects.
        """
        if self._yolo_world is None:
            raise RuntimeError("YOLO-World not loaded")

        result = self._yolo_world.predict(
            input_path,
            prompt="license plate",
            confidence=confidence,
            iou_threshold=iou_threshold,
        )

        # Convert Detection objects to PlateDetection
        plates: list[PlateDetection] = []
        for det in result.detections:
            plates.append(
                PlateDetection(
                    bbox=det.bbox,
                    confidence=det.confidence,
                    text=None,  # OCR is out of scope
                    text_confidence=None,
                )
            )
        return plates

    def _predict_specialized(
        self,
        input_path: str,
        confidence: float,
        iou_threshold: float,
    ) -> list[PlateDetection]:
        """
        Predict using specialized plate detection model.

        Args:
            input_path: Path to input image.
            confidence: Minimum confidence threshold.
            iou_threshold: IoU threshold for NMS.

        Returns:
            List of PlateDetection objects.
        """
        if self._specialized_model is None:
            raise RuntimeError("Specialized model not loaded")

        raw_results = self._specialized_model.predict(
            input_path,
            conf=confidence,
            iou=iou_threshold,
            device=self._device,
            verbose=False,
        )
        result: Any = raw_results[0]

        plates: list[PlateDetection] = []

        if result.boxes is not None:
            for box in result.boxes:
                xywhn = box.xywhn[0]
                plates.append(
                    PlateDetection(
                        bbox=BBox(
                            x=float(xywhn[0]),
                            y=float(xywhn[1]),
                            width=float(xywhn[2]),
                            height=float(xywhn[3]),
                        ),
                        confidence=float(box.conf[0]),
                        text=None,
                        text_confidence=None,
                    )
                )

        return plates

    def predict_batch(
        self,
        input_paths: list[str],
        progress_callback: ProgressCallback | None = None,
        *,
        confidence: float = 0.3,
        validate_aspect_ratio: bool = True,
        iou_threshold: float = 0.45,
        **kwargs: Any,
    ) -> list[PlateDetectionResult]:
        """
        Batch plate detection with progress tracking.

        Args:
            input_paths: List of paths to input images.
            progress_callback: Optional callback with (current, total) progress.
            confidence: Minimum confidence threshold (0-1).
            validate_aspect_ratio: If True, filter by plate aspect ratio.
            iou_threshold: IoU threshold for NMS.
            **kwargs: Additional arguments (ignored).

        Returns:
            List of PlateDetectionResult, one per input image.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        results: list[PlateDetectionResult] = []
        total = len(input_paths)

        for i, path in enumerate(input_paths):
            result = self.predict(
                path,
                confidence=confidence,
                validate_aspect_ratio=validate_aspect_ratio,
                iou_threshold=iou_threshold,
            )
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results


def get_plate_detector(
    region: str | None = None,
    use_specialized: bool = False,
    force_yolo_world: bool = False,
) -> PlateDetectorWrapper:
    """
    Get plate detector configured for region.

    Factory function that returns a plate detector with region-specific
    aspect ratio configuration.

    Args:
        region: Plate region ("eu", "us", "china", None for auto/default).
        use_specialized: Use specialized model if available.
        force_yolo_world: Always use YOLO-World even if specialized exists.

    Returns:
        Configured PlateDetectorWrapper (unloaded - call load() before use).

    Example:
        detector = get_plate_detector(region="eu")
        detector.load()
        result = detector.predict("car.jpg")
        print(f"Found {result.count} plates")
        detector.unload()
    """
    # Override use_specialized if forcing YOLO-World
    if force_yolo_world:
        use_specialized = False

    detector = PlateDetectorWrapper(use_specialized=use_specialized)

    # Apply region-specific configuration
    if region and region in PLATE_REGIONS:
        config = PLATE_REGIONS[region]
        min_ratio, max_ratio = config["aspect_ratio_range"]
        detector.MIN_ASPECT_RATIO = min_ratio
        detector.MAX_ASPECT_RATIO = max_ratio
        logger.info(
            "Configured plate detector for %s region (aspect ratio: %.1f-%.1f)",
            region,
            min_ratio,
            max_ratio,
        )
    else:
        logger.info(
            "Using default plate detector (aspect ratio: %.1f-%.1f)",
            detector.MIN_ASPECT_RATIO,
            detector.MAX_ASPECT_RATIO,
        )

    return detector
