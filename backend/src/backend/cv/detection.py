"""
YOLO11 and RT-DETR object detection wrappers.

This module provides YOLO11m as the primary detector and RT-DETR-l as a
fallback for higher accuracy requirements. Both use the ultralytics library.

Implements spec: 03-cv-models/01-yolo11-detection
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from backend.cv.base import BaseModelWrapper, ModelInfo, ProgressCallback, register_model
from backend.cv.types import BBox, Detection, DetectionResult

if TYPE_CHECKING:
    from ultralytics import RTDETR, YOLO

logger = logging.getLogger(__name__)


# COCO class names (80 classes)
COCO_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def get_class_indices(class_names: list[str] | None) -> list[int] | None:
    """
    Convert class names to COCO indices.

    Args:
        class_names: List of class names to convert. None returns None.

    Returns:
        List of COCO indices, or None if input is None or empty.
    """
    if not class_names:
        return None

    indices: list[int] = []
    for name in class_names:
        name_lower = name.lower().strip()
        if name_lower in COCO_CLASSES:
            indices.append(COCO_CLASSES.index(name_lower))
        else:
            # Try case-insensitive match
            for i, coco_name in enumerate(COCO_CLASSES):
                if coco_name.lower() == name_lower:
                    indices.append(i)
                    break
            else:
                logger.warning("Unknown class name: %s (not in COCO classes)", name)

    return indices if indices else None


def _convert_ultralytics_result(result: Any) -> list[Detection]:
    """
    Convert ultralytics result to Detection list.

    Args:
        result: Single ultralytics prediction result.

    Returns:
        List of Detection objects.
    """
    detections: list[Detection] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0])

        # Ensure class_id is valid
        if cls_id < 0 or cls_id >= len(COCO_CLASSES):
            logger.warning("Invalid class ID: %d, skipping", cls_id)
            continue

        # Get normalized xywh (center format) - ultralytics uses xywhn for normalized
        xywhn = box.xywhn[0]

        detections.append(
            Detection(
                class_id=cls_id,
                class_name=COCO_CLASSES[cls_id],
                bbox=BBox(
                    x=float(xywhn[0]),
                    y=float(xywhn[1]),
                    width=float(xywhn[2]),
                    height=float(xywhn[3]),
                ),
                confidence=float(box.conf[0]),
            )
        )

    return detections


@register_model
class YOLO11Wrapper(BaseModelWrapper[DetectionResult]):
    """
    YOLO11m object detection wrapper.

    Primary detector offering excellent speed-accuracy trade-off for
    real-time detection. Supports 80 COCO classes.

    Attributes:
        info: Model metadata with VRAM requirements and capabilities.
    """

    info = ModelInfo(
        name="yolo11m",
        description="YOLO11m object detector (COCO 80 classes)",
        vram_required_mb=2500,
        supports_batching=True,
        supports_gpu=True,
        source="ultralytics",
        version="11.0",
    )

    def __init__(self) -> None:
        """Initialize YOLO11 wrapper."""
        super().__init__()
        self._yolo: YOLO | None = None

    def _load_model(self, device: str) -> None:
        """
        Load YOLO11m model.

        Args:
            device: Target device ("cuda", "cpu", "mps").
        """
        from ultralytics import YOLO

        from backend.cv.download import download_model, get_model_path

        # Get model path - ultralytics auto-downloads if not present
        model_path = get_model_path("yolo11m")
        if not model_path.exists():
            download_model("yolo11m")

        logger.info("Loading YOLO11m from %s", model_path)
        self._yolo = YOLO(str(model_path))

        # Move to device if CUDA
        if device == "cuda":
            import torch

            if torch.cuda.is_available():
                self._yolo.to(device)
            else:
                logger.warning("CUDA requested but not available, using CPU")
                device = "cpu"

        # Warm up model for consistent inference times
        self._warmup(device)

        # Store reference for base class compatibility
        self._model = self._yolo

    def _warmup(self, device: str) -> None:
        """
        Warm up model with dummy inference.

        Args:
            device: Current device.
        """
        import numpy as np

        # Create dummy image tensor (640x640 RGB)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._yolo.predict(dummy, verbose=False, device=device)  # type: ignore[union-attr]
        logger.debug("YOLO11m warmed up on %s", device)

    def _unload_model(self) -> None:
        """Unload model and free memory."""
        if self._yolo is not None:
            del self._yolo
            self._yolo = None

    def predict(
        self,
        input_path: str,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        iou_threshold: float = 0.45,
        **kwargs: Any,
    ) -> DetectionResult:
        """
        Run object detection on an image.

        Args:
            input_path: Path to input image.
            confidence: Minimum confidence threshold (0-1).
            classes: List of class names to detect (None = all COCO classes).
            iou_threshold: IoU threshold for NMS.
            **kwargs: Additional arguments (ignored).

        Returns:
            DetectionResult with list of detections.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded or self._yolo is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert class names to indices
        class_indices = get_class_indices(classes)

        start = time.perf_counter()
        results = self._yolo.predict(
            input_path,
            conf=confidence,
            iou=iou_threshold,
            classes=class_indices,
            device=self._device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Convert results (predict returns a list, we want first item)
        detections = _convert_ultralytics_result(results[0])

        return DetectionResult(
            detections=detections,
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
        classes: list[str] | None = None,
        iou_threshold: float = 0.45,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> list[DetectionResult]:
        """
        Batch inference with progress tracking.

        Args:
            input_paths: List of paths to input images.
            progress_callback: Optional callback with (current, total) progress.
            confidence: Minimum confidence threshold (0-1).
            classes: List of class names to detect (None = all COCO classes).
            iou_threshold: IoU threshold for NMS.
            batch_size: Number of images per mini-batch.
            **kwargs: Additional arguments (ignored).

        Returns:
            List of DetectionResult, one per input image.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded or self._yolo is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        class_indices = get_class_indices(classes)
        results_list: list[DetectionResult] = []
        total = len(input_paths)

        # Process in mini-batches for memory efficiency
        for i in range(0, total, batch_size):
            batch = input_paths[i : i + batch_size]

            start = time.perf_counter()
            batch_results = self._yolo.predict(
                batch,
                conf=confidence,
                iou=iou_threshold,
                classes=class_indices,
                device=self._device,
                verbose=False,
            )
            batch_time = (time.perf_counter() - start) * 1000

            # Convert each result
            per_image_time = batch_time / len(batch)
            for j, res in enumerate(batch_results):
                detections = _convert_ultralytics_result(res)
                results_list.append(
                    DetectionResult(
                        detections=detections,
                        image_path=batch[j],
                        processing_time_ms=per_image_time,
                        model_name=self.info.name,
                    )
                )

            # Progress callback
            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return results_list


@register_model
class RTDETRWrapper(BaseModelWrapper[DetectionResult]):
    """
    RT-DETR transformer-based detection wrapper.

    Fallback detector offering higher accuracy than YOLO11 for
    precision-critical tasks. Uses transformer architecture.

    Attributes:
        info: Model metadata with VRAM requirements and capabilities.
    """

    info = ModelInfo(
        name="rtdetr-l",
        description="RT-DETR-l transformer detector (higher accuracy)",
        vram_required_mb=3500,
        supports_batching=True,
        supports_gpu=True,
        source="ultralytics",
        version="1.0",
    )

    def __init__(self) -> None:
        """Initialize RT-DETR wrapper."""
        super().__init__()
        self._rtdetr: RTDETR | None = None

    def _load_model(self, device: str) -> None:
        """
        Load RT-DETR model.

        Args:
            device: Target device ("cuda", "cpu", "mps").
        """
        from ultralytics import RTDETR

        from backend.cv.download import get_model_path

        # Get model path - ultralytics auto-downloads if not present
        model_path = get_model_path("rtdetr-l")

        logger.info("Loading RT-DETR-l from %s", model_path)
        self._rtdetr = RTDETR(str(model_path))

        # Move to device if CUDA
        if device == "cuda":
            import torch

            if torch.cuda.is_available():
                self._rtdetr.to(device)
            else:
                logger.warning("CUDA requested but not available, using CPU")

        # Store reference for base class compatibility
        self._model = self._rtdetr

    def _unload_model(self) -> None:
        """Unload model and free memory."""
        if self._rtdetr is not None:
            del self._rtdetr
            self._rtdetr = None

    def predict(
        self,
        input_path: str,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        iou_threshold: float = 0.45,
        **kwargs: Any,
    ) -> DetectionResult:
        """
        Run RT-DETR detection on an image.

        Args:
            input_path: Path to input image.
            confidence: Minimum confidence threshold (0-1).
            classes: List of class names to detect (None = all COCO classes).
            iou_threshold: IoU threshold for NMS.
            **kwargs: Additional arguments (ignored).

        Returns:
            DetectionResult with list of detections.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded or self._rtdetr is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        class_indices = get_class_indices(classes)

        start = time.perf_counter()
        results = self._rtdetr.predict(
            input_path,
            conf=confidence,
            iou=iou_threshold,
            classes=class_indices,
            device=self._device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        detections = _convert_ultralytics_result(results[0])

        return DetectionResult(
            detections=detections,
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
        classes: list[str] | None = None,
        iou_threshold: float = 0.45,
        batch_size: int = 4,  # Smaller batch for RT-DETR due to higher VRAM usage
        **kwargs: Any,
    ) -> list[DetectionResult]:
        """
        Batch inference with progress tracking.

        Args:
            input_paths: List of paths to input images.
            progress_callback: Optional callback with (current, total) progress.
            confidence: Minimum confidence threshold (0-1).
            classes: List of class names to detect (None = all COCO classes).
            iou_threshold: IoU threshold for NMS.
            batch_size: Number of images per mini-batch (default 4 for RT-DETR).
            **kwargs: Additional arguments (ignored).

        Returns:
            List of DetectionResult, one per input image.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded or self._rtdetr is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        class_indices = get_class_indices(classes)
        results_list: list[DetectionResult] = []
        total = len(input_paths)

        for i in range(0, total, batch_size):
            batch = input_paths[i : i + batch_size]

            start = time.perf_counter()
            batch_results = self._rtdetr.predict(
                batch,
                conf=confidence,
                iou=iou_threshold,
                classes=class_indices,
                device=self._device,
                verbose=False,
            )
            batch_time = (time.perf_counter() - start) * 1000

            per_image_time = batch_time / len(batch)
            for j, res in enumerate(batch_results):
                detections = _convert_ultralytics_result(res)
                results_list.append(
                    DetectionResult(
                        detections=detections,
                        image_path=batch[j],
                        processing_time_ms=per_image_time,
                        model_name=self.info.name,
                    )
                )

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return results_list


def get_detector(
    prefer_accuracy: bool = False,
    force_model: str | None = None,
) -> BaseModelWrapper[DetectionResult]:
    """
    Get appropriate detector based on requirements and available resources.

    Factory function that selects the best detector for the given requirements.
    By default returns YOLO11m for speed. With prefer_accuracy=True, returns
    RT-DETR if sufficient VRAM is available.

    Args:
        prefer_accuracy: If True, prefer RT-DETR over YOLO11 (requires more VRAM).
        force_model: Force specific model ("yolo11m" or "rtdetr-l").

    Returns:
        Appropriate detector wrapper (unloaded - call load() before use).

    Example:
        detector = get_detector(prefer_accuracy=True)
        detector.load()
        result = detector.predict("image.jpg", classes=["person", "car"])
        detector.unload()
    """
    from backend.cv.device import get_available_vram_mb

    # Force specific model if requested
    if force_model == "yolo11m":
        logger.info("Returning YOLO11m detector (forced)")
        return YOLO11Wrapper()
    elif force_model == "rtdetr-l":
        logger.info("Returning RT-DETR-l detector (forced)")
        return RTDETRWrapper()

    # Check if accuracy is preferred and VRAM is sufficient
    if prefer_accuracy:
        available = get_available_vram_mb()
        if available >= RTDETRWrapper.info.vram_required_mb:
            logger.info(
                "Selecting RT-DETR for accuracy (VRAM available: %dMB)",
                available,
            )
            return RTDETRWrapper()
        logger.info(
            "RT-DETR needs %dMB VRAM, only %dMB available, using YOLO11",
            RTDETRWrapper.info.vram_required_mb,
            available,
        )

    # Default to YOLO11m
    logger.info("Returning YOLO11m detector (default)")
    return YOLO11Wrapper()
