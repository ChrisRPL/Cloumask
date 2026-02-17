"""
YOLO-World and GroundingDINO open-vocabulary detection wrappers.

This module provides open-vocabulary object detection that supports arbitrary
text prompts (not limited to fixed COCO classes). YOLO-World is the primary
detector for speed, with GroundingDINO as a fallback for higher accuracy.

Implements spec: 03-cv-models/04-yolo-world-openvocab
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch

from backend.cv.base import BaseModelWrapper, ModelInfo, ProgressCallback, register_model
from backend.cv.types import BBox, Detection, DetectionResult

if TYPE_CHECKING:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from ultralytics import YOLOWorld

logger = logging.getLogger(__name__)


def _as_positive_int(value: Any) -> int | None:
    """Best-effort conversion of model metadata values to positive ints."""
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _extract_model_stride(model: Any) -> int | None:
    """Extract stride from ultralytics model internals when available."""
    stride_candidate = getattr(model, "stride", None)
    if stride_candidate is None:
        inner = getattr(model, "model", None)
        stride_candidate = getattr(inner, "stride", None) if inner is not None else None

    if stride_candidate is None:
        return None

    if hasattr(stride_candidate, "max"):
        try:
            max_value = stride_candidate.max()
            if hasattr(max_value, "item"):
                max_value = max_value.item()
            return _as_positive_int(max_value)
        except Exception:
            pass

    if isinstance(stride_candidate, (list, tuple)):
        values = [_as_positive_int(v) for v in stride_candidate]
        values = [v for v in values if v is not None]
        return max(values) if values else None

    return _as_positive_int(stride_candidate)


def _resolve_aligned_imgsz(model: Any, base: int = 640) -> int:
    """
    Resolve an image size aligned to model stride.

    Prevents repetitive Ultralytics warnings such as:
    "imgsz=[640] must be multiple of max stride 14..."
    """
    stride = _extract_model_stride(model) or 32
    base_size = _as_positive_int(base) or 640
    return ((base_size + stride - 1) // stride) * stride


def _parse_prompt(prompt: str) -> list[str]:
    """
    Parse comma-separated prompt into list of class names.

    Args:
        prompt: Comma-separated class descriptions (e.g., "red car, person, dog").

    Returns:
        List of stripped class names.
    """
    return [c.strip() for c in prompt.split(",") if c.strip()]


def _convert_yoloworld_result(
    result: Any,
    classes: list[str],
) -> list[Detection]:
    """
    Convert YOLO-World result to Detection list.

    Args:
        result: Single YOLO-World prediction result.
        classes: List of class names used for detection.

    Returns:
        List of Detection objects with user-provided class names.
    """
    detections: list[Detection] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0])

        # Ensure class_id is valid for our custom classes
        if cls_id < 0 or cls_id >= len(classes):
            logger.warning("Invalid class ID: %d (max: %d), skipping", cls_id, len(classes) - 1)
            continue

        # Get normalized xywh (center format)
        xywhn = box.xywhn[0]

        detections.append(
            Detection(
                class_id=cls_id,
                class_name=classes[cls_id],  # Map back to user's class name
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
class YOLOWorldWrapper(BaseModelWrapper[DetectionResult]):
    """
    YOLO-World open-vocabulary object detection wrapper.

    Supports arbitrary text prompts for detection, not limited to fixed COCO
    classes. Caches class embeddings for efficiency with repeated prompts.

    Attributes:
        info: Model metadata with VRAM requirements and capabilities.
    """

    info = ModelInfo(
        name="yolo-world-l",
        description="YOLO-World open-vocabulary detector",
        vram_required_mb=4000,
        supports_batching=True,
        supports_gpu=True,
        source="ultralytics",
        version="1.0",
    )

    def __init__(self) -> None:
        """Initialize YOLO-World wrapper."""
        super().__init__()
        self._yoloworld: YOLOWorld | None = None
        self._current_classes: list[str] = []
        self._imgsz: int = 640

    def _load_model(self, device: str) -> None:
        """
        Load YOLO-World model.

        Args:
            device: Target device ("cuda", "cpu", "mps").
        """
        from ultralytics import YOLOWorld

        from backend.cv.download import download_model, get_model_path

        # Get model path - download if not present
        model_path = get_model_path("yolo-world-l")
        if not model_path.exists():
            download_model("yolo-world-l")

        logger.info("Loading YOLO-World-l from %s", model_path)
        self._yoloworld = YOLOWorld(str(model_path))

        # Move to device if CUDA
        if device == "cuda":
            if torch.cuda.is_available():
                self._yoloworld.to(device)
            else:
                logger.warning("CUDA requested but not available, using CPU")
                device = "cpu"

        self._imgsz = _resolve_aligned_imgsz(self._yoloworld)

        # Warm up model for consistent inference times
        self._warmup(device)

        # Store reference for base class compatibility
        self._model = self._yoloworld

    def _warmup(self, device: str) -> None:
        """
        Warm up model with dummy inference.

        Args:
            device: Current device.
        """
        import numpy as np

        # Set a dummy class for warmup
        self._yoloworld.set_classes(["object"])  # type: ignore[union-attr]
        self._current_classes = ["object"]

        # Create dummy image tensor (aligned to model stride)
        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._predict_with_optional_imgsz(
            dummy,
            verbose=False,
            device=device,
        )

        logger.debug("YOLO-World warmed up on %s", device)

    def _predict_with_optional_imgsz(self, source: Any, **kwargs: Any) -> Any:
        """
        Run predict while tolerating YOLOWorld variants without `imgsz` kwarg.

        Some mocked/older API surfaces reject `imgsz`. Retry once without it.
        """
        if self._yoloworld is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        predict_kwargs = dict(kwargs)
        predict_kwargs.setdefault("imgsz", self._imgsz)

        try:
            return self._yoloworld.predict(source, **predict_kwargs)
        except TypeError as exc:
            message = str(exc)
            if "imgsz" not in message or "unexpected keyword argument" not in message:
                raise
            logger.debug("YOLOWorld.predict does not accept imgsz, retrying without it")
            predict_kwargs.pop("imgsz", None)
            return self._yoloworld.predict(source, **predict_kwargs)

    def _unload_model(self) -> None:
        """Unload model and free memory."""
        if self._yoloworld is not None:
            del self._yoloworld
            self._yoloworld = None
            self._current_classes = []

    def set_classes(self, classes: list[str]) -> None:
        """
        Set custom classes for detection.

        Caches embeddings for efficiency - only updates if classes change.

        Args:
            classes: List of class names to detect.
        """
        if classes != self._current_classes:
            logger.debug("Setting new classes: %s", classes)
            self._yoloworld.set_classes(classes)  # type: ignore[union-attr]
            self._current_classes = classes.copy()

    def predict(
        self,
        input_path: str,
        *,
        prompt: str = "object",
        confidence: float = 0.3,
        iou_threshold: float = 0.45,
        **kwargs: Any,
    ) -> DetectionResult:
        """
        Run open-vocabulary detection on an image.

        Args:
            input_path: Path to input image.
            prompt: Comma-separated class descriptions (e.g., "red car, person").
            confidence: Minimum confidence threshold (0-1).
            iou_threshold: IoU threshold for NMS.
            **kwargs: Additional arguments (ignored).

        Returns:
            DetectionResult with list of detections.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded or self._yoloworld is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Parse prompt into classes and set them
        classes = _parse_prompt(prompt)
        if not classes:
            classes = ["object"]
        self.set_classes(classes)

        start = time.perf_counter()
        results = self._predict_with_optional_imgsz(
            input_path,
            conf=confidence,
            iou=iou_threshold,
            device=self._device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Convert results (predict returns a list, we want first item)
        detections = _convert_yoloworld_result(results[0], classes)

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
        prompt: str = "object",
        confidence: float = 0.3,
        iou_threshold: float = 0.45,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> list[DetectionResult]:
        """
        Batch inference with progress tracking.

        Args:
            input_paths: List of paths to input images.
            progress_callback: Optional callback with (current, total) progress.
            prompt: Comma-separated class descriptions.
            confidence: Minimum confidence threshold (0-1).
            iou_threshold: IoU threshold for NMS.
            batch_size: Number of images per mini-batch.
            **kwargs: Additional arguments (ignored).

        Returns:
            List of DetectionResult, one per input image.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded or self._yoloworld is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Parse and set classes once for entire batch
        classes = _parse_prompt(prompt)
        if not classes:
            classes = ["object"]
        self.set_classes(classes)

        results_list: list[DetectionResult] = []
        total = len(input_paths)

        # Process in mini-batches for memory efficiency
        for i in range(0, total, batch_size):
            batch = input_paths[i : i + batch_size]

            start = time.perf_counter()
            batch_results = self._predict_with_optional_imgsz(
                batch,
                conf=confidence,
                iou=iou_threshold,
                device=self._device,
                verbose=False,
            )
            batch_time = (time.perf_counter() - start) * 1000

            # Convert each result
            per_image_time = batch_time / len(batch)
            for j, res in enumerate(batch_results):
                detections = _convert_yoloworld_result(res, classes)
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
class GroundingDINOWrapper(BaseModelWrapper[DetectionResult]):
    """
    GroundingDINO open-vocabulary detection wrapper.

    Higher accuracy fallback for YOLO-World. Uses transformer architecture
    and supports natural language prompts. Slower but more precise.

    Attributes:
        info: Model metadata with VRAM requirements and capabilities.
    """

    info = ModelInfo(
        name="groundingdino",
        description="GroundingDINO open-vocabulary detector (higher accuracy)",
        vram_required_mb=5000,
        supports_batching=False,  # GroundingDINO processes one at a time
        supports_gpu=True,
        source="huggingface",
        version="1.0",
    )

    # HuggingFace model ID
    MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(self) -> None:
        """Initialize GroundingDINO wrapper."""
        super().__init__()
        self._gdino: AutoModelForZeroShotObjectDetection | None = None
        self._processor: AutoProcessor | None = None

    def _load_model(self, device: str) -> None:
        """
        Load GroundingDINO model.

        Args:
            device: Target device ("cuda", "cpu", "mps").
        """
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        logger.info("Loading GroundingDINO from %s", self.MODEL_ID)

        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(self.MODEL_ID)

        # Move to device if GPU requested
        if device == "cuda":
            if torch.cuda.is_available():
                self._gdino.to(device)  # type: ignore[union-attr]
            else:
                logger.warning("CUDA requested but not available, using CPU")
                device = "cpu"
        elif device == "mps":
            if torch.backends.mps.is_available():
                self._gdino.to(device)  # type: ignore[union-attr]
            else:
                logger.warning("MPS requested but not available, using CPU")
                device = "cpu"

        # Store reference for base class compatibility
        self._model = self._gdino

    def _unload_model(self) -> None:
        """Unload model and free memory."""
        if self._gdino is not None:
            del self._gdino
            self._gdino = None
        if self._processor is not None:
            del self._processor
            self._processor = None

    @staticmethod
    def _format_prompt(prompt: str) -> str:
        """
        Convert comma-separated prompt to GroundingDINO format.

        GroundingDINO expects period-separated classes ending with period.
        E.g., "car, person" -> "car . person ."

        Args:
            prompt: Comma-separated prompt.

        Returns:
            Period-separated prompt for GroundingDINO.
        """
        classes = _parse_prompt(prompt)
        if not classes:
            return "object ."
        return " . ".join(classes) + " ."

    def predict(
        self,
        input_path: str,
        *,
        prompt: str = "object",
        confidence: float = 0.3,
        text_threshold: float | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        """
        Run GroundingDINO detection on an image.

        Args:
            input_path: Path to input image.
            prompt: Comma-separated class descriptions or natural language.
            confidence: Minimum box confidence threshold (0-1).
            text_threshold: Text confidence threshold (defaults to confidence).
            **kwargs: Additional arguments (ignored).

        Returns:
            DetectionResult with list of detections.

        Raises:
            RuntimeError: If model not loaded.
        """
        if not self.is_loaded or self._gdino is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from PIL import Image

        # Load and convert image
        image = Image.open(input_path).convert("RGB")
        w, h = image.size

        # Format prompt for GroundingDINO
        text = self._format_prompt(prompt)
        text_thresh = text_threshold if text_threshold is not None else confidence

        start = time.perf_counter()

        # Process inputs
        inputs = self._processor(images=image, text=text, return_tensors="pt")  # type: ignore[operator]
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._gdino(**inputs)  # type: ignore[operator]

        # Post-process results
        results = self._processor.post_process_grounded_object_detection(  # type: ignore[attr-defined]
            outputs,
            inputs["input_ids"],
            box_threshold=confidence,
            text_threshold=text_thresh,
            target_sizes=[(h, w)],
        )[0]

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Convert to our types
        detections: list[Detection] = []
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"],
            strict=True,
        ):
            x1, y1, x2, y2 = box.cpu().numpy()

            # Convert xyxy to normalized center format
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            detections.append(
                Detection(
                    class_id=0,  # GroundingDINO doesn't use fixed class IDs
                    class_name=label,
                    bbox=BBox(x=cx, y=cy, width=bw, height=bh),
                    confidence=float(score),
                )
            )

        return DetectionResult(
            detections=detections,
            image_path=input_path,
            processing_time_ms=elapsed_ms,
            model_name=self.info.name,
        )


def get_openvocab_detector(
    prefer_accuracy: bool = False,
    force_model: str | None = None,
) -> BaseModelWrapper[DetectionResult]:
    """
    Get appropriate open-vocabulary detector based on requirements.

    Factory function that selects the best detector for the given requirements.
    By default returns YOLO-World for speed. With prefer_accuracy=True, returns
    GroundingDINO if sufficient VRAM is available.

    Args:
        prefer_accuracy: If True, prefer GroundingDINO over YOLO-World.
        force_model: Force specific model ("yoloworld" or "groundingdino").

    Returns:
        Appropriate detector wrapper (unloaded - call load() before use).

    Example:
        detector = get_openvocab_detector(prefer_accuracy=True)
        detector.load()
        result = detector.predict("image.jpg", prompt="red car, person")
        detector.unload()
    """
    from backend.cv.device import get_available_vram_mb

    # Force specific model if requested
    if force_model == "yoloworld":
        logger.info("Returning YOLO-World detector (forced)")
        return YOLOWorldWrapper()
    elif force_model == "groundingdino":
        logger.info("Returning GroundingDINO detector (forced)")
        return GroundingDINOWrapper()

    # Check if accuracy is preferred and VRAM is sufficient
    if prefer_accuracy:
        available = get_available_vram_mb()
        if available >= GroundingDINOWrapper.info.vram_required_mb:
            logger.info(
                "Selecting GroundingDINO for accuracy (VRAM available: %dMB)",
                available,
            )
            return GroundingDINOWrapper()
        logger.info(
            "GroundingDINO needs %dMB VRAM, only %dMB available, using YOLO-World",
            GroundingDINOWrapper.info.vram_required_mb,
            available,
        )

    # Default to YOLO-World
    logger.info("Returning YOLO-World detector (default)")
    return YOLOWorldWrapper()
