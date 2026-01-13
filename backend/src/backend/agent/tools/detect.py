"""Detection tool for object detection in images.

Uses YOLO11 (primary) or RT-DETR (fallback) for real inference.
Supports 80 COCO classes with configurable confidence threshold.
Auto-selects YOLO-World for open-vocabulary detection (non-COCO classes).
Quality mode routes through SAM3 for superior detection-via-segmentation.

Implements spec: 03-cv-models/01-yolo11-detection, 03-cv-models/08-cv-tools
Integration points: backend/cv/detection.py, backend/cv/openvocab.py, backend/cv/segmentation.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.constants import IMAGE_EXTENSIONS
from backend.agent.tools.registry import register_tool

if TYPE_CHECKING:
    from backend.cv.types import SegmentationResult

logger = logging.getLogger(__name__)


def _needs_openvocab(classes: list[str] | None, coco_classes: set[str]) -> bool:
    """
    Check if any requested class requires open-vocabulary detection.

    Args:
        classes: List of class names to detect.
        coco_classes: Set of lowercase COCO class names.

    Returns:
        True if any class is not in COCO classes.
    """
    if not classes:
        return False
    return any(c.lower() not in coco_classes for c in classes)


def _masks_to_detections(
    seg_result: SegmentationResult,
    classes: list[str],
) -> tuple[list[dict], dict[str, int]]:
    """
    Convert segmentation masks to detection-style bounding boxes.

    Args:
        seg_result: Segmentation result with masks.
        classes: Class names used in the prompt.

    Returns:
        Tuple of (list of detection dicts, class counts dict).
    """
    import numpy as np

    detections = []
    class_counts: dict[str, int] = {}

    for i, mask in enumerate(seg_result.masks):
        # Get mask data as numpy array
        mask_np = mask.to_numpy()

        # Find bounding box from mask
        coords = np.where(mask_np > 0)
        if len(coords[0]) == 0:
            continue

        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        h, w = mask_np.shape

        # Normalized center format
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # Assign class name (cycle through if more masks than classes)
        class_name = classes[i % len(classes)] if classes and len(classes) > 0 else "object"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        detections.append({
            "class_name": class_name,
            "confidence": mask.confidence,
            "bbox": {"x": cx, "y": cy, "width": bw, "height": bh},
        })

    return detections, class_counts


@register_tool
class DetectTool(BaseTool):
    """Detect objects in images using YOLO11, YOLO-World, or SAM3."""

    name = "detect"
    description = """Run object detection on images to find and label objects.

Model Selection:
- COCO classes (person, car, etc.): Uses YOLO11m (fast) or RT-DETR (accurate)
- Custom classes (red car, delivery truck): Auto-selects YOLO-World (open-vocab)
- quality=True: Routes through SAM3 for superior detection (~8GB VRAM)

Examples:
- detect(path, classes=["person", "car"]) → YOLO11
- detect(path, classes=["red car", "delivery truck"]) → YOLO-World (auto)
- detect(path, classes=["person"], quality=True) → SAM3"""
    category = ToolCategory.DETECTION

    parameters = [
        ToolParameter(
            name="input_path",
            type=str,
            description="Path to input image or directory",
            required=True,
        ),
        ToolParameter(
            name="classes",
            type=list,
            description="List of class names to detect (e.g., ['person', 'car']). "
            "Supports COCO classes + any custom descriptions for open-vocab detection.",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="confidence",
            type=float,
            description="Minimum confidence threshold (0-1)",
            required=False,
            default=0.5,
        ),
        ToolParameter(
            name="prefer_accuracy",
            type=bool,
            description="Use RT-DETR/GroundingDINO for higher accuracy (slower, more VRAM)",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="quality",
            type=bool,
            description="Use SAM3 for superior detection-via-segmentation (~8GB VRAM, slower but best results)",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="save_annotations",
            type=bool,
            description="Whether to save detection annotations",
            required=False,
            default=True,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        classes: list[str] | None = None,
        confidence: float = 0.5,
        prefer_accuracy: bool = False,
        quality: bool = False,
        save_annotations: bool = True,
    ) -> ToolResult:
        """
        Execute object detection using CV models.

        Automatically selects the best model based on parameters:
        - quality=True: SAM3 (detection-via-segmentation, best results)
        - Non-COCO classes: YOLO-World (open-vocabulary)
        - COCO classes: YOLO11 (fast) or RT-DETR (accurate)

        Args:
            input_path: Path to input image or directory.
            classes: List of class names to detect.
            confidence: Minimum confidence threshold (0-1).
            prefer_accuracy: Use RT-DETR/GroundingDINO instead of YOLO11/YOLO-World.
            quality: Use SAM3 for superior detection-via-segmentation.
            save_annotations: Whether to save detection annotations.

        Returns:
            ToolResult with detection statistics.
        """
        from backend.cv.detection import COCO_CLASSES

        input_p = Path(input_path)

        if not input_p.exists():
            return error_result(f"Input not found: {input_path}")

        if confidence < 0 or confidence > 1:
            return error_result(
                f"Invalid confidence: {confidence}. Must be between 0 and 1."
            )

        # Collect image files
        image_paths = self._collect_image_files(input_p)
        if not image_paths:
            return error_result("No image files found")

        # Build lowercase COCO classes set for comparison
        coco_lower = {c.lower() for c in COCO_CLASSES}

        # Determine detection mode
        use_quality = quality
        use_openvocab = _needs_openvocab(classes, coco_lower) if not quality else False

        try:
            if use_quality:
                # SAM3 quality mode: detection-via-segmentation
                return await self._execute_sam3(
                    image_paths, classes, confidence, save_annotations
                )
            elif use_openvocab:
                # Open-vocabulary mode: YOLO-World or GroundingDINO
                return await self._execute_openvocab(
                    image_paths, classes, confidence, prefer_accuracy, save_annotations
                )
            else:
                # Standard COCO mode: YOLO11 or RT-DETR
                return await self._execute_coco(
                    image_paths, classes, confidence, prefer_accuracy, save_annotations
                )

        except ImportError as e:
            logger.exception("CV dependencies not installed")
            return error_result(
                f"CV dependencies not installed: {e}. "
                "Install with: pip install -r requirements-cv.txt"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory: %s", e)
                return error_result(
                    f"GPU out of memory. Try: reduce batch size, disable quality mode, "
                    f"or close other GPU applications. Error: {e}"
                )
            logger.exception("Detection failed")
            return error_result(f"Detection failed: {e}")
        except Exception as e:
            logger.exception("Detection failed")
            return error_result(f"Detection failed: {e}")

    async def _execute_coco(
        self,
        image_paths: list[str],
        classes: list[str] | None,
        confidence: float,
        prefer_accuracy: bool,
        save_annotations: bool,
    ) -> ToolResult:
        """Execute detection using YOLO11 or RT-DETR for COCO classes."""
        from backend.cv.detection import COCO_CLASSES, get_detector

        # Validate COCO class names
        if classes:
            coco_lower = {c.lower() for c in COCO_CLASSES}
            invalid = [c for c in classes if c.lower() not in coco_lower]
            if invalid:
                return error_result(
                    f"Invalid COCO class names: {invalid}. "
                    f"Valid classes include: {COCO_CLASSES[:10]}... (80 total). "
                    "For custom classes, they will auto-route to YOLO-World."
                )

        # Get and load detector
        detector = get_detector(prefer_accuracy=prefer_accuracy)
        detector.load()

        try:
            # Run detection with progress reporting
            results = detector.predict_batch(
                image_paths,
                progress_callback=lambda curr, total: self.report_progress(
                    curr, total, f"Detecting in image {curr}/{total}"
                ),
                confidence=confidence,
                classes=classes,
            )

            # Aggregate results
            total_detections = 0
            class_counts: dict[str, int] = {}
            total_confidence = 0.0

            for result in results:
                total_detections += result.count
                for det in result.detections:
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                    total_confidence += det.confidence

            avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0

            if save_annotations:
                logger.debug("Annotation saving not yet implemented")

            return success_result(
                {
                    "files_processed": len(image_paths),
                    "count": total_detections,
                    "classes": class_counts,
                    "confidence_threshold": confidence,
                    "confidence": round(avg_confidence, 3),
                    "model": detector.info.name,
                    "mode": "coco",
                    "annotations_saved": False,
                }
            )
        finally:
            detector.unload()

    async def _execute_openvocab(
        self,
        image_paths: list[str],
        classes: list[str] | None,
        confidence: float,
        prefer_accuracy: bool,
        save_annotations: bool,
    ) -> ToolResult:
        """Execute detection using YOLO-World or GroundingDINO for open-vocabulary."""
        from backend.cv.openvocab import get_openvocab_detector

        # Get and load open-vocab detector
        detector = get_openvocab_detector(prefer_accuracy=prefer_accuracy)
        detector.load()

        # Build prompt from classes
        prompt = ", ".join(classes) if classes else "object"

        try:
            # Run detection with progress reporting
            results = detector.predict_batch(
                image_paths,
                progress_callback=lambda curr, total: self.report_progress(
                    curr, total, f"Detecting (open-vocab) {curr}/{total}"
                ),
                prompt=prompt,
                confidence=confidence,
            )

            # Aggregate results
            total_detections = 0
            class_counts: dict[str, int] = {}
            total_confidence = 0.0

            for result in results:
                total_detections += result.count
                for det in result.detections:
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                    total_confidence += det.confidence

            avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0

            if save_annotations:
                logger.debug("Annotation saving not yet implemented")

            return success_result(
                {
                    "files_processed": len(image_paths),
                    "count": total_detections,
                    "classes": class_counts,
                    "confidence_threshold": confidence,
                    "confidence": round(avg_confidence, 3),
                    "model": detector.info.name,
                    "mode": "openvocab",
                    "prompt": prompt,
                    "annotations_saved": False,
                }
            )
        finally:
            detector.unload()

    async def _execute_sam3(
        self,
        image_paths: list[str],
        classes: list[str] | None,
        confidence: float,
        save_annotations: bool,
    ) -> ToolResult:
        """Execute detection using SAM3 (detection-via-segmentation)."""
        from backend.cv.segmentation import get_segmenter

        # Get and load SAM3 segmenter
        segmenter = get_segmenter(prompt_type="text")
        segmenter.load()

        # Build prompt from classes
        prompt = ", ".join(classes) if classes else "object"

        try:
            total_detections = 0
            class_counts: dict[str, int] = {}
            total_confidence = 0.0
            total = len(image_paths)

            for i, path in enumerate(image_paths):
                self.report_progress(i + 1, total, f"Segmenting (SAM3) {i + 1}/{total}")

                result = segmenter.predict(path, prompt=prompt, confidence=confidence)

                # Convert masks to detection format
                detections, counts = _masks_to_detections(
                    result, classes if classes else ["object"]
                )

                total_detections += len(detections)
                for class_name, count in counts.items():
                    class_counts[class_name] = class_counts.get(class_name, 0) + count
                for det in detections:
                    total_confidence += det["confidence"]

            avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0

            if save_annotations:
                logger.debug("Annotation saving not yet implemented")

            return success_result(
                {
                    "files_processed": len(image_paths),
                    "count": total_detections,
                    "classes": class_counts,
                    "confidence_threshold": confidence,
                    "confidence": round(avg_confidence, 3),
                    "model": segmenter.info.name,
                    "mode": "sam3",
                    "prompt": prompt,
                    "annotations_saved": False,
                }
            )
        finally:
            segmenter.unload()

    def _collect_image_files(self, path: Path) -> list[str]:
        """
        Collect all image file paths.

        Args:
            path: Input path (file or directory).

        Returns:
            List of image file paths as strings.
        """
        if path.is_file():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                return [str(path)]
            return []

        files: list[str] = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(str(f) for f in path.glob(f"**/*{ext}"))
        return files
