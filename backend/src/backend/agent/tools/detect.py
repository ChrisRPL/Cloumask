"""Detection tool for object detection in images.

Uses YOLO11 (primary) or RT-DETR (fallback) for real inference.
Supports 80 COCO classes with configurable confidence threshold.

Implements spec: 03-cv-models/01-yolo11-detection
Integration point: backend/cv/detection.py
"""

from __future__ import annotations

import logging
from pathlib import Path

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

logger = logging.getLogger(__name__)


@register_tool
class DetectTool(BaseTool):
    """Detect objects in images using YOLO11 or RT-DETR."""

    name = "detect"
    description = """Run object detection on images to find and label objects.
Uses YOLO11m (fast) or RT-DETR (accurate) models.
Supports 80 COCO classes including vehicles, people, animals, furniture, etc."""
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
            description="List of COCO class names to detect (e.g., ['person', 'car']). "
            "None for all 80 classes.",
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
            description="Use RT-DETR for higher accuracy (slower, more VRAM)",
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
        save_annotations: bool = True,
    ) -> ToolResult:
        """
        Execute object detection using CV models.

        Args:
            input_path: Path to input image or directory.
            classes: List of COCO class names to detect.
            confidence: Minimum confidence threshold (0-1).
            prefer_accuracy: Use RT-DETR instead of YOLO11.
            save_annotations: Whether to save detection annotations.

        Returns:
            ToolResult with detection statistics.
        """
        from backend.cv.detection import COCO_CLASSES, get_detector

        input_p = Path(input_path)

        if not input_p.exists():
            return error_result(f"Input not found: {input_path}")

        if confidence < 0 or confidence > 1:
            return error_result(
                f"Invalid confidence: {confidence}. Must be between 0 and 1."
            )

        # Validate class names if provided
        if classes:
            invalid = [c for c in classes if c.lower() not in [cc.lower() for cc in COCO_CLASSES]]
            if invalid:
                return error_result(
                    f"Invalid class names: {invalid}. "
                    f"Valid classes include: {COCO_CLASSES[:10]}... (80 total)"
                )

        # Collect image files
        image_paths = self._collect_image_files(input_p)
        if not image_paths:
            return error_result("No image files found")

        try:
            # Get and load detector
            detector = get_detector(prefer_accuracy=prefer_accuracy)
            detector.load()

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

            # Unload to free memory
            detector.unload()

            # TODO: Implement annotation saving when save_annotations=True
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
                    "annotations_saved": False,  # Not yet implemented
                }
            )

        except ImportError as e:
            logger.exception("CV dependencies not installed")
            return error_result(
                f"CV dependencies not installed: {e}. "
                "Install with: pip install -r requirements-cv.txt"
            )
        except Exception as e:
            logger.exception("Detection failed")
            return error_result(f"Detection failed: {e}")

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
