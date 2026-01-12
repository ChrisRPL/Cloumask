"""Detection tool for object detection in images.

This is a STUB implementation that returns mock data for testing.
Real implementation will integrate with YOLO11 inference.

Integration point: backend/cv/detection.py
"""

import random
from pathlib import Path

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.constants import DEFAULT_DETECTION_CLASSES, IMAGE_EXTENSIONS
from backend.agent.tools.registry import register_tool


@register_tool
class DetectTool(BaseTool):
    """Detect objects in images using YOLO."""

    name = "detect"
    description = """Run object detection on images to find and label objects.
Supports common classes like vehicles, people, and can use custom prompts."""
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
            description="List of object classes to detect",
            required=False,
            default=list(DEFAULT_DETECTION_CLASSES),  # Create a copy to avoid mutation
        ),
        ToolParameter(
            name="confidence",
            type=float,
            description="Minimum confidence threshold (0-1)",
            required=False,
            default=0.5,
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
        classes: list | None = None,
        confidence: float = 0.5,
        save_annotations: bool = True,
    ) -> ToolResult:
        """
        STUB: Returns mock detection results.

        TODO: Replace with YOLO11 inference.
        Integration point: backend/cv/detection.py
        """
        classes = classes if classes is not None else list(DEFAULT_DETECTION_CLASSES)
        input_p = Path(input_path)

        if not input_p.exists():
            return error_result(f"Input not found: {input_path}")

        # Validate confidence threshold
        if confidence < 0 or confidence > 1:
            return error_result(
                f"Invalid confidence: {confidence}. Must be between 0 and 1."
            )

        # Count files
        file_count = self._count_image_files(input_p)

        if file_count == 0:
            return error_result("No image files found")

        # Generate mock detections with deterministic seed
        seed = hash(input_path) % (2**32)
        rng = random.Random(seed)

        class_counts: dict[str, int] = {}
        total_detections = 0

        # Limit to 5 classes for mock data
        for cls in classes[:5]:
            count = rng.randint(file_count, file_count * 10)
            class_counts[cls] = count
            total_detections += count

        # Simulate processing with progress reporting
        for i in range(file_count):
            self.report_progress(
                i + 1, file_count, f"Detecting in file {i + 1}/{file_count}"
            )

        return success_result(
            {
                "files_processed": file_count,
                "count": total_detections,
                "classes": class_counts,
                "confidence_threshold": confidence,
                "confidence": round(0.82 + rng.uniform(-0.05, 0.1), 3),
                "annotations_saved": save_annotations,
                "_stub": True,
                "_integration_point": "backend/cv/detection.py",
            }
        )

    def _count_image_files(self, path: Path) -> int:
        """Count image files in path."""
        if path.is_file():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                return 1
            return 0

        count = 0
        for ext in IMAGE_EXTENSIONS:
            count += sum(1 for _ in path.glob(f"**/*{ext}"))
        return count
