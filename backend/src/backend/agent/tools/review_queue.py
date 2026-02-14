"""
Review Queue tool for populating review items from detection results.

This tool takes detection results (YOLO format) and creates review items
with annotations for human-in-the-loop verification and correction.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.registry import register_tool
from backend.api.models.review import (
    Annotation,
    BoundingBox,
    ImageDimensions,
    ReviewItem,
    ReviewStatus,
)
from backend.cv.utils.thumbnail import generate_thumbnail, get_image_dimensions

logger = logging.getLogger(__name__)


def parse_yolo_annotation(annotation_path: Path, img_width: int, img_height: int) -> list[dict]:
    """
    Parse YOLO format annotation file.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values are normalized (0-1).

    Args:
        annotation_path: Path to .txt annotation file
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of annotation dicts with normalized coordinates
    """
    annotations = []

    # Default COCO classes - in production this should be configurable
    coco_classes = [
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

    try:
        with annotation_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                confidence = float(parts[5]) if len(parts) > 5 else 1.0

                # Convert from center to top-left
                x = x_center - (width / 2)
                y = y_center - (height / 2)

                # Get class name
                class_name = (
                    coco_classes[class_id] if class_id < len(coco_classes) else f"class_{class_id}"
                )

                annotations.append(
                    {
                        "label": class_name,
                        "confidence": confidence,
                        "bbox": {
                            "x": max(0, min(1, x)),
                            "y": max(0, min(1, y)),
                            "width": max(0, min(1, width)),
                            "height": max(0, min(1, height)),
                        },
                    }
                )
    except Exception as e:
        logger.error(f"Error parsing annotation file {annotation_path}: {e}")

    return annotations


@register_tool
class ReviewQueueTool(BaseTool):
    """Populate review queue with detection results for human verification."""

    name = "review"
    description = """Populate the review queue with detection results for human verification.
Use this after detection to allow users to review and correct annotations before export.
Reads YOLO-format annotations and creates review items with bounding boxes."""
    category = ToolCategory.UTILITY

    parameters = [
        ToolParameter(
            name="source_path",
            type=str,
            description="Path to directory containing detection results (YOLO .txt files)",
            required=True,
        ),
        ToolParameter(
            name="image_dir",
            type=str,
            description="Path to directory containing source images",
            required=True,
        ),
        ToolParameter(
            name="execution_id",
            type=str,
            description="Execution ID to associate with review items",
            required=False,
            default=None,
        ),
    ]

    async def execute(
        self,
        source_path: str,
        image_dir: str,
        execution_id: str | None = None,
    ) -> ToolResult:
        """
        Populate review queue from detection results.

        Args:
            source_path: Path to YOLO annotations directory
            image_dir: Path to source images
            execution_id: Optional execution ID for grouping

        Returns:
            ToolResult with count of items created
        """
        source_p = Path(source_path)
        image_p = Path(image_dir)

        # Validate paths
        if not source_p.exists():
            return error_result(f"Source path not found: {source_path}")
        if not image_p.exists():
            return error_result(f"Image directory not found: {image_dir}")

        # Generate execution ID if not provided
        if execution_id is None:
            execution_id = f"exec_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        created_count = 0
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        # Import here to avoid circular dependency
        from backend.api.routes.review import _review_items, _save_to_disk

        self.report_progress(0, 100, "Scanning for images and annotations...")

        # Find all images and their corresponding annotations
        for img_file in image_p.glob("*"):
            if img_file.suffix.lower() not in valid_extensions:
                continue

            try:
                # Look for corresponding annotation file
                annotation_file = source_p / f"{img_file.stem}.txt"

                # Get image dimensions
                width, height = get_image_dimensions(img_file)

                # Generate thumbnail
                thumbnail_url = generate_thumbnail(img_file)

                # Parse annotations if they exist
                annotations = []
                if annotation_file.exists():
                    yolo_annots = parse_yolo_annotation(annotation_file, width, height)

                    for annot in yolo_annots:
                        annotation = Annotation(
                            id=str(uuid.uuid4()),
                            type="bbox",
                            label=annot["label"],
                            confidence=annot["confidence"],
                            bbox=BoundingBox(**annot["bbox"]),
                            color="#166534",  # Default green color
                            visible=True,
                        )
                        annotations.append(annotation)

                # Create review item
                item = ReviewItem(
                    id=str(uuid.uuid4()),
                    file_path=str(img_file.absolute()),
                    file_name=img_file.name,
                    dimensions=ImageDimensions(width=width, height=height),
                    thumbnail_url=thumbnail_url,
                    annotations=annotations,
                    original_annotations=annotations.copy(),  # Store copy for reset
                    status=ReviewStatus.PENDING,
                    reviewed_at=None,
                    flagged=False,
                    flag_reason=None,
                )

                _review_items[item.id] = item
                created_count += 1

                self.report_progress(
                    created_count,
                    created_count + 10,  # Approximate total
                    f"Created review item for {img_file.name} with {len(annotations)} annotations",
                )

            except Exception as e:
                logger.error(f"Error creating review item for {img_file}: {e}")
                continue

        # Save to disk
        _save_to_disk(execution_id)

        self.report_progress(100, 100, f"Created {created_count} review items")

        return success_result(
            {
                "created_count": created_count,
                "execution_id": execution_id,
                "source_path": str(source_p),
                "image_dir": str(image_p),
            },
            review_items=created_count,
        )
