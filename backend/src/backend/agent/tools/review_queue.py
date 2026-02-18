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
from backend.data.yolo_annotations import (
    build_annotation_index,
    find_annotation_for_image,
    find_yolo_labels_dir,
    load_yolo_class_names,
    parse_yolo_annotation_file,
)

logger = logging.getLogger(__name__)


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
        ToolParameter(
            name="project_id",
            type=str,
            description="Optional project ID to isolate review items",
            required=False,
            default=None,
        ),
    ]

    async def execute(
        self,
        source_path: str,
        image_dir: str,
        execution_id: str | None = None,
        project_id: str | None = None,
    ) -> ToolResult:
        """
        Populate review queue from detection results.

        Args:
            source_path: Path to YOLO annotations directory
            image_dir: Path to source images
            execution_id: Optional execution ID for grouping
            project_id: Optional project ID for grouping

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

        labels_dir = find_yolo_labels_dir(source_p)
        if labels_dir is None:
            return error_result(f"No YOLO labels found under source path: {source_path}")

        class_names = load_yolo_class_names(source_p)
        exact_index, normalized_index = build_annotation_index(labels_dir)

        # Generate execution ID if not provided
        if execution_id is None:
            execution_id = f"exec_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        created_count = 0
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = sorted(
            img_file
            for img_file in image_p.rglob("*")
            if img_file.is_file() and img_file.suffix.lower() in valid_extensions
        )

        # Import here to avoid circular dependency
        from backend.api.routes.review import _review_items, _save_to_disk

        self.report_progress(0, 100, "Scanning for images and annotations...")

        # Find all images and their corresponding annotations
        total_images = max(len(images), 1)
        for index, img_file in enumerate(images, start=1):
            try:
                # Look for corresponding annotation file
                annotation_file = find_annotation_for_image(
                    img_file,
                    exact_index,
                    normalized_index,
                )

                # Get image dimensions
                width, height = get_image_dimensions(img_file)

                # Generate thumbnail
                thumbnail_url = generate_thumbnail(img_file)

                # Parse annotations if they exist
                annotations = []
                if annotation_file and annotation_file.exists():
                    yolo_annots = parse_yolo_annotation_file(annotation_file, class_names)

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
                    execution_id=execution_id,
                    project_id=project_id,
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
                    index,
                    total_images,
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
                "project_id": project_id,
                "source_path": str(source_p),
                "image_dir": str(image_p),
            },
            review_items=created_count,
        )
