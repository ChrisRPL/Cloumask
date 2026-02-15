"""Review queue API routes for human-in-the-loop annotation correction."""

import json
import uuid
from datetime import UTC
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import ValidationError

from backend.api.models.review import (
    Annotation,
    AnnotationCreate,
    AnnotationUpdate,
    BatchRequest,
    BatchResponse,
    BoundingBox,
    ImageDimensions,
    ReviewItem,
    ReviewItemsResponse,
    ReviewItemUpdate,
    ReviewStatus,
    SuccessResponse,
)
from backend.cv.utils.thumbnail import generate_thumbnail, get_image_dimensions

router = APIRouter(prefix="/api/review", tags=["review"])

_VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# In-memory storage (replace with SQLite/PostgreSQL in production)
# Format: {item_id: ReviewItem}
_review_items: dict[str, ReviewItem] = {}


def _get_item_or_404(item_id: str) -> ReviewItem:
    """Get review item by ID or raise 404."""
    if item_id not in _review_items:
        raise HTTPException(status_code=404, detail=f"Review item {item_id} not found")
    return _review_items[item_id]


def _save_to_disk(execution_id: str):
    """Save review items to disk for persistence (JSON format)."""
    # This would be replaced with proper database in production
    data_dir = Path("data/review")
    data_dir.mkdir(parents=True, exist_ok=True)

    items_data = [item.model_dump(mode="json") for item in _review_items.values()]
    with (data_dir / f"{execution_id}.json").open("w") as f:
        json.dump(items_data, f, indent=2)


def _load_from_disk(execution_id: str):
    """Load review items from disk."""
    data_dir = Path("data/review")
    filepath = data_dir / f"{execution_id}.json"

    if filepath.exists():
        with filepath.open() as f:
            items_data = json.load(f)
            for item_data in items_data:
                try:
                    item = ReviewItem.model_validate(item_data)
                    _review_items[item.id] = item
                except ValidationError:
                    # Skip invalid items
                    continue


@router.get("/items", response_model=ReviewItemsResponse)
async def get_review_items(  # noqa: B008
    execution_id: str = Query(..., description="Execution ID to load items from"),
    status: ReviewStatus | None = Query(None, description="Filter by review status"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(50, ge=1, le=200, description="Maximum items to return"),
):
    """
    Load review items from execution results with optional filtering and pagination.

    Args:
        execution_id: ID of the execution to load items from
        status: Optional filter by review status
        skip: Number of items to skip (pagination)
        limit: Maximum number of items to return

    Returns:
        ReviewItemsResponse with paginated items and total count
    """
    # Load from disk if not in memory
    if not _review_items:
        _load_from_disk(execution_id)

    # Filter by status if specified
    items = list(_review_items.values())
    if status:
        items = [item for item in items if item.status == status]

    total = len(items)

    # Apply pagination
    paginated_items = items[skip : skip + limit]

    return ReviewItemsResponse(items=paginated_items, total=total, skip=skip, limit=limit)


@router.get("/items/{item_id}", response_model=ReviewItem)
async def get_review_item(item_id: str):
    """
    Get a single review item by ID.

    Args:
        item_id: Unique review item ID

    Returns:
        ReviewItem with full details

    Raises:
        HTTPException: 404 if item not found
    """
    return _get_item_or_404(item_id)


@router.get("/image")
async def get_local_image(
    path: str = Query(..., description="Absolute or workspace-relative image path"),
):
    """
    Serve a local image file for browser-based previews.

    This endpoint is used by the web UI (non-Tauri mode) where direct file:// access
    is not available.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    else:
        resolved = resolved.resolve()

    if resolved.suffix.lower() not in _VALID_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image extension: {resolved.suffix}",
        )

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {resolved}")

    return FileResponse(resolved)


@router.put("/items/{item_id}", response_model=ReviewItem)
async def update_review_item(item_id: str, updates: ReviewItemUpdate):
    """
    Update review item status and/or annotations.

    Args:
        item_id: Unique review item ID
        updates: Partial updates to apply

    Returns:
        Updated ReviewItem

    Raises:
        HTTPException: 404 if item not found
    """
    item = _get_item_or_404(item_id)

    # Apply updates
    update_data = updates.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(item, field, value)

    # Update reviewed timestamp if status changed
    if "status" in update_data:
        from datetime import datetime

        item.reviewed_at = datetime.now(UTC)

    _review_items[item_id] = item

    return item


@router.put("/items/{item_id}/annotations/{annotation_id}", response_model=Annotation)
async def update_annotation(item_id: str, annotation_id: str, updates: AnnotationUpdate):
    """
    Update a single annotation within a review item.

    Args:
        item_id: Review item ID
        annotation_id: Annotation ID to update
        updates: Partial annotation updates

    Returns:
        Updated Annotation

    Raises:
        HTTPException: 404 if item or annotation not found
    """
    item = _get_item_or_404(item_id)

    # Find annotation
    annotation = next((ann for ann in item.annotations if ann.id == annotation_id), None)
    if not annotation:
        raise HTTPException(
            status_code=404,
            detail=f"Annotation {annotation_id} not found in item {item_id}",
        )

    # Apply updates
    update_data = updates.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(annotation, field, value)

    # Mark item as modified
    item.status = ReviewStatus.MODIFIED

    _review_items[item_id] = item

    return annotation


@router.post("/items/{item_id}/annotations", response_model=Annotation)
async def add_annotation(item_id: str, annotation: AnnotationCreate):
    """
    Add a new annotation to a review item.

    Args:
        item_id: Review item ID
        annotation: New annotation to add

    Returns:
        Created Annotation with generated ID

    Raises:
        HTTPException: 404 if item not found
    """
    item = _get_item_or_404(item_id)

    # Create annotation with generated ID
    new_annotation = Annotation(id=str(uuid.uuid4()), **annotation.model_dump())

    item.annotations.append(new_annotation)
    item.status = ReviewStatus.MODIFIED

    _review_items[item_id] = item

    return new_annotation


@router.delete("/items/{item_id}/annotations/{annotation_id}")
async def delete_annotation(item_id: str, annotation_id: str):
    """
    Delete an annotation from a review item.

    Args:
        item_id: Review item ID
        annotation_id: Annotation ID to delete

    Returns:
        SuccessResponse

    Raises:
        HTTPException: 404 if item or annotation not found
    """
    item = _get_item_or_404(item_id)

    # Find and remove annotation
    original_count = len(item.annotations)
    item.annotations = [ann for ann in item.annotations if ann.id != annotation_id]

    if len(item.annotations) == original_count:
        raise HTTPException(
            status_code=404,
            detail=f"Annotation {annotation_id} not found in item {item_id}",
        )

    item.status = ReviewStatus.MODIFIED

    _review_items[item_id] = item

    return SuccessResponse(success=True, message="Annotation deleted successfully")


@router.post("/batch-approve", response_model=BatchResponse)
async def batch_approve(request: BatchRequest):
    """
    Batch approve multiple review items.

    Args:
        request: List of item IDs to approve

    Returns:
        BatchResponse with success/failure counts
    """
    from datetime import datetime

    success_count = 0
    failed_ids = []

    for item_id in request.item_ids:
        if item_id in _review_items:
            item = _review_items[item_id]
            item.status = ReviewStatus.APPROVED
            item.reviewed_at = datetime.now(UTC)
            success_count += 1
        else:
            failed_ids.append(item_id)

    return BatchResponse(
        success_count=success_count,
        failed_count=len(failed_ids),
        failed_ids=failed_ids,
    )


@router.post("/batch-reject", response_model=BatchResponse)
async def batch_reject(request: BatchRequest):
    """
    Batch reject multiple review items.

    Args:
        request: List of item IDs to reject

    Returns:
        BatchResponse with success/failure counts
    """
    from datetime import datetime

    success_count = 0
    failed_ids = []

    for item_id in request.item_ids:
        if item_id in _review_items:
            item = _review_items[item_id]
            item.status = ReviewStatus.REJECTED
            item.reviewed_at = datetime.now(UTC)
            success_count += 1
        else:
            failed_ids.append(item_id)

    return BatchResponse(
        success_count=success_count,
        failed_count=len(failed_ids),
        failed_ids=failed_ids,
    )


# Utility endpoint for testing - populate review queue with sample data
@router.post("/seed")
async def seed_review_items(execution_id: str, image_dir: str):
    """
    Seed the review queue with items from a directory (for testing).

    Args:
        execution_id: Execution ID to associate items with
        image_dir: Directory containing images to create review items from

    Returns:
        Number of items created
    """

    image_path = Path(image_dir)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image directory not found: {image_dir}")

    created_count = 0
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for img_file in image_path.glob("*"):
        if img_file.suffix.lower() not in valid_extensions:
            continue

        try:
            # Generate thumbnail
            thumbnail_url = generate_thumbnail(img_file)

            # Get dimensions
            width, height = get_image_dimensions(img_file)

            # Create review item

            item = ReviewItem(
                id=str(uuid.uuid4()),
                file_path=str(img_file.absolute()),
                file_name=img_file.name,
                dimensions=ImageDimensions(width=width, height=height),
                thumbnail_url=thumbnail_url,
                annotations=[],
                original_annotations=[],
                status=ReviewStatus.PENDING,
                reviewed_at=None,
                flagged=False,
                flag_reason=None,
            )

            _review_items[item.id] = item
            created_count += 1

        except Exception as e:
            # Skip files that can't be processed
            print(f"Error processing {img_file}: {e}")
            continue

    # Save to disk
    _save_to_disk(execution_id)

    return {"created_count": created_count, "execution_id": execution_id}


@router.post("/import-annotations")
async def import_annotations(
    execution_id: str,
    annotations_dir: str,
    image_dir: str,
):
    """
    Import YOLO-format annotations into existing review items.

    Args:
        execution_id: Execution ID associated with review items
        annotations_dir: Directory containing YOLO .txt annotation files
        image_dir: Directory containing source images

    Returns:
        Count of items updated with annotations
    """
    import uuid

    annot_path = Path(annotations_dir)
    img_path = Path(image_dir)

    if not annot_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Annotations directory not found: {annotations_dir}"
        )
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image directory not found: {image_dir}")

    # Load existing items
    if not _review_items:
        _load_from_disk(execution_id)

    updated_count = 0

    # Create mapping of filename to review item
    file_to_item = {item.file_name: item for item in _review_items.values()}

    # Parse and import annotations
    for annot_file in annot_path.glob("*.txt"):
        # Find corresponding image
        image_name = annot_file.stem
        image_file = None

        # Try common image extensions
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            candidate = img_path / f"{image_name}{ext}"
            if candidate.exists():
                image_file = candidate
                break

        if not image_file or image_file.name not in file_to_item:
            continue

        item = file_to_item[image_file.name]

        try:
            # Parse YOLO annotations
            annotations = []

            # COCO classes mapping
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

            with annot_file.open() as f:
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
                        coco_classes[class_id]
                        if class_id < len(coco_classes)
                        else f"class_{class_id}"
                    )

                    annotation = Annotation(
                        id=str(uuid.uuid4()),
                        type="bbox",
                        label=class_name,
                        confidence=confidence,
                        bbox=BoundingBox(
                            x=max(0, min(1, x)),
                            y=max(0, min(1, y)),
                            width=max(0, min(1, width)),
                            height=max(0, min(1, height)),
                        ),
                        polygon=None,
                        mask_url=None,
                        color="#166534",
                        visible=True,
                    )
                    annotations.append(annotation)

            # Update item with annotations
            if annotations:
                item.annotations = annotations
                item.original_annotations = annotations.copy()
                updated_count += 1

        except Exception as e:
            print(f"Error importing annotations for {annot_file}: {e}")
            continue

    # Save updated items
    if updated_count > 0:
        _save_to_disk(execution_id)

    return {
        "updated_count": updated_count,
        "execution_id": execution_id,
        "total_items": len(_review_items),
    }
