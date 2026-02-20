"""Review queue data models for annotation correction workflow."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ReviewStatus(str, Enum):
    """Status of a review item."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class Point(BaseModel):
    """2D point for polygon annotations."""

    x: float = Field(..., ge=0, le=1, description="Normalized X coordinate (0-1)")
    y: float = Field(..., ge=0, le=1, description="Normalized Y coordinate (0-1)")


class BoundingBox(BaseModel):
    """Bounding box with normalized coordinates."""

    x: float = Field(..., ge=0, le=1, description="Normalized X coordinate (0-1)")
    y: float = Field(..., ge=0, le=1, description="Normalized Y coordinate (0-1)")
    width: float = Field(..., gt=0, le=1, description="Normalized width (0-1)")
    height: float = Field(..., gt=0, le=1, description="Normalized height (0-1)")


class Annotation(BaseModel):
    """Annotation on an image (bbox, polygon, or mask)."""

    id: str = Field(..., description="Unique annotation ID")
    type: Literal["bbox", "polygon", "mask"] = Field(
        ..., description="Type of annotation"
    )
    label: str = Field(..., description="Class/category label")
    confidence: float = Field(
        ..., ge=0, le=1, description="Detection confidence score"
    )
    bbox: BoundingBox | None = Field(None, description="Bounding box (if type=bbox)")
    polygon: list[Point] | None = Field(
        None, description="Polygon points (if type=polygon)"
    )
    mask_url: str | None = Field(
        None, description="URL to mask image (if type=mask)"
    )
    color: str = Field("#166534", description="Display color (hex)")
    visible: bool = Field(True, description="Visibility toggle")


class ImageDimensions(BaseModel):
    """Image dimensions in pixels."""

    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")


class ReviewItem(BaseModel):
    """Single item in the review queue."""

    id: str = Field(..., description="Unique review item ID")
    execution_id: str | None = Field(
        None, description="Execution ID used for review queue isolation"
    )
    project_id: str | None = Field(
        None, description="Project ID used for review queue isolation"
    )
    file_path: str = Field(..., description="Absolute path to source image file")
    file_name: str = Field(..., description="Image filename")
    dimensions: ImageDimensions = Field(..., description="Image dimensions")
    thumbnail_url: str = Field(
        ..., description="Base64-encoded thumbnail data URL for list view"
    )
    annotations: list[Annotation] = Field(
        default_factory=list, description="Current annotations"
    )
    original_annotations: list[Annotation] = Field(
        default_factory=list, description="Original annotations for reset"
    )
    status: ReviewStatus = Field(
        ReviewStatus.PENDING, description="Review status"
    )
    reviewed_at: datetime | None = Field(
        None, description="Timestamp when reviewed"
    )
    flagged: bool = Field(False, description="Whether item is flagged for attention")
    flag_reason: str | None = Field(
        None, description="Reason for flagging (if flagged=True)"
    )


class ReviewItemUpdate(BaseModel):
    """Update payload for review item."""

    status: ReviewStatus | None = None
    annotations: list[Annotation] | None = None
    flagged: bool | None = None
    flag_reason: str | None = None


class AnnotationCreate(BaseModel):
    """Payload for creating a new annotation."""

    type: Literal["bbox", "polygon", "mask"]
    label: str
    confidence: float = Field(1.0, ge=0, le=1)
    bbox: BoundingBox | None = None
    polygon: list[Point] | None = None
    mask_url: str | None = None
    color: str = "#166534"
    visible: bool = True


class AnnotationUpdate(BaseModel):
    """Payload for updating an annotation."""

    label: str | None = None
    confidence: float | None = Field(None, ge=0, le=1)
    bbox: BoundingBox | None = None
    polygon: list[Point] | None = None
    color: str | None = None
    visible: bool | None = None


class BatchRequest(BaseModel):
    """Request payload for batch operations."""

    item_ids: list[str] = Field(..., min_length=1, description="List of item IDs")


class BatchResponse(BaseModel):
    """Response for batch operations."""

    success_count: int = Field(..., description="Number of successfully processed items")
    failed_count: int = Field(0, description="Number of failed items")
    failed_ids: list[str] = Field(
        default_factory=list, description="IDs of failed items"
    )


class ReviewItemsResponse(BaseModel):
    """Response containing multiple review items with pagination."""

    items: list[ReviewItem] = Field(..., description="List of review items")
    total: int = Field(..., description="Total number of items matching query")
    skip: int = Field(0, description="Number of items skipped")
    limit: int = Field(50, description="Maximum items returned")


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")
