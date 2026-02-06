"""
Fused annotation format for linked 2D and 3D annotations.

Provides data models for representing sensor fusion results where
2D image detections are matched with 3D LiDAR detections.

Implements spec: 05-point-cloud/05-2d-3d-fusion
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

# Import at runtime for Pydantic - can't use TYPE_CHECKING for annotated fields
from backend.cv.types import BBox, Detection3D


class FusedAnnotation(BaseModel):
    """Linked 2D and 3D annotation for a single object.

    Represents a detection that has been matched between camera (2D)
    and LiDAR (3D) modalities, or a detection from only one modality.

    Attributes:
        bbox_2d: 2D bounding box in normalized [0-1] coordinates.
        detection_3d: Full 3D detection, None if 2D-only.
        class_id: Integer class identifier.
        class_name: Human-readable class name.
        confidence_2d: Confidence from 2D detector.
        confidence_3d: Confidence from 3D detector, None if 2D-only.
        track_id: Track ID for multi-frame consistency.
        occlusion: Occlusion level (0=visible, 1=partly, 2=largely, 3=fully).
        truncation: Fraction of object outside image bounds [0-1].
        iou_2d_3d: IoU between projected 3D box and 2D detection.
        depth_meters: Distance from camera to object center.
    """

    # 2D bounding box (normalized coordinates)
    bbox_2d: "BBox" = Field(..., description="2D bounding box")

    # 3D detection (optional - may be 2D-only)
    detection_3d: "Detection3D | None" = Field(
        None, description="3D detection if available"
    )

    # Class information
    class_id: int = Field(..., ge=0, description="Class ID")
    class_name: str = Field(..., min_length=1, description="Class name")

    # Confidence scores
    confidence_2d: float = Field(..., ge=0.0, le=1.0, description="2D confidence")
    confidence_3d: float | None = Field(
        None, ge=0.0, le=1.0, description="3D confidence"
    )

    # Tracking
    track_id: int | None = Field(None, description="Track ID across frames")

    # Occlusion and truncation (KITTI-style)
    occlusion: int = Field(0, ge=0, le=3, description="Occlusion level 0-3")
    truncation: float = Field(0.0, ge=0.0, le=1.0, description="Truncation ratio")

    # Fusion metadata
    iou_2d_3d: float | None = Field(
        None, ge=0.0, le=1.0, description="IoU between projected 3D and 2D box"
    )
    depth_meters: float | None = Field(None, description="Distance from camera")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_3d(self) -> bool:
        """Check if 3D annotation is available."""
        return self.detection_3d is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """Combined confidence (average of 2D and 3D if both present)."""
        if self.confidence_3d is not None:
            return (self.confidence_2d + self.confidence_3d) / 2
        return self.confidence_2d


class FusedAnnotationResult(BaseModel):
    """Complete fused annotation result for a frame.

    Contains all fused annotations for a single synchronized frame
    along with metadata about the source data.

    Attributes:
        annotations: List of fused annotations.
        image_path: Path to source image.
        pointcloud_path: Path to source point cloud.
        calibration_path: Path to calibration file.
        frame_id: Frame index in sequence.
        timestamp: Frame timestamp.
        processing_time_ms: Time taken for fusion.
    """

    annotations: list[FusedAnnotation] = Field(default_factory=list)
    image_path: str = Field("", description="Path to image")
    pointcloud_path: str | None = Field(None, description="Path to point cloud")
    calibration_path: str | None = Field(None, description="Path to calibration")
    frame_id: int | None = Field(None, description="Frame index")
    timestamp: float | None = Field(None, description="Timestamp")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of annotations in result."""
        return len(self.annotations)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count_with_3d(self) -> int:
        """Number of annotations with 3D data."""
        return sum(1 for a in self.annotations if a.has_3d)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count_2d_only(self) -> int:
        """Number of annotations with only 2D data."""
        return sum(1 for a in self.annotations if not a.has_3d)
