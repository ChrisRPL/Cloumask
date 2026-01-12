"""
Core Pydantic types for computer vision detection and segmentation results.

This module defines the data models used throughout the CV pipeline for
representing bounding boxes, detections, masks, and other CV primitives.
All types are serializable and designed for both internal use and API responses.

Implements spec: 03-cv-models/00-infrastructure
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field, computed_field

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BBox(BaseModel):
    """
    Bounding box in normalized coordinates [0-1].

    Uses center-based format (YOLO style) internally but provides
    conversion methods for other common formats.

    Attributes:
        x: Center x coordinate, normalized [0-1].
        y: Center y coordinate, normalized [0-1].
        width: Box width, normalized [0-1].
        height: Box height, normalized [0-1].
    """

    x: float = Field(..., ge=0.0, le=1.0, description="Center x coordinate")
    y: float = Field(..., ge=0.0, le=1.0, description="Center y coordinate")
    width: float = Field(..., ge=0.0, le=1.0, description="Box width")
    height: float = Field(..., ge=0.0, le=1.0, description="Box height")

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """
        Convert to absolute pixel coordinates (x1, y1, x2, y2).

        Args:
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates.
        """
        half_w = self.width / 2
        half_h = self.height / 2

        x1 = int((self.x - half_w) * img_w)
        y1 = int((self.y - half_h) * img_h)
        x2 = int((self.x + half_w) * img_w)
        y2 = int((self.y + half_h) * img_h)

        # Clamp to image bounds (0-indexed, so max is size-1)
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        return x1, y1, x2, y2

    def to_xywh(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """
        Convert to absolute pixel coordinates (x, y, width, height).

        Args:
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            Tuple of (x, y, width, height) where (x, y) is top-left corner.
        """
        x1, y1, x2, y2 = self.to_xyxy(img_w, img_h)
        return x1, y1, x2 - x1, y2 - y1

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> BBox:
        """
        Create BBox from absolute pixel coordinates.

        Args:
            x1, y1: Top-left corner in pixels.
            x2, y2: Bottom-right corner in pixels.
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            BBox in normalized center format.
        """
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        x = (x1 + x2) / 2 / img_w
        y = (y1 + y2) / 2 / img_h

        return cls(x=x, y=y, width=width, height=height)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def area(self) -> float:
        """Normalized area of the bounding box."""
        return self.width * self.height


class Detection(BaseModel):
    """
    Single object detection result.

    Represents one detected object with its bounding box, class information,
    and confidence score.

    Attributes:
        class_id: Integer class identifier from the model.
        class_name: Human-readable class name.
        bbox: Bounding box in normalized coordinates.
        confidence: Detection confidence score [0-1].
    """

    class_id: int = Field(..., ge=0, description="Class ID from model")
    class_name: str = Field(..., min_length=1, description="Human-readable class name")
    bbox: BBox = Field(..., description="Bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class DetectionResult(BaseModel):
    """
    Complete detection result for an image.

    Contains all detections found in an image along with metadata
    about the processing.

    Attributes:
        detections: List of individual detections.
        image_path: Path to the processed image.
        processing_time_ms: Time taken for inference in milliseconds.
        model_name: Name of the model used for detection.
    """

    detections: list[Detection] = Field(default_factory=list)
    image_path: str = Field(..., description="Path to processed image")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in ms")
    model_name: str = Field(..., description="Model used for detection")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of detections found."""
        return len(self.detections)

    def filter_by_confidence(self, min_confidence: float) -> DetectionResult:
        """
        Return a new result with only detections above threshold.

        Args:
            min_confidence: Minimum confidence threshold [0-1].

        Returns:
            New DetectionResult with filtered detections.
        """
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return DetectionResult(
            detections=filtered,
            image_path=self.image_path,
            processing_time_ms=self.processing_time_ms,
            model_name=self.model_name,
        )

    def filter_by_class(self, class_names: list[str]) -> DetectionResult:
        """
        Return a new result with only specified classes.

        Args:
            class_names: List of class names to keep.

        Returns:
            New DetectionResult with filtered detections.
        """
        class_set = set(class_names)
        filtered = [d for d in self.detections if d.class_name in class_set]
        return DetectionResult(
            detections=filtered,
            image_path=self.image_path,
            processing_time_ms=self.processing_time_ms,
            model_name=self.model_name,
        )


class Mask(BaseModel):
    """
    Segmentation mask for a single instance.

    Stores binary mask data in compressed format for efficient serialization.
    Provides methods for numpy array conversion.

    Attributes:
        data: zlib-compressed binary mask data.
        width: Mask width in pixels.
        height: Mask height in pixels.
        confidence: Segmentation confidence score [0-1].
    """

    model_config = {"arbitrary_types_allowed": True}

    data: bytes = Field(..., description="Compressed mask data")
    width: int = Field(..., gt=0, description="Mask width in pixels")
    height: int = Field(..., gt=0, description="Mask height in pixels")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    @classmethod
    def from_numpy(cls, arr: NDArray[np.uint8], confidence: float) -> Mask:
        """
        Create Mask from numpy array.

        Args:
            arr: Binary numpy array (HxW) with values 0 or 1/255.
            confidence: Segmentation confidence score.

        Returns:
            Mask with compressed data.
        """
        # Ensure binary mask
        binary_arr = (arr > 0).astype(np.uint8)
        # Compress using zlib for efficient storage
        compressed = zlib.compress(binary_arr.tobytes(), level=6)

        return cls(
            data=compressed,
            width=arr.shape[1],
            height=arr.shape[0],
            confidence=confidence,
        )

    def to_numpy(self) -> NDArray[np.uint8]:
        """
        Decompress and return mask as numpy array.

        Returns:
            Binary numpy array (HxW) with values 0 or 1.
        """
        decompressed = zlib.decompress(self.data)
        arr = np.frombuffer(decompressed, dtype=np.uint8)
        return arr.reshape((self.height, self.width))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def area_pixels(self) -> int:
        """Total area of the mask in pixels (computed on demand)."""
        return int(np.sum(self.to_numpy()))


class SegmentationResult(BaseModel):
    """
    Complete segmentation result for an image.

    Contains all masks found in an image along with metadata
    about the processing.

    Attributes:
        masks: List of segmentation masks.
        image_path: Path to the processed image.
        processing_time_ms: Time taken for inference in milliseconds.
        model_name: Name of the model used for segmentation.
        prompt: Optional text prompt used for segmentation.
    """

    masks: list[Mask] = Field(default_factory=list)
    image_path: str = Field(..., description="Path to processed image")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in ms")
    model_name: str = Field(..., description="Model used for segmentation")
    prompt: str | None = Field(None, description="Text prompt if used")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of masks in result."""
        return len(self.masks)


class FaceDetection(BaseModel):
    """
    Face detection result with optional landmarks.

    Represents a detected face with bounding box, confidence,
    and optional 5-point facial landmarks.

    Attributes:
        bbox: Bounding box around the face.
        confidence: Detection confidence score [0-1].
        landmarks: Optional 5-point landmarks (eyes, nose, mouth corners).
    """

    bbox: BBox = Field(..., description="Face bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    landmarks: list[tuple[float, float]] | None = Field(
        None, description="5-point landmarks in normalized coordinates"
    )

    def get_eye_distance(self, img_w: int, img_h: int) -> float | None:
        """
        Calculate inter-eye distance in pixels.

        Args:
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            Distance between eyes in pixels, or None if landmarks unavailable.
        """
        if not self.landmarks or len(self.landmarks) < 2:
            return None

        left_eye = self.landmarks[0]
        right_eye = self.landmarks[1]

        dx: float = (right_eye[0] - left_eye[0]) * img_w
        dy: float = (right_eye[1] - left_eye[1]) * img_h
        distance: float = (dx**2 + dy**2) ** 0.5

        return distance


class FaceDetectionResult(BaseModel):
    """
    Complete face detection result for an image.

    Attributes:
        faces: List of detected faces.
        image_path: Path to the processed image.
        processing_time_ms: Time taken for inference in milliseconds.
        model_name: Name of the model used for detection.
    """

    faces: list[FaceDetection] = Field(default_factory=list)
    image_path: str = Field(..., description="Path to processed image")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in ms")
    model_name: str = Field(..., description="Model used for detection")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of faces detected."""
        return len(self.faces)


class Detection3D(BaseModel):
    """
    3D bounding box detection for point cloud data.

    Represents a detected 3D object with its 3D bounding box,
    dimensions, and orientation.

    Attributes:
        class_id: Integer class identifier from the model.
        class_name: Human-readable class name.
        center: 3D center coordinates (x, y, z) in meters.
        dimensions: Box dimensions (length, width, height) in meters.
        rotation: Yaw angle in radians.
        confidence: Detection confidence score [0-1].
    """

    class_id: int = Field(..., ge=0, description="Class ID from model")
    class_name: str = Field(..., min_length=1, description="Human-readable class name")
    center: tuple[float, float, float] = Field(..., description="3D center (x, y, z) in meters")
    dimensions: tuple[float, float, float] = Field(
        ..., description="Dimensions (length, width, height) in meters"
    )
    rotation: float = Field(..., description="Yaw angle in radians")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def volume(self) -> float:
        """Volume of the 3D bounding box in cubic meters."""
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]


class Detection3DResult(BaseModel):
    """
    Complete 3D detection result for a point cloud.

    Attributes:
        detections: List of 3D detections.
        pointcloud_path: Path to the processed point cloud.
        processing_time_ms: Time taken for inference in milliseconds.
        model_name: Name of the model used for detection.
    """

    detections: list[Detection3D] = Field(default_factory=list)
    pointcloud_path: str = Field(..., description="Path to processed point cloud")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in ms")
    model_name: str = Field(..., description="Model used for detection")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of 3D detections found."""
        return len(self.detections)


class PlateDetection(BaseModel):
    """
    License plate detection result.

    Represents a detected license plate with optional OCR text.

    Attributes:
        bbox: Bounding box around the plate.
        confidence: Detection confidence score [0-1].
        text: Optional recognized plate text.
        text_confidence: Optional OCR confidence score.
    """

    bbox: BBox = Field(..., description="Plate bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    text: str | None = Field(None, description="Recognized plate text")
    text_confidence: float | None = Field(None, ge=0.0, le=1.0)


class PlateDetectionResult(BaseModel):
    """
    Complete license plate detection result for an image.

    Attributes:
        plates: List of detected plates.
        image_path: Path to the processed image.
        processing_time_ms: Time taken for inference in milliseconds.
        model_name: Name of the model used for detection.
    """

    plates: list[PlateDetection] = Field(default_factory=list)
    image_path: str = Field(..., description="Path to processed image")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in ms")
    model_name: str = Field(..., description="Model used for detection")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of plates detected."""
        return len(self.plates)
