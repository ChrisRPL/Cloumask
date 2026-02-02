"""
ROS bag data types for extracted sensor messages.

This module defines dataclasses for ROS bag metadata and extracted
sensor data including PointCloud2, Image, and CameraInfo messages.

Implements spec: 05-point-cloud/03-rosbag-extraction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class BagFormat(str, Enum):
    """ROS bag file format."""

    ROS1 = "ros1"
    ROS2 = "ros2"


@dataclass
class TopicInfo:
    """Information about a ROS topic.

    Attributes:
        name: Topic name (e.g., "/velodyne_points").
        msg_type: Message type (e.g., "sensor_msgs/msg/PointCloud2").
        message_count: Number of messages on this topic.
        frequency_hz: Estimated message frequency, None if unknown.
    """

    name: str
    msg_type: str
    message_count: int
    frequency_hz: float | None = None


@dataclass
class BagInfo:
    """Metadata about a ROS bag file.

    Attributes:
        path: Absolute path to the bag file.
        format: Bag format ("ros1" or "ros2").
        duration_sec: Total duration in seconds.
        start_time: Start timestamp (UNIX epoch seconds).
        end_time: End timestamp (UNIX epoch seconds).
        topics: List of topics in the bag.
        message_count: Total message count across all topics.
    """

    path: str
    format: BagFormat
    duration_sec: float
    start_time: float
    end_time: float
    topics: list[TopicInfo]
    message_count: int = 0


@dataclass
class PointCloud2Message:
    """Extracted PointCloud2 sensor data.

    Attributes:
        timestamp: Message timestamp (UNIX epoch seconds).
        frame_id: TF frame ID (e.g., "velodyne", "lidar_top").
        points: Point coordinates (N, 3) float64 in meters.
        intensity: Optional intensity values (N,) float32.
        rgb: Optional RGB colors (N, 3) uint8.
        fields: List of field names in original message.
    """

    timestamp: float
    frame_id: str
    points: NDArray[np.float64]  # (N, 3)
    intensity: NDArray[np.float32] | None = None  # (N,)
    rgb: NDArray[np.uint8] | None = None  # (N, 3)
    fields: list[str] = field(default_factory=list)

    @property
    def point_count(self) -> int:
        """Number of points in the cloud."""
        return self.points.shape[0]


@dataclass
class ImageMessage:
    """Extracted Image sensor data.

    Attributes:
        timestamp: Message timestamp (UNIX epoch seconds).
        frame_id: TF frame ID (e.g., "camera_front").
        image: Image data (H, W, C) uint8.
        encoding: Original encoding (e.g., "rgb8", "bgr8", "mono8").
        width: Image width in pixels.
        height: Image height in pixels.
    """

    timestamp: float
    frame_id: str
    image: NDArray[np.uint8]  # (H, W, C)
    encoding: str
    width: int
    height: int


@dataclass
class CameraInfoMessage:
    """Extracted CameraInfo sensor data.

    Attributes:
        timestamp: Message timestamp (UNIX epoch seconds).
        frame_id: TF frame ID.
        width: Image width in pixels.
        height: Image height in pixels.
        K: Intrinsic camera matrix (3, 3) float64.
        D: Distortion coefficients (N,) float64.
        R: Rectification matrix (3, 3) float64.
        P: Projection matrix (3, 4) float64.
        distortion_model: Distortion model name.
    """

    timestamp: float
    frame_id: str
    width: int
    height: int
    K: NDArray[np.float64]  # (3, 3)
    D: NDArray[np.float64]  # (N,)
    R: NDArray[np.float64]  # (3, 3)
    P: NDArray[np.float64]  # (3, 4)
    distortion_model: str = "plumb_bob"


@dataclass
class SyncedFrame:
    """Synchronized multi-sensor frame.

    Represents a single time-aligned frame from multiple sensors.

    Attributes:
        timestamp: Reference timestamp (from primary sensor).
        pointcloud: Optional point cloud data.
        image: Optional image data.
        camera_info: Optional camera calibration.
        sync_error_ms: Maximum timestamp difference in milliseconds.
        frame_index: Frame index in sequence.
    """

    timestamp: float
    pointcloud: PointCloud2Message | None
    image: ImageMessage | None
    camera_info: CameraInfoMessage | None
    sync_error_ms: float
    frame_index: int = 0


# Pydantic models for API request/response


class TopicInfoResponse(BaseModel):
    """API response for a single topic."""

    name: str = Field(..., description="Topic name")
    msg_type: str = Field(..., description="Message type")
    message_count: int = Field(..., ge=0, description="Number of messages")
    frequency_hz: float | None = Field(None, description="Estimated frequency in Hz")


class BagInfoResponse(BaseModel):
    """API response model for bag metadata."""

    path: str = Field(..., description="Path to bag file")
    format: str = Field(..., description="Bag format: ros1 or ros2")
    duration_sec: float = Field(..., ge=0, description="Duration in seconds")
    start_time: float = Field(..., description="Start timestamp (epoch)")
    end_time: float = Field(..., description="End timestamp (epoch)")
    message_count: int = Field(..., ge=0, description="Total message count")
    topics: list[TopicInfoResponse] = Field(..., description="List of topic info")


class ExtractPointcloudRequest(BaseModel):
    """Request to extract point clouds from bag."""

    bag_path: str = Field(..., description="Path to ROS bag file")
    topic: str = Field(..., description="PointCloud2 topic name")
    output_dir: str = Field(..., description="Output directory path")
    max_frames: int | None = Field(None, ge=1, description="Max frames to extract")
    output_format: str = Field("pcd", description="Output format: pcd, ply, npy")
    skip_existing: bool = Field(False, description="Skip if output exists")


class ExtractImagesRequest(BaseModel):
    """Request to extract images from bag."""

    bag_path: str = Field(..., description="Path to ROS bag file")
    topic: str = Field(..., description="Image topic name")
    output_dir: str = Field(..., description="Output directory path")
    max_frames: int | None = Field(None, ge=1, description="Max frames to extract")
    output_format: str = Field("png", description="Output format: png, jpg")
    skip_existing: bool = Field(False, description="Skip if output exists")


class ExtractSyncedRequest(BaseModel):
    """Request to extract synchronized frames."""

    bag_path: str = Field(..., description="Path to ROS bag file")
    pointcloud_topic: str = Field(..., description="PointCloud2 topic")
    image_topic: str = Field(..., description="Image topic")
    camera_info_topic: str | None = Field(None, description="CameraInfo topic")
    output_dir: str = Field(..., description="Output directory path")
    max_frames: int | None = Field(None, ge=1, description="Max frames to extract")
    max_sync_error_ms: float = Field(50.0, ge=0, description="Max sync error in ms")
    pointcloud_format: str = Field("pcd", description="Point cloud format")
    image_format: str = Field("png", description="Image format")


class ExtractionResult(BaseModel):
    """Result of extraction operation."""

    extracted_count: int = Field(..., ge=0, description="Frames extracted")
    output_dir: str = Field(..., description="Output directory")
    processing_time_sec: float = Field(..., ge=0, description="Processing time")
    files: list[str] = Field(default_factory=list, description="Output files")


class SyncedExtractionResult(BaseModel):
    """Result of synchronized extraction."""

    extracted_count: int = Field(..., ge=0, description="Synced frames extracted")
    output_dir: str = Field(..., description="Output directory")
    processing_time_sec: float = Field(..., ge=0, description="Processing time")
    average_sync_error_ms: float = Field(..., ge=0, description="Avg sync error")
    pointcloud_files: list[str] = Field(default_factory=list, description="Point cloud files")
    image_files: list[str] = Field(default_factory=list, description="Image files")
