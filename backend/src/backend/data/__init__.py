"""Data loaders and exporters for Cloumask.

This module provides data loading and extraction utilities including:
- ROS bag parsing and sensor data extraction
- Format conversion utilities (planned for 06-data-pipeline)
"""

from backend.data.ros_types import (
    BagFormat,
    BagInfo,
    BagInfoResponse,
    CameraInfoMessage,
    ExtractionResult,
    ExtractImagesRequest,
    ExtractPointcloudRequest,
    ExtractSyncedRequest,
    ImageMessage,
    PointCloud2Message,
    SyncedExtractionResult,
    SyncedFrame,
    TopicInfo,
    TopicInfoResponse,
)
from backend.data.rosbag_parser import (
    RosbagParseError,
    RosbagParser,
    get_parser,
)

__all__ = [
    # Enums
    "BagFormat",
    # Dataclasses
    "TopicInfo",
    "BagInfo",
    "PointCloud2Message",
    "ImageMessage",
    "CameraInfoMessage",
    "SyncedFrame",
    # Pydantic models
    "TopicInfoResponse",
    "BagInfoResponse",
    "ExtractPointcloudRequest",
    "ExtractImagesRequest",
    "ExtractSyncedRequest",
    "ExtractionResult",
    "SyncedExtractionResult",
    # Parser
    "RosbagParser",
    "RosbagParseError",
    "get_parser",
]
