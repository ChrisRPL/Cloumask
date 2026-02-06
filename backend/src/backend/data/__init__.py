"""Data loaders and exporters for Cloumask.

This module provides data loading and extraction utilities including:
- ROS bag parsing and sensor data extraction
- Core data models for label representation (BBox, Label, Sample, Dataset)
- Format conversion utilities
"""

from backend.data.models import (
    BBox,
    BBoxFormat,
    BBoxSchema,
    Dataset,
    DatasetStatsSchema,
    Label,
    LabelSchema,
    Sample,
    SampleSchema,
)
from backend.data.ros_types import (
    BagFormat,
    BagInfo,
    BagInfoResponse,
    CameraInfoMessage,
    ExtractImagesRequest,
    ExtractionResult,
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
    # Data Pipeline Models
    "BBox",
    "BBoxFormat",
    "Label",
    "Sample",
    "Dataset",
    # Data Pipeline Pydantic Schemas
    "BBoxSchema",
    "LabelSchema",
    "SampleSchema",
    "DatasetStatsSchema",
    # ROS Enums
    "BagFormat",
    # ROS Dataclasses
    "TopicInfo",
    "BagInfo",
    "PointCloud2Message",
    "ImageMessage",
    "CameraInfoMessage",
    "SyncedFrame",
    # ROS Pydantic models
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
