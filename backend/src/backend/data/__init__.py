"""Data loaders and exporters for Cloumask.

This module provides data loading and extraction utilities including:
- ROS bag parsing and sensor data extraction
- Core data models for label representation (BBox, Label, Sample, Dataset)
- Format loaders and exporters (YOLO, COCO, KITTI, VOC, CVAT)
- Format conversion utilities
"""

from backend.data.duplicates import (
    DuplicateDetector,
    DuplicateGroup,
    DuplicateResult,
    EmbeddingComparator,
    PerceptualHasher,
    find_duplicates,
)
from backend.data.formats import (
    FormatExporter,
    FormatLoader,
    FormatRegistry,
    convert,
    detect_format,
    get_exporter,
    get_loader,
    list_formats,
)
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
from backend.data.qa import (
    IssueType,
    LabelQA,
    QAIssue,
    QAResult,
    run_qa,
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
from backend.data.splitting import (
    DatasetSplitter,
    SplitResult,
    create_folds,
    cross_validation_indices,
    split_dataset,
    split_indices,
    stratified_split_indices,
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
    # Format handling
    "FormatLoader",
    "FormatExporter",
    "FormatRegistry",
    "get_loader",
    "get_exporter",
    "detect_format",
    "list_formats",
    "convert",
    # Duplicate detection
    "PerceptualHasher",
    "EmbeddingComparator",
    "DuplicateGroup",
    "DuplicateResult",
    "DuplicateDetector",
    "find_duplicates",
    # Label QA
    "IssueType",
    "QAIssue",
    "QAResult",
    "LabelQA",
    "run_qa",
    # Dataset splitting
    "SplitResult",
    "DatasetSplitter",
    "split_indices",
    "stratified_split_indices",
    "cross_validation_indices",
    "split_dataset",
    "create_folds",
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
