"""Shared constants for agent tools.

This module provides common constants used across multiple tools,
ensuring consistency in file type detection and categorization.
"""

# Supported file extensions by type
# These are used for categorizing files in scan operations and
# determining which files can be processed by various tools.

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".gif",
})

VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".wmv",
    ".flv",
})

POINTCLOUD_EXTENSIONS: frozenset[str] = frozenset({
    ".las",
    ".laz",
    ".pcd",
    ".ply",
    ".xyz",
    ".pts",
    ".e57",
    ".bin",  # KITTI binary format
})

ANNOTATION_EXTENSIONS: frozenset[str] = frozenset({
    ".json",
    ".xml",
    ".txt",
    ".csv",
    ".yaml",
    ".yml",
})

# Export format constants
SUPPORTED_EXPORT_FORMATS: list[str] = [
    "yolo",
    "coco",
    "pascal",
    "labelme",
    "cvat",
]

# Default 2D detection classes (common COCO subset)
DEFAULT_DETECTION_CLASSES: list[str] = [
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
]

# 3D detection classes (KITTI/nuScenes)
DETECTION_3D_CLASSES: list[str] = [
    "Car",
    "Pedestrian",
    "Cyclist",
]

# Progress reporting throttle interval
PROGRESS_THROTTLE_INTERVAL: int = 100  # Report every N files
