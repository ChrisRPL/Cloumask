"""Format loaders and exporters.

Supports YOLO, COCO, KITTI, Pascal VOC, CVAT, nuScenes, OpenLABEL.
"""

from backend.data.formats.base import (
    FormatExporter,
    FormatLoader,
    FormatRegistry,
    ProgressCallback,
    convert,
    detect_format,
    get_exporter,
    get_loader,
    list_formats,
)
from backend.data.formats.fused_annotation import (
    FusedAnnotation,
    FusedAnnotationResult,
)

# Import format modules to trigger registration
# (uncomment as they are implemented)
from backend.data.formats import yolo  # noqa: F401
from backend.data.formats import coco  # noqa: F401
from backend.data.formats import kitti  # noqa: F401
# from backend.data.formats import voc
# from backend.data.formats import cvat
# from backend.data.formats import nuscenes
# from backend.data.formats import openlabel

__all__ = [
    # Base classes
    "FormatLoader",
    "FormatExporter",
    "FormatRegistry",
    "ProgressCallback",
    # Convenience functions
    "get_loader",
    "get_exporter",
    "detect_format",
    "list_formats",
    "convert",
    # Fusion formats
    "FusedAnnotation",
    "FusedAnnotationResult",
    # Format loaders
    "YoloLoader",
    "CocoLoader",
    "KittiLoader",
]
