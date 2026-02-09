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
from backend.data.formats.coco import CocoExporter, CocoLoader
from backend.data.formats.cvat import CvatLoader
from backend.data.formats.fused_annotation import (
    FusedAnnotation,
    FusedAnnotationResult,
)
from backend.data.formats.kitti import KITTI_CLASSES, KittiLoader
from backend.data.formats.nuscenes import NUSCENES_CAMERAS, NuscenesLoader
from backend.data.formats.openlabel import OpenlabelLoader
from backend.data.formats.voc import VOC_CLASSES, VocLoader
from backend.data.formats.yolo import YoloExporter, YoloLoader

# Remaining loaders/exporters:
# from backend.data.formats import nuscenes

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
    "YoloExporter",
    "YoloLoader",
    "CocoExporter",
    "CocoLoader",
    "CvatLoader",
    "KittiLoader",
    "NuscenesLoader",
    "OpenlabelLoader",
    "VocLoader",
    "VOC_CLASSES",
    "KITTI_CLASSES",
    "NUSCENES_CAMERAS",
]
