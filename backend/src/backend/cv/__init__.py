"""
Computer vision model wrappers and infrastructure.

This package provides the foundation for CV model integration including:
- Core types for detection and segmentation results
- Device management and VRAM monitoring
- Model lifecycle management (loading, unloading, caching)
- Model download infrastructure

Implements spec: 03-cv-models/00-infrastructure

Example:
    from backend.cv import ModelManager, DetectionResult, get_device_info

    # Get device info
    info = get_device_info()
    print(f"GPU: {info.cuda_device_name}, VRAM: {info.vram_available_mb}MB")

    # Get a model (lazy loaded)
    manager = ModelManager()
    yolo = manager.get("yolo11m")

    # Run detection
    result: DetectionResult = yolo.predict("image.jpg")
    print(f"Found {result.count} objects")

    # Cleanup
    manager.unload("yolo11m")
"""

from backend.cv.base import (
    BaseModelWrapper,
    ModelInfo,
    ModelState,
    ProgressCallback,
    register_model,
)
from backend.cv.detection import (
    COCO_CLASSES,
    RTDETRWrapper,
    YOLO11Wrapper,
    get_class_indices,
    get_detector,
)
from backend.cv.faces import (
    SCRFDWrapper,
    YuNetWrapper,
    get_face_detector,
)
from backend.cv.device import (
    CUDAOOMHandler,
    DeviceInfo,
    VRAMInfo,
    clear_gpu_memory,
    get_available_vram_mb,
    get_device_info,
    get_gpu_memory_summary,
    get_vram_info,
    get_vram_usage,
    select_device,
)
from backend.cv.download import (
    MODEL_REGISTRY,
    ModelRegistryEntry,
    ModelSource,
    delete_model,
    download_model,
    get_model_path,
    get_model_size_mb,
    get_models_dir,
    get_total_downloaded_size_mb,
    is_model_downloaded,
    list_available_models,
    list_downloaded_models,
)
from backend.cv.download import (
    register_model as register_model_download,
)
from backend.cv.manager import ModelManager, get_model_manager
from backend.cv.openvocab import (
    GroundingDINOWrapper,
    YOLOWorldWrapper,
    get_openvocab_detector,
)
from backend.cv.plates import (
    PLATE_REGIONS,
    PlateDetectorWrapper,
    get_plate_detector,
)
from backend.cv.segmentation import (
    MobileSAMWrapper,
    SAM2Wrapper,
    SAM3Wrapper,
    get_segmenter,
)
from backend.cv.types import (
    BBox,
    Detection,
    Detection3D,
    Detection3DResult,
    DetectionResult,
    FaceDetection,
    FaceDetectionResult,
    Mask,
    PlateDetection,
    PlateDetectionResult,
    SegmentationResult,
)
from backend.cv.anonymization import (
    AnonymizationConfig,
    AnonymizationMode,
    AnonymizationPipeline,
    AnonymizationResult,
    anonymize,
)

__all__ = [
    # Types
    "BBox",
    "Detection",
    "DetectionResult",
    "Mask",
    "SegmentationResult",
    "FaceDetection",
    "FaceDetectionResult",
    "Detection3D",
    "Detection3DResult",
    "PlateDetection",
    "PlateDetectionResult",
    # Base
    "BaseModelWrapper",
    "ModelInfo",
    "ModelState",
    "ProgressCallback",
    "register_model",
    # Manager
    "ModelManager",
    "get_model_manager",
    # Device
    "VRAMInfo",
    "DeviceInfo",
    "CUDAOOMHandler",
    "get_vram_usage",
    "get_vram_info",
    "get_available_vram_mb",
    "get_device_info",
    "select_device",
    "clear_gpu_memory",
    "get_gpu_memory_summary",
    # Download
    "MODEL_REGISTRY",
    "ModelRegistryEntry",
    "ModelSource",
    "get_models_dir",
    "get_model_path",
    "is_model_downloaded",
    "download_model",
    "list_available_models",
    "list_downloaded_models",
    "get_model_size_mb",
    "get_total_downloaded_size_mb",
    "delete_model",
    "register_model_download",
    # Detection
    "COCO_CLASSES",
    "YOLO11Wrapper",
    "RTDETRWrapper",
    "get_detector",
    "get_class_indices",
    # Segmentation
    "SAM3Wrapper",
    "SAM2Wrapper",
    "MobileSAMWrapper",
    "get_segmenter",
    # Face Detection
    "SCRFDWrapper",
    "YuNetWrapper",
    "get_face_detector",
    # Open-Vocabulary Detection
    "YOLOWorldWrapper",
    "GroundingDINOWrapper",
    "get_openvocab_detector",
    # Plate Detection
    "PlateDetectorWrapper",
    "get_plate_detector",
    "PLATE_REGIONS",
    # Anonymization
    "AnonymizationConfig",
    "AnonymizationMode",
    "AnonymizationPipeline",
    "AnonymizationResult",
    "anonymize",
]
