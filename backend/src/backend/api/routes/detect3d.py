"""
FastAPI routes for 3D object detection operations.

Provides REST endpoints for 3D object detection in point clouds using
PV-RCNN++ and CenterPoint models via OpenPCDet.

Implements spec: 05-point-cloud/04-3d-detection
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.cv.detection_3d import (
    DETECTION_3D_CLASSES,
    CenterPointWrapper,
    CoordinateSystem,
    PVRCNNWrapper,
    get_3d_detector,
)
from backend.cv.types import Detection3DResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detect3d", tags=["3D Detection"])

# Allowed point cloud file extensions for security validation
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pcd", ".ply", ".bin", ".las", ".laz"})

# Module-level model cache for preloading (thread-safe access)
_loaded_models: dict[str, PVRCNNWrapper | CenterPointWrapper] = {}
_models_lock = threading.Lock()


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class Detect3DRequest(BaseModel):
    """Request model for 3D object detection."""

    input_path: str = Field(..., description="Path to point cloud file (.pcd, .ply, .bin, .las)")
    model: Literal["auto", "pvrcnn++", "centerpoint"] = Field(
        "auto", description="Model to use: 'auto', 'pvrcnn++', or 'centerpoint'"
    )
    confidence: float = Field(
        0.3, ge=0.0, le=1.0, description="Minimum confidence threshold [0-1]"
    )
    classes: list[str] | None = Field(
        None, description="Classes to detect (None = all: Car, Pedestrian, Cyclist)"
    )
    coordinate_system: Literal["kitti", "nuscenes", "waymo"] = Field(
        "kitti", description="Input point cloud coordinate system"
    )


class ModelLoadRequest(BaseModel):
    """Request model for preloading a 3D detection model."""

    model: Literal["pvrcnn++", "centerpoint"] = Field(
        ..., description="Model to load: 'pvrcnn++' or 'centerpoint'"
    )
    device: Literal["auto", "cuda", "cpu"] = Field(
        "auto", description="Device to load model on"
    )


class ModelUnloadRequest(BaseModel):
    """Request model for unloading a 3D detection model."""

    model: Literal["pvrcnn++", "centerpoint"] = Field(
        ..., description="Model to unload: 'pvrcnn++' or 'centerpoint'"
    )


class ModelInfo(BaseModel):
    """Information about a 3D detection model."""

    name: str = Field(..., description="Model identifier")
    description: str = Field(..., description="Model description")
    loaded: bool = Field(..., description="Whether the model is currently loaded")
    vram_required_mb: int = Field(..., description="VRAM required in megabytes")
    classes: list[str] = Field(..., description="Supported object classes")
    benchmark: str = Field(..., description="Benchmark performance")


class ModelsListResponse(BaseModel):
    """Response model for listing available 3D detection models."""

    models: list[ModelInfo] = Field(..., description="Available 3D detection models")


class LoadResponse(BaseModel):
    """Response model for model load/unload operations."""

    success: bool = Field(..., description="Whether the operation succeeded")
    model: str = Field(..., description="Model name")
    message: str = Field(..., description="Status message")


class ClassesResponse(BaseModel):
    """Response model for supported object classes."""

    classes: list[str] = Field(..., description="Supported object classes")


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _validate_input_path(input_path: str) -> Path:
    """
    Validate input path for security and existence.

    Args:
        input_path: Path to point cloud file.

    Returns:
        Resolved Path object.

    Raises:
        HTTPException: If path is invalid, has wrong extension, or doesn't exist.
    """
    path = Path(input_path).resolve()

    # Validate file extension to prevent arbitrary file access
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{path.suffix}'. "
            f"Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Point cloud file not found: {input_path}"
        )

    if not path.is_file():
        raise HTTPException(
            status_code=400, detail=f"Path is not a file: {input_path}"
        )

    return path


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """
    List available 3D object detection models.

    Returns information about PV-RCNN++ and CenterPoint models including
    VRAM requirements, supported classes, and current load status.
    """
    with _models_lock:
        pvrcnn_loaded = "pvrcnn++" in _loaded_models
        centerpoint_loaded = "centerpoint" in _loaded_models

    models = [
        ModelInfo(
            name="pvrcnn++",
            description="PV-RCNN++ 3D detector via OpenPCDet (highest accuracy)",
            loaded=pvrcnn_loaded,
            vram_required_mb=PVRCNNWrapper.info.vram_required_mb,
            classes=list(DETECTION_3D_CLASSES),
            benchmark="KITTI 84% 3D AP",
        ),
        ModelInfo(
            name="centerpoint",
            description="CenterPoint 3D detector via OpenPCDet (faster, lower VRAM)",
            loaded=centerpoint_loaded,
            vram_required_mb=CenterPointWrapper.info.vram_required_mb,
            classes=list(DETECTION_3D_CLASSES),
            benchmark="KITTI 79% 3D AP",
        ),
    ]
    return ModelsListResponse(models=models)


@router.post("/infer", response_model=Detection3DResult)
async def detect_3d(request: Detect3DRequest) -> Detection3DResult:
    """
    Run 3D object detection on a point cloud.

    Detects vehicles, pedestrians, and cyclists in LiDAR point clouds
    using PV-RCNN++ or CenterPoint models.

    Supported input formats: .pcd, .ply, .bin (KITTI), .las, .laz
    """
    try:
        # Validate input path (security check + existence)
        validated_path = _validate_input_path(request.input_path)

        # Validate classes if provided
        if request.classes:
            invalid = set(request.classes) - set(DETECTION_3D_CLASSES)
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid classes: {invalid}. Valid: {DETECTION_3D_CLASSES}",
                )

        # Get or create detector (thread-safe cache access)
        requested_model = request.model
        with _models_lock:
            if requested_model == "auto":
                detector = get_3d_detector(prefer_accuracy=True)
                actual_model_name = detector.info.name
            elif requested_model in _loaded_models:
                detector = _loaded_models[requested_model]
                actual_model_name = requested_model
            else:
                detector = get_3d_detector(force_model=requested_model)
                actual_model_name = requested_model

            # Ensure model is loaded (inside lock for cache consistency)
            if not detector.is_loaded:
                logger.info("Loading 3D detector: %s", actual_model_name)
                detector.load()
                _loaded_models[actual_model_name] = detector  # type: ignore[assignment]

        # Parse coordinate system
        coord_system = CoordinateSystem(request.coordinate_system)

        # Run detection (outside lock - inference can be concurrent)
        result = detector.predict(
            str(validated_path),
            confidence=request.confidence,
            classes=request.classes,
            coordinate_system=coord_system,
        )

        logger.info(
            "3D detection on %s: found %d objects in %.1fms",
            validated_path,
            result.count,
            result.processing_time_ms,
        )

        return result

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        # OpenPCDet not installed or model not found
        logger.exception("3D detection runtime error")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("3D detection failed: %s", request.input_path)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/load", response_model=LoadResponse)
async def load_model(request: ModelLoadRequest) -> LoadResponse:
    """
    Preload a 3D detection model into memory.

    Loading models ahead of time reduces latency for the first inference.
    Models remain loaded until explicitly unloaded or the server restarts.
    """
    try:
        model_name = request.model

        with _models_lock:
            # Check if already loaded
            if model_name in _loaded_models and _loaded_models[model_name].is_loaded:
                return LoadResponse(
                    success=True,
                    model=model_name,
                    message=f"Model '{model_name}' is already loaded",
                )

            # Create and load the model
            detector = PVRCNNWrapper() if model_name == "pvrcnn++" else CenterPointWrapper()

            device = request.device if request.device != "auto" else "auto"
            detector.load(device=device)

            _loaded_models[model_name] = detector

        logger.info("Loaded 3D detector: %s on %s", model_name, detector.device)

        return LoadResponse(
            success=True,
            model=model_name,
            message=f"Model '{model_name}' loaded successfully on {detector.device}",
        )

    except RuntimeError as e:
        logger.exception("Failed to load 3D detector: %s", request.model)
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to load model: %s", request.model)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/unload", response_model=LoadResponse)
async def unload_model(request: ModelUnloadRequest) -> LoadResponse:
    """
    Unload a 3D detection model from memory.

    Frees GPU/CPU memory used by the model. The model will need to be
    reloaded before the next inference.
    """
    model_name = request.model

    with _models_lock:
        if model_name not in _loaded_models:
            return LoadResponse(
                success=True,
                model=model_name,
                message=f"Model '{model_name}' is not loaded",
            )

        try:
            detector = _loaded_models[model_name]
            detector.unload()
            del _loaded_models[model_name]

            logger.info("Unloaded 3D detector: %s", model_name)

            return LoadResponse(
                success=True,
                model=model_name,
                message=f"Model '{model_name}' unloaded successfully",
            )

        except Exception as e:
            logger.exception("Failed to unload model: %s", model_name)
            raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/classes", response_model=ClassesResponse)
async def get_supported_classes() -> ClassesResponse:
    """
    Get the list of supported object classes for 3D detection.

    Returns the KITTI/nuScenes classes supported by both PV-RCNN++ and CenterPoint.
    """
    return ClassesResponse(classes=list(DETECTION_3D_CLASSES))
