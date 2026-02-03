"""
FastAPI routes for 3D object detection operations.

Provides REST endpoints for 3D object detection in point clouds using
PV-RCNN++ and CenterPoint models via OpenPCDet.

Implements spec: 05-point-cloud/04-3d-detection
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
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

# Module-level model cache for preloading
_loaded_models: dict[str, PVRCNNWrapper | CenterPointWrapper] = {}


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
    models = [
        ModelInfo(
            name="pvrcnn++",
            description="PV-RCNN++ 3D detector via OpenPCDet (highest accuracy)",
            loaded="pvrcnn++" in _loaded_models,
            vram_required_mb=PVRCNNWrapper.info.vram_required_mb,
            classes=list(DETECTION_3D_CLASSES),
            benchmark="KITTI 84% 3D AP",
        ),
        ModelInfo(
            name="centerpoint",
            description="CenterPoint 3D detector via OpenPCDet (faster, lower VRAM)",
            loaded="centerpoint" in _loaded_models,
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
        # Validate input path
        if not Path(request.input_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Point cloud file not found: {request.input_path}"
            )

        # Validate classes if provided
        if request.classes:
            invalid = set(request.classes) - set(DETECTION_3D_CLASSES)
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid classes: {invalid}. Valid: {DETECTION_3D_CLASSES}",
                )

        # Get or create detector
        model_name = request.model
        if model_name == "auto":
            detector = get_3d_detector(prefer_accuracy=True)
            model_name = detector.info.name
        elif model_name in _loaded_models:
            detector = _loaded_models[model_name]
        else:
            detector = get_3d_detector(force_model=model_name)

        # Ensure model is loaded
        if not detector.is_loaded:
            logger.info("Loading 3D detector: %s", model_name)
            detector.load()
            _loaded_models[model_name] = detector  # type: ignore[assignment]

        # Parse coordinate system
        coord_system = CoordinateSystem(request.coordinate_system)

        # Run detection
        result = detector.predict(
            request.input_path,
            confidence=request.confidence,
            classes=request.classes,
            coordinate_system=coord_system,
        )

        logger.info(
            "3D detection on %s: found %d objects in %.1fms",
            request.input_path,
            result.count,
            result.processing_time_ms,
        )

        return result

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

        # Check if already loaded
        if model_name in _loaded_models and _loaded_models[model_name].is_loaded:
            return LoadResponse(
                success=True,
                model=model_name,
                message=f"Model '{model_name}' is already loaded",
            )

        # Create and load the model
        if model_name == "pvrcnn++":
            detector = PVRCNNWrapper()
        else:
            detector = CenterPointWrapper()

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


@router.get("/classes")
async def get_supported_classes() -> dict[str, list[str]]:
    """
    Get the list of supported object classes for 3D detection.

    Returns the KITTI/nuScenes classes supported by both PV-RCNN++ and CenterPoint.
    """
    return {"classes": list(DETECTION_3D_CLASSES)}
