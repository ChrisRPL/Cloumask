"""
FastAPI routes for 2D-3D sensor fusion operations.

Provides REST endpoints for:
- Loading calibration from various formats
- Projecting 3D points/boxes to 2D
- Lifting 2D boxes to 3D using point cloud
- Fusing 2D and 3D detections

Implements spec: 05-point-cloud/05-2d-3d-fusion
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.cv.detection_3d import PointCloudLoader
from backend.cv.types import BBox, Detection, Detection3D
from backend.data.calibration import CameraCalibration
from backend.data.formats.fused_annotation import FusedAnnotation, FusedAnnotationResult
from backend.data.fusion import fuse_detections
from backend.data.projection import (
    lift_2d_to_3d,
    project_bbox3d_to_2d,
    project_detections_3d_to_2d,
    project_points_to_image,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fusion", tags=["2D-3D Fusion"])

# Allowed file extensions for security validation
ALLOWED_CALIB_EXTENSIONS: frozenset[str] = frozenset({".txt", ".json", ".yaml", ".yml"})
ALLOWED_PC_EXTENSIONS: frozenset[str] = frozenset({".pcd", ".ply", ".bin", ".las", ".laz"})

# Calibration cache (thread-safe, bounded)
_CACHE_MAX_SIZE = 100
_calibration_cache: dict[str, CameraCalibration] = {}
_cache_lock = threading.Lock()


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class LoadCalibrationRequest(BaseModel):
    """Request to load calibration file."""

    calibration_path: str = Field(..., description="Path to calibration file")
    format: Literal["kitti", "nuscenes", "ros", "json"] = Field(
        "kitti", description="Calibration file format"
    )
    camera_id: int = Field(2, ge=0, le=3, description="Camera ID for KITTI format")


class CalibrationResponse(BaseModel):
    """Response with loaded calibration info."""

    success: bool
    width: int
    height: int
    has_extrinsic: bool
    has_distortion: bool
    source_format: str


class ProjectPointsRequest(BaseModel):
    """Request to project 3D points to 2D."""

    points_path: str = Field(..., description="Path to point cloud file")
    calibration_path: str = Field(..., description="Path to calibration file")
    calibration_format: Literal["kitti", "nuscenes", "ros", "json"] = Field("kitti")
    camera_id: int = Field(2, ge=0, le=3, description="Camera ID for KITTI")
    max_points: int | None = Field(None, ge=1, description="Max points to project")


class ProjectPointsResponse(BaseModel):
    """Response with projected 2D points."""

    points_2d: list[tuple[float, float]]
    valid_count: int
    total_count: int
    processing_time_ms: float


class Detection3DInput(BaseModel):
    """3D detection input for API requests."""

    class_id: int = Field(..., ge=0)
    class_name: str = Field(..., min_length=1)
    center: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    rotation: float
    confidence: float = Field(..., ge=0.0, le=1.0)


class ProjectBoxesRequest(BaseModel):
    """Request to project 3D boxes to 2D."""

    boxes_3d: list[Detection3DInput] = Field(..., description="3D detections")
    calibration_path: str = Field(..., description="Path to calibration file")
    calibration_format: Literal["kitti", "nuscenes", "ros", "json"] = Field("kitti")
    camera_id: int = Field(2, ge=0, le=3)


class ProjectedBox(BaseModel):
    """Projected 2D box result."""

    class_name: str
    confidence: float
    box_2d: tuple[float, float, float, float] | None  # x_min, y_min, x_max, y_max
    visible: bool


class ProjectBoxesResponse(BaseModel):
    """Response with projected 2D boxes."""

    boxes: list[ProjectedBox]
    visible_count: int
    total_count: int
    processing_time_ms: float


class LiftBoxesRequest(BaseModel):
    """Request to lift 2D boxes to 3D."""

    boxes_2d: list[tuple[float, float, float, float]] = Field(
        ..., description="2D boxes [x_min, y_min, x_max, y_max] in pixels"
    )
    points_path: str = Field(..., description="Path to point cloud")
    calibration_path: str = Field(..., description="Path to calibration")
    calibration_format: Literal["kitti", "nuscenes", "ros", "json"] = Field("kitti")
    camera_id: int = Field(2, ge=0, le=3)
    class_names: list[str] | None = Field(None, description="Class names per box")
    class_priors: dict[str, dict] | None = Field(None, description="Size priors")


class LiftedBox(BaseModel):
    """Lifted 3D box result."""

    box_2d: tuple[float, float, float, float]
    detection_3d: Detection3DInput | None
    lifted: bool
    point_count: int = 0


class LiftBoxesResponse(BaseModel):
    """Response with lifted 3D boxes."""

    boxes: list[LiftedBox]
    lifted_count: int
    total_count: int
    processing_time_ms: float


class Detection2DInput(BaseModel):
    """2D detection input for fusion."""

    class_id: int = Field(..., ge=0)
    class_name: str
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized center x")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized center y")
    width: float = Field(..., ge=0.0, le=1.0, description="Normalized width")
    height: float = Field(..., ge=0.0, le=1.0, description="Normalized height")
    confidence: float = Field(..., ge=0.0, le=1.0)


class FuseRequest(BaseModel):
    """Request to fuse 2D and 3D detections."""

    detections_2d: list[Detection2DInput] = Field(..., description="2D detections")
    detections_3d: list[Detection3DInput] = Field(..., description="3D detections")
    calibration_path: str = Field(..., description="Path to calibration")
    calibration_format: Literal["kitti", "nuscenes", "ros", "json"] = Field("kitti")
    camera_id: int = Field(2, ge=0, le=3)
    iou_threshold: float = Field(0.3, ge=0.0, le=1.0)
    class_match_required: bool = Field(False)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _get_calibration(
    path: str,
    format: str,
    camera_id: int = 2,
) -> CameraCalibration:
    """Get calibration from cache or load from file."""
    cache_key = f"{path}:{format}:{camera_id}"

    with _cache_lock:
        if cache_key in _calibration_cache:
            return _calibration_cache[cache_key]

    # Validate path and extension for security
    calib_path = Path(path).resolve()
    if calib_path.suffix.lower() not in ALLOWED_CALIB_EXTENSIONS:
        raise ValueError(
            f"Invalid calibration file extension: {calib_path.suffix}. "
            f"Allowed: {', '.join(sorted(ALLOWED_CALIB_EXTENSIONS))}"
        )
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    # Load based on format
    if format == "kitti":
        calib = CameraCalibration.from_kitti(str(calib_path), camera_id)
    elif format == "nuscenes":
        import json

        with open(calib_path) as f:
            data = json.load(f)
        calib = CameraCalibration.from_nuscenes(data)
    elif format == "ros":
        import json

        from backend.data.ros_types import CameraInfoMessage

        with open(calib_path) as f:
            data = json.load(f)
        msg = CameraInfoMessage(**data)
        calib = CameraCalibration.from_ros(msg)
    elif format == "json":
        calib = CameraCalibration.from_json(str(calib_path))
    else:
        raise ValueError(f"Unknown calibration format: {format}")

    with _cache_lock:
        # Bounded cache: remove oldest entry if at capacity
        if len(_calibration_cache) >= _CACHE_MAX_SIZE:
            oldest_key = next(iter(_calibration_cache))
            del _calibration_cache[oldest_key]
        _calibration_cache[cache_key] = calib

    return calib


def _load_pointcloud(path: str) -> "NDArray":
    """Load point cloud using existing loader."""
    import numpy as np
    from numpy.typing import NDArray

    pc_path = Path(path).resolve()

    # Validate extension for security
    if pc_path.suffix.lower() not in ALLOWED_PC_EXTENSIONS:
        raise ValueError(
            f"Invalid point cloud extension: {pc_path.suffix}. "
            f"Allowed: {', '.join(sorted(ALLOWED_PC_EXTENSIONS))}"
        )
    if not pc_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    return PointCloudLoader.load(str(pc_path))


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post("/load_calibration", response_model=CalibrationResponse)
async def load_calibration(request: LoadCalibrationRequest) -> CalibrationResponse:
    """Load and validate a calibration file."""
    try:
        calib = _get_calibration(
            request.calibration_path,
            request.format,
            request.camera_id,
        )

        return CalibrationResponse(
            success=True,
            width=calib.width,
            height=calib.height,
            has_extrinsic=calib.has_extrinsic,
            has_distortion=calib.has_distortion,
            source_format=calib.source_format,
        )
    except (FileNotFoundError, ValueError) as e:
        status = 404 if isinstance(e, FileNotFoundError) else 400
        raise HTTPException(status_code=status, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to load calibration: %s", request.calibration_path)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/project_points", response_model=ProjectPointsResponse)
async def project_points(request: ProjectPointsRequest) -> ProjectPointsResponse:
    """Project 3D point cloud to 2D image coordinates."""
    import numpy as np

    try:
        start = time.perf_counter()

        # Load calibration and point cloud
        calib = _get_calibration(
            request.calibration_path,
            request.calibration_format,
            request.camera_id,
        )
        points = _load_pointcloud(request.points_path)

        # Optionally subsample
        if request.max_points and len(points) > request.max_points:
            indices = np.random.choice(len(points), request.max_points, replace=False)
            points = points[indices]

        # Project points
        points_2d, valid = project_points_to_image(points[:, :3], calib)

        # Filter to valid only
        valid_points = points_2d[valid]

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ProjectPointsResponse(
            points_2d=[(float(p[0]), float(p[1])) for p in valid_points],
            valid_count=int(valid.sum()),
            total_count=len(points),
            processing_time_ms=elapsed_ms,
        )
    except (FileNotFoundError, ValueError) as e:
        status = 404 if isinstance(e, FileNotFoundError) else 400
        raise HTTPException(status_code=status, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to project points")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/project_boxes", response_model=ProjectBoxesResponse)
async def project_boxes(request: ProjectBoxesRequest) -> ProjectBoxesResponse:
    """Project 3D bounding boxes to 2D image boxes."""
    try:
        start = time.perf_counter()

        calib = _get_calibration(
            request.calibration_path,
            request.calibration_format,
            request.camera_id,
        )

        # Convert input to Detection3D
        detections = [
            Detection3D(
                class_id=d.class_id,
                class_name=d.class_name,
                center=d.center,
                dimensions=d.dimensions,
                rotation=d.rotation,
                confidence=d.confidence,
            )
            for d in request.boxes_3d
        ]

        # Project
        results = project_detections_3d_to_2d(detections, calib)

        boxes = []
        visible_count = 0
        for det, box_2d in results:
            visible = box_2d is not None
            if visible:
                visible_count += 1
            boxes.append(
                ProjectedBox(
                    class_name=det.class_name,
                    confidence=det.confidence,
                    box_2d=box_2d,
                    visible=visible,
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ProjectBoxesResponse(
            boxes=boxes,
            visible_count=visible_count,
            total_count=len(detections),
            processing_time_ms=elapsed_ms,
        )
    except (FileNotFoundError, ValueError) as e:
        status = 404 if isinstance(e, FileNotFoundError) else 400
        raise HTTPException(status_code=status, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to project boxes")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/lift_boxes", response_model=LiftBoxesResponse)
async def lift_boxes(request: LiftBoxesRequest) -> LiftBoxesResponse:
    """Lift 2D bounding boxes to 3D using point cloud depth."""
    try:
        start = time.perf_counter()

        # Validate class_names length if provided
        if request.class_names and len(request.class_names) != len(request.boxes_2d):
            raise HTTPException(
                status_code=400,
                detail=f"class_names length ({len(request.class_names)}) "
                f"must match boxes_2d length ({len(request.boxes_2d)})",
            )

        calib = _get_calibration(
            request.calibration_path,
            request.calibration_format,
            request.camera_id,
        )
        points = _load_pointcloud(request.points_path)

        boxes = []
        lifted_count = 0

        for i, box_2d in enumerate(request.boxes_2d):
            class_name = request.class_names[i] if request.class_names else None

            det3d = lift_2d_to_3d(
                box_2d,
                points,
                calib,
                class_name=class_name,
                class_priors=request.class_priors,
            )

            lifted = det3d is not None
            if lifted:
                lifted_count += 1

            boxes.append(
                LiftedBox(
                    box_2d=box_2d,
                    detection_3d=Detection3DInput(
                        class_id=det3d.class_id,
                        class_name=det3d.class_name,
                        center=det3d.center,
                        dimensions=det3d.dimensions,
                        rotation=det3d.rotation,
                        confidence=det3d.confidence,
                    )
                    if det3d
                    else None,
                    lifted=lifted,
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return LiftBoxesResponse(
            boxes=boxes,
            lifted_count=lifted_count,
            total_count=len(request.boxes_2d),
            processing_time_ms=elapsed_ms,
        )
    except (FileNotFoundError, ValueError) as e:
        status = 404 if isinstance(e, FileNotFoundError) else 400
        raise HTTPException(status_code=status, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to lift boxes")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/fuse", response_model=FusedAnnotationResult)
async def fuse(request: FuseRequest) -> FusedAnnotationResult:
    """Fuse 2D and 3D detections into linked annotations."""
    try:
        start = time.perf_counter()

        calib = _get_calibration(
            request.calibration_path,
            request.calibration_format,
            request.camera_id,
        )

        # Convert inputs to internal types
        detections_2d = [
            Detection(
                class_id=d.class_id,
                class_name=d.class_name,
                bbox=BBox(x=d.x, y=d.y, width=d.width, height=d.height),
                confidence=d.confidence,
            )
            for d in request.detections_2d
        ]

        detections_3d = [
            Detection3D(
                class_id=d.class_id,
                class_name=d.class_name,
                center=d.center,
                dimensions=d.dimensions,
                rotation=d.rotation,
                confidence=d.confidence,
            )
            for d in request.detections_3d
        ]

        # Fuse
        annotations = fuse_detections(
            detections_2d,
            detections_3d,
            calib,
            iou_threshold=request.iou_threshold,
            class_match_required=request.class_match_required,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return FusedAnnotationResult(
            annotations=annotations,
            image_path="",
            pointcloud_path=None,
            calibration_path=request.calibration_path,
            processing_time_ms=elapsed_ms,
        )
    except (FileNotFoundError, ValueError) as e:
        status = 404 if isinstance(e, FileNotFoundError) else 400
        raise HTTPException(status_code=status, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to fuse detections")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/calibration_cache")
async def clear_calibration_cache() -> dict[str, str]:
    """Clear the calibration cache."""
    with _cache_lock:
        count = len(_calibration_cache)
        _calibration_cache.clear()
    return {"message": f"Cleared {count} cached calibrations"}
