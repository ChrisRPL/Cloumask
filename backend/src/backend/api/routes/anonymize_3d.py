"""
FastAPI routes for 3D point cloud anonymization.

Provides REST endpoints for anonymizing faces in point clouds using
multi-view projection and face detection.

Implements spec: 05-point-cloud/07-anonymization-3d (API section)
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.cv.anonymization_3d import Anonymization3DResult, PointCloudAnonymizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/anonymize3d", tags=["3D Anonymization"])

# Allowed point cloud extensions for security
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pcd", ".ply", ".las", ".laz", ".bin"})

# Module-level anonymizer instance (lazy-loaded, thread-safe)
_anonymizer: PointCloudAnonymizer | None = None
_anonymizer_lock = threading.Lock()


# -----------------------------------------------------------------------------
# Request / Response Models
# -----------------------------------------------------------------------------


class Anonymize3DRequest(BaseModel):
    """Request to anonymize a 3D point cloud."""

    input_path: str = Field(..., description="Path to input point cloud file")
    output_path: str = Field(..., description="Path for anonymized output file")
    mode: Literal["remove", "noise"] = Field(
        "remove", description="Anonymization mode: remove points or add noise"
    )
    num_views: int = Field(
        8, ge=1, le=32, description="Number of virtual camera viewpoints"
    )
    face_confidence: float = Field(
        0.4, ge=0.0, le=1.0, description="Face detection confidence threshold"
    )
    face_margin: float = Field(
        1.2, ge=1.0, le=3.0, description="Factor to expand face bounding boxes"
    )
    noise_sigma: float = Field(
        0.1, gt=0.0, description="Noise std deviation in metres (noise mode only)"
    )
    verify: bool = Field(
        True, description="Re-detect after anonymization to verify removal"
    )
    resolution_width: int = Field(
        640, ge=64, le=1920, description="Virtual camera image width"
    )
    resolution_height: int = Field(
        480, ge=64, le=1080, description="Virtual camera image height"
    )


class Anonymize3DResponse(BaseModel):
    """Response with anonymization results."""

    output_path: str
    original_point_count: int
    anonymized_point_count: int
    face_regions_found: int
    points_removed: int
    points_noised: int
    verification_passed: bool
    processing_time_ms: float
    views_processed: int
    mode: str


class Verify3DRequest(BaseModel):
    """Request to verify a point cloud contains no detectable faces."""

    input_path: str = Field(..., description="Path to point cloud to verify")
    num_views: int = Field(
        8, ge=1, le=32, description="Number of virtual camera viewpoints"
    )
    face_confidence: float = Field(
        0.4, ge=0.0, le=1.0, description="Face detection confidence threshold"
    )


class Verify3DResponse(BaseModel):
    """Verification result."""

    passed: bool = Field(..., description="True if no faces detected")
    input_path: str
    views_checked: int


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _validate_pc_path(path_str: str, must_exist: bool = True) -> Path:
    """Validate a point cloud file path."""
    path = Path(path_str).resolve()
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid extension '{path.suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )
    if must_exist and not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path_str}")
    return path


def _get_anonymizer() -> PointCloudAnonymizer:
    """Get or create the module-level PointCloudAnonymizer."""
    global _anonymizer
    with _anonymizer_lock:
        if _anonymizer is None or not _anonymizer.is_loaded:
            _anonymizer = PointCloudAnonymizer()
            _anonymizer.load("auto")
        return _anonymizer


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post("/process", response_model=Anonymize3DResponse)
async def anonymize_3d(request: Anonymize3DRequest) -> Anonymize3DResponse:
    """
    Anonymize faces in a 3D point cloud.

    Projects the cloud to multiple virtual camera views, detects faces
    in each rendered depth image, then removes or noises the
    corresponding 3D points.
    """
    try:
        input_path = _validate_pc_path(request.input_path, must_exist=True)
        output_path = _validate_pc_path(request.output_path, must_exist=False)

        anonymizer = _get_anonymizer()

        result: Anonymization3DResult = anonymizer.anonymize(
            pcd_path=str(input_path),
            output_path=str(output_path),
            mode=request.mode,
            num_views=request.num_views,
            face_confidence=request.face_confidence,
            face_margin=request.face_margin,
            noise_sigma=request.noise_sigma,
            verify=request.verify,
            resolution=(request.resolution_width, request.resolution_height),
        )

        return Anonymize3DResponse(
            output_path=result.output_path,
            original_point_count=result.original_point_count,
            anonymized_point_count=result.anonymized_point_count,
            face_regions_found=result.face_regions_found,
            points_removed=result.points_removed,
            points_noised=result.points_noised,
            verification_passed=result.verification_passed,
            processing_time_ms=result.processing_time_ms,
            views_processed=result.views_processed,
            mode=result.mode,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        logger.exception("3D anonymization runtime error")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error during 3D anonymization")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/verify", response_model=Verify3DResponse)
async def verify_3d(request: Verify3DRequest) -> Verify3DResponse:
    """
    Verify that a point cloud has no detectable faces.

    Useful for post-anonymization quality assurance.
    """
    try:
        input_path = _validate_pc_path(request.input_path, must_exist=True)

        import numpy as np
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(input_path))
        if pcd.is_empty():
            return Verify3DResponse(
                passed=True,
                input_path=str(input_path),
                views_checked=0,
            )

        anonymizer = _get_anonymizer()
        passed = anonymizer._verify(
            pcd,
            request.num_views,
            (640, 480),
            request.face_confidence,
        )

        return Verify3DResponse(
            passed=passed,
            input_path=str(input_path),
            views_checked=request.num_views,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Verification error")
        raise HTTPException(status_code=500, detail=str(e)) from e
