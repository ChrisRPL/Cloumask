"""
FastAPI routes for point cloud processing operations.

Provides REST endpoints for point cloud processing including metadata
extraction, downsampling, filtering, and normal estimation.

Implements spec: 05-point-cloud/02-python-open3d
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.cv.pointcloud import PointCloudProcessor
from backend.cv.types import PointCloudProcessingResult, PointCloudStats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pointcloud", tags=["Point Cloud"])


# Request Models


class DownsampleRequest(BaseModel):
    """Request model for downsampling operations."""

    input_path: str = Field(..., description="Path to input point cloud file")
    output_path: str = Field(..., description="Path for output point cloud file")
    method: Literal["voxel", "random"] = Field(
        ..., description="Downsampling method: 'voxel' or 'random'"
    )
    voxel_size: float | None = Field(
        None, ge=0.001, description="Voxel size in meters (required for 'voxel' method)"
    )
    target_count: int | None = Field(
        None, ge=1, description="Target point count (required for 'random' method)"
    )


class FilterRequest(BaseModel):
    """Request model for outlier filtering operations."""

    input_path: str = Field(..., description="Path to input point cloud file")
    output_path: str = Field(..., description="Path for output point cloud file")
    method: Literal["statistical", "radius"] = Field(
        ..., description="Filtering method: 'statistical' or 'radius'"
    )
    nb_neighbors: int = Field(
        20, ge=1, description="Number of neighbors for statistical filtering"
    )
    std_ratio: float = Field(
        2.0, gt=0, description="Standard deviation ratio for statistical filtering"
    )
    radius: float | None = Field(
        None, gt=0, description="Search radius in meters (required for 'radius' method)"
    )
    min_neighbors: int = Field(
        2, ge=1, description="Minimum neighbors for radius filtering"
    )


class NormalsRequest(BaseModel):
    """Request model for normal estimation."""

    input_path: str = Field(..., description="Path to input point cloud file")
    output_path: str = Field(..., description="Path for output point cloud file")
    search_radius: float = Field(
        0.1, gt=0, description="Search radius for neighbor lookup in meters"
    )
    max_nn: int = Field(30, ge=1, description="Maximum number of neighbors to consider")


class ConvertRequest(BaseModel):
    """Request model for format conversion."""

    input_path: str = Field(..., description="Path to input point cloud file")
    output_path: str = Field(..., description="Path for output point cloud file")
    write_ascii: bool = Field(False, description="Write output in ASCII format")


# Endpoints


@router.get("/stats", response_model=PointCloudStats)
async def get_stats(
    path: str = Query(..., description="Path to point cloud file"),
) -> PointCloudStats:
    """
    Get metadata and statistics for a point cloud file.

    Returns point count, spatial bounds, and available attributes.
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        processor = PointCloudProcessor()
        stats = processor.get_stats(path)

        logger.info("Retrieved stats for %s: %d points", path, stats.point_count)
        return stats

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error getting stats for %s", path)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/downsample", response_model=PointCloudProcessingResult)
async def downsample(request: DownsampleRequest) -> PointCloudProcessingResult:
    """
    Downsample a point cloud using voxel grid or random sampling.

    Voxel method groups points into voxels and averages them.
    Random method randomly samples points to reach target count.
    """
    try:
        # Validate input path
        if not Path(request.input_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Input file not found: {request.input_path}"
            )

        # Validate method-specific parameters
        if request.method == "voxel" and request.voxel_size is None:
            raise HTTPException(
                status_code=400,
                detail="voxel_size is required for 'voxel' method",
            )
        if request.method == "random" and request.target_count is None:
            raise HTTPException(
                status_code=400,
                detail="target_count is required for 'random' method",
            )

        processor = PointCloudProcessor()

        if request.method == "voxel":
            result = processor.process_file(
                request.input_path,
                request.output_path,
                "voxel",
                voxel_size=request.voxel_size,
            )
        else:
            result = processor.process_file(
                request.input_path,
                request.output_path,
                "random",
                target_count=request.target_count,
            )

        logger.info(
            "Downsampled %s: %d -> %d points",
            request.input_path,
            result.original_count,
            result.result_count,
        )
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error downsampling %s", request.input_path)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/filter", response_model=PointCloudProcessingResult)
async def filter_outliers(request: FilterRequest) -> PointCloudProcessingResult:
    """
    Remove outliers from a point cloud.

    Statistical method removes points with abnormal neighbor distances.
    Radius method removes points with too few neighbors within radius.
    """
    try:
        # Validate input path
        if not Path(request.input_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Input file not found: {request.input_path}"
            )

        # Validate method-specific parameters
        if request.method == "radius" and request.radius is None:
            raise HTTPException(
                status_code=400,
                detail="radius is required for 'radius' method",
            )

        processor = PointCloudProcessor()

        if request.method == "statistical":
            result = processor.process_file(
                request.input_path,
                request.output_path,
                "statistical",
                nb_neighbors=request.nb_neighbors,
                std_ratio=request.std_ratio,
            )
        else:
            result = processor.process_file(
                request.input_path,
                request.output_path,
                "radius",
                radius=request.radius,
                min_neighbors=request.min_neighbors,
            )

        logger.info(
            "Filtered %s: %d -> %d points (removed %d)",
            request.input_path,
            result.original_count,
            result.result_count,
            result.original_count - result.result_count,
        )
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error filtering %s", request.input_path)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/normals", response_model=PointCloudProcessingResult)
async def estimate_normals(request: NormalsRequest) -> PointCloudProcessingResult:
    """
    Estimate surface normals for a point cloud.

    Uses hybrid KDTree search combining radius and k-nearest neighbors.
    """
    try:
        # Validate input path
        if not Path(request.input_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Input file not found: {request.input_path}"
            )

        processor = PointCloudProcessor()
        result = processor.process_file(
            request.input_path,
            request.output_path,
            "normals",
            search_radius=request.search_radius,
            max_nn=request.max_nn,
        )

        logger.info(
            "Estimated normals for %s: %d points",
            request.input_path,
            result.result_count,
        )
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error estimating normals for %s", request.input_path)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/convert", response_model=PointCloudProcessingResult)
async def convert_format(request: ConvertRequest) -> PointCloudProcessingResult:
    """
    Convert a point cloud to a different format.

    Supports conversion between PCD, PLY, LAS, LAZ, and BIN formats.
    Output format is determined by the output_path file extension.
    """
    import time

    try:
        # Validate input path
        if not Path(request.input_path).exists():
            raise HTTPException(
                status_code=404, detail=f"Input file not found: {request.input_path}"
            )

        processor = PointCloudProcessor()

        start_time = time.perf_counter()

        # Load and save to new format
        pcd = processor.load(request.input_path)
        original_count = len(pcd.points)

        processor.save(pcd, request.output_path, write_ascii=request.write_ascii)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        input_format = Path(request.input_path).suffix.lower().lstrip(".")
        output_format = Path(request.output_path).suffix.lower().lstrip(".")

        result = PointCloudProcessingResult(
            output_path=request.output_path,
            original_count=original_count,
            result_count=original_count,  # No points lost in conversion
            operation="convert",
            processing_time_ms=elapsed_ms,
            parameters={
                "input_format": input_format,
                "output_format": output_format,
                "write_ascii": request.write_ascii,
            },
        )

        logger.info(
            "Converted %s to %s: %d points",
            request.input_path,
            request.output_path,
            original_count,
        )
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error converting %s", request.input_path)
        raise HTTPException(status_code=500, detail=str(e)) from e
