"""Point cloud processing tool for geometry operations.

Provides downsampling, filtering, and normal estimation operations on point clouds
using Open3D. Supports PCD, PLY, LAS/LAZ, and BIN formats.

Implements spec: 05-point-cloud/02-python-open3d, 05-point-cloud/08-agent-tools
Integration point: backend/cv/pointcloud.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.registry import register_tool

# Use processor's supported formats to avoid mismatch with constants
SUPPORTED_INPUT_FORMATS: frozenset[str] = frozenset({".pcd", ".ply", ".las", ".laz", ".bin"})
SUPPORTED_OUTPUT_FORMATS: frozenset[str] = frozenset({".pcd", ".ply"})

logger = logging.getLogger(__name__)


# Supported processing operations
PROCESSING_OPERATIONS: list[str] = [
    "voxel",
    "random",
    "statistical",
    "radius",
    "normals",
]


@register_tool
class ProcessPointCloudTool(BaseTool):
    """Process point clouds with downsampling, filtering, or normal estimation."""

    name = "process_pointcloud"
    description = """Process point cloud with geometry operations.

Operations:
- voxel: Voxel grid downsampling (requires voxel_size, e.g., 0.05 for 5cm)
- random: Random subsampling to target_count points
- statistical: Remove statistical outliers (nb_neighbors, std_ratio)
- radius: Remove isolated points (radius, min_neighbors)
- normals: Estimate surface normals (search_radius, max_nn)

Supported Formats:
- PCD, PLY, LAS/LAZ, KITTI BIN

Examples:
- process_pointcloud(input, output, "voxel", voxel_size=0.1)
- process_pointcloud(input, output, "random", target_count=100000)
- process_pointcloud(input, output, "statistical", nb_neighbors=20, std_ratio=2.0)
- process_pointcloud(input, output, "radius", radius=0.1, min_neighbors=5)
- process_pointcloud(input, output, "normals", search_radius=0.1)"""
    category = ToolCategory.UTILITY

    parameters = [
        ToolParameter(
            name="input_path",
            type=str,
            description="Path to input point cloud file",
            required=True,
        ),
        ToolParameter(
            name="output_path",
            type=str,
            description="Path for output point cloud file (.pcd or .ply)",
            required=True,
        ),
        ToolParameter(
            name="operation",
            type=str,
            description="Operation: voxel, random, statistical, radius, or normals",
            required=True,
            enum_values=PROCESSING_OPERATIONS,
        ),
        ToolParameter(
            name="voxel_size",
            type=float,
            description="Voxel size in meters for 'voxel' operation",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="target_count",
            type=int,
            description="Target point count for 'random' operation",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="nb_neighbors",
            type=int,
            description="Number of neighbors for 'statistical' operation",
            required=False,
            default=20,
        ),
        ToolParameter(
            name="std_ratio",
            type=float,
            description="Standard deviation ratio for 'statistical' operation",
            required=False,
            default=2.0,
        ),
        ToolParameter(
            name="radius",
            type=float,
            description="Search radius in meters for 'radius' operation",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="min_neighbors",
            type=int,
            description="Minimum neighbors for 'radius' operation",
            required=False,
            default=2,
        ),
        ToolParameter(
            name="search_radius",
            type=float,
            description="Search radius for 'normals' operation",
            required=False,
            default=0.1,
        ),
        ToolParameter(
            name="max_nn",
            type=int,
            description="Maximum neighbors for 'normals' operation",
            required=False,
            default=30,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        output_path: str,
        operation: Literal["voxel", "random", "statistical", "radius", "normals"],
        voxel_size: float | None = None,
        target_count: int | None = None,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
        radius: float | None = None,
        min_neighbors: int = 2,
        search_radius: float = 0.1,
        max_nn: int = 30,
    ) -> ToolResult:
        """
        Execute point cloud processing operation.

        Args:
            input_path: Path to input point cloud.
            output_path: Path for output point cloud.
            operation: Processing operation to apply.
            voxel_size: Voxel size for 'voxel' operation.
            target_count: Target count for 'random' operation.
            nb_neighbors: Neighbor count for 'statistical'.
            std_ratio: Std ratio for 'statistical'.
            radius: Search radius for 'radius' operation.
            min_neighbors: Min neighbors for 'radius'.
            search_radius: Search radius for 'normals'.
            max_nn: Max neighbors for 'normals'.

        Returns:
            ToolResult with processing statistics.
        """
        from backend.cv.pointcloud import PointCloudProcessor

        input_p = Path(input_path)
        output_p = Path(output_path)

        # Validate input file
        if not input_p.exists():
            return error_result(f"Input file not found: {input_path}")

        if not input_p.is_file():
            return error_result("Input must be a file, not a directory")

        if input_p.suffix.lower() not in SUPPORTED_INPUT_FORMATS:
            return error_result(
                f"Unsupported input format: {input_p.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_INPUT_FORMATS))}"
            )

        # Validate output format
        if output_p.suffix.lower() not in SUPPORTED_OUTPUT_FORMATS:
            return error_result(
                f"Unsupported output format: {output_p.suffix}. "
                f"Supported for writing: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}"
            )

        # Validate operation
        if operation not in PROCESSING_OPERATIONS:
            return error_result(
                f"Unknown operation: {operation}. "
                f"Supported: {', '.join(PROCESSING_OPERATIONS)}"
            )

        # Validate operation-specific parameters
        if operation == "voxel" and voxel_size is None:
            return error_result("voxel_size is required for 'voxel' operation")
        if operation == "voxel" and voxel_size is not None and voxel_size <= 0:
            return error_result("voxel_size must be positive")

        if operation == "random" and target_count is None:
            return error_result("target_count is required for 'random' operation")
        if operation == "random" and target_count is not None and target_count <= 0:
            return error_result("target_count must be positive")

        if operation == "radius" and radius is None:
            return error_result("radius is required for 'radius' operation")
        if operation == "radius" and radius is not None and radius <= 0:
            return error_result("radius must be positive")

        try:
            self.report_progress(1, 2, f"Processing point cloud with {operation}...")

            processor = PointCloudProcessor()

            # Build operation parameters
            params: dict[str, float | int] = {}
            if operation == "voxel":
                params["voxel_size"] = voxel_size  # type: ignore[assignment]
            elif operation == "random":
                params["target_count"] = target_count  # type: ignore[assignment]
            elif operation == "statistical":
                params["nb_neighbors"] = nb_neighbors
                params["std_ratio"] = std_ratio
            elif operation == "radius":
                params["radius"] = radius  # type: ignore[assignment]
                params["min_neighbors"] = min_neighbors
            elif operation == "normals":
                params["search_radius"] = search_radius
                params["max_nn"] = max_nn

            # Execute processing
            result = processor.process_file(input_path, output_path, operation, **params)

            self.report_progress(2, 2, "Processing complete")

            return success_result(
                {
                    "input_path": input_path,
                    "output_path": result.output_path,
                    "operation": result.operation,
                    "original_count": result.original_count,
                    "result_count": result.result_count,
                    "reduction_ratio": round(result.reduction_ratio, 4),
                    "processing_time_ms": round(result.processing_time_ms, 2),
                    "parameters": result.parameters,
                }
            )

        except ImportError as e:
            logger.exception("Open3D not installed")
            return error_result(
                f"Open3D not installed: {e}. "
                "Install with: pip install open3d"
            )
        except FileNotFoundError as e:
            return error_result(str(e))
        except ValueError as e:
            return error_result(f"Invalid parameter: {e}")
        except Exception as e:
            logger.exception("Point cloud processing failed")
            return error_result(f"Processing failed: {e}")


@register_tool
class PointCloudStatsTool(BaseTool):
    """Get statistics and metadata about a point cloud file."""

    name = "pointcloud_stats"
    description = """Get metadata and statistics for a point cloud file.

Returns:
- Point count
- Spatial bounds (min/max coordinates)
- Available attributes (colors, normals, intensity)
- File format

Supported Formats:
- PCD, PLY, LAS/LAZ, KITTI BIN

Examples:
- pointcloud_stats("/data/scan.pcd")
- pointcloud_stats("/data/lidar/frame001.las")"""
    category = ToolCategory.UTILITY

    parameters = [
        ToolParameter(
            name="path",
            type=str,
            description="Path to point cloud file",
            required=True,
        ),
    ]

    async def execute(self, path: str) -> ToolResult:
        """
        Get statistics for a point cloud file.

        Args:
            path: Path to point cloud file.

        Returns:
            ToolResult with point cloud statistics.
        """
        from backend.cv.pointcloud import PointCloudProcessor

        file_path = Path(path)

        # Validate input
        if not file_path.exists():
            return error_result(f"File not found: {path}")

        if not file_path.is_file():
            return error_result("Path must be a file, not a directory")

        if file_path.suffix.lower() not in SUPPORTED_INPUT_FORMATS:
            return error_result(
                f"Unsupported format: {file_path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_INPUT_FORMATS))}"
            )

        try:
            processor = PointCloudProcessor()
            stats = processor.get_stats(path)

            return success_result(
                {
                    "path": stats.file_path,
                    "format": stats.file_format,
                    "point_count": stats.point_count,
                    "bounds": {
                        "min": {
                            "x": round(stats.bounds_min[0], 3),
                            "y": round(stats.bounds_min[1], 3),
                            "z": round(stats.bounds_min[2], 3),
                        },
                        "max": {
                            "x": round(stats.bounds_max[0], 3),
                            "y": round(stats.bounds_max[1], 3),
                            "z": round(stats.bounds_max[2], 3),
                        },
                    },
                    "extent": {
                        "x": round(stats.extent[0], 3),
                        "y": round(stats.extent[1], 3),
                        "z": round(stats.extent[2], 3),
                    },
                    "center": {
                        "x": round(stats.center[0], 3),
                        "y": round(stats.center[1], 3),
                        "z": round(stats.center[2], 3),
                    },
                    "has_colors": stats.has_colors,
                    "has_normals": stats.has_normals,
                    "has_intensity": stats.has_intensity,
                }
            )

        except ImportError as e:
            logger.exception("Open3D not installed")
            return error_result(f"Open3D not installed: {e}")
        except FileNotFoundError as e:
            return error_result(str(e))
        except ValueError as e:
            return error_result(f"Invalid file: {e}")
        except Exception as e:
            logger.exception("Failed to get point cloud stats")
            return error_result(f"Failed to get stats: {e}")
