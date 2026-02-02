"""
Open3D-based point cloud processing operations.

This module provides geometry operations for point cloud data including
loading, saving, downsampling, filtering, and normal estimation.

Uses lazy imports for Open3D to avoid loading the heavy library until needed.

Implements spec: 05-point-cloud/02-python-open3d
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.cv.types import PointCloudProcessingResult, PointCloudStats

if TYPE_CHECKING:
    import open3d as o3d

logger = logging.getLogger(__name__)


class PointCloudProcessor:
    """
    Open3D-based point cloud processing operations.

    Provides geometry operations including loading, saving, downsampling,
    filtering, and normal estimation. Uses lazy imports for Open3D to
    avoid loading the library until actually needed.

    Supported formats:
        - PCD: Point Cloud Data format
        - PLY: Polygon File Format
        - LAS/LAZ: LiDAR data formats (via laspy)
        - BIN: KITTI binary format

    Example:
        processor = PointCloudProcessor()
        stats = processor.get_stats("scan.pcd")
        print(f"Points: {stats.point_count}")

        pcd = processor.load("scan.pcd")
        downsampled = processor.downsample_voxel(pcd, voxel_size=0.05)
        processor.save(downsampled, "scan_downsampled.pcd")
    """

    SUPPORTED_FORMATS: set[str] = {".pcd", ".ply", ".las", ".laz", ".bin"}

    def load(self, path: str | Path) -> o3d.geometry.PointCloud:
        """
        Load point cloud from file.

        Supports PCD, PLY, LAS/LAZ, and KITTI BIN formats. Format is
        auto-detected from file extension.

        Args:
            path: Path to the point cloud file.

        Returns:
            Open3D PointCloud object.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If format is not supported or file is invalid.
        """
        import open3d as o3d

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {ext}. Supported: {self.SUPPORTED_FORMATS}"
            )

        logger.debug("Loading point cloud from %s", path)

        if ext in {".las", ".laz"}:
            return self._load_las(path)
        elif ext == ".bin":
            return self._load_bin(path)
        else:
            # PCD and PLY handled natively by Open3D
            pcd = o3d.io.read_point_cloud(str(path))
            if pcd.is_empty():
                raise ValueError(f"Empty or invalid point cloud file: {path}")
            logger.info("Loaded %d points from %s", len(pcd.points), path)
            return pcd

    def save(
        self,
        pcd: o3d.geometry.PointCloud,
        path: str | Path,
        write_ascii: bool = False,
    ) -> None:
        """
        Save point cloud to file.

        Format is auto-detected from file extension. Supports PCD and PLY.

        Args:
            pcd: Open3D PointCloud object to save.
            path: Output file path.
            write_ascii: If True, write in ASCII format (larger but readable).

        Raises:
            ValueError: If output format is not supported for writing.
        """
        import open3d as o3d

        path = Path(path)
        ext = path.suffix.lower()

        if ext not in {".pcd", ".ply"}:
            raise ValueError(
                f"Cannot write to format: {ext}. Supported for writing: .pcd, .ply"
            )

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        success = o3d.io.write_point_cloud(
            str(path),
            pcd,
            write_ascii=write_ascii,
        )

        if not success:
            raise ValueError(f"Failed to write point cloud to {path}")

        logger.info("Saved %d points to %s", len(pcd.points), path)

    def get_stats(self, path: str | Path) -> PointCloudStats:
        """
        Get statistics and metadata about a point cloud file.

        Args:
            path: Path to the point cloud file.

        Returns:
            PointCloudStats with point count, bounds, and attributes.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is not supported.
        """
        path = Path(path)
        pcd = self.load(path)

        points = np.asarray(pcd.points)

        # Compute bounds
        if len(points) > 0:
            bounds_min = tuple(float(v) for v in points.min(axis=0))
            bounds_max = tuple(float(v) for v in points.max(axis=0))
        else:
            bounds_min = (0.0, 0.0, 0.0)
            bounds_max = (0.0, 0.0, 0.0)

        # Check for intensity in colors (we store intensity in red channel for some formats)
        has_intensity = pcd.has_colors()

        return PointCloudStats(
            point_count=len(points),
            bounds_min=bounds_min,  # type: ignore[arg-type]
            bounds_max=bounds_max,  # type: ignore[arg-type]
            has_colors=pcd.has_colors(),
            has_normals=pcd.has_normals(),
            has_intensity=has_intensity,
            file_path=str(path),
            file_format=path.suffix.lower().lstrip("."),
        )

    def downsample_voxel(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float,
    ) -> o3d.geometry.PointCloud:
        """
        Downsample point cloud using voxel grid filtering.

        Points within each voxel are averaged to produce a single point.
        This is deterministic and preserves spatial distribution.

        Args:
            pcd: Input point cloud.
            voxel_size: Size of voxel in meters (e.g., 0.05 for 5cm voxels).

        Returns:
            Downsampled point cloud.

        Raises:
            ValueError: If voxel_size is not positive.
        """
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")

        original_count = len(pcd.points)
        result = pcd.voxel_down_sample(voxel_size)
        new_count = len(result.points)

        logger.info(
            "Voxel downsampling: %d -> %d points (%.1f%% reduction, voxel=%.3fm)",
            original_count,
            new_count,
            (1 - new_count / max(original_count, 1)) * 100,
            voxel_size,
        )

        return result

    def downsample_random(
        self,
        pcd: o3d.geometry.PointCloud,
        target_count: int,
    ) -> o3d.geometry.PointCloud:
        """
        Downsample point cloud by random sampling.

        Randomly selects points to reach the target count. Non-deterministic
        but preserves original point positions.

        Args:
            pcd: Input point cloud.
            target_count: Desired number of points in output.

        Returns:
            Downsampled point cloud.

        Raises:
            ValueError: If target_count is not positive.
        """
        if target_count <= 0:
            raise ValueError(f"target_count must be positive, got {target_count}")

        original_count = len(pcd.points)

        # If already at or below target, return as-is
        if original_count <= target_count:
            logger.info(
                "Random downsampling: %d points already <= target %d, no change",
                original_count,
                target_count,
            )
            return pcd

        # Calculate ratio and downsample
        ratio = target_count / original_count
        result = pcd.random_down_sample(ratio)
        new_count = len(result.points)

        logger.info(
            "Random downsampling: %d -> %d points (target was %d)",
            original_count,
            new_count,
            target_count,
        )

        return result

    def remove_statistical_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
    ) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Remove statistical outliers from point cloud.

        Points are considered outliers if their average distance to neighbors
        is more than std_ratio standard deviations from the mean.

        Args:
            pcd: Input point cloud.
            nb_neighbors: Number of neighbors to consider for each point.
            std_ratio: Standard deviation multiplier for outlier threshold.

        Returns:
            Tuple of (filtered point cloud, boolean mask of inliers).

        Raises:
            ValueError: If parameters are invalid.
        """
        if nb_neighbors <= 0:
            raise ValueError(f"nb_neighbors must be positive, got {nb_neighbors}")
        if std_ratio <= 0:
            raise ValueError(f"std_ratio must be positive, got {std_ratio}")

        original_count = len(pcd.points)
        result, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        new_count = len(result.points)
        removed = original_count - new_count

        logger.info(
            "Statistical outlier removal: %d -> %d points (removed %d, %.1f%%)",
            original_count,
            new_count,
            removed,
            (removed / max(original_count, 1)) * 100,
        )

        # Convert indices to boolean mask
        mask = np.zeros(original_count, dtype=bool)
        mask[inlier_indices] = True

        return result, mask

    def remove_radius_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        radius: float,
        min_neighbors: int = 2,
    ) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Remove radius-based outliers from point cloud.

        Points with fewer than min_neighbors within the given radius
        are considered outliers and removed.

        Args:
            pcd: Input point cloud.
            radius: Search radius in meters.
            min_neighbors: Minimum number of neighbors required.

        Returns:
            Tuple of (filtered point cloud, boolean mask of inliers).

        Raises:
            ValueError: If parameters are invalid.
        """
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        if min_neighbors <= 0:
            raise ValueError(f"min_neighbors must be positive, got {min_neighbors}")

        original_count = len(pcd.points)
        result, inlier_indices = pcd.remove_radius_outlier(
            nb_points=min_neighbors,
            radius=radius,
        )
        new_count = len(result.points)
        removed = original_count - new_count

        logger.info(
            "Radius outlier removal: %d -> %d points (removed %d, radius=%.3fm)",
            original_count,
            new_count,
            removed,
            radius,
        )

        # Convert indices to boolean mask
        mask = np.zeros(original_count, dtype=bool)
        mask[inlier_indices] = True

        return result, mask

    def estimate_normals(
        self,
        pcd: o3d.geometry.PointCloud,
        search_radius: float = 0.1,
        max_nn: int = 30,
    ) -> o3d.geometry.PointCloud:
        """
        Estimate surface normals for each point.

        Uses hybrid KDTree search combining radius and k-nearest neighbors
        for robust normal estimation.

        Args:
            pcd: Input point cloud.
            search_radius: Search radius for neighbors in meters.
            max_nn: Maximum number of neighbors to consider.

        Returns:
            Point cloud with normals computed (modifies in-place and returns).

        Raises:
            ValueError: If parameters are invalid.
        """
        import open3d as o3d

        if search_radius <= 0:
            raise ValueError(f"search_radius must be positive, got {search_radius}")
        if max_nn <= 0:
            raise ValueError(f"max_nn must be positive, got {max_nn}")

        point_count = len(pcd.points)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius,
                max_nn=max_nn,
            )
        )

        # Orient normals consistently (towards camera/origin)
        pcd.orient_normals_consistent_tangent_plane(k=max_nn)

        logger.info(
            "Estimated normals for %d points (radius=%.3fm, max_nn=%d)",
            point_count,
            search_radius,
            max_nn,
        )

        return pcd

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        operation: str,
        **params: Any,
    ) -> PointCloudProcessingResult:
        """
        Process a point cloud file with a specified operation.

        Convenience method that loads, processes, saves, and returns results.

        Args:
            input_path: Path to input point cloud.
            output_path: Path for output point cloud.
            operation: Operation name (voxel, random, statistical, radius, normals).
            **params: Operation-specific parameters.

        Returns:
            PointCloudProcessingResult with operation details.

        Raises:
            ValueError: If operation is unknown or parameters are invalid.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        start_time = time.perf_counter()

        # Load input
        pcd = self.load(input_path)
        original_count = len(pcd.points)

        # Apply operation
        if operation == "voxel":
            voxel_size = params.get("voxel_size", 0.05)
            result_pcd = self.downsample_voxel(pcd, voxel_size)
            op_params = {"voxel_size": voxel_size}
        elif operation == "random":
            target_count = params.get("target_count", original_count // 2)
            result_pcd = self.downsample_random(pcd, target_count)
            op_params = {"target_count": target_count}
        elif operation == "statistical":
            nb_neighbors = params.get("nb_neighbors", 20)
            std_ratio = params.get("std_ratio", 2.0)
            result_pcd, _ = self.remove_statistical_outliers(pcd, nb_neighbors, std_ratio)
            op_params = {"nb_neighbors": nb_neighbors, "std_ratio": std_ratio}
        elif operation == "radius":
            radius = params.get("radius", 0.1)
            min_neighbors = params.get("min_neighbors", 2)
            result_pcd, _ = self.remove_radius_outliers(pcd, radius, min_neighbors)
            op_params = {"radius": radius, "min_neighbors": min_neighbors}
        elif operation == "normals":
            search_radius = params.get("search_radius", 0.1)
            max_nn = params.get("max_nn", 30)
            result_pcd = self.estimate_normals(pcd, search_radius, max_nn)
            op_params = {"search_radius": search_radius, "max_nn": max_nn}
        else:
            raise ValueError(
                f"Unknown operation: {operation}. "
                "Supported: voxel, random, statistical, radius, normals"
            )

        # Save output
        self.save(result_pcd, output_path)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return PointCloudProcessingResult(
            output_path=str(output_path),
            original_count=original_count,
            result_count=len(result_pcd.points),
            operation=operation,
            processing_time_ms=elapsed_ms,
            parameters=op_params,
        )

    def _load_las(self, path: Path) -> o3d.geometry.PointCloud:
        """
        Load LAS/LAZ file using laspy.

        Args:
            path: Path to LAS/LAZ file.

        Returns:
            Open3D PointCloud with positions and optional colors.
        """
        import laspy
        import open3d as o3d

        las = laspy.read(str(path))

        # Extract coordinates
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Add colors if available (RGB values in LAS are 16-bit)
        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            colors = (
                np.vstack([las.red, las.green, las.blue]).T.astype(np.float64) / 65535.0
            )
            pcd.colors = o3d.utility.Vector3dVector(colors)

        logger.info("Loaded %d points from LAS file: %s", len(points), path)

        return pcd

    def _load_bin(self, path: Path) -> o3d.geometry.PointCloud:
        """
        Load KITTI binary format.

        KITTI binary files store points as float32: [x, y, z, intensity]

        Args:
            path: Path to .bin file.

        Returns:
            Open3D PointCloud with positions and intensity as grayscale colors.
        """
        import open3d as o3d

        # Read binary data
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))

        # Store intensity as grayscale colors
        intensity = points[:, 3:4]
        # Normalize intensity to [0, 1] if needed
        if intensity.max() > 1.0:
            intensity = intensity / intensity.max()
        colors = np.hstack([intensity, intensity, intensity]).astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        logger.info("Loaded %d points from BIN file: %s", len(points), path)

        return pcd


# Module-level convenience function
def get_processor() -> PointCloudProcessor:
    """
    Get a PointCloudProcessor instance.

    Returns:
        New PointCloudProcessor instance.
    """
    return PointCloudProcessor()
