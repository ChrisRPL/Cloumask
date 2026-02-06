"""
Tests for point cloud processing module.

Tests loading, saving, downsampling, filtering, and normal estimation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip all tests if Open3D is not available
open3d = pytest.importorskip("open3d")

from backend.cv.pointcloud import PointCloudProcessor, get_processor
from backend.cv.types import PointCloudProcessingResult, PointCloudStats


@pytest.fixture
def processor() -> PointCloudProcessor:
    """Create a PointCloudProcessor instance."""
    return PointCloudProcessor()


@pytest.fixture
def sample_pcd(tmp_path: Path) -> Path:
    """Create a sample PCD file with random points."""
    import open3d as o3d

    # Create point cloud with 1000 random points
    np.random.seed(42)
    points = np.random.rand(1000, 3).astype(np.float64)
    points = (points - 0.5) * 10  # Scale to [-5, 5] range

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add colors
    colors = np.random.rand(1000, 3).astype(np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save to file
    path = tmp_path / "sample.pcd"
    o3d.io.write_point_cloud(str(path), pcd)

    return path


@pytest.fixture
def sample_ply(tmp_path: Path) -> Path:
    """Create a sample PLY file with random points."""
    import open3d as o3d

    np.random.seed(123)
    points = np.random.rand(500, 3).astype(np.float64) * 5

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    path = tmp_path / "sample.ply"
    o3d.io.write_point_cloud(str(path), pcd)

    return path


@pytest.fixture
def noisy_pcd(tmp_path: Path) -> Path:
    """Create a point cloud with outliers for filtering tests."""
    import open3d as o3d

    np.random.seed(42)

    # Create main cluster of points
    main_points = np.random.randn(900, 3).astype(np.float64) * 0.5

    # Add outliers far from main cluster
    outliers = np.random.rand(100, 3).astype(np.float64) * 20 - 10

    points = np.vstack([main_points, outliers])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    path = tmp_path / "noisy.pcd"
    o3d.io.write_point_cloud(str(path), pcd)

    return path


class TestPointCloudProcessorLoad:
    """Tests for loading point clouds."""

    def test_load_pcd(self, processor: PointCloudProcessor, sample_pcd: Path) -> None:
        """Should load PCD file successfully."""
        pcd = processor.load(sample_pcd)

        assert pcd is not None
        assert len(pcd.points) == 1000
        assert pcd.has_colors()

    def test_load_ply(self, processor: PointCloudProcessor, sample_ply: Path) -> None:
        """Should load PLY file successfully."""
        pcd = processor.load(sample_ply)

        assert pcd is not None
        assert len(pcd.points) == 500

    def test_load_nonexistent_file(self, processor: PointCloudProcessor) -> None:
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            processor.load("/nonexistent/path/file.pcd")

    def test_load_unsupported_format(
        self, processor: PointCloudProcessor, tmp_path: Path
    ) -> None:
        """Should raise ValueError for unsupported format."""
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("dummy")

        with pytest.raises(ValueError, match="Unsupported format"):
            processor.load(unsupported)


class TestPointCloudProcessorSave:
    """Tests for saving point clouds."""

    def test_save_pcd(
        self, processor: PointCloudProcessor, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should save to PCD format."""
        pcd = processor.load(sample_pcd)
        output_path = tmp_path / "output.pcd"

        processor.save(pcd, output_path)

        assert output_path.exists()
        reloaded = processor.load(output_path)
        assert len(reloaded.points) == 1000

    def test_save_ply(
        self, processor: PointCloudProcessor, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should save to PLY format."""
        pcd = processor.load(sample_pcd)
        output_path = tmp_path / "output.ply"

        processor.save(pcd, output_path)

        assert output_path.exists()
        reloaded = processor.load(output_path)
        assert len(reloaded.points) == 1000

    def test_save_unsupported_format(
        self, processor: PointCloudProcessor, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should raise ValueError for unsupported write format."""
        pcd = processor.load(sample_pcd)

        with pytest.raises(ValueError, match="Cannot write"):
            processor.save(pcd, tmp_path / "output.las")

    def test_save_creates_parent_directory(
        self, processor: PointCloudProcessor, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should create parent directories if they don't exist."""
        pcd = processor.load(sample_pcd)
        output_path = tmp_path / "nested" / "dir" / "output.pcd"

        processor.save(pcd, output_path)

        assert output_path.exists()


class TestPointCloudProcessorStats:
    """Tests for getting point cloud statistics."""

    def test_get_stats(self, processor: PointCloudProcessor, sample_pcd: Path) -> None:
        """Should return correct statistics."""
        stats = processor.get_stats(sample_pcd)

        assert isinstance(stats, PointCloudStats)
        assert stats.point_count == 1000
        assert stats.has_colors is True
        assert stats.has_normals is False
        assert stats.file_format == "pcd"
        assert str(sample_pcd) in stats.file_path

        # Check bounds are reasonable
        assert len(stats.bounds_min) == 3
        assert len(stats.bounds_max) == 3
        assert all(stats.bounds_max[i] >= stats.bounds_min[i] for i in range(3))

    def test_stats_extent(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should compute extent correctly."""
        stats = processor.get_stats(sample_pcd)

        extent = stats.extent
        assert len(extent) == 3
        assert all(e >= 0 for e in extent)

    def test_stats_center(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should compute center correctly."""
        stats = processor.get_stats(sample_pcd)

        center = stats.center
        assert len(center) == 3


class TestVoxelDownsampling:
    """Tests for voxel grid downsampling."""

    def test_voxel_downsample(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should reduce point count with voxel downsampling."""
        pcd = processor.load(sample_pcd)
        original_count = len(pcd.points)

        result = processor.downsample_voxel(pcd, voxel_size=1.0)

        assert len(result.points) < original_count
        assert len(result.points) > 0

    def test_voxel_downsample_small_size(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should preserve more points with smaller voxel size."""
        pcd = processor.load(sample_pcd)

        result_small = processor.downsample_voxel(pcd, voxel_size=0.1)
        result_large = processor.downsample_voxel(pcd, voxel_size=1.0)

        assert len(result_small.points) > len(result_large.points)

    def test_voxel_downsample_invalid_size(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should raise ValueError for invalid voxel size."""
        pcd = processor.load(sample_pcd)

        with pytest.raises(ValueError, match="positive"):
            processor.downsample_voxel(pcd, voxel_size=0)

        with pytest.raises(ValueError, match="positive"):
            processor.downsample_voxel(pcd, voxel_size=-1.0)


class TestRandomDownsampling:
    """Tests for random downsampling."""

    def test_random_downsample(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should reduce to approximately target count."""
        pcd = processor.load(sample_pcd)

        result = processor.downsample_random(pcd, target_count=500)

        # Allow some tolerance in random sampling
        assert len(result.points) <= 550
        assert len(result.points) >= 450

    def test_random_downsample_target_higher(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should return unchanged if target >= current count."""
        pcd = processor.load(sample_pcd)
        original_count = len(pcd.points)

        result = processor.downsample_random(pcd, target_count=2000)

        assert len(result.points) == original_count

    def test_random_downsample_invalid_count(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should raise ValueError for invalid target count."""
        pcd = processor.load(sample_pcd)

        with pytest.raises(ValueError, match="positive"):
            processor.downsample_random(pcd, target_count=0)


class TestStatisticalOutlierRemoval:
    """Tests for statistical outlier removal."""

    def test_statistical_outlier_removal(
        self, processor: PointCloudProcessor, noisy_pcd: Path
    ) -> None:
        """Should remove outliers from noisy point cloud."""
        pcd = processor.load(noisy_pcd)
        original_count = len(pcd.points)

        result, mask = processor.remove_statistical_outliers(
            pcd, nb_neighbors=20, std_ratio=2.0
        )

        assert len(result.points) < original_count
        assert len(mask) == original_count
        assert mask.sum() == len(result.points)

    def test_statistical_outlier_removal_invalid_params(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should raise ValueError for invalid parameters."""
        pcd = processor.load(sample_pcd)

        with pytest.raises(ValueError, match="positive"):
            processor.remove_statistical_outliers(pcd, nb_neighbors=0)

        with pytest.raises(ValueError, match="positive"):
            processor.remove_statistical_outliers(pcd, std_ratio=-1.0)


class TestRadiusOutlierRemoval:
    """Tests for radius-based outlier removal."""

    def test_radius_outlier_removal(
        self, processor: PointCloudProcessor, noisy_pcd: Path
    ) -> None:
        """Should remove isolated points."""
        pcd = processor.load(noisy_pcd)
        original_count = len(pcd.points)

        result, mask = processor.remove_radius_outliers(
            pcd, radius=1.0, min_neighbors=5
        )

        assert len(result.points) < original_count
        assert len(mask) == original_count
        assert mask.sum() == len(result.points)

    def test_radius_outlier_removal_invalid_params(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should raise ValueError for invalid parameters."""
        pcd = processor.load(sample_pcd)

        with pytest.raises(ValueError, match="positive"):
            processor.remove_radius_outliers(pcd, radius=0)

        with pytest.raises(ValueError, match="positive"):
            processor.remove_radius_outliers(pcd, radius=1.0, min_neighbors=0)


class TestNormalEstimation:
    """Tests for normal estimation."""

    def test_estimate_normals(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should estimate normals for all points."""
        pcd = processor.load(sample_pcd)

        result = processor.estimate_normals(pcd, search_radius=1.0, max_nn=30)

        assert result.has_normals()
        normals = np.asarray(result.normals)
        assert normals.shape == (1000, 3)

    def test_normals_are_unit_vectors(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should produce unit-length normal vectors."""
        pcd = processor.load(sample_pcd)

        result = processor.estimate_normals(pcd, search_radius=1.0, max_nn=30)

        normals = np.asarray(result.normals)
        magnitudes = np.linalg.norm(normals, axis=1)

        # All normals should have magnitude ~1.0
        assert np.allclose(magnitudes, 1.0, atol=0.001)

    def test_estimate_normals_invalid_params(
        self, processor: PointCloudProcessor, sample_pcd: Path
    ) -> None:
        """Should raise ValueError for invalid parameters."""
        pcd = processor.load(sample_pcd)

        with pytest.raises(ValueError, match="positive"):
            processor.estimate_normals(pcd, search_radius=0)

        with pytest.raises(ValueError, match="positive"):
            processor.estimate_normals(pcd, max_nn=0)


class TestProcessFile:
    """Tests for process_file convenience method."""

    def test_process_file_voxel(
        self, processor: PointCloudProcessor, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should process file with voxel downsampling."""
        output_path = tmp_path / "output.pcd"

        result = processor.process_file(
            sample_pcd, output_path, "voxel", voxel_size=1.0
        )

        assert isinstance(result, PointCloudProcessingResult)
        assert result.operation == "voxel"
        assert result.original_count == 1000
        assert result.result_count < 1000
        assert Path(result.output_path).exists()
        assert result.processing_time_ms > 0

    def test_process_file_random(
        self, processor: PointCloudProcessor, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should process file with random downsampling."""
        output_path = tmp_path / "output.pcd"

        result = processor.process_file(
            sample_pcd, output_path, "random", target_count=500
        )

        assert result.operation == "random"
        assert result.result_count <= 550  # Allow some tolerance

    def test_process_file_unknown_operation(
        self, processor: PointCloudProcessor, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should raise ValueError for unknown operation."""
        with pytest.raises(ValueError, match="Unknown operation"):
            processor.process_file(sample_pcd, tmp_path / "out.pcd", "invalid")

    def test_processing_result_reduction_ratio(
        self, processor: PointCloudProcessor, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should compute reduction_ratio correctly."""
        result = processor.process_file(
            sample_pcd, tmp_path / "out.pcd", "voxel", voxel_size=2.0
        )

        expected_ratio = 1.0 - (result.result_count / result.original_count)
        assert abs(result.reduction_ratio - expected_ratio) < 0.001


class TestGetProcessor:
    """Tests for get_processor factory function."""

    def test_get_processor_returns_instance(self) -> None:
        """Should return a PointCloudProcessor instance."""
        processor = get_processor()

        assert isinstance(processor, PointCloudProcessor)
