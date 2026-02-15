"""
API integration tests for point cloud endpoints.

Tests the FastAPI routes for point cloud processing operations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Skip all tests if Open3D is not available
open3d = pytest.importorskip("open3d")

from backend.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_pcd(tmp_path: Path) -> Path:
    """Create a sample PCD file for testing."""
    import open3d as o3d

    np.random.seed(42)
    points = np.random.rand(500, 3).astype(np.float64) * 10

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.random.rand(500, 3).astype(np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    path = tmp_path / "test.pcd"
    o3d.io.write_point_cloud(str(path), pcd)

    return path


class TestStatsEndpoint:
    """Tests for GET /pointcloud/stats endpoint."""

    def test_get_stats_success(self, client: TestClient, sample_pcd: Path) -> None:
        """Should return stats for valid point cloud."""
        response = client.get(f"/pointcloud/stats?path={sample_pcd}")

        assert response.status_code == 200
        data = response.json()
        assert data["point_count"] == 500
        assert data["has_colors"] is True
        assert data["has_normals"] is False
        assert data["file_format"] == "pcd"
        assert len(data["bounds_min"]) == 3
        assert len(data["bounds_max"]) == 3

    def test_get_stats_file_not_found(self, client: TestClient) -> None:
        """Should return 404 for missing file."""
        response = client.get("/pointcloud/stats?path=/nonexistent/file.pcd")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_stats_missing_path(self, client: TestClient) -> None:
        """Should return 422 for missing path parameter."""
        response = client.get("/pointcloud/stats")

        assert response.status_code == 422


class TestDownsampleEndpoint:
    """Tests for POST /pointcloud/downsample endpoint."""

    def test_downsample_voxel_success(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should downsample using voxel method."""
        output_path = tmp_path / "output.pcd"

        response = client.post(
            "/pointcloud/downsample",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(output_path),
                "method": "voxel",
                "voxel_size": 2.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["operation"] == "voxel"
        assert data["original_count"] == 500
        assert data["result_count"] < 500
        assert data["processing_time_ms"] > 0
        assert Path(data["output_path"]).exists()

    def test_downsample_random_success(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should downsample using random method."""
        output_path = tmp_path / "output.pcd"

        response = client.post(
            "/pointcloud/downsample",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(output_path),
                "method": "random",
                "target_count": 200,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["operation"] == "random"
        assert data["result_count"] <= 220  # Allow tolerance

    def test_downsample_voxel_missing_size(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should return 400 when voxel_size missing for voxel method."""
        response = client.post(
            "/pointcloud/downsample",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(tmp_path / "out.pcd"),
                "method": "voxel",
            },
        )

        assert response.status_code == 400
        assert "voxel_size" in response.json()["detail"].lower()

    def test_downsample_random_missing_count(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should return 400 when target_count missing for random method."""
        response = client.post(
            "/pointcloud/downsample",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(tmp_path / "out.pcd"),
                "method": "random",
            },
        )

        assert response.status_code == 400
        assert "target_count" in response.json()["detail"].lower()

    def test_downsample_file_not_found(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Should return 404 for missing input file."""
        response = client.post(
            "/pointcloud/downsample",
            json={
                "input_path": "/nonexistent/file.pcd",
                "output_path": str(tmp_path / "out.pcd"),
                "method": "voxel",
                "voxel_size": 1.0,
            },
        )

        assert response.status_code == 404


class TestFilterEndpoint:
    """Tests for POST /pointcloud/filter endpoint."""

    def test_filter_statistical_success(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should filter using statistical method."""
        output_path = tmp_path / "output.pcd"

        response = client.post(
            "/pointcloud/filter",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(output_path),
                "method": "statistical",
                "nb_neighbors": 20,
                "std_ratio": 2.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["operation"] == "statistical"
        assert Path(data["output_path"]).exists()

    def test_filter_radius_success(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should filter using radius method."""
        output_path = tmp_path / "output.pcd"

        response = client.post(
            "/pointcloud/filter",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(output_path),
                "method": "radius",
                "radius": 1.0,
                "min_neighbors": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["operation"] == "radius"

    def test_filter_radius_missing_radius(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should return 400 when radius missing for radius method."""
        response = client.post(
            "/pointcloud/filter",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(tmp_path / "out.pcd"),
                "method": "radius",
            },
        )

        assert response.status_code == 400
        assert "radius" in response.json()["detail"].lower()


class TestNormalsEndpoint:
    """Tests for POST /pointcloud/normals endpoint."""

    def test_estimate_normals_success(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should estimate normals successfully."""
        output_path = tmp_path / "output.pcd"

        response = client.post(
            "/pointcloud/normals",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(output_path),
                "search_radius": 2.0,
                "max_nn": 30,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["operation"] == "normals"
        assert data["result_count"] == 500  # Point count unchanged

        # Verify normals were computed
        import open3d as o3d

        result_pcd = o3d.io.read_point_cloud(str(output_path))
        assert result_pcd.has_normals()

    def test_estimate_normals_default_params(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should use default parameters if not provided."""
        output_path = tmp_path / "output.pcd"

        response = client.post(
            "/pointcloud/normals",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(output_path),
            },
        )

        assert response.status_code == 200
        assert response.json()["operation"] == "normals"

    def test_estimate_normals_accepts_radius_alias(
        self,
        client: TestClient,
        sample_pcd: Path,
        tmp_path: Path,
    ) -> None:
        """Should accept legacy `radius` field as alias for `search_radius`."""
        output_path = tmp_path / "output_alias.pcd"

        response = client.post(
            "/pointcloud/normals",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(output_path),
                "radius": 2.0,
                "max_nn": 16,
            },
        )

        assert response.status_code == 200
        assert response.json()["operation"] == "normals"
        assert response.json()["parameters"]["search_radius"] == 2.0


class TestConvertEndpoint:
    """Tests for POST /pointcloud/convert endpoint."""

    def test_convert_pcd_to_ply(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should convert PCD to PLY format."""
        output_path = tmp_path / "output.ply"

        response = client.post(
            "/pointcloud/convert",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(output_path),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["operation"] == "convert"
        assert data["result_count"] == 500  # No points lost
        assert output_path.exists()

    def test_convert_with_ascii(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should support ASCII output option."""
        output_path = tmp_path / "output.ply"

        response = client.post(
            "/pointcloud/convert",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(output_path),
                "write_ascii": True,
            },
        )

        assert response.status_code == 200
        assert response.json()["parameters"]["write_ascii"] is True

    def test_convert_file_not_found(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Should return 404 for missing input file."""
        response = client.post(
            "/pointcloud/convert",
            json={
                "input_path": "/nonexistent/file.pcd",
                "output_path": str(tmp_path / "out.ply"),
            },
        )

        assert response.status_code == 404


class TestValidationErrors:
    """Tests for request validation errors."""

    def test_downsample_invalid_method(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should return 422 for invalid method value."""
        response = client.post(
            "/pointcloud/downsample",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(tmp_path / "out.pcd"),
                "method": "invalid",
                "voxel_size": 1.0,
            },
        )

        assert response.status_code == 422

    def test_downsample_invalid_voxel_size(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should return 422 for negative voxel size."""
        response = client.post(
            "/pointcloud/downsample",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(tmp_path / "out.pcd"),
                "method": "voxel",
                "voxel_size": -1.0,
            },
        )

        assert response.status_code == 422

    def test_filter_invalid_method(
        self, client: TestClient, sample_pcd: Path, tmp_path: Path
    ) -> None:
        """Should return 422 for invalid filter method."""
        response = client.post(
            "/pointcloud/filter",
            json={
                "input_path": str(sample_pcd),
                "output_path": str(tmp_path / "out.pcd"),
                "method": "unknown",
            },
        )

        assert response.status_code == 422
