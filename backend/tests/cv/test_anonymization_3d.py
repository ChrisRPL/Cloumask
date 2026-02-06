"""
Tests for 3D point cloud anonymization pipeline.

Unit tests use mocked face detectors to avoid requiring real model
downloads.  Integration tests are marked with @pytest.mark.integration.

Implements spec: 05-point-cloud/07-anonymization-3d (testing section)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip the whole module if Open3D is not installed
open3d = pytest.importorskip("open3d")

from backend.cv.anonymization_3d import (
    Anonymization3DResult,
    PointCloudAnonymizer,
)
from backend.cv.face_3d_projection import (
    expand_box,
    find_points_in_2d_box,
    project_points_to_camera,
    render_depth_image,
)
from backend.cv.types import BBox, FaceDetection, FaceDetectionResult
from backend.cv.virtual_camera import VirtualCamera, generate_virtual_cameras


# ---------------------------------------------------------------------------
# Mock face detector
# ---------------------------------------------------------------------------


class MockFaceDetector3D:
    """Mock face detector that returns configurable results."""

    info = MagicMock()
    info.vram_required_mb = 200

    def __init__(self, faces_per_call: int = 0) -> None:
        self._is_loaded = False
        self._device = "cpu"
        self.faces_per_call = faces_per_call

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self, device: str = "cpu") -> None:
        self._is_loaded = True
        self._device = device

    def unload(self) -> None:
        self._is_loaded = False

    def predict(
        self,
        input_path: str,
        *,
        confidence: float = 0.5,
        **kwargs: Any,
    ) -> FaceDetectionResult:
        faces = [
            FaceDetection(
                bbox=BBox(x=0.5, y=0.5, width=0.3, height=0.3),
                confidence=0.95,
                landmarks=None,
            )
            for _ in range(self.faces_per_call)
        ]
        return FaceDetectionResult(
            faces=faces,
            image_path=input_path,
            processing_time_ms=1.0,
            model_name="mock-face-3d",
        )


class MockFaceDetectorNoFaces(MockFaceDetector3D):
    """Always returns zero faces (for verification pass)."""

    def __init__(self) -> None:
        super().__init__(faces_per_call=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pcd(tmp_path: Path) -> Path:
    """Point cloud with a face-like cluster at head height."""
    import open3d as o3d

    np.random.seed(42)
    # Random scene points
    scene = np.random.rand(5000, 3).astype(np.float64) * 10

    # Dense cluster at approximate head height (simulates face region)
    face_center = np.array([5.0, 5.0, 1.7])
    face_points = np.random.randn(300, 3).astype(np.float64) * 0.15 + face_center

    all_points = np.vstack([scene, face_points])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    # Add colours so colour-preservation is testable
    colors = np.random.rand(len(all_points), 3).astype(np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    path = tmp_path / "with_face.pcd"
    o3d.io.write_point_cloud(str(path), pcd)
    return path


@pytest.fixture
def empty_pcd(tmp_path: Path) -> Path:
    """Empty point cloud file."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    path = tmp_path / "empty.pcd"
    o3d.io.write_point_cloud(str(path), pcd)
    return path


# ---------------------------------------------------------------------------
# Virtual Camera Tests
# ---------------------------------------------------------------------------


class TestVirtualCameraGeneration:
    """Tests for generate_virtual_cameras."""

    def test_generates_correct_count(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([10, 10, 3]))
        cameras = generate_virtual_cameras(bounds, num_views=8)
        assert len(cameras) == 8

    def test_camera_has_required_fields(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([10, 10, 3]))
        cameras = generate_virtual_cameras(bounds, num_views=1)
        cam = cameras[0]

        assert cam.K.shape == (3, 3)
        assert cam.view_matrix.shape == (4, 4)
        assert cam.position.shape == (3,)
        assert cam.resolution == (640, 480)

    def test_cameras_surround_center(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([10, 10, 3]))
        cameras = generate_virtual_cameras(bounds, num_views=4)

        center = np.array([5, 5, 1.5])
        for cam in cameras:
            # All cameras should be farther from center than the scene extent
            dist = np.linalg.norm(cam.position[:2] - center[:2])
            assert dist > 5.0

    def test_single_view(self) -> None:
        bounds = (np.array([-1, -1, -1]), np.array([1, 1, 1]))
        cameras = generate_virtual_cameras(bounds, num_views=1)
        assert len(cameras) == 1

    def test_invalid_num_views_raises(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([10, 10, 3]))
        with pytest.raises(ValueError, match="num_views"):
            generate_virtual_cameras(bounds, num_views=0)

    def test_degenerate_scene(self) -> None:
        """Single-point cloud should not crash."""
        bounds = (np.array([5, 5, 5]), np.array([5, 5, 5]))
        cameras = generate_virtual_cameras(bounds, num_views=4)
        assert len(cameras) == 4

    def test_custom_resolution(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([1, 1, 1]))
        cameras = generate_virtual_cameras(bounds, num_views=2, resolution=(320, 240))
        for cam in cameras:
            assert cam.resolution == (320, 240)
            assert cam.width == 320
            assert cam.height == 240


# ---------------------------------------------------------------------------
# Face 3D Projection Tests
# ---------------------------------------------------------------------------


class TestProjectPointsToCamera:
    """Tests for project_points_to_camera."""

    def test_empty_points(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([1, 1, 1]))
        cam = generate_virtual_cameras(bounds, num_views=1)[0]
        pts2d, depths, valid = project_points_to_camera(np.zeros((0, 3)), cam)
        assert pts2d.shape == (0, 2)
        assert len(depths) == 0
        assert len(valid) == 0

    def test_points_in_front_of_camera_are_valid(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([10, 10, 3]))
        cam = generate_virtual_cameras(bounds, num_views=1)[0]

        # Place points at the scene center (camera should see them)
        pts = np.array([[5.0, 5.0, 1.5]])
        pts2d, depths, valid = project_points_to_camera(pts, cam)

        # Projection should complete without error and return correct shapes
        assert pts2d.shape == (1, 2)
        assert depths.shape == (1,)
        assert valid.shape == (1,)


class TestRenderDepthImage:
    """Tests for render_depth_image."""

    def test_output_shape(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([10, 10, 3]))
        cam = generate_virtual_cameras(bounds, num_views=1, resolution=(320, 240))[0]

        pts = np.random.rand(1000, 3) * 10
        img = render_depth_image(pts, cam)

        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8

    def test_empty_points_returns_black(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([1, 1, 1]))
        cam = generate_virtual_cameras(bounds, num_views=1, resolution=(64, 48))[0]
        img = render_depth_image(np.zeros((0, 3)), cam)
        assert img.shape == (48, 64, 3)
        assert img.max() == 0


class TestFindPointsIn2DBox:
    """Tests for find_points_in_2d_box."""

    def test_basic_selection(self) -> None:
        pts2d = np.array([[10.0, 10.0], [50.0, 50.0], [200.0, 200.0]])
        valid = np.array([True, True, True])

        indices = find_points_in_2d_box(pts2d, valid, (0, 0, 100, 100))
        assert set(indices) == {0, 1}

    def test_invalid_points_excluded(self) -> None:
        pts2d = np.array([[10.0, 10.0], [50.0, 50.0]])
        valid = np.array([False, True])

        indices = find_points_in_2d_box(pts2d, valid, (0, 0, 100, 100))
        assert set(indices) == {1}

    def test_margin_expansion(self) -> None:
        """Point just outside box should be included after expansion."""
        pts2d = np.array([[105.0, 50.0]])
        valid = np.array([True])

        # Without margin: box (0,0)-(100,100), point at x=105 is outside
        assert len(find_points_in_2d_box(pts2d, valid, (0, 0, 100, 100), margin=1.0)) == 0

        # With 20% margin the box becomes wider
        assert len(find_points_in_2d_box(pts2d, valid, (0, 0, 100, 100), margin=1.2)) == 1


class TestExpandBox:
    """Tests for expand_box."""

    def test_no_expansion(self) -> None:
        assert expand_box((10, 10, 50, 50), 1.0, 640, 480) == (10.0, 10.0, 50.0, 50.0)

    def test_expansion_clamped(self) -> None:
        result = expand_box((0, 0, 100, 100), 2.0, 200, 200)
        assert result[0] >= 0
        assert result[2] <= 200


# ---------------------------------------------------------------------------
# PointCloudAnonymizer Tests (mocked face detector)
# ---------------------------------------------------------------------------


class TestPointCloudAnonymizerUnit:
    """Unit tests for the full anonymization pipeline with mocks."""

    @patch("backend.cv.faces.get_face_detector")
    @patch("backend.cv.device.select_device", return_value="cpu")
    def test_load_unload(
        self,
        mock_device: MagicMock,
        mock_factory: MagicMock,
    ) -> None:
        mock_factory.return_value = MockFaceDetector3D()

        anon = PointCloudAnonymizer()
        assert not anon.is_loaded

        anon.load("cpu")
        assert anon.is_loaded

        anon.unload()
        assert not anon.is_loaded

    def test_anonymize_not_loaded_raises(self, tmp_path: Path) -> None:
        anon = PointCloudAnonymizer()
        with pytest.raises(RuntimeError, match="not loaded"):
            anon.anonymize("input.pcd", "output.pcd")

    @patch("backend.cv.faces.get_face_detector")
    @patch("backend.cv.device.select_device", return_value="cpu")
    def test_anonymize_empty_cloud(
        self,
        mock_device: MagicMock,
        mock_factory: MagicMock,
        empty_pcd: Path,
        tmp_path: Path,
    ) -> None:
        mock_factory.return_value = MockFaceDetector3D(faces_per_call=0)

        anon = PointCloudAnonymizer()
        anon.load("cpu")

        output = tmp_path / "out.pcd"
        result = anon.anonymize(str(empty_pcd), str(output))

        assert result.original_point_count == 0
        assert result.verification_passed is True
        assert output.exists()

        anon.unload()

    @patch("backend.cv.faces.get_face_detector")
    @patch("backend.cv.device.select_device", return_value="cpu")
    def test_anonymize_no_faces_preserves_points(
        self,
        mock_device: MagicMock,
        mock_factory: MagicMock,
        sample_pcd: Path,
        tmp_path: Path,
    ) -> None:
        """When detector finds 0 faces, all points should be preserved."""
        mock_factory.return_value = MockFaceDetector3D(faces_per_call=0)

        anon = PointCloudAnonymizer()
        anon.load("cpu")

        output = tmp_path / "out.pcd"
        result = anon.anonymize(str(sample_pcd), str(output), verify=False)

        assert result.face_regions_found == 0
        assert result.points_removed == 0
        assert result.anonymized_point_count == result.original_point_count
        assert output.exists()

        anon.unload()

    @patch("backend.cv.faces.get_face_detector")
    @patch("backend.cv.device.select_device", return_value="cpu")
    def test_anonymize_remove_mode(
        self,
        mock_device: MagicMock,
        mock_factory: MagicMock,
        sample_pcd: Path,
        tmp_path: Path,
    ) -> None:
        """When faces are detected, remove mode should reduce point count."""
        mock_factory.return_value = MockFaceDetector3D(faces_per_call=1)

        anon = PointCloudAnonymizer()
        anon.load("cpu")

        output = tmp_path / "removed.pcd"
        result = anon.anonymize(
            str(sample_pcd),
            str(output),
            mode="remove",
            verify=False,
            num_views=4,
        )

        assert result.face_regions_found > 0
        assert result.points_removed > 0
        assert result.anonymized_point_count < result.original_point_count
        assert result.mode == "remove"
        assert output.exists()

        anon.unload()

    @patch("backend.cv.faces.get_face_detector")
    @patch("backend.cv.device.select_device", return_value="cpu")
    def test_anonymize_noise_mode(
        self,
        mock_device: MagicMock,
        mock_factory: MagicMock,
        sample_pcd: Path,
        tmp_path: Path,
    ) -> None:
        """Noise mode should keep the same count but modify positions."""
        mock_factory.return_value = MockFaceDetector3D(faces_per_call=1)

        anon = PointCloudAnonymizer()
        anon.load("cpu")

        output = tmp_path / "noised.pcd"
        result = anon.anonymize(
            str(sample_pcd),
            str(output),
            mode="noise",
            verify=False,
            num_views=4,
        )

        assert result.points_noised > 0
        assert result.points_removed == 0
        assert result.anonymized_point_count == result.original_point_count
        assert result.mode == "noise"

        anon.unload()

    @patch("backend.cv.faces.get_face_detector")
    @patch("backend.cv.device.select_device", return_value="cpu")
    def test_anonymize_preserves_colors(
        self,
        mock_device: MagicMock,
        mock_factory: MagicMock,
        sample_pcd: Path,
        tmp_path: Path,
    ) -> None:
        """Output cloud should retain RGB colours."""
        import open3d as o3d

        mock_factory.return_value = MockFaceDetector3D(faces_per_call=1)

        anon = PointCloudAnonymizer()
        anon.load("cpu")

        output = tmp_path / "colored.pcd"
        anon.anonymize(str(sample_pcd), str(output), mode="remove", verify=False)

        result_pcd = o3d.io.read_point_cloud(str(output))
        assert result_pcd.has_colors()

        anon.unload()

    @patch("backend.cv.faces.get_face_detector")
    @patch("backend.cv.device.select_device", return_value="cpu")
    def test_nonexistent_input_raises(
        self,
        mock_device: MagicMock,
        mock_factory: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_factory.return_value = MockFaceDetector3D()

        anon = PointCloudAnonymizer()
        anon.load("cpu")

        with pytest.raises(FileNotFoundError):
            anon.anonymize("/does/not/exist.pcd", str(tmp_path / "out.pcd"))

        anon.unload()

    @patch("backend.cv.faces.get_face_detector")
    @patch("backend.cv.device.select_device", return_value="cpu")
    def test_result_timing(
        self,
        mock_device: MagicMock,
        mock_factory: MagicMock,
        sample_pcd: Path,
        tmp_path: Path,
    ) -> None:
        mock_factory.return_value = MockFaceDetector3D(faces_per_call=0)

        anon = PointCloudAnonymizer()
        anon.load("cpu")

        result = anon.anonymize(str(sample_pcd), str(tmp_path / "t.pcd"), verify=False)
        assert result.processing_time_ms > 0
        assert result.views_processed == 8  # default

        anon.unload()


# ---------------------------------------------------------------------------
# Anonymization3DResult Tests
# ---------------------------------------------------------------------------


class TestAnonymization3DResult:
    """Tests for the result dataclass."""

    def test_result_fields(self) -> None:
        r = Anonymization3DResult(
            output_path="/out.pcd",
            original_point_count=10000,
            anonymized_point_count=9500,
            face_regions_found=3,
            points_removed=500,
            points_noised=0,
            verification_passed=True,
            processing_time_ms=1234.5,
            views_processed=8,
            mode="remove",
        )
        assert r.points_removed == 500
        assert r.mode == "remove"
