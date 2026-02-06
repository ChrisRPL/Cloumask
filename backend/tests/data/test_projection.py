"""
Tests for 3D-2D projection and 2D-3D lifting functions.

Tests projection accuracy, performance, and edge cases.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from backend.cv.types import Detection3D
from backend.data.calibration import CameraCalibration
from backend.data.projection import (
    lift_2d_to_3d,
    project_bbox3d_to_2d,
    project_detections_3d_to_2d,
    project_points_to_image,
    transform_points,
)


@pytest.fixture
def simple_calib() -> CameraCalibration:
    """Simple pinhole calibration without distortion."""
    K = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # Identity transform (camera = LiDAR frame)
    T = np.eye(4)

    return CameraCalibration(
        K=K,
        D=np.zeros(5),
        width=640,
        height=480,
        T_cam_lidar=T,
    )


@pytest.fixture
def kitti_like_calib() -> CameraCalibration:
    """KITTI-like calibration with extrinsic transform."""
    K = np.array(
        [
            [721.5, 0.0, 609.6],
            [0.0, 721.5, 172.9],
            [0.0, 0.0, 1.0],
        ]
    )
    # Typical KITTI velodyne-to-camera transform
    T = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -0.08],
            [1.0, 0.0, 0.0, -0.27],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    return CameraCalibration(
        K=K,
        D=np.zeros(5),
        width=1242,
        height=375,
        T_cam_lidar=T,
    )


@pytest.fixture
def sample_detection() -> Detection3D:
    """Sample 3D detection for testing.
    
    Uses camera coordinate system where z is forward (depth).
    """
    return Detection3D(
        class_id=0,
        class_name="Car",
        center=(0.0, 0.0, 10.0),  # 10m ahead in camera frame (z is forward)
        dimensions=(4.5, 1.8, 1.5),  # Typical car dimensions (l, w, h)
        rotation=0.0,
        confidence=0.9,
    )


class TestTransformPoints:
    """Tests for point transformation."""

    def test_identity_transform(self) -> None:
        """Identity transform should not change points."""
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        T = np.eye(4)

        result = transform_points(points, T)

        np.testing.assert_array_almost_equal(result, points)

    def test_translation(self) -> None:
        """Test translation transform."""
        points = np.array([[0, 0, 0]], dtype=np.float64)
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]

        result = transform_points(points, T)

        np.testing.assert_array_almost_equal(result, [[1, 2, 3]])

    def test_empty_points(self) -> None:
        """Test with empty point array."""
        points = np.zeros((0, 3), dtype=np.float64)
        T = np.eye(4)

        result = transform_points(points, T)

        assert result.shape == (0, 3)


class TestProjectPointsToImage:
    """Tests for 3D to 2D point projection."""

    def test_center_point_projects_to_center(self, simple_calib: CameraCalibration) -> None:
        """Point on optical axis projects to image center."""
        # Point at (0, 0, 10) in camera frame should project to (cx, cy)
        points = np.array([[0.0, 0.0, 10.0]])
        points_2d, valid = project_points_to_image(points, simple_calib)

        assert valid[0]
        np.testing.assert_almost_equal(points_2d[0, 0], 320.0, decimal=1)
        np.testing.assert_almost_equal(points_2d[0, 1], 240.0, decimal=1)

    def test_points_behind_camera_filtered(self, simple_calib: CameraCalibration) -> None:
        """Points behind camera (z <= 0) should be marked invalid."""
        points = np.array([[0.0, 0.0, -10.0], [0.0, 0.0, 0.0]])
        points_2d, valid = project_points_to_image(points, simple_calib)

        assert not valid[0]
        assert not valid[1]

    def test_points_outside_image_filtered(self, simple_calib: CameraCalibration) -> None:
        """Points projecting outside image should be marked invalid."""
        # Point far to the left
        points = np.array([[-100.0, 0.0, 10.0]])
        points_2d, valid = project_points_to_image(points, simple_calib)

        assert not valid[0]

    def test_filter_options(self, simple_calib: CameraCalibration) -> None:
        """Test filter_behind and filter_outside options."""
        points = np.array([[0.0, 0.0, -10.0]])

        # With filter_behind=False, point should still be valid (allows negative z)
        points_2d, valid = project_points_to_image(
            points, simple_calib, filter_behind=False
        )
        # Point is behind camera but filter_behind=False so it's not filtered
        assert valid[0]  # Valid because filter_behind=False

    def test_performance_100k_points(self, simple_calib: CameraCalibration) -> None:
        """Projection of 100K points should complete in <100ms."""
        np.random.seed(42)
        points = np.random.randn(100000, 3)
        points[:, 2] = np.abs(points[:, 2]) + 1  # All in front of camera

        start = time.perf_counter()
        points_2d, valid = project_points_to_image(points, simple_calib)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Projection took {elapsed_ms:.1f}ms, should be <100ms"
        assert points_2d.shape == (100000, 2)

    def test_empty_points(self, simple_calib: CameraCalibration) -> None:
        """Test with empty point array."""
        points = np.zeros((0, 3), dtype=np.float64)
        points_2d, valid = project_points_to_image(points, simple_calib)

        assert points_2d.shape == (0, 2)
        assert valid.shape == (0,)


class TestProjectBBox3DTo2D:
    """Tests for 3D bounding box projection."""

    def test_visible_box_projects(
        self, simple_calib: CameraCalibration, sample_detection: Detection3D
    ) -> None:
        """Visible 3D box should project to 2D rectangle."""
        box_2d = project_bbox3d_to_2d(sample_detection, simple_calib)

        assert box_2d is not None
        x_min, y_min, x_max, y_max = box_2d
        assert x_max > x_min
        assert y_max > y_min

    def test_box_behind_camera_returns_none(self, simple_calib: CameraCalibration) -> None:
        """Box entirely behind camera should return None."""
        detection = Detection3D(
            class_id=0,
            class_name="Car",
            center=(-10.0, 0.0, 0.0),  # Behind camera
            dimensions=(4.5, 1.8, 1.5),
            rotation=0.0,
            confidence=0.9,
        )
        box_2d = project_bbox3d_to_2d(detection, simple_calib)

        assert box_2d is None

    def test_box_clipped_to_image(self, simple_calib: CameraCalibration) -> None:
        """Projected box should be clipped to image bounds."""
        # Box at edge of FOV
        detection = Detection3D(
            class_id=0,
            class_name="Car",
            center=(5.0, -5.0, 0.0),  # Far to the right
            dimensions=(4.5, 1.8, 1.5),
            rotation=0.0,
            confidence=0.9,
        )
        box_2d = project_bbox3d_to_2d(detection, simple_calib)

        if box_2d is not None:
            x_min, y_min, x_max, y_max = box_2d
            assert x_min >= 0
            assert y_min >= 0
            assert x_max <= simple_calib.width
            assert y_max <= simple_calib.height


class TestDetection3DToCorners:
    """Tests for Detection3D.to_corners() method."""

    def test_corners_shape(self, sample_detection: Detection3D) -> None:
        """to_corners() should return (8, 3) array."""
        corners = sample_detection.to_corners()
        assert corners.shape == (8, 3)

    def test_corners_extent_matches_dimensions(self, sample_detection: Detection3D) -> None:
        """Corner extent should match box dimensions."""
        corners = sample_detection.to_corners()
        extent = corners.max(axis=0) - corners.min(axis=0)

        np.testing.assert_almost_equal(extent[0], 4.5, decimal=1)  # length
        np.testing.assert_almost_equal(extent[1], 1.8, decimal=1)  # width
        np.testing.assert_almost_equal(extent[2], 1.5, decimal=1)  # height

    def test_corners_centered(self, sample_detection: Detection3D) -> None:
        """Corner center should match detection center."""
        corners = sample_detection.to_corners()
        center = corners.mean(axis=0)

        # Detection center is at (0, 0, 10) in camera frame
        np.testing.assert_almost_equal(center[0], 0.0, decimal=1)
        np.testing.assert_almost_equal(center[1], 0.0, decimal=1)
        np.testing.assert_almost_equal(center[2], 10.0, decimal=1)

    def test_corners_with_rotation(self) -> None:
        """Corners should be rotated when yaw is non-zero."""
        detection = Detection3D(
            class_id=0,
            class_name="Car",
            center=(0.0, 0.0, 0.0),
            dimensions=(4.0, 2.0, 1.5),
            rotation=np.pi / 2,  # 90 degrees
            confidence=0.9,
        )
        corners = detection.to_corners()
        extent = corners.max(axis=0) - corners.min(axis=0)

        # After 90-degree rotation, length and width should swap
        np.testing.assert_almost_equal(extent[0], 2.0, decimal=1)  # was width
        np.testing.assert_almost_equal(extent[1], 4.0, decimal=1)  # was length


class TestProjectDetections3DTo2D:
    """Tests for batch projection."""

    def test_batch_projection(
        self, simple_calib: CameraCalibration, sample_detection: Detection3D
    ) -> None:
        """Batch projection should return results for all detections."""
        detections = [sample_detection, sample_detection]
        results = project_detections_3d_to_2d(detections, simple_calib)

        assert len(results) == 2
        for det, box_2d in results:
            assert det == sample_detection
            assert box_2d is not None


class TestLift2DTo3D:
    """Tests for 2D to 3D lifting."""

    def test_lift_with_points(self, simple_calib: CameraCalibration) -> None:
        """Test lifting 2D box with point cloud."""
        # Create points roughly forming a car shape at z=10m (in front of camera)
        # With identity T_cam_lidar, z is the forward axis
        np.random.seed(42)
        points = np.array(
            [
                [-0.5, -0.5, 8.0],
                [0.5, -0.5, 8.0],
                [-0.5, 0.5, 8.0],
                [0.5, 0.5, 8.0],
                [-0.5, -0.5, 10.0],
                [0.5, -0.5, 10.0],
                [-0.5, 0.5, 10.0],
                [0.5, 0.5, 10.0],
                [0.0, 0.0, 12.0],
            ]
        )

        # Project to get 2D box
        points_2d, valid = project_points_to_image(points, simple_calib)
        valid_2d = points_2d[valid]
        assert len(valid_2d) > 0, "No points projected into image"
        box_2d = (
            float(valid_2d[:, 0].min()),
            float(valid_2d[:, 1].min()),
            float(valid_2d[:, 0].max()),
            float(valid_2d[:, 1].max()),
        )

        # Lift back
        det3d = lift_2d_to_3d(box_2d, points, simple_calib)

        assert det3d is not None
        # Center z should be near the median of points (around 10m)
        assert 7 < det3d.center[2] < 13

    def test_lift_insufficient_points(self, simple_calib: CameraCalibration) -> None:
        """Test that lifting fails with insufficient points."""
        points = np.array([[10.0, 0.0, 0.0]])  # Only 1 point
        box_2d = (300, 220, 340, 260)

        det3d = lift_2d_to_3d(box_2d, points, simple_calib, min_points=3)

        assert det3d is None

    def test_lift_with_class_prior(self, simple_calib: CameraCalibration) -> None:
        """Test lifting with class-specific size prior."""
        # Points in front of camera (positive z) - with identity transform, z is forward
        points = np.array(
            [
                [-0.5, -0.5, 10.0],
                [0.5, -0.5, 10.0],
                [0.0, 0.5, 10.0],
                [0.0, 0.0, 11.0],
            ]
        )

        points_2d, valid = project_points_to_image(points, simple_calib)
        valid_2d = points_2d[valid]
        assert len(valid_2d) > 0, "No points projected into image"
        box_2d = (
            float(valid_2d[:, 0].min()),
            float(valid_2d[:, 1].min()),
            float(valid_2d[:, 0].max()),
            float(valid_2d[:, 1].max()),
        )

        class_priors = {"Car": {"dimensions": (4.5, 1.8, 1.5)}}

        det3d = lift_2d_to_3d(
            box_2d, points, simple_calib, class_name="Car", class_priors=class_priors
        )

        assert det3d is not None
        assert det3d.dimensions == (4.5, 1.8, 1.5)
        assert det3d.class_name == "Car"
