"""
Tests for 2D-3D detection fusion module.

Tests IoU computation and detection matching pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.cv.types import BBox, Detection, Detection3D
from backend.data.calibration import CameraCalibration
from backend.data.fusion import compute_iou_2d, fuse_detections


@pytest.fixture
def simple_calib() -> CameraCalibration:
    """Simple pinhole calibration for testing."""
    K = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return CameraCalibration(
        K=K,
        D=np.zeros(5),
        width=640,
        height=480,
        T_cam_lidar=np.eye(4),
    )


@pytest.fixture
def sample_detection_2d() -> Detection:
    """Sample 2D detection centered in image."""
    # Bbox centered at (0.5, 0.5) will overlap with 3D box at (0, 0, 10)
    # which projects to image center (320, 240) = normalized (0.5, 0.5)
    return Detection(
        class_id=0,
        class_name="Car",
        bbox=BBox(x=0.5, y=0.5, width=0.3, height=0.25),  # Centered, wider for overlap
        confidence=0.9,
    )


@pytest.fixture
def sample_detection_3d() -> Detection3D:
    """Sample 3D detection that projects to center."""
    return Detection3D(
        class_id=0,
        class_name="Car",
        center=(0.0, 0.0, 10.0),  # Projects to image center
        dimensions=(4.5, 1.8, 1.5),
        rotation=0.0,
        confidence=0.85,
    )


class TestComputeIoU2D:
    """Tests for 2D IoU computation."""

    def test_identical_boxes_iou_1(self) -> None:
        """Identical boxes should have IoU = 1.0."""
        box = (100.0, 100.0, 200.0, 200.0)
        iou = compute_iou_2d(box, box)
        assert iou == 1.0

    def test_no_overlap_iou_0(self) -> None:
        """Non-overlapping boxes should have IoU = 0.0."""
        box1 = (0.0, 0.0, 100.0, 100.0)
        box2 = (200.0, 200.0, 300.0, 300.0)
        iou = compute_iou_2d(box1, box2)
        assert iou == 0.0

    def test_partial_overlap(self) -> None:
        """Test partial overlap IoU calculation."""
        box1 = (0.0, 0.0, 100.0, 100.0)
        box2 = (50.0, 50.0, 150.0, 150.0)

        iou = compute_iou_2d(box1, box2)

        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        expected_iou = 2500.0 / 17500.0
        np.testing.assert_almost_equal(iou, expected_iou, decimal=3)

    def test_one_inside_other(self) -> None:
        """Test when one box is inside the other."""
        outer = (0.0, 0.0, 100.0, 100.0)
        inner = (25.0, 25.0, 75.0, 75.0)

        iou = compute_iou_2d(outer, inner)

        # Inner area = 50x50 = 2500
        # Outer area = 100x100 = 10000
        # Intersection = 2500, Union = 10000
        expected_iou = 2500.0 / 10000.0
        np.testing.assert_almost_equal(iou, expected_iou, decimal=3)

    def test_edge_touching(self) -> None:
        """Boxes that only touch at edge should have IoU = 0."""
        box1 = (0.0, 0.0, 100.0, 100.0)
        box2 = (100.0, 0.0, 200.0, 100.0)  # Touches at x=100

        iou = compute_iou_2d(box1, box2)
        assert iou == 0.0


class TestFuseDetections:
    """Tests for detection fusion pipeline."""

    def test_matching_detections(
        self,
        simple_calib: CameraCalibration,
        sample_detection_2d: Detection,
        sample_detection_3d: Detection3D,
    ) -> None:
        """Test fusion with matching 2D and 3D detections."""
        # Use low IoU threshold to ensure matching
        fused = fuse_detections(
            [sample_detection_2d],
            [sample_detection_3d],
            simple_calib,
            iou_threshold=0.1,
        )

        assert len(fused) == 1
        assert fused[0].has_3d
        assert fused[0].confidence_2d == 0.9
        assert fused[0].confidence_3d == 0.85

    def test_unmatched_2d_preserved(
        self, simple_calib: CameraCalibration, sample_detection_2d: Detection
    ) -> None:
        """Unmatched 2D detections should be preserved."""
        fused = fuse_detections(
            [sample_detection_2d],
            [],  # No 3D detections
            simple_calib,
        )

        assert len(fused) == 1
        assert not fused[0].has_3d
        assert fused[0].class_name == "Car"

    def test_unmatched_3d_preserved(
        self, simple_calib: CameraCalibration, sample_detection_3d: Detection3D
    ) -> None:
        """Unmatched 3D detections should be preserved with projected 2D box."""
        fused = fuse_detections(
            [],  # No 2D detections
            [sample_detection_3d],
            simple_calib,
        )

        assert len(fused) == 1
        assert fused[0].has_3d
        assert fused[0].detection_3d == sample_detection_3d

    def test_class_match_required(self, simple_calib: CameraCalibration) -> None:
        """Test fusion with class_match_required=True."""
        det_2d = Detection(
            class_id=0,
            class_name="Car",
            bbox=BBox(x=0.5, y=0.5, width=0.2, height=0.15),
            confidence=0.9,
        )
        det_3d = Detection3D(
            class_id=1,
            class_name="Pedestrian",  # Different class
            center=(0.0, 0.0, 10.0),
            dimensions=(0.5, 0.5, 1.7),
            rotation=0.0,
            confidence=0.8,
        )

        fused = fuse_detections(
            [det_2d],
            [det_3d],
            simple_calib,
            iou_threshold=0.1,
            class_match_required=True,
        )

        # Should have 2 unmatched detections
        assert len(fused) == 2
        assert sum(1 for f in fused if f.has_3d) == 1
        assert sum(1 for f in fused if not f.has_3d) == 1

    def test_multiple_detections_greedy_matching(
        self, simple_calib: CameraCalibration
    ) -> None:
        """Test greedy matching with multiple detections."""
        # Two 2D detections
        det_2d_1 = Detection(
            class_id=0,
            class_name="Car",
            bbox=BBox(x=0.3, y=0.5, width=0.15, height=0.1),
            confidence=0.9,
        )
        det_2d_2 = Detection(
            class_id=0,
            class_name="Car",
            bbox=BBox(x=0.7, y=0.5, width=0.15, height=0.1),
            confidence=0.85,
        )

        # Two 3D detections at different positions
        det_3d_1 = Detection3D(
            class_id=0,
            class_name="Car",
            center=(-3.0, 0.0, 10.0),  # Projects to left
            dimensions=(4.5, 1.8, 1.5),
            rotation=0.0,
            confidence=0.88,
        )
        det_3d_2 = Detection3D(
            class_id=0,
            class_name="Car",
            center=(3.0, 0.0, 10.0),  # Projects to right
            dimensions=(4.5, 1.8, 1.5),
            rotation=0.0,
            confidence=0.82,
        )

        fused = fuse_detections(
            [det_2d_1, det_2d_2],
            [det_3d_1, det_3d_2],
            simple_calib,
            iou_threshold=0.1,
        )

        # All detections should result in annotations
        assert len(fused) >= 2

    def test_empty_inputs(self, simple_calib: CameraCalibration) -> None:
        """Test fusion with empty inputs."""
        fused = fuse_detections([], [], simple_calib)
        assert len(fused) == 0

    def test_depth_calculation(
        self,
        simple_calib: CameraCalibration,
        sample_detection_2d: Detection,
        sample_detection_3d: Detection3D,
    ) -> None:
        """Test that depth is calculated for matched detections."""
        fused = fuse_detections(
            [sample_detection_2d],
            [sample_detection_3d],
            simple_calib,
            iou_threshold=0.1,
        )

        assert len(fused) == 1
        # With identity transform, depth = z coordinate = 10.0
        assert fused[0].depth_meters is not None
        np.testing.assert_almost_equal(fused[0].depth_meters, 10.0, decimal=1)


class TestFusedAnnotation:
    """Tests for FusedAnnotation model."""

    def test_has_3d_property(
        self,
        simple_calib: CameraCalibration,
        sample_detection_2d: Detection,
        sample_detection_3d: Detection3D,
    ) -> None:
        """Test has_3d computed property."""
        fused_with_3d = fuse_detections(
            [sample_detection_2d],
            [sample_detection_3d],
            simple_calib,
            iou_threshold=0.1,
        )
        fused_without_3d = fuse_detections(
            [sample_detection_2d], [], simple_calib
        )

        assert fused_with_3d[0].has_3d is True
        assert fused_without_3d[0].has_3d is False

    def test_confidence_property(
        self,
        simple_calib: CameraCalibration,
        sample_detection_2d: Detection,
        sample_detection_3d: Detection3D,
    ) -> None:
        """Test combined confidence computation."""
        fused = fuse_detections(
            [sample_detection_2d],
            [sample_detection_3d],
            simple_calib,
            iou_threshold=0.1,
        )

        # Combined confidence = (0.9 + 0.85) / 2 = 0.875
        expected = (0.9 + 0.85) / 2
        np.testing.assert_almost_equal(fused[0].confidence, expected, decimal=3)
