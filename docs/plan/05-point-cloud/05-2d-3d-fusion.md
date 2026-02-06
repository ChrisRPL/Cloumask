# 2D-3D Sensor Fusion

> **Status:** 🟢 Complete
> **Priority:** P1 (High)
> **Dependencies:** 03-rosbag-extraction, 04-3d-detection
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Implement camera-LiDAR sensor fusion pipeline including calibration matrix handling, 3D-to-2D point/box projection, 2D-to-3D detection lifting with depth lookup, and multi-sensor timestamp synchronization. Support KITTI, nuScenes, and custom calibration formats.

## Goals

- [ ] Parse calibration matrices from KITTI, nuScenes, ROS formats
- [ ] Project 3D points to 2D image pixels
- [ ] Project 3D bounding boxes to 2D image boxes
- [ ] Lift 2D detections to 3D using point cloud depth
- [ ] Handle camera distortion (radial + tangential)
- [ ] Create fused annotation format linking 2D and 3D labels

## Technical Design

### Calibration Data Structures

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class CameraCalibration:
    """Camera intrinsic and extrinsic calibration."""
    # Camera intrinsic matrix (3, 3)
    K: np.ndarray
    # Distortion coefficients [k1, k2, p1, p2, k3, ...]
    D: np.ndarray
    # Image dimensions
    width: int
    height: int
    # Camera to LiDAR transform (4, 4) - extrinsic
    T_cam_lidar: np.ndarray | None = None
    # Rectification matrix (3, 3) - for stereo
    R: np.ndarray | None = None
    # Projection matrix (3, 4) - for stereo
    P: np.ndarray | None = None

    @classmethod
    def from_kitti(cls, calib_path: str, camera_id: int = 2) -> "CameraCalibration":
        """Load calibration from KITTI format."""
        with open(calib_path, 'r') as f:
            lines = f.readlines()

        calib = {}
        for line in lines:
            key, values = line.strip().split(':')
            calib[key] = np.array([float(v) for v in values.split()])

        P = calib[f'P{camera_id}'].reshape(3, 4)
        K = P[:3, :3]

        # Velodyne to camera transform
        R0_rect = calib['R0_rect'].reshape(3, 3)
        Tr_velo_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
        T_cam_lidar = np.eye(4)
        T_cam_lidar[:3, :4] = R0_rect @ Tr_velo_cam

        return cls(
            K=K,
            D=np.zeros(5),  # KITTI images are already undistorted
            width=1242,
            height=375,
            T_cam_lidar=T_cam_lidar,
            P=P,
        )

    @classmethod
    def from_nuscenes(cls, sensor_data: dict) -> "CameraCalibration":
        """Load calibration from nuScenes format."""
        K = np.array(sensor_data['camera_intrinsic'])
        T_cam_lidar = np.array(sensor_data['lidar_to_camera'])

        return cls(
            K=K,
            D=np.zeros(5),
            width=1600,
            height=900,
            T_cam_lidar=T_cam_lidar,
        )

    @classmethod
    def from_ros_camera_info(cls, msg) -> "CameraCalibration":
        """Load calibration from ROS CameraInfo message."""
        return cls(
            K=np.array(msg.K).reshape(3, 3),
            D=np.array(msg.D),
            width=msg.width,
            height=msg.height,
            R=np.array(msg.R).reshape(3, 3) if msg.R else None,
            P=np.array(msg.P).reshape(3, 4) if msg.P else None,
        )


@dataclass
class FusedAnnotation:
    """Linked 2D and 3D annotation."""
    # 2D bounding box [x_min, y_min, x_max, y_max]
    bbox_2d: tuple[float, float, float, float]
    # 3D bounding box (from BBox3D)
    bbox_3d: dict  # center, dimensions, rotation
    # Object class
    class_name: str
    # Confidence score
    score: float
    # Track ID for multi-frame consistency
    track_id: int | None = None
    # Occlusion level (0-3)
    occlusion: int = 0
    # Truncation (0-1)
    truncation: float = 0.0
```

### Projection Functions

```python
def project_points_to_image(
    points_3d: np.ndarray,
    calib: CameraCalibration,
    filter_behind: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) array of 3D points in LiDAR frame
        calib: Camera calibration data
        filter_behind: Remove points behind camera

    Returns:
        points_2d: (M, 2) array of 2D pixel coordinates
        valid_mask: (N,) boolean mask of valid projections
    """
    # Transform to camera frame
    if calib.T_cam_lidar is not None:
        points_cam = transform_points(points_3d, calib.T_cam_lidar)
    else:
        points_cam = points_3d

    # Filter points behind camera
    valid_mask = points_cam[:, 2] > 0
    if filter_behind:
        points_cam = points_cam[valid_mask]

    # Apply distortion correction if needed
    if calib.D is not None and np.any(calib.D != 0):
        points_2d = undistort_points(points_cam, calib.K, calib.D)
    else:
        # Simple pinhole projection
        points_2d = points_cam[:, :2] / points_cam[:, 2:3]
        points_2d = (calib.K[:2, :2] @ points_2d.T + calib.K[:2, 2:3]).T

    # Filter points outside image
    in_image = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < calib.width) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < calib.height)
    )

    return points_2d, valid_mask & in_image


def project_bbox3d_to_2d(
    bbox_3d: "BBox3D",
    calib: CameraCalibration
) -> tuple[float, float, float, float] | None:
    """
    Project 3D bounding box to 2D image box.

    Args:
        bbox_3d: 3D bounding box
        calib: Camera calibration data

    Returns:
        (x_min, y_min, x_max, y_max) or None if not visible
    """
    # Get 8 corner points
    corners_3d = bbox_3d.to_corners()

    # Project corners
    corners_2d, valid = project_points_to_image(corners_3d, calib)

    if not np.any(valid):
        return None

    # Get bounding rectangle
    x_min = float(corners_2d[:, 0].min())
    y_min = float(corners_2d[:, 1].min())
    x_max = float(corners_2d[:, 0].max())
    y_max = float(corners_2d[:, 1].max())

    # Clip to image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(calib.width, x_max)
    y_max = min(calib.height, y_max)

    return (x_min, y_min, x_max, y_max)


def lift_2d_to_3d(
    bbox_2d: tuple[float, float, float, float],
    points_3d: np.ndarray,
    calib: CameraCalibration,
    class_prior: dict | None = None
) -> "BBox3D":
    """
    Lift 2D bounding box to 3D using point cloud depth.

    Args:
        bbox_2d: (x_min, y_min, x_max, y_max) in pixels
        points_3d: (N, 3) point cloud in LiDAR frame
        calib: Camera calibration data
        class_prior: Optional class-specific size priors

    Returns:
        BBox3D with estimated 3D position and dimensions
    """
    # Project all points to image
    points_2d, valid = project_points_to_image(points_3d, calib)

    # Find points inside 2D box
    x_min, y_min, x_max, y_max = bbox_2d
    in_box = (
        (points_2d[:, 0] >= x_min) & (points_2d[:, 0] <= x_max) &
        (points_2d[:, 1] >= y_min) & (points_2d[:, 1] <= y_max)
    )

    box_points = points_3d[valid][in_box]

    if len(box_points) < 3:
        # Not enough points, use image center and default depth
        return None

    # Estimate center as median of points
    center = np.median(box_points, axis=0)

    # Estimate dimensions from point spread or class prior
    if class_prior:
        dimensions = class_prior.get("dimensions", (4.0, 1.8, 1.5))
    else:
        point_range = box_points.max(axis=0) - box_points.min(axis=0)
        dimensions = tuple(point_range)

    return BBox3D(
        center=tuple(center),
        dimensions=dimensions,
        rotation=0.0,  # Would need PCA or learned orientation
        class_name="unknown",
        score=0.5,
    )
```

## Implementation Tasks

- [ ] **Implement calibration parsers**
  - [ ] KITTI format parser
  - [ ] nuScenes format parser
  - [ ] ROS CameraInfo parser
  - [ ] Custom JSON format support

- [ ] **Implement 3D→2D projection**
  - [ ] Point projection with pinhole model
  - [ ] Distortion handling (radial + tangential)
  - [ ] Bounding box projection
  - [ ] Visibility checking

- [ ] **Implement 2D→3D lifting**
  - [ ] Point cloud frustum filtering
  - [ ] Depth-based position estimation
  - [ ] Class-specific size priors
  - [ ] Orientation estimation (PCA or learned)

- [ ] **Implement fusion pipeline**
  - [ ] Match 2D and 3D detections by IoU
  - [ ] Create fused annotations
  - [ ] Handle occluded objects

- [ ] **FastAPI integration**
  - [ ] Create routes/fusion.py
  - [ ] Projection endpoint
  - [ ] Lifting endpoint
  - [ ] Fusion endpoint

## Files to Create/Modify

| Path | Action | Purpose |
|------|--------|---------|
| `backend/data/calibration.py` | Create | Calibration parsers |
| `backend/data/fusion.py` | Create | Fusion pipeline |
| `backend/data/projection.py` | Create | 3D↔2D projection functions |
| `backend/data/formats/fused_annotation.py` | Create | Fused annotation format |
| `backend/api/routes/fusion.py` | Create | FastAPI endpoints |
| `backend/tests/data/test_fusion.py` | Create | Unit tests |
| `backend/tests/data/test_calibration.py` | Create | Calibration tests |

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/fusion/project_points` | Project 3D points to 2D |
| POST | `/fusion/project_boxes` | Project 3D boxes to 2D |
| POST | `/fusion/lift_boxes` | Lift 2D boxes to 3D |
| POST | `/fusion/fuse` | Create fused annotations |
| POST | `/fusion/load_calibration` | Load calibration file |

### Request/Response Models

```python
class ProjectPointsRequest(BaseModel):
    points_path: str  # Path to point cloud file
    calibration_path: str
    calibration_format: str = "kitti"  # "kitti" | "nuscenes" | "ros"

class ProjectBoxesRequest(BaseModel):
    boxes_3d: list[dict]  # List of BBox3D as dicts
    calibration_path: str
    calibration_format: str = "kitti"

class LiftBoxesRequest(BaseModel):
    boxes_2d: list[tuple[float, float, float, float]]
    points_path: str
    calibration_path: str
    class_priors: dict | None = None

class FuseRequest(BaseModel):
    boxes_2d: list[dict]  # 2D detections with scores
    boxes_3d: list[dict]  # 3D detections (BBox3D)
    calibration_path: str
    iou_threshold: float = 0.3
```

## Acceptance Criteria

- [ ] 3D bounding box corners project within 5px of ground truth
- [ ] 2D detections lifted to 3D are within 0.5m of actual position
- [ ] Parse calibration from KITTI, nuScenes, and ROS formats
- [ ] Handle radial distortion correctly (verify with checkerboard)
- [ ] Fused output JSON contains linked 2D box, 3D box, and track ID
- [ ] Projection runs in <10ms for 100K points
- [ ] `pytest tests/data/test_fusion.py -v` passes

## Testing Strategy

```python
import pytest
import numpy as np
from data.calibration import CameraCalibration
from data.projection import project_points_to_image, project_bbox3d_to_2d


@pytest.fixture
def kitti_calib():
    """Load sample KITTI calibration."""
    return CameraCalibration.from_kitti("fixtures/kitti_calib.txt")


def test_point_projection(kitti_calib):
    # Known test point
    points_3d = np.array([[10.0, 0.0, 0.0]])  # 10m ahead
    points_2d, valid = project_points_to_image(points_3d, kitti_calib)

    # Should project near image center
    assert valid[0]
    assert 500 < points_2d[0, 0] < 700
    assert 150 < points_2d[0, 1] < 250


def test_bbox_projection(kitti_calib):
    from cv.detection_3d import BBox3D

    bbox = BBox3D(
        center=(20.0, 0.0, -1.0),
        dimensions=(4.5, 1.8, 1.5),
        rotation=0.0,
        class_name="Car",
        score=0.9
    )

    bbox_2d = project_bbox3d_to_2d(bbox, kitti_calib)
    assert bbox_2d is not None
    x_min, y_min, x_max, y_max = bbox_2d
    assert x_max > x_min
    assert y_max > y_min


def test_calibration_roundtrip():
    """Test that projection and lifting are consistent."""
    # Project a known 3D point, lift it back
    pass
```

## Calibration Format Examples

### KITTI Format
```
P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 ...
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 ...
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 ...
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 ...
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 ...
```

### nuScenes Format (JSON)
```json
{
  "camera_intrinsic": [[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1]],
  "lidar_to_camera": [[0.01, -1.0, 0.0, 0.0], ...]
}
```

### ROS CameraInfo
```yaml
width: 1920
height: 1080
K: [1500, 0, 960, 0, 1500, 540, 0, 0, 1]
D: [-0.1, 0.05, 0, 0, 0]
```

## Performance Considerations

- Use vectorized numpy operations for batch projection
- Cache calibration matrices (don't reload for each frame)
- Consider using OpenCV's `projectPoints` for distortion
- Pre-filter points by depth before full projection
- Use KD-tree for efficient frustum queries in lifting

## Related Sub-Specs

- [03-rosbag-extraction.md](./03-rosbag-extraction.md) - Source of synced frames
- [04-3d-detection.md](./04-3d-detection.md) - 3D boxes to project
- [07-anonymization-3d.md](./07-anonymization-3d.md) - Uses projection for face detection
- [08-agent-tools.md](./08-agent-tools.md) - project_3d_to_2d agent tool
