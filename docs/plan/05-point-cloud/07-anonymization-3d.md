# 3D Point Cloud Anonymization

> **Status:** 🔴 Not Started
> **Priority:** P1 (High)
> **Dependencies:** 03-cv-models (SCRFD), 05-2d-3d-fusion, 02-python-open3d
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Remove or mask points belonging to faces and persons in 3D point clouds. Uses 2D face detection in projected virtual views, then lifts detected regions back to 3D to identify and remove/noise the corresponding points.

## Goals

- [ ] Project point cloud to multiple virtual camera views
- [ ] Detect faces in projected 2D views using SCRFD
- [ ] Lift 2D face regions back to 3D point indices
- [ ] Remove face points from point cloud (clean anonymization)
- [ ] Add noise to face points (preserving structure)
- [ ] Validate no faces remain in anonymized output

## Technical Design

### Anonymization Pipeline

```
Point Cloud (N points)
         │
         ▼
┌────────────────────┐
│ Project to Virtual │  Generate 4-8 views (front, sides, etc.)
│   Camera Views     │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  SCRFD Face Det    │  Detect faces in each 2D view
│   (per view)       │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  Lift 2D Boxes     │  Map face bboxes back to 3D point indices
│    to 3D           │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  Anonymize Points  │  Remove or add noise to face points
│   (mode)           │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│   Verification     │  Re-project and check no faces detected
│     Pass           │
└────────────────────┘
         │
         ▼
Anonymized Point Cloud
```

### Virtual Camera Generation

```python
import numpy as np


def generate_virtual_cameras(
    bounds: tuple[np.ndarray, np.ndarray],
    num_views: int = 8,
    resolution: tuple[int, int] = (640, 480)
) -> list[dict]:
    """
    Generate virtual camera poses around the point cloud.

    Args:
        bounds: (min_xyz, max_xyz) of point cloud
        num_views: Number of viewpoints
        resolution: Image resolution (width, height)

    Returns:
        List of camera configs with intrinsics and extrinsics
    """
    min_xyz, max_xyz = bounds
    center = (min_xyz + max_xyz) / 2
    size = np.max(max_xyz - min_xyz)
    radius = size * 1.5

    cameras = []

    for i in range(num_views):
        # Distribute cameras around the scene
        angle = 2 * np.pi * i / num_views
        height_offset = size * 0.3 * (i % 2 - 0.5)  # Alternate heights

        cam_pos = np.array([
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            center[2] + height_offset
        ])

        # Camera looks at center
        look_at = center
        up = np.array([0, 0, 1])

        # Build view matrix
        z_axis = (cam_pos - look_at) / np.linalg.norm(cam_pos - look_at)
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        view_matrix = np.eye(4)
        view_matrix[:3, 0] = x_axis
        view_matrix[:3, 1] = y_axis
        view_matrix[:3, 2] = z_axis
        view_matrix[:3, 3] = cam_pos

        # Intrinsics (simple pinhole)
        fx = fy = resolution[0]  # ~90 degree FOV
        cx, cy = resolution[0] / 2, resolution[1] / 2
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        cameras.append({
            'K': K,
            'view_matrix': view_matrix,
            'position': cam_pos,
            'resolution': resolution,
        })

    return cameras
```

### Anonymization Class

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import open3d as o3d


@dataclass
class AnonymizationResult:
    """Result of point cloud anonymization."""
    output_path: str
    original_point_count: int
    anonymized_point_count: int
    face_regions_found: int
    points_removed: int
    points_noised: int
    verification_passed: bool
    processing_time_ms: float


class PointCloudAnonymizer:
    """Anonymize faces in 3D point clouds."""

    def __init__(
        self,
        face_detector,  # SCRFD face detector from 03-cv-models
        fusion_module,  # From 05-2d-3d-fusion
    ):
        self.face_detector = face_detector
        self.fusion = fusion_module

    def anonymize(
        self,
        pcd_path: str | Path,
        output_path: str | Path,
        mode: str = "remove",  # "remove" | "noise" | "blur"
        num_views: int = 8,
        face_margin: float = 1.2,  # Expand face regions by 20%
        noise_sigma: float = 0.1,  # Noise magnitude for "noise" mode
        verify: bool = True,
    ) -> AnonymizationResult:
        """
        Anonymize faces in point cloud.

        Args:
            pcd_path: Input point cloud path
            output_path: Output path for anonymized cloud
            mode: Anonymization mode
            num_views: Number of virtual camera views
            face_margin: Factor to expand face bounding boxes
            noise_sigma: Noise standard deviation for "noise" mode
            verify: Run verification pass

        Returns:
            AnonymizationResult with statistics
        """
        import time
        start_time = time.time()

        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
        original_count = len(points)

        # Get bounds
        bounds = (points.min(axis=0), points.max(axis=0))

        # Generate virtual cameras
        cameras = generate_virtual_cameras(bounds, num_views)

        # Detect faces in each view and collect point indices
        face_point_indices = set()
        face_regions_found = 0

        for cam in cameras:
            # Project points to this view
            points_2d = self._project_points(points, cam)

            # Create depth image for rendering
            depth_image = self._render_depth(points, cam)

            # Detect faces in the rendered view
            face_boxes = self.face_detector.detect(depth_image)
            face_regions_found += len(face_boxes)

            # Lift each face box back to 3D
            for box in face_boxes:
                # Expand box by margin
                expanded_box = self._expand_box(box, face_margin)

                # Find points inside 2D box
                in_box = self._points_in_box(points_2d, expanded_box)
                face_point_indices.update(np.where(in_box)[0])

        # Apply anonymization
        face_indices = np.array(list(face_point_indices))

        if mode == "remove":
            # Remove face points
            mask = np.ones(len(points), dtype=bool)
            mask[face_indices] = False
            new_points = points[mask]
            points_removed = len(face_indices)
            points_noised = 0

            if pcd.has_colors():
                new_colors = np.asarray(pcd.colors)[mask]

        elif mode == "noise":
            # Add noise to face points
            new_points = points.copy()
            noise = np.random.normal(0, noise_sigma, (len(face_indices), 3))
            new_points[face_indices] += noise
            points_removed = 0
            points_noised = len(face_indices)
            new_colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        # Create output point cloud
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_points)
        if new_colors is not None:
            new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

        # Save
        o3d.io.write_point_cloud(str(output_path), new_pcd)

        # Verification pass
        verification_passed = True
        if verify:
            verification_passed = self._verify(new_pcd, num_views)

        processing_time = (time.time() - start_time) * 1000

        return AnonymizationResult(
            output_path=str(output_path),
            original_point_count=original_count,
            anonymized_point_count=len(new_points),
            face_regions_found=face_regions_found,
            points_removed=points_removed,
            points_noised=points_noised,
            verification_passed=verification_passed,
            processing_time_ms=processing_time,
        )

    def _project_points(
        self,
        points: np.ndarray,
        camera: dict
    ) -> np.ndarray:
        """Project 3D points to 2D using camera parameters."""
        # Transform to camera frame
        view = np.linalg.inv(camera['view_matrix'])
        points_cam = (view[:3, :3] @ points.T + view[:3, 3:4]).T

        # Project using intrinsics
        K = camera['K']
        points_2d = (K @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]

        return points_2d

    def _render_depth(
        self,
        points: np.ndarray,
        camera: dict
    ) -> np.ndarray:
        """Render depth image for face detection."""
        # Simple point splatting for visualization
        W, H = camera['resolution']
        depth = np.zeros((H, W), dtype=np.float32)

        points_2d = self._project_points(points, camera)

        # Get depth values
        view = np.linalg.inv(camera['view_matrix'])
        points_cam = (view[:3, :3] @ points.T + view[:3, 3:4]).T
        depths = points_cam[:, 2]

        # Splat points to image
        valid = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H) &
            (depths > 0)
        )

        u = points_2d[valid, 0].astype(int)
        v = points_2d[valid, 1].astype(int)
        depth[v, u] = np.maximum(depth[v, u], 1.0 / depths[valid])

        # Normalize to 0-255 for face detector
        depth_normalized = (depth / depth.max() * 255).astype(np.uint8)
        return np.stack([depth_normalized] * 3, axis=-1)  # RGB format

    def _verify(self, pcd: o3d.geometry.PointCloud, num_views: int) -> bool:
        """Verify no faces detected in anonymized cloud."""
        points = np.asarray(pcd.points)
        bounds = (points.min(axis=0), points.max(axis=0))
        cameras = generate_virtual_cameras(bounds, num_views)

        for cam in cameras:
            depth_image = self._render_depth(points, cam)
            faces = self.face_detector.detect(depth_image)
            if len(faces) > 0:
                return False

        return True
```

## Implementation Tasks

- [ ] **Implement virtual camera generation**
  - [ ] Multiple viewpoints around scene
  - [ ] Configurable resolution and FOV
  - [ ] Coverage validation

- [ ] **Implement depth rendering**
  - [ ] Point splatting
  - [ ] Surface reconstruction (optional)
  - [ ] Intensity/color visualization

- [ ] **Integrate face detection**
  - [ ] SCRFD detector from 03-cv-models
  - [ ] Batch processing for multiple views
  - [ ] Confidence thresholding

- [ ] **Implement 3D lifting**
  - [ ] 2D box to 3D point indices
  - [ ] Margin expansion for safety
  - [ ] Frustum-based selection

- [ ] **Implement anonymization modes**
  - [ ] Point removal (clean)
  - [ ] Point noise addition
  - [ ] Point displacement (blur)

- [ ] **Implement verification**
  - [ ] Re-project and detect
  - [ ] Coverage metrics
  - [ ] Report generation

- [ ] **FastAPI integration**
  - [ ] Create routes/anonymize_3d.py
  - [ ] Progress streaming
  - [ ] Batch endpoint

## Files to Create/Modify

| Path | Action | Purpose |
|------|--------|---------|
| `backend/cv/anonymization_3d.py` | Create | Main anonymizer class |
| `backend/cv/face_3d_projection.py` | Create | 2D↔3D face region mapping |
| `backend/cv/virtual_camera.py` | Create | Virtual camera generation |
| `backend/api/routes/anonymize_3d.py` | Create | FastAPI endpoints |
| `backend/tests/cv/test_anonymization_3d.py` | Create | Unit tests |

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/anonymize3d/process` | Anonymize point cloud |
| GET | `/anonymize3d/status/{job_id}` | Get job status |
| POST | `/anonymize3d/verify` | Verify anonymization |

### Request/Response Models

```python
class Anonymize3DRequest(BaseModel):
    input_path: str
    output_path: str
    mode: str = "remove"  # "remove" | "noise"
    num_views: int = 8
    face_margin: float = 1.2
    verify: bool = True

class Anonymize3DResponse(BaseModel):
    output_path: str
    original_point_count: int
    anonymized_point_count: int
    face_regions_found: int
    points_removed: int
    verification_passed: bool
    processing_time_ms: float
```

## Acceptance Criteria

- [ ] Face points successfully removed from point cloud (0 points in face regions)
- [ ] Non-face points preserved (scene structure maintained)
- [ ] Processing completes in <2 seconds for 100K point cloud
- [ ] Verification pass confirms no faces detectable in output
- [ ] Multiple viewpoints ensure 360-degree coverage
- [ ] "noise" mode preserves approximate geometry
- [ ] `pytest tests/cv/test_anonymization_3d.py -v` passes

## Testing Strategy

```python
import pytest
import numpy as np
import open3d as o3d
from cv.anonymization_3d import PointCloudAnonymizer, generate_virtual_cameras


@pytest.fixture
def sample_pcd_with_face(tmp_path):
    """Create point cloud with face-like cluster."""
    # Random scene points
    scene = np.random.rand(10000, 3) * 10

    # Face-like cluster (small sphere at head height)
    face_center = np.array([5, 5, 1.7])  # Head height
    face_points = np.random.randn(500, 3) * 0.15 + face_center

    all_points = np.vstack([scene, face_points])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    path = tmp_path / "with_face.pcd"
    o3d.io.write_point_cloud(str(path), pcd)
    return path


def test_virtual_cameras():
    bounds = (np.array([0, 0, 0]), np.array([10, 10, 3]))
    cameras = generate_virtual_cameras(bounds, num_views=8)

    assert len(cameras) == 8
    for cam in cameras:
        assert 'K' in cam
        assert 'view_matrix' in cam


def test_anonymization_removes_faces(sample_pcd_with_face, tmp_path, face_detector):
    anonymizer = PointCloudAnonymizer(face_detector, None)

    output_path = tmp_path / "anonymized.pcd"
    result = anonymizer.anonymize(
        sample_pcd_with_face,
        output_path,
        mode="remove",
        verify=True
    )

    assert result.face_regions_found > 0
    assert result.points_removed > 0
    assert result.verification_passed
```

## Privacy Considerations

- Always verify anonymization before sharing data
- Consider multiple viewpoints for complete coverage
- Face detection confidence threshold affects recall/precision trade-off
- Body detection (future) may be needed for full anonymization
- GDPR compliance may require additional validation steps

## Performance Considerations

- Virtual camera generation is fast (<1ms)
- Depth rendering scales with point count (optimize with spatial indexing)
- Face detection runs per view (batch for GPU efficiency)
- Point index lookup can be optimized with KD-tree
- Consider parallel view processing

## Related Sub-Specs

- [02-python-open3d.md](./02-python-open3d.md) - Point cloud loading/saving
- [03-cv-models (SCRFD)](../03-cv-models/SPEC.md) - Face detection
- [05-2d-3d-fusion.md](./05-2d-3d-fusion.md) - Projection utilities
- [08-agent-tools.md](./08-agent-tools.md) - anonymize_pointcloud tool
