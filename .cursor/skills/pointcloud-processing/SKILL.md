---
name: pointcloud-processing
description: Guide for working with 3D point cloud data in Cloumask including I/O, processing, 2D-3D fusion, and visualization. Use when processing point clouds, projecting between 2D and 3D, or working with multi-sensor data.
---

# Point Cloud Processing

## Quick Start

When working with point clouds:

1. **Rust side**: Use `pasture` library for I/O (PCD, PLY, LAS/LAZ)
2. **Python side**: Use `Open3D` for processing and `OpenPCDet` for 3D detection
3. **Projection**: Use `backend.data.projection` for 2D-3D fusion
4. **Frontend**: Use Three.js with point cloud state management

## Point Cloud I/O (Rust)

Point clouds are read/written using the `pasture` library:

```rust
use crate::pointcloud::io::{read_pointcloud, read_metadata};
use crate::pointcloud::types::{PointCloudData, PointCloudMetadata};

// Read metadata (fast, no point loading)
let metadata = read_metadata("cloud.pcd")?;

// Read full point cloud (for small files)
let cloud = read_pointcloud("cloud.pcd")?;

// Stream large files
stream_pointcloud(app, "large_cloud.laz", config).await?;
```

## Supported Formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| PCD | ✓ | ✓ | Point Cloud Data (ASCII/binary) |
| PLY | ✓ | ✓ | Polygon File Format |
| LAS/LAZ | ✓ | ✓ | LiDAR standard, compressed |
| ROS bags | ✓ | - | Extract PointCloud2 topics |
| E57 | ✓ | ✓ | ASTM scanner format |

## Point Cloud Processing (Python)

Use `Open3D` for processing operations:

```python
import open3d as o3d
from backend.cv.pointcloud import PointCloudProcessor

# Load point cloud
pcd = o3d.io.read_point_cloud("cloud.pcd")

# Processing operations
pcd = pcd.voxel_down_sample(voxel_size=0.05)  # Decimation
pcd = pcd.remove_statistical_outlier(20, 2.0)  # Noise removal
pcd = pcd.estimate_normals()  # Normal estimation

# Save
o3d.io.write_point_cloud("processed.pcd", pcd)
```

## 3D Object Detection

Use `OpenPCDet` wrappers for 3D detection:

```python
from backend.cv.detection_3d import get_3d_detector, Detection3D

# Get detector (PV-RCNN++ or CenterPoint)
detector = get_3d_detector()

# Load point cloud
from backend.cv.detection_3d import PointCloudLoader
loader = PointCloudLoader()
points = loader.load("cloud.pcd")  # Returns (N, 4) array [x, y, z, intensity]

# Run detection
detections: list[Detection3D] = detector.predict(points)

# Each detection has:
# - center: (x, y, z)
# - dimensions: (l, w, h)
# - rotation: yaw angle
# - class_name: "car", "pedestrian", etc.
# - confidence: float
```

## 2D-3D Projection

Project 3D points to 2D image coordinates:

```python
from backend.data.projection import project_points_to_image
from backend.data.calibration import CameraCalibration

# Load calibration
calib = CameraCalibration.from_file("calibration.json")

# Project 3D points to 2D
points_3d = np.array([[10, 5, 20], [15, 8, 25]])  # (N, 3) in LiDAR frame
points_2d, valid = project_points_to_image(points_3d, calib)

# points_2d: (N, 2) pixel coordinates (u, v)
# valid: (N,) boolean mask
```

## 3D Bounding Box to 2D

Project 3D detection boxes to 2D image boxes:

```python
from backend.data.projection import project_bbox3d_to_2d

# Project 3D box to 2D
bbox_2d = project_bbox3d_to_2d(detection_3d, calib)

# Returns (x_min, y_min, x_max, y_max) or None if not visible
if bbox_2d:
    x_min, y_min, x_max, y_max = bbox_2d
```

## 2D to 3D Lifting

Lift 2D detections to 3D using point cloud depth:

```python
from backend.data.projection import lift_2d_to_3d

# Lift 2D box to 3D
bbox_2d = (100, 200, 300, 400)  # (x_min, y_min, x_max, y_max)
detection_3d = lift_2d_to_3d(
    bbox_2d,
    points_3d,  # (N, 3+) point cloud
    calib,
    class_name="car",
    min_points=3,  # Minimum points in frustum
)

# Returns Detection3D or None if insufficient points
```

## Sensor Fusion

Fuse 2D and 3D detections:

```python
from backend.data.fusion import fuse_detections, FusedAnnotation

# Fuse detections
fused = fuse_detections(
    detections_2d,      # List of 2D Detection
    detections_3d,      # List of 3D Detection3D
    calib,
    iou_threshold=0.3,  # IoU threshold for matching
    class_match_required=False,  # Require class match
)

# Returns list of FusedAnnotation with:
# - detection_2d: 2D detection (or None)
# - detection_3d: 3D detection (or None)
# - match_confidence: float
```

## Camera Calibration

Load and use camera calibration:

```python
from backend.data.calibration import CameraCalibration

# Load from file
calib = CameraCalibration.from_file("calibration.json")

# Or create manually
calib = CameraCalibration(
    K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),  # Intrinsic matrix
    D=np.array([k1, k2, p1, p2, k3]),  # Distortion coefficients
    width=1920,
    height=1080,
    T_cam_lidar=T,  # 4x4 transformation matrix (optional)
)

# Check if distortion present
if calib.has_distortion:
    # Uses OpenCV projection with distortion
    ...
```

## ROS Bag Parsing

Extract point clouds from ROS bags:

```python
from backend.data.rosbag_parser import RosbagParser

parser = RosbagParser("bag_file.bag")

# Extract PointCloud2 messages
pointclouds = parser.extract_pointcloud(
    topic="/velodyne_points",
    output_dir="output/",
    start_time=0.0,
    end_time=100.0,
)

# Returns list of extracted file paths
```

## Coordinate Systems

Handle coordinate system conversions:

```python
from backend.cv.detection_3d import convert_coordinates, CoordinateSystem

# Convert between coordinate systems
points_kitti = convert_coordinates(
    points_lidar,
    from_system=CoordinateSystem.LIDAR,
    to_system=CoordinateSystem.KITTI,
)

# Supported systems:
# - LIDAR: Standard LiDAR frame (x forward, y left, z up)
# - KITTI: KITTI dataset frame
# - NUSCENES: nuScenes dataset frame
```

## Point Cloud Anonymization

Remove points near detected faces/plates:

```python
from backend.cv.pointcloud import PointCloudProcessor

processor = PointCloudProcessor()

# Anonymize point cloud
result = processor.anonymize(
    pointcloud_path="cloud.pcd",
    detection_results=face_detections,  # List of 2D detections
    calib=calib,
    radius_m=2.0,  # Remove points within radius
)

# Returns anonymized point cloud path
```

## Frontend Visualization

Point clouds are visualized using Three.js:

```typescript
import { getPointCloudState } from '$lib/stores/pointcloud.svelte';

// Get state
const state = getPointCloudState();

// Load point cloud
state.setFile({
    name: 'cloud.pcd',
    path: '/path/to/cloud.pcd',
    format: 'pcd',
    pointCount: 1000000,
    sizeBytes: 100000000,
    bounds: {
        min: { x: -50, y: -50, z: -5 },
        max: { x: 50, y: 50, z: 5 },
    },
});

// Update visualization
state.setColorMode('height');  // 'height' | 'intensity' | 'rgb' | 'classification'
state.setPointSize(2);
state.setShowBoundingBoxes(true);
```

## Streaming Large Files

For large point clouds, use streaming:

```rust
// Rust: Stream point cloud
#[tauri::command]
pub async fn stream_pointcloud(
    app: AppHandle,
    path: String,
    config: Option<StreamConfig>,
) -> Result<PointCloudMetadata, String> {
    // Emit chunks asynchronously
    tokio::spawn(async move {
        for chunk in stream_chunks(&path).await {
            app.emit("pointcloud:chunk", chunk).unwrap();
        }
        app.emit("pointcloud:complete", ()).unwrap();
    });
    
    Ok(metadata)
}
```

```typescript
// Frontend: Listen for chunks
import { listen } from '@tauri-apps/api/event';

const unlisten = await listen<PointCloudChunk>('pointcloud:chunk', (event) => {
    // Add chunk to viewer
    viewer.addChunk(event.payload);
});
```

## Common Patterns

### Decimation

Reduce point count for faster processing:

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("cloud.pcd")
pcd_down = pcd.voxel_down_sample(voxel_size=0.1)  # 10cm voxels
o3d.io.write_point_cloud("decimated.pcd", pcd_down)
```

### Frustum Query

Find points inside 2D bounding box:

```python
from backend.data.projection import project_points_to_image

points_2d, valid = project_points_to_image(points_3d, calib)
in_box = (
    valid
    & (points_2d[:, 0] >= x_min)
    & (points_2d[:, 0] <= x_max)
    & (points_2d[:, 1] >= y_min)
    & (points_2d[:, 1] <= y_max)
)
frustum_points = points_3d[in_box]
```

### Multi-Sensor Sync

Synchronize camera and LiDAR timestamps:

```python
from backend.data.rosbag_parser import RosbagParser

parser = RosbagParser("bag.bag")

# Extract synced frames
synced = parser.extract_synced_frames(
    image_topic="/camera/image_raw",
    pointcloud_topic="/velodyne/points",
    output_dir="synced/",
    max_time_diff=0.1,  # 100ms tolerance
)
```

## Best Practices

1. **Use streaming for large files** - Don't load entire point cloud into memory
2. **Validate calibration** - Check calibration before projection
3. **Handle coordinate systems** - Be explicit about coordinate frames
4. **Filter points early** - Remove invalid points before processing
5. **Use appropriate voxel size** - Balance quality vs performance
6. **Check point counts** - Validate minimum points for operations
7. **Handle edge cases** - Empty clouds, invalid projections, etc.

## Additional Resources

- See `src-tauri/src/pointcloud/` for Rust I/O
- See `backend/src/backend/data/projection.py` for projection functions
- See `backend/src/backend/data/fusion.py` for sensor fusion
- See `src/lib/stores/pointcloud.svelte.ts` for frontend state
