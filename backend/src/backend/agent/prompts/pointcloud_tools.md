# Point Cloud Tools

## When to use each tool:

### pointcloud_stats
Use when user wants to:
- Inspect a point cloud file
- Know how many points are in a scan
- Check available attributes (colors, intensity, normals)
- Understand the spatial extent (bounding box)
- Determine the file format before processing

### process_pointcloud
Use when user wants to:
- Downsample a large point cloud (voxel or random)
- Remove outlier points (statistical or radius filtering)
- Estimate surface normals
- Reduce file size before other operations

### detect_3d
Use when user wants to:
- Find objects in a LiDAR scan
- Detect cars, pedestrians, cyclists in point clouds
- Get 3D bounding boxes with positions, sizes, and rotations
- Analyze autonomous driving data (KITTI, nuScenes, Waymo)

### project_3d_to_2d
Use when user wants to:
- Show 3D detections on a camera image
- Create 2D labels from 3D annotations
- Fuse LiDAR and camera data
- Visualize sensor fusion results
- Verify sensor calibration alignment

### anonymize_pointcloud
Use when user wants to:
- Remove faces from street scans
- Prepare point cloud data for public release
- Comply with privacy regulations (GDPR)
- Anonymize pedestrian data in 3D scans
- Choose between removing face points or adding noise

### extract_rosbag
Use when user wants to:
- Convert ROS bag files to standard formats
- Extract frames from a recording
- Get synchronized LiDAR+camera pairs
- Process autonomous driving recordings
- Inspect ROS bag contents and topics

## Tool chaining examples:

### 1. "Analyze the KITTI scan and find all cars"
```
pointcloud_stats(path) -> detect_3d(path, classes=["Car"])
```

### 2. "Show detected objects on the camera"
```
detect_3d(path) -> project_3d_to_2d(calibration, detections)
```

### 3. "Extract and anonymize the rosbag data"
```
extract_rosbag(bag_path, output_dir) -> anonymize_pointcloud(pcd_path, output)
```

### 4. "Downsample the scan, then detect objects"
```
process_pointcloud(input, output, "voxel", voxel_size=0.1) -> detect_3d(output)
```

### 5. "Extract frames and detect pedestrians with camera overlay"
```
extract_rosbag(bag, out) -> detect_3d(pc, classes=["Pedestrian"]) -> project_3d_to_2d(calib, dets)
```

## Common parameter patterns:

- For KITTI data: coordinate_system="kitti", calibration_format="kitti"
- For nuScenes data: coordinate_system="nuscenes", calibration_format="nuscenes"
- For privacy tasks: mode="remove" (cleaner) or mode="noise" (preserves density)
- For large point clouds: start with pointcloud_stats to check size, then consider downsampling
- For ROS bags: use sync_sensors=True when both LiDAR and camera data are needed
