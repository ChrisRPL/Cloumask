# Point Cloud Module

> **Status:** 🟢 Complete
> **Priority:** P1 (High)
> **Dependencies:** 01-foundation, 03-cv-models

## Overview

Support 3D point cloud data across the stack: Rust (pasture) for efficient I/O, Python (Open3D, OpenPCDet) for processing and ML, Frontend (Three.js) for visualization. Enable 2D-3D sensor fusion.

## Goals

- [ ] Read/write PCD, PLY, LAS/LAZ formats
- [ ] Parse ROS bags for PointCloud2 topics
- [ ] Three.js viewer with orbit controls
- [ ] 3D object detection (PV-RCNN++)
- [ ] Camera-LiDAR projection and fusion
- [ ] Point cloud anonymization

## Technical Design

### Stack Distribution

| Layer | Library | Responsibility |
|-------|---------|----------------|
| Rust | pasture | Fast I/O, streaming, format conversion |
| Python | Open3D | Geometry ops, downsampling, filtering |
| Python | PDAL | Complex format pipelines |
| Python | OpenPCDet | 3D object detection |
| Python | rosbag | ROS bag extraction |
| Frontend | Three.js | Visualization, interaction |

### Supported Formats

| Format | Read | Write | Library |
|--------|------|-------|---------|
| PCD | ✓ | ✓ | pasture, Open3D |
| PLY | ✓ | ✓ | pasture, Open3D |
| LAS/LAZ | ✓ | ✓ | pasture, laspy |
| E57 | ✓ | ✓ | PDAL |
| ROS bags | ✓ | - | rosbag |
| nuScenes | ✓ | ✓ | Custom |

### 2D-3D Fusion Pipeline
```
Camera Image + LiDAR Points
         │
         ▼
    Calibration (intrinsic + extrinsic)
         │
         ▼
    Timestamp Sync
         │
         ▼
    3D Detection (PV-RCNN++)
         │
         ▼
    Project 3D→2D
         │
         ▼
    Fused Labels (2D boxes + 3D boxes)
```

## Implementation Tasks

- [ ] **Rust I/O (pasture)**
  - [ ] PCD/PLY reader with streaming
  - [ ] LAS/LAZ support via pasture-io
  - [ ] Point cloud decimation
  - [ ] Format conversion commands

- [ ] **Python Processing (Open3D)**
  - [ ] Point cloud loading/saving
  - [ ] Downsampling (voxel, random)
  - [ ] Filtering (statistical, radius)
  - [ ] Normal estimation

- [ ] **ROS Bag Extraction**
  - [ ] Topic discovery
  - [ ] PointCloud2 extraction
  - [ ] Camera image extraction
  - [ ] Timestamp alignment

- [ ] **3D Detection (OpenPCDet)**
  - [ ] PV-RCNN++ integration
  - [ ] CenterPoint fallback
  - [ ] Inference pipeline
  - [ ] 3D bounding box output

- [ ] **2D-3D Fusion**
  - [ ] Calibration matrix handling
  - [ ] 3D→2D projection
  - [ ] 2D→3D lifting (with depth)
  - [ ] Multi-sensor sync

- [ ] **Frontend Viewer (Three.js)**
  - [ ] Point cloud rendering
  - [ ] Color by: intensity, height, RGB
  - [ ] 3D bounding box visualization
  - [ ] Orbit/pan/zoom controls
  - [ ] LOD for large clouds

- [ ] **Agent Tools**
  - [ ] `parse_pointcloud` tool
  - [ ] `detect_3d` tool
  - [ ] `project_3d_to_2d` tool
  - [ ] `anonymize_pointcloud` tool

## Acceptance Criteria

- [ ] Load 10M point PCD file in <2 seconds
- [ ] Three.js viewer renders with 60fps for 1M points
- [ ] PV-RCNN++ detects cars in KITTI sample
- [ ] 3D boxes project correctly to camera image
- [ ] ROS bag extraction produces synced frames
- [ ] Point cloud anonymization removes face points

## Files to Create/Modify

```
src-tauri/
├── src/
│   └── pointcloud/
│       ├── mod.rs          # Module exports
│       ├── io.rs           # pasture I/O
│       ├── convert.rs      # Format conversion
│       └── decimate.rs     # Downsampling

backend/
├── cv/
│   ├── pointcloud.py       # Open3D operations
│   └── detection_3d.py     # OpenPCDet wrapper
├── data/
│   ├── rosbag_parser.py    # ROS bag extraction
│   └── fusion.py           # 2D-3D fusion
├── agent/
│   └── tools/
│       ├── pointcloud.py   # parse_pointcloud
│       ├── detect_3d.py    # detect_3d
│       └── fusion.py       # project_3d_to_2d

src/lib/
├── components/
│   └── PointCloud/
│       ├── Viewer.svelte   # Main viewer
│       ├── Controls.svelte # UI controls
│       └── renderer.ts     # Three.js setup
```

## Sub-Specs

Detailed implementation specs for each component:

| # | Spec | Description | Status |
|---|------|-------------|--------|
| 1 | [01-rust-io.md](./01-rust-io.md) | Rust pasture I/O (PCD, PLY, LAS) | 🔴 Not Started |
| 2 | [02-python-open3d.md](./02-python-open3d.md) | Open3D geometry processing | 🔴 Not Started |
| 3 | [03-rosbag-extraction.md](./03-rosbag-extraction.md) | ROS bag parsing & extraction | 🔴 Not Started |
| 4 | [04-3d-detection.md](./04-3d-detection.md) | OpenPCDet (PV-RCNN++, CenterPoint) | 🔴 Not Started |
| 5 | [05-2d-3d-fusion.md](./05-2d-3d-fusion.md) | Camera-LiDAR sensor fusion | 🔴 Not Started |
| 6 | [06-threejs-viewer.md](./06-threejs-viewer.md) | Three.js frontend visualization | 🔴 Not Started |
| 7 | [07-anonymization-3d.md](./07-anonymization-3d.md) | 3D point cloud anonymization | 🔴 Not Started |
| 8 | [08-agent-tools.md](./08-agent-tools.md) | LangGraph agent tools | 🟢 Complete |

### Implementation Order

```
Phase 1 (parallel):  01-rust-io + 06-threejs-viewer
Phase 2:             02-python-open3d
Phase 3 (parallel):  03-rosbag-extraction + 04-3d-detection
Phase 4:             05-2d-3d-fusion
Phase 5:             07-anonymization-3d
Phase 6:             08-agent-tools
```

### Dependency Graph

```
01-foundation    03-cv-models
     │               │
     ▼               │
01-rust-io ──────────┤
     │               │
     ├───────────────┼──────────────┐
     │               │              │
     ▼               ▼              ▼
06-threejs    02-python-open3d    (external)
                    │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
03-rosbag-extract    04-3d-detection
         │                   │
         └─────────┬─────────┘
                   │
                   ▼
           05-2d-3d-fusion
                   │
                   ▼
         07-anonymization-3d
                   │
                   ▼
           08-agent-tools
```
