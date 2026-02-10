# Point Cloud Module

> **Status:** 🟢 Complete (Implemented; release QA backlog remains)
> **Priority:** P1 (High)
> **Dependencies:** 01-foundation, 03-cv-models

## Overview

Support 3D point cloud data across the stack: Rust (pasture) for efficient I/O, Python (Open3D, OpenPCDet) for processing and ML, Frontend (Three.js) for visualization. Enable 2D-3D sensor fusion.

## Goals

- [x] Read/write PCD, PLY, LAS/LAZ formats
- [x] Parse ROS bags for PointCloud2 topics
- [x] Three.js viewer with orbit controls
- [x] 3D object detection (PV-RCNN++)
- [x] Camera-LiDAR projection and fusion
- [x] Point cloud anonymization

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

- [x] **Rust I/O (pasture)**
  - [x] PCD/PLY reader with streaming
  - [x] LAS/LAZ support via pasture-io
  - [x] Point cloud decimation
  - [x] Format conversion commands

- [x] **Python Processing (Open3D)**
  - [x] Point cloud loading/saving
  - [x] Downsampling (voxel, random)
  - [x] Filtering (statistical, radius)
  - [x] Normal estimation

- [x] **ROS Bag Extraction**
  - [x] Topic discovery
  - [x] PointCloud2 extraction
  - [x] Camera image extraction
  - [x] Timestamp alignment

- [x] **3D Detection (OpenPCDet)**
  - [x] PV-RCNN++ integration
  - [x] CenterPoint fallback
  - [x] Inference pipeline
  - [x] 3D bounding box output

- [x] **2D-3D Fusion**
  - [x] Calibration matrix handling
  - [x] 3D→2D projection
  - [x] 2D→3D lifting (with depth)
  - [x] Multi-sensor sync

- [x] **Frontend Viewer (Three.js)**
  - [x] Point cloud rendering
  - [x] Color by: intensity, height, RGB
  - [x] 3D bounding box visualization
  - [x] Orbit/pan/zoom controls
  - [x] LOD for large clouds

- [x] **Agent Tools**
  - [x] `parse_pointcloud` tool
  - [x] `detect_3d` tool
  - [x] `project_3d_to_2d` tool
  - [x] `anonymize_pointcloud` tool

## Acceptance Criteria

- [x] Load 10M point PCD file in <2 seconds
- [x] Three.js viewer renders with 60fps for 1M points
- [x] PV-RCNN++ detects cars in KITTI sample
- [x] 3D boxes project correctly to camera image
- [x] ROS bag extraction produces synced frames
- [x] Point cloud anonymization removes face points

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
| 1 | [01-rust-io.md](./01-rust-io.md) | Rust pasture I/O (PCD, PLY, LAS) | 🟢 Complete |
| 2 | [02-python-open3d.md](./02-python-open3d.md) | Open3D geometry processing | 🟢 Complete |
| 3 | [03-rosbag-extraction.md](./03-rosbag-extraction.md) | ROS bag parsing & extraction | 🟢 Complete |
| 4 | [04-3d-detection.md](./04-3d-detection.md) | OpenPCDet (PV-RCNN++, CenterPoint) | 🟢 Complete |
| 5 | [05-2d-3d-fusion.md](./05-2d-3d-fusion.md) | Camera-LiDAR sensor fusion | 🟢 Complete |
| 6 | [06-threejs-viewer.md](./06-threejs-viewer.md) | Three.js frontend visualization | 🟢 Complete |
| 7 | [07-anonymization-3d.md](./07-anonymization-3d.md) | 3D point cloud anonymization | 🟢 Complete |
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
