# Point Cloud Agent Tools

> **Status:** 🔴 Not Started
> **Priority:** P1 (High)
> **Dependencies:** 02-agent-system (@tool decorator), all other 05-point-cloud sub-specs
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

LangGraph tool wrappers for all point cloud operations, enabling the conversational AI agent to process 3D data. Tools include: `parse_pointcloud`, `detect_3d`, `project_3d_to_2d`, `anonymize_pointcloud`, and `extract_rosbag`.

## Goals

- [ ] Implement `parse_pointcloud` tool for file inspection
- [ ] Implement `detect_3d` tool for 3D object detection
- [ ] Implement `project_3d_to_2d` tool for sensor fusion
- [ ] Implement `anonymize_pointcloud` tool for privacy
- [ ] Implement `extract_rosbag` tool for ROS data extraction
- [ ] Write comprehensive docstrings for LLM tool selection

## Technical Design

### Tool Pattern (from 02-agent-system)

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Base result for all point cloud tools."""
    success: bool
    message: str
    data: dict | None = None
```

### Tool Definitions

```python
# backend/agent/tools/pointcloud.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path

from cv.pointcloud import PointCloudProcessor
from data.rosbag_parser import RosbagParser


class PointCloudMetadata(BaseModel):
    """Metadata about a point cloud file."""
    path: str
    format: str
    point_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    has_colors: bool
    has_intensity: bool
    has_normals: bool
    file_size_mb: float
    attributes: list[str]


@tool
def parse_pointcloud(
    path: str = Field(description="Path to the point cloud file (PCD, PLY, LAS, LAZ)")
) -> PointCloudMetadata:
    """
    Parse and inspect a point cloud file without loading all points into memory.

    Returns metadata including:
    - Point count and bounding box
    - Available attributes (colors, intensity, normals)
    - File format and size

    Use this tool to understand a point cloud before processing.
    Example: "What's in /data/scan.pcd?"
    """
    processor = PointCloudProcessor()
    pcd = processor.load(path)
    stats = processor.get_stats(pcd)

    file_path = Path(path)
    return PointCloudMetadata(
        path=path,
        format=file_path.suffix.lower().strip('.'),
        point_count=stats.point_count,
        bounds_min=stats.bounds_min,
        bounds_max=stats.bounds_max,
        has_colors=stats.has_colors,
        has_intensity=stats.has_intensity,
        has_normals=stats.has_normals,
        file_size_mb=file_path.stat().st_size / (1024 * 1024),
        attributes=["XYZ"] +
                   (["RGB"] if stats.has_colors else []) +
                   (["intensity"] if stats.has_intensity else []) +
                   (["normals"] if stats.has_normals else []),
    )
```

```python
# backend/agent/tools/detect_3d.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

from cv.detection_3d import Detection3DModel, BBox3D, get_best_available_model


class Detection3DOutput(BaseModel):
    """Result of 3D object detection."""
    boxes: list[dict]  # List of BBox3D as dicts
    model_used: str
    inference_time_ms: float
    point_count: int
    summary: str  # Human-readable summary


@tool
def detect_3d(
    path: str = Field(description="Path to point cloud file"),
    model: Literal["auto", "pvrcnn++", "centerpoint"] = Field(
        default="auto",
        description="Detection model: 'auto' selects best for available GPU"
    ),
    confidence: float = Field(
        default=0.3,
        ge=0.0, le=1.0,
        description="Minimum confidence threshold for detections"
    ),
    classes: list[str] | None = Field(
        default=None,
        description="Filter to specific classes (e.g., ['Car', 'Pedestrian'])"
    )
) -> Detection3DOutput:
    """
    Detect 3D objects (cars, pedestrians, cyclists) in a point cloud.

    Returns 3D bounding boxes with:
    - Position (x, y, z center)
    - Dimensions (length, width, height)
    - Rotation (yaw angle)
    - Class and confidence score

    Supports autonomous driving datasets like KITTI and nuScenes.
    Example: "Find all cars in the lidar scan"
    Example: "Detect pedestrians with confidence > 0.5"
    """
    # Select model
    model_name = model if model != "auto" else get_best_available_model()

    # Load and run detection
    detector = Detection3DModel(model_name)

    # Load point cloud
    import numpy as np
    from cv.pointcloud import PointCloudProcessor
    processor = PointCloudProcessor()
    pcd = processor.load(path)
    points = np.asarray(pcd.points)

    # Add intensity if not present
    if points.shape[1] < 4:
        points = np.hstack([points, np.ones((len(points), 1))])

    # Run detection
    result = detector.detect(points, score_threshold=confidence)

    # Filter by class if specified
    boxes = result.boxes
    if classes:
        boxes = [b for b in boxes if b.class_name in classes]

    # Generate summary
    class_counts = {}
    for box in boxes:
        class_counts[box.class_name] = class_counts.get(box.class_name, 0) + 1

    summary_parts = [f"{count} {cls}" for cls, count in class_counts.items()]
    summary = f"Detected {', '.join(summary_parts)}" if summary_parts else "No objects detected"

    # Unload model to free memory
    detector.unload()

    return Detection3DOutput(
        boxes=[{
            "center": box.center,
            "dimensions": box.dimensions,
            "rotation": box.rotation,
            "class": box.class_name,
            "score": box.score,
        } for box in boxes],
        model_used=model_name,
        inference_time_ms=result.inference_time_ms,
        point_count=len(points),
        summary=summary,
    )
```

```python
# backend/agent/tools/fusion.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal


class ProjectionOutput(BaseModel):
    """Result of 3D to 2D projection."""
    boxes_2d: list[dict]  # {x_min, y_min, x_max, y_max, class, score}
    calibration_format: str
    image_size: tuple[int, int]
    visible_count: int
    total_count: int


@tool
def project_3d_to_2d(
    boxes_3d: list[dict] = Field(
        description="List of 3D boxes from detect_3d tool"
    ),
    calibration_path: str = Field(
        description="Path to calibration file (KITTI, nuScenes, or JSON format)"
    ),
    calibration_format: Literal["kitti", "nuscenes", "json"] = Field(
        default="kitti",
        description="Format of calibration file"
    ),
    image_path: str | None = Field(
        default=None,
        description="Optional image to overlay boxes on"
    )
) -> ProjectionOutput:
    """
    Project 3D bounding boxes to 2D image coordinates.

    Useful for:
    - Visualizing 3D detections on camera images
    - Creating 2D annotations from 3D labels
    - Multi-modal data analysis

    Requires calibration file with camera intrinsics and extrinsics.
    Example: "Project the detected cars onto the camera image"
    """
    from data.calibration import CameraCalibration
    from data.projection import project_bbox3d_to_2d
    from cv.detection_3d import BBox3D

    # Load calibration
    if calibration_format == "kitti":
        calib = CameraCalibration.from_kitti(calibration_path)
    elif calibration_format == "nuscenes":
        import json
        with open(calibration_path) as f:
            data = json.load(f)
        calib = CameraCalibration.from_nuscenes(data)
    else:
        # Custom JSON format
        import json
        with open(calibration_path) as f:
            data = json.load(f)
        calib = CameraCalibration(**data)

    # Project each box
    boxes_2d = []
    visible_count = 0

    for box_dict in boxes_3d:
        bbox_3d = BBox3D(
            center=tuple(box_dict["center"]),
            dimensions=tuple(box_dict["dimensions"]),
            rotation=box_dict["rotation"],
            class_name=box_dict["class"],
            score=box_dict["score"],
        )

        result = project_bbox3d_to_2d(bbox_3d, calib)
        if result is not None:
            x_min, y_min, x_max, y_max = result
            boxes_2d.append({
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "class": bbox_3d.class_name,
                "score": bbox_3d.score,
            })
            visible_count += 1

    return ProjectionOutput(
        boxes_2d=boxes_2d,
        calibration_format=calibration_format,
        image_size=(calib.width, calib.height),
        visible_count=visible_count,
        total_count=len(boxes_3d),
    )
```

```python
# backend/agent/tools/anonymize_3d.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal


class AnonymizationOutput(BaseModel):
    """Result of point cloud anonymization."""
    output_path: str
    original_points: int
    anonymized_points: int
    faces_found: int
    points_affected: int
    verified: bool
    mode: str


@tool
def anonymize_pointcloud(
    input_path: str = Field(description="Path to input point cloud"),
    output_path: str = Field(description="Path to save anonymized point cloud"),
    mode: Literal["remove", "noise"] = Field(
        default="remove",
        description="'remove' deletes face points, 'noise' adds random displacement"
    ),
    verify: bool = Field(
        default=True,
        description="Run verification to confirm no faces remain"
    )
) -> AnonymizationOutput:
    """
    Anonymize faces in a 3D point cloud for privacy compliance.

    Modes:
    - 'remove': Completely removes points in face regions (cleaner but sparse)
    - 'noise': Adds random noise to face points (preserves density)

    Verification re-checks the output to confirm no detectable faces remain.
    Example: "Anonymize the street scan for GDPR compliance"
    Example: "Remove faces from the point cloud before sharing"
    """
    from cv.anonymization_3d import PointCloudAnonymizer
    from cv.detection import FaceDetector  # SCRFD from 03-cv-models

    # Initialize detector and anonymizer
    face_detector = FaceDetector()
    anonymizer = PointCloudAnonymizer(face_detector, fusion_module=None)

    # Run anonymization
    result = anonymizer.anonymize(
        pcd_path=input_path,
        output_path=output_path,
        mode=mode,
        verify=verify,
    )

    return AnonymizationOutput(
        output_path=result.output_path,
        original_points=result.original_point_count,
        anonymized_points=result.anonymized_point_count,
        faces_found=result.face_regions_found,
        points_affected=result.points_removed + result.points_noised,
        verified=result.verification_passed,
        mode=mode,
    )
```

```python
# backend/agent/tools/rosbag.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal


class RosbagInfo(BaseModel):
    """Information about a ROS bag file."""
    path: str
    format: str  # "ros1" | "ros2"
    duration_sec: float
    topics: list[dict]  # {name, msg_type, count}
    pointcloud_topics: list[str]
    image_topics: list[str]


class RosbagExtractionOutput(BaseModel):
    """Result of ROS bag extraction."""
    output_dir: str
    frames_extracted: int
    pointcloud_files: list[str]
    image_files: list[str]
    sync_error_avg_ms: float


@tool
def extract_rosbag(
    bag_path: str = Field(description="Path to ROS bag file (.bag, .db3, .mcap)"),
    output_dir: str = Field(description="Directory to save extracted data"),
    pointcloud_topic: str | None = Field(
        default=None,
        description="PointCloud2 topic to extract (auto-detected if not specified)"
    ),
    image_topic: str | None = Field(
        default=None,
        description="Image topic to extract (auto-detected if not specified)"
    ),
    max_frames: int = Field(
        default=100,
        description="Maximum number of frames to extract"
    ),
    sync_sensors: bool = Field(
        default=True,
        description="Synchronize point clouds with camera images by timestamp"
    )
) -> RosbagExtractionOutput:
    """
    Extract point clouds and images from a ROS bag file.

    Supports:
    - ROS1 (.bag) and ROS2 (.db3, .mcap) formats
    - Automatic topic discovery for sensor data
    - Timestamp synchronization between LiDAR and camera

    Use this to convert ROS recordings to standard formats for processing.
    Example: "Extract 50 frames from the KITTI bag file"
    Example: "Get synced lidar and camera data from the rosbag"
    """
    from data.rosbag_parser import RosbagParser
    import os

    parser = RosbagParser(bag_path)
    bag_info = parser.get_info()

    # Auto-detect topics if not specified
    if pointcloud_topic is None:
        pc_topics = [t for t in bag_info.topics if "PointCloud2" in t.msg_type]
        pointcloud_topic = pc_topics[0].name if pc_topics else None

    if image_topic is None:
        img_topics = [t for t in bag_info.topics if "Image" in t.msg_type]
        image_topic = img_topics[0].name if img_topics else None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    pointcloud_files = []
    image_files = []
    sync_errors = []

    if sync_sensors and pointcloud_topic and image_topic:
        # Extract synchronized frames
        frames = parser.sync_frames(
            pointcloud_topic=pointcloud_topic,
            image_topic=image_topic,
            max_sync_error_ms=50.0,
        )

        for i, frame in enumerate(frames[:max_frames]):
            # Save point cloud
            pc_path = os.path.join(output_dir, f"frame_{i:06d}.pcd")
            # ... save logic
            pointcloud_files.append(pc_path)

            # Save image
            img_path = os.path.join(output_dir, f"frame_{i:06d}.png")
            # ... save logic
            image_files.append(img_path)

            sync_errors.append(frame.sync_error_ms)
    else:
        # Extract separately
        if pointcloud_topic:
            messages = parser.extract_pointcloud2(pointcloud_topic, max_frames)
            for i, msg in enumerate(messages):
                pc_path = os.path.join(output_dir, f"pc_{i:06d}.pcd")
                # ... save logic
                pointcloud_files.append(pc_path)

        if image_topic:
            messages = parser.extract_images(image_topic, max_frames)
            for i, msg in enumerate(messages):
                img_path = os.path.join(output_dir, f"img_{i:06d}.png")
                # ... save logic
                image_files.append(img_path)

    avg_sync_error = sum(sync_errors) / len(sync_errors) if sync_errors else 0.0

    return RosbagExtractionOutput(
        output_dir=output_dir,
        frames_extracted=len(pointcloud_files),
        pointcloud_files=pointcloud_files,
        image_files=image_files,
        sync_error_avg_ms=avg_sync_error,
    )
```

### Tool Registration

```python
# backend/agent/tools/__init__.py

from .pointcloud import parse_pointcloud
from .detect_3d import detect_3d
from .fusion import project_3d_to_2d
from .anonymize_3d import anonymize_pointcloud
from .rosbag import extract_rosbag

POINTCLOUD_TOOLS = [
    parse_pointcloud,
    detect_3d,
    project_3d_to_2d,
    anonymize_pointcloud,
    extract_rosbag,
]
```

### Prompt Guidance

```markdown
<!-- backend/agent/prompts/pointcloud_tools.md -->

# Point Cloud Tools

## When to use each tool:

### parse_pointcloud
Use when user wants to:
- Inspect a point cloud file
- Know how many points are in a scan
- Check available attributes (colors, intensity)
- Understand the spatial extent (bounding box)

### detect_3d
Use when user wants to:
- Find objects in a LiDAR scan
- Detect cars, pedestrians, cyclists
- Get 3D bounding boxes with positions and sizes
- Analyze autonomous driving data

### project_3d_to_2d
Use when user wants to:
- Show 3D detections on a camera image
- Create 2D labels from 3D annotations
- Fuse LiDAR and camera data
- Visualize sensor fusion results

### anonymize_pointcloud
Use when user wants to:
- Remove faces from street scans
- Prepare data for public release
- Comply with privacy regulations (GDPR)
- Anonymize pedestrian point clouds

### extract_rosbag
Use when user wants to:
- Convert ROS bag to standard formats
- Extract frames from a recording
- Get synchronized LiDAR+camera pairs
- Process autonomous driving datasets

## Tool chaining examples:

1. "Analyze the KITTI scan and find all cars"
   -> parse_pointcloud(path) -> detect_3d(path, classes=["Car"])

2. "Show detected objects on the camera"
   -> detect_3d(path) -> project_3d_to_2d(boxes, calib)

3. "Extract and anonymize the rosbag data"
   -> extract_rosbag(bag) -> anonymize_pointcloud(pcd)
```

## Implementation Tasks

- [ ] **Implement parse_pointcloud tool**
  - [ ] File format detection
  - [ ] Metadata extraction without full load
  - [ ] Comprehensive docstring

- [ ] **Implement detect_3d tool**
  - [ ] Model selection logic
  - [ ] Class filtering
  - [ ] Human-readable summary

- [ ] **Implement project_3d_to_2d tool**
  - [ ] Calibration format handling
  - [ ] Visibility filtering
  - [ ] Optional image overlay

- [ ] **Implement anonymize_pointcloud tool**
  - [ ] Mode selection
  - [ ] Verification flag
  - [ ] Progress reporting

- [ ] **Implement extract_rosbag tool**
  - [ ] Auto topic discovery
  - [ ] Sync vs async extraction
  - [ ] Output file naming

- [ ] **Create prompt guidance**
  - [ ] When to use each tool
  - [ ] Tool chaining examples
  - [ ] Common parameter patterns

- [ ] **Testing**
  - [ ] Unit tests for each tool
  - [ ] Integration tests with agent
  - [ ] Docstring validation

## Files to Create/Modify

| Path | Action | Purpose |
|------|--------|---------|
| `backend/agent/tools/pointcloud.py` | Create | parse_pointcloud tool |
| `backend/agent/tools/detect_3d.py` | Create | detect_3d tool |
| `backend/agent/tools/fusion.py` | Create | project_3d_to_2d tool |
| `backend/agent/tools/anonymize_3d.py` | Create | anonymize_pointcloud tool |
| `backend/agent/tools/rosbag.py` | Create | extract_rosbag tool |
| `backend/agent/tools/__init__.py` | Modify | Export POINTCLOUD_TOOLS |
| `backend/agent/prompts/pointcloud_tools.md` | Create | Tool selection guidance |
| `backend/tests/agent/test_pointcloud_tools.py` | Create | Unit tests |

## Acceptance Criteria

- [ ] Agent executes "parse /data/scan.pcd" and returns point count, bounds, attributes
- [ ] Agent executes "detect cars in the point cloud" and returns 3D bounding boxes
- [ ] Agent executes "project to camera" with 3D boxes and calibration
- [ ] Agent executes "anonymize the scan" and produces verified output
- [ ] Agent executes "extract frames from rosbag" with auto topic detection
- [ ] All tools return structured `BaseModel` objects, not raw strings
- [ ] Tool docstrings enable LLM to select appropriate tool for user requests
- [ ] `pytest tests/agent/test_pointcloud_tools.py -v` passes

## Testing Strategy

```python
import pytest
from agent.tools import (
    parse_pointcloud,
    detect_3d,
    project_3d_to_2d,
    anonymize_pointcloud,
    extract_rosbag,
)


def test_parse_pointcloud_docstring():
    """Tool docstrings should be descriptive for LLM."""
    assert "metadata" in parse_pointcloud.description.lower()
    assert "point count" in parse_pointcloud.description.lower()


def test_detect_3d_docstring():
    """Tool docstrings should mention supported classes."""
    assert "car" in detect_3d.description.lower()
    assert "pedestrian" in detect_3d.description.lower()


def test_tool_returns_pydantic():
    """All tools should return Pydantic models."""
    # Mock invocation and check return type
    pass


def test_tool_chain_detect_and_project():
    """Test typical tool chain."""
    # 1. Detect 3D objects
    # 2. Project to 2D
    # Verify data flows correctly
    pass
```

## Agent Integration Example

```python
# Example LangGraph node using these tools
from langgraph.graph import StateGraph
from agent.tools import POINTCLOUD_TOOLS

def create_pointcloud_agent():
    tools = POINTCLOUD_TOOLS

    # Create tool node
    tool_node = ToolNode(tools)

    # Create graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge("agent", "tools")
    workflow.add_conditional_edges("tools", should_continue)

    return workflow.compile()
```

## Related Sub-Specs

- [02-python-open3d.md](./02-python-open3d.md) - parse_pointcloud backend
- [03-rosbag-extraction.md](./03-rosbag-extraction.md) - extract_rosbag backend
- [04-3d-detection.md](./04-3d-detection.md) - detect_3d backend
- [05-2d-3d-fusion.md](./05-2d-3d-fusion.md) - project_3d_to_2d backend
- [07-anonymization-3d.md](./07-anonymization-3d.md) - anonymize_pointcloud backend
