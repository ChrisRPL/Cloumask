# ROS Bag Extraction

> **Status:** 🟢 Complete
> **Priority:** P1 (High)
> **Dependencies:** 02-python-open3d
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Parse ROS1 (.bag) and ROS2 (.db3/.mcap) bag files to extract sensor data. Discover topics, extract PointCloud2 and Image messages, and align timestamps across multiple sensors for synchronized multi-modal data processing.

## Goals

- [ ] Discover all topics with message types and counts
- [ ] Extract `sensor_msgs/PointCloud2` to numpy/Open3D format
- [ ] Extract `sensor_msgs/Image` with associated `CameraInfo`
- [ ] Align timestamps across sensors (nearest-neighbor + interpolation)
- [ ] Support ROS1 (.bag) and ROS2 (.db3, .mcap) formats
- [ ] Create synchronized frame iterator for multi-sensor workflows

## Technical Design

### Dependencies

```txt
# requirements.txt
rosbags>=0.9.0          # Pure Python ROS bag reader (ROS1 + ROS2)
rosbags-dataframe>=0.9.0  # DataFrame utilities
numpy>=1.24.0
opencv-python>=4.8.0
```

### Data Structures

```python
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class TopicInfo:
    """Information about a ROS topic."""
    name: str
    msg_type: str
    message_count: int
    frequency_hz: float | None = None


@dataclass
class BagInfo:
    """Metadata about a ROS bag."""
    path: str
    format: str  # "ros1" | "ros2"
    duration_sec: float
    start_time: float
    end_time: float
    topics: list[TopicInfo]


@dataclass
class PointCloud2Message:
    """Extracted PointCloud2 data."""
    timestamp: float
    frame_id: str
    points: np.ndarray  # (N, 3) float64
    intensity: np.ndarray | None = None  # (N,) float32
    rgb: np.ndarray | None = None  # (N, 3) uint8
    fields: list[str] = field(default_factory=list)


@dataclass
class ImageMessage:
    """Extracted Image data."""
    timestamp: float
    frame_id: str
    image: np.ndarray  # (H, W, C) uint8
    encoding: str
    width: int
    height: int


@dataclass
class CameraInfoMessage:
    """Extracted CameraInfo data."""
    timestamp: float
    frame_id: str
    width: int
    height: int
    K: np.ndarray  # (3, 3) intrinsic matrix
    D: np.ndarray  # distortion coefficients
    R: np.ndarray  # (3, 3) rectification matrix
    P: np.ndarray  # (3, 4) projection matrix


@dataclass
class SyncedFrame:
    """Synchronized multi-sensor frame."""
    timestamp: float
    pointcloud: PointCloud2Message | None
    image: ImageMessage | None
    camera_info: CameraInfoMessage | None
    sync_error_ms: float  # Max timestamp difference in ms
```

### RosbagParser Class

```python
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


class RosbagParser:
    """Parse ROS1 and ROS2 bags."""

    def __init__(self, bag_path: str | Path):
        self.bag_path = Path(bag_path)
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)

    def get_info(self) -> BagInfo:
        """Get bag metadata and topic list."""
        with AnyReader([self.bag_path]) as reader:
            topics = []
            for conn in reader.connections:
                count = sum(1 for _ in reader.messages([conn]))
                topics.append(TopicInfo(
                    name=conn.topic,
                    msg_type=conn.msgtype,
                    message_count=count,
                ))
            return BagInfo(
                path=str(self.bag_path),
                format=self._detect_format(),
                duration_sec=reader.duration / 1e9,
                start_time=reader.start_time / 1e9,
                end_time=reader.end_time / 1e9,
                topics=topics,
            )

    def extract_pointcloud2(
        self,
        topic: str,
        max_messages: int | None = None
    ) -> list[PointCloud2Message]:
        """Extract PointCloud2 messages from topic."""
        messages = []
        with AnyReader([self.bag_path]) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            for i, (conn, timestamp, rawdata) in enumerate(reader.messages(connections)):
                if max_messages and i >= max_messages:
                    break
                msg = reader.deserialize(rawdata, conn.msgtype)
                messages.append(self._convert_pointcloud2(msg, timestamp))
        return messages

    def extract_images(
        self,
        topic: str,
        max_messages: int | None = None
    ) -> list[ImageMessage]:
        """Extract Image messages from topic."""
        messages = []
        with AnyReader([self.bag_path]) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            for i, (conn, timestamp, rawdata) in enumerate(reader.messages(connections)):
                if max_messages and i >= max_messages:
                    break
                msg = reader.deserialize(rawdata, conn.msgtype)
                messages.append(self._convert_image(msg, timestamp))
        return messages

    def sync_frames(
        self,
        pointcloud_topic: str,
        image_topic: str,
        camera_info_topic: str | None = None,
        max_sync_error_ms: float = 50.0
    ) -> list[SyncedFrame]:
        """Create synchronized frames from multiple topics."""
        pointclouds = self.extract_pointcloud2(pointcloud_topic)
        images = self.extract_images(image_topic)
        camera_infos = (
            self.extract_camera_info(camera_info_topic)
            if camera_info_topic else []
        )

        return self._nearest_neighbor_sync(
            pointclouds, images, camera_infos, max_sync_error_ms
        )

    def _convert_pointcloud2(self, msg, timestamp: int) -> PointCloud2Message:
        """Convert ROS PointCloud2 to numpy arrays."""
        # Parse point cloud fields
        import struct
        points = []
        intensity = []
        # ... field parsing logic
        return PointCloud2Message(
            timestamp=timestamp / 1e9,
            frame_id=msg.header.frame_id,
            points=np.array(points),
            intensity=np.array(intensity) if intensity else None,
            fields=[f.name for f in msg.fields],
        )

    def _nearest_neighbor_sync(
        self,
        pointclouds: list[PointCloud2Message],
        images: list[ImageMessage],
        camera_infos: list[CameraInfoMessage],
        max_error_ms: float
    ) -> list[SyncedFrame]:
        """Synchronize by nearest timestamp."""
        frames = []
        for pc in pointclouds:
            # Find nearest image
            nearest_img = min(images, key=lambda i: abs(i.timestamp - pc.timestamp))
            error_ms = abs(nearest_img.timestamp - pc.timestamp) * 1000

            if error_ms <= max_error_ms:
                frames.append(SyncedFrame(
                    timestamp=pc.timestamp,
                    pointcloud=pc,
                    image=nearest_img,
                    camera_info=None,  # Find nearest camera_info similarly
                    sync_error_ms=error_ms,
                ))
        return frames
```

## Implementation Tasks

- [ ] **Setup rosbags library**
  - [ ] Add rosbags to requirements.txt
  - [ ] Create data/rosbag_parser.py module
  - [ ] Test with sample ROS1 and ROS2 bags

- [ ] **Implement bag inspection**
  - [ ] get_info() for topic discovery
  - [ ] Format detection (ROS1 vs ROS2)
  - [ ] Duration and timestamp extraction

- [ ] **Implement PointCloud2 extraction**
  - [ ] Parse binary point cloud data
  - [ ] Handle common field layouts (XYZ, XYZI, XYZRGB)
  - [ ] Convert to numpy/Open3D format

- [ ] **Implement Image extraction**
  - [ ] Handle common encodings (rgb8, bgr8, mono8, 16UC1)
  - [ ] Convert to numpy array
  - [ ] Extract compressed images if present

- [ ] **Implement CameraInfo extraction**
  - [ ] Parse intrinsic matrix K
  - [ ] Parse distortion coefficients D
  - [ ] Parse projection matrix P

- [ ] **Implement timestamp synchronization**
  - [ ] Nearest-neighbor matching
  - [ ] Configurable max sync error
  - [ ] Interpolation for smoother sync (optional)

- [ ] **FastAPI integration**
  - [ ] Create routes/rosbag.py
  - [ ] Endpoint for bag info
  - [ ] Endpoint for frame extraction
  - [ ] Streaming endpoint for large bags

## Files to Create/Modify

| Path | Action | Purpose |
|------|--------|---------|
| `backend/requirements.txt` | Modify | Add rosbags dependencies |
| `backend/data/rosbag_parser.py` | Create | Main ROS bag parser |
| `backend/data/ros_types.py` | Create | PointCloud2, Image converters |
| `backend/data/__init__.py` | Modify | Export rosbag modules |
| `backend/api/routes/rosbag.py` | Create | FastAPI endpoints |
| `backend/api/main.py` | Modify | Register rosbag router |
| `backend/tests/data/test_rosbag.py` | Create | Unit tests |

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/rosbag/info` | Get bag metadata and topics |
| GET | `/rosbag/topics` | List all topics with types |
| POST | `/rosbag/extract/pointcloud` | Extract point clouds |
| POST | `/rosbag/extract/images` | Extract camera images |
| POST | `/rosbag/extract/synced` | Extract synchronized frames |

### Request/Response Models

```python
class BagInfoRequest(BaseModel):
    path: str

class ExtractPointcloudRequest(BaseModel):
    bag_path: str
    topic: str
    output_dir: str
    max_frames: int | None = None
    output_format: str = "pcd"  # "pcd" | "ply" | "npy"

class ExtractSyncedRequest(BaseModel):
    bag_path: str
    pointcloud_topic: str
    image_topic: str
    camera_info_topic: str | None = None
    output_dir: str
    max_frames: int | None = None
    max_sync_error_ms: float = 50.0
```

## Acceptance Criteria

- [ ] List all topics from a KITTI/nuScenes .bag file with message counts
- [ ] Extract 100 synced lidar+camera frames in <30 seconds
- [ ] Timestamp alignment error <50ms between sensors
- [ ] Handle ROS1 (.bag) format without ROS installation
- [ ] Handle ROS2 (.db3, .mcap) format without ROS installation
- [ ] Extract point clouds with all available fields (XYZ, intensity, RGB)
- [ ] Images extracted with correct color space (RGB not BGR)
- [ ] `pytest tests/data/test_rosbag.py -v` passes

## Testing Strategy

```python
import pytest
from data.rosbag_parser import RosbagParser


@pytest.fixture
def sample_ros1_bag(tmp_path):
    """Download or create sample ROS1 bag for testing."""
    # Use a small sample bag from KITTI or create synthetic
    return tmp_path / "sample.bag"


def test_bag_info(sample_ros1_bag):
    parser = RosbagParser(sample_ros1_bag)
    info = parser.get_info()
    assert info.duration_sec > 0
    assert len(info.topics) > 0


def test_pointcloud_extraction(sample_ros1_bag):
    parser = RosbagParser(sample_ros1_bag)
    info = parser.get_info()
    pc_topics = [t for t in info.topics if "PointCloud2" in t.msg_type]
    if pc_topics:
        messages = parser.extract_pointcloud2(pc_topics[0].name, max_messages=5)
        assert len(messages) <= 5
        assert messages[0].points.shape[1] == 3


def test_sync_frames(sample_ros1_bag):
    parser = RosbagParser(sample_ros1_bag)
    frames = parser.sync_frames(
        pointcloud_topic="/velodyne_points",
        image_topic="/camera/image_raw",
        max_sync_error_ms=50.0
    )
    for frame in frames:
        assert frame.sync_error_ms <= 50.0
```

## Common ROS Topic Patterns

| Dataset | PointCloud2 Topic | Image Topic | CameraInfo Topic |
|---------|-------------------|-------------|------------------|
| KITTI | `/velodyne_points` | `/camera2/image_raw` | `/camera2/camera_info` |
| nuScenes | `/lidar_top` | `/cam_front/image_raw` | `/cam_front/camera_info` |
| Waymo | `/lidar/top/pointcloud` | `/camera/front/image` | `/camera/front/camera_info` |
| Custom | `/points` | `/image` | `/camera_info` |

## Performance Considerations

- Use memory mapping for large bag files
- Extract messages lazily (generator pattern)
- Cache deserialized typestore for repeated access
- Consider parallel extraction for multi-topic bags
- Stream extracted frames to disk for memory efficiency

## Related Sub-Specs

- [02-python-open3d.md](./02-python-open3d.md) - Point cloud processing
- [05-2d-3d-fusion.md](./05-2d-3d-fusion.md) - Uses synced frames for fusion
- [08-agent-tools.md](./08-agent-tools.md) - extract_rosbag agent tool
