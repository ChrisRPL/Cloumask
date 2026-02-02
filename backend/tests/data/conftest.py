"""
Shared test fixtures for ROS bag testing.

Creates mock ROS messages and parser components for testing
without requiring actual ROS bag files.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def temp_bag_path(tmp_path: Path) -> Path:
    """Create a temporary path for a fake bag file."""
    bag_path = tmp_path / "test.bag"
    # Create a file with ROS1 magic bytes
    with open(bag_path, "wb") as f:
        f.write(b"#ROSBAG V2.0\n")
        f.write(b"\x00" * 100)  # Padding
    return bag_path


@pytest.fixture
def temp_ros2_bag_path(tmp_path: Path) -> Path:
    """Create a temporary path for a fake ROS2 bag file."""
    bag_path = tmp_path / "test.db3"
    bag_path.touch()
    return bag_path


@pytest.fixture
def sample_pointcloud_msg() -> MagicMock:
    """Create a mock PointCloud2 message with XYZ and intensity data."""
    msg = MagicMock()
    msg.header = MagicMock()
    msg.header.frame_id = "velodyne"

    # Create fields for X, Y, Z, intensity
    field_x = MagicMock()
    field_x.name = "x"
    field_x.offset = 0
    field_x.datatype = 7  # FLOAT32
    field_x.count = 1

    field_y = MagicMock()
    field_y.name = "y"
    field_y.offset = 4
    field_y.datatype = 7
    field_y.count = 1

    field_z = MagicMock()
    field_z.name = "z"
    field_z.offset = 8
    field_z.datatype = 7
    field_z.count = 1

    field_i = MagicMock()
    field_i.name = "intensity"
    field_i.offset = 12
    field_i.datatype = 7
    field_i.count = 1

    msg.fields = [field_x, field_y, field_z, field_i]
    msg.point_step = 16
    msg.row_step = 160
    msg.height = 1
    msg.width = 10

    # Create binary point data (10 points)
    data = b""
    for i in range(10):
        data += struct.pack("<ffff", float(i), float(i) * 0.5, float(i) * 0.1, float(i) * 10)
    msg.data = data

    return msg


@pytest.fixture
def sample_pointcloud_rgb_msg() -> MagicMock:
    """Create a mock PointCloud2 message with XYZ and RGB data."""
    msg = MagicMock()
    msg.header = MagicMock()
    msg.header.frame_id = "lidar"

    # Create fields for X, Y, Z, rgb
    field_x = MagicMock()
    field_x.name = "x"
    field_x.offset = 0
    field_x.datatype = 7
    field_x.count = 1

    field_y = MagicMock()
    field_y.name = "y"
    field_y.offset = 4
    field_y.datatype = 7
    field_y.count = 1

    field_z = MagicMock()
    field_z.name = "z"
    field_z.offset = 8
    field_z.datatype = 7
    field_z.count = 1

    field_rgb = MagicMock()
    field_rgb.name = "rgb"
    field_rgb.offset = 12
    field_rgb.datatype = 6  # UINT32
    field_rgb.count = 1

    msg.fields = [field_x, field_y, field_z, field_rgb]
    msg.point_step = 16
    msg.row_step = 80
    msg.height = 1
    msg.width = 5

    # Create binary point data (5 points with RGB)
    data = b""
    for i in range(5):
        x, y, z = float(i), float(i) * 2, float(i) * 3
        # Pack RGB as uint32: R << 16 | G << 8 | B
        r, g, b = (i * 50) % 256, (i * 100) % 256, (i * 150) % 256
        rgb_packed = (r << 16) | (g << 8) | b
        data += struct.pack("<fffI", x, y, z, rgb_packed)
    msg.data = data

    return msg


@pytest.fixture
def sample_image_msg() -> MagicMock:
    """Create a mock Image message with RGB8 encoding."""
    msg = MagicMock()
    msg.header = MagicMock()
    msg.header.frame_id = "camera_front"
    msg.encoding = "rgb8"
    msg.width = 64
    msg.height = 48

    # Create RGB image data
    np.random.seed(42)
    image_data = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    msg.data = image_data.tobytes()

    return msg


@pytest.fixture
def sample_image_bgr_msg() -> MagicMock:
    """Create a mock Image message with BGR8 encoding."""
    msg = MagicMock()
    msg.header = MagicMock()
    msg.header.frame_id = "camera_rear"
    msg.encoding = "bgr8"
    msg.width = 32
    msg.height = 24

    np.random.seed(123)
    image_data = np.random.randint(0, 255, (24, 32, 3), dtype=np.uint8)
    msg.data = image_data.tobytes()

    return msg


@pytest.fixture
def sample_image_mono_msg() -> MagicMock:
    """Create a mock Image message with mono8 encoding."""
    msg = MagicMock()
    msg.header = MagicMock()
    msg.header.frame_id = "camera_ir"
    msg.encoding = "mono8"
    msg.width = 32
    msg.height = 24

    np.random.seed(456)
    image_data = np.random.randint(0, 255, (24, 32), dtype=np.uint8)
    msg.data = image_data.tobytes()

    return msg


@pytest.fixture
def sample_camera_info_msg() -> MagicMock:
    """Create a mock CameraInfo message."""
    msg = MagicMock()
    msg.header = MagicMock()
    msg.header.frame_id = "camera_front"
    msg.width = 640
    msg.height = 480
    msg.distortion_model = "plumb_bob"

    # Sample intrinsics (typical camera matrix)
    msg.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
    msg.d = [0.1, -0.2, 0.0, 0.0, 0.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    return msg


@pytest.fixture
def mock_anyreader_context() -> Generator[MagicMock, None, None]:
    """Mock the rosbags AnyReader for testing without actual bag files."""
    with patch("backend.data.rosbag_parser.AnyReader", autospec=True) as mock_reader_class:
        mock_reader = MagicMock()
        mock_reader_class.return_value.__enter__.return_value = mock_reader

        # Setup default values
        mock_reader.duration = 10_000_000_000  # 10 seconds in nanoseconds
        mock_reader.start_time = 1000_000_000_000
        mock_reader.end_time = 1010_000_000_000
        mock_reader.connections = []

        yield mock_reader


@pytest.fixture
def mock_connection_pointcloud() -> MagicMock:
    """Create a mock connection for PointCloud2 topic."""
    conn = MagicMock()
    conn.topic = "/velodyne_points"
    conn.msgtype = "sensor_msgs/msg/PointCloud2"
    return conn


@pytest.fixture
def mock_connection_image() -> MagicMock:
    """Create a mock connection for Image topic."""
    conn = MagicMock()
    conn.topic = "/camera/image_raw"
    conn.msgtype = "sensor_msgs/msg/Image"
    return conn


@pytest.fixture
def mock_connection_camera_info() -> MagicMock:
    """Create a mock connection for CameraInfo topic."""
    conn = MagicMock()
    conn.topic = "/camera/camera_info"
    conn.msgtype = "sensor_msgs/msg/CameraInfo"
    return conn
