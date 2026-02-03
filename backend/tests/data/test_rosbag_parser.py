"""
Unit tests for RosbagParser class.

Tests parser initialization, format detection, and message conversion
using mocked rosbags library.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.data.ros_types import BagFormat
from backend.data.rosbag_parser import RosbagParseError, RosbagParser

if TYPE_CHECKING:
    pass


def _has_rosbags() -> bool:
    """Check if rosbags library is available for testing."""
    try:
        import rosbags.highlevel  # noqa: F401

        return True
    except ImportError:
        return False


class TestRosbagParserInit:
    """Tests for RosbagParser initialization."""

    def test_init_file_not_found(self, tmp_path: Path) -> None:
        """Parser raises FileNotFoundError for missing file."""
        fake_path = tmp_path / "nonexistent.bag"
        with pytest.raises(FileNotFoundError, match="Bag file not found"):
            RosbagParser(fake_path)

    def test_init_ros1_format(self, temp_bag_path: Path) -> None:
        """Parser detects ROS1 format from .bag extension."""
        parser = RosbagParser(temp_bag_path)
        assert parser.format == BagFormat.ROS1

    def test_init_ros2_format(self, temp_ros2_bag_path: Path) -> None:
        """Parser detects ROS2 format from .db3 extension."""
        parser = RosbagParser(temp_ros2_bag_path)
        assert parser.format == BagFormat.ROS2

    def test_init_unsupported_format(self, tmp_path: Path) -> None:
        """Parser raises ValueError for unsupported format."""
        fake_path = tmp_path / "test.txt"
        fake_path.touch()
        with pytest.raises(ValueError, match="Unsupported bag format"):
            RosbagParser(fake_path)

    def test_init_ros1_magic_bytes(self, tmp_path: Path) -> None:
        """Parser detects ROS1 from magic bytes."""
        # Create file with unknown extension but ROS1 magic bytes
        bag_path = tmp_path / "test.unknown"
        with bag_path.open("wb") as f:
            f.write(b"#ROSBAG V2.0\n")
            f.write(b"\x00" * 100)
        parser = RosbagParser(bag_path)
        assert parser.format == BagFormat.ROS1


class TestPointCloud2Conversion:
    """Tests for PointCloud2 message conversion."""

    def test_convert_xyz(self, temp_bag_path: Path, sample_pointcloud_msg: MagicMock) -> None:
        """Converts XYZ coordinates correctly."""
        parser = RosbagParser(temp_bag_path)
        timestamp_ns = 1_000_000_000_000  # 1000 seconds

        result = parser._convert_pointcloud2(sample_pointcloud_msg, timestamp_ns)

        assert result.timestamp == 1000.0
        assert result.frame_id == "velodyne"
        assert result.points.shape == (10, 3)
        assert result.point_count == 10
        # Verify first point coordinates
        np.testing.assert_almost_equal(result.points[0], [0.0, 0.0, 0.0])
        np.testing.assert_almost_equal(result.points[1], [1.0, 0.5, 0.1])

    def test_convert_intensity(
        self, temp_bag_path: Path, sample_pointcloud_msg: MagicMock
    ) -> None:
        """Converts intensity values correctly."""
        parser = RosbagParser(temp_bag_path)
        result = parser._convert_pointcloud2(sample_pointcloud_msg, 0)

        assert result.intensity is not None
        assert result.intensity.shape == (10,)
        assert result.intensity[0] == 0.0
        assert result.intensity[1] == 10.0

    def test_convert_rgb(
        self, temp_bag_path: Path, sample_pointcloud_rgb_msg: MagicMock
    ) -> None:
        """Converts RGB colors correctly."""
        parser = RosbagParser(temp_bag_path)
        result = parser._convert_pointcloud2(sample_pointcloud_rgb_msg, 0)

        assert result.rgb is not None
        assert result.rgb.shape == (5, 3)
        assert result.rgb.dtype == np.uint8

    def test_fields_list(self, temp_bag_path: Path, sample_pointcloud_msg: MagicMock) -> None:
        """Captures field names from message."""
        parser = RosbagParser(temp_bag_path)
        result = parser._convert_pointcloud2(sample_pointcloud_msg, 0)

        assert "x" in result.fields
        assert "y" in result.fields
        assert "z" in result.fields
        assert "intensity" in result.fields


class TestImageConversion:
    """Tests for Image message conversion."""

    def test_convert_rgb8(self, temp_bag_path: Path, sample_image_msg: MagicMock) -> None:
        """Converts RGB8 image correctly."""
        parser = RosbagParser(temp_bag_path)
        timestamp_ns = 2_000_000_000_000

        result = parser._convert_image(sample_image_msg, timestamp_ns)

        assert result.timestamp == 2000.0
        assert result.frame_id == "camera_front"
        assert result.encoding == "rgb8"
        assert result.width == 64
        assert result.height == 48
        assert result.image.shape == (48, 64, 3)

    def test_convert_bgr8(self, temp_bag_path: Path, sample_image_bgr_msg: MagicMock) -> None:
        """Converts BGR8 image to RGB correctly."""
        parser = RosbagParser(temp_bag_path)
        result = parser._convert_image(sample_image_bgr_msg, 0)

        # Image should be converted to RGB
        assert result.image.shape == (24, 32, 3)

    def test_convert_mono8(self, temp_bag_path: Path, sample_image_mono_msg: MagicMock) -> None:
        """Converts mono8 image to 3-channel correctly."""
        parser = RosbagParser(temp_bag_path)
        result = parser._convert_image(sample_image_mono_msg, 0)

        # Mono should be expanded to 3 channels
        assert result.image.shape == (24, 32, 3)
        # All channels should be equal for mono image
        assert np.array_equal(result.image[:, :, 0], result.image[:, :, 1])
        assert np.array_equal(result.image[:, :, 1], result.image[:, :, 2])


class TestCameraInfoConversion:
    """Tests for CameraInfo message conversion."""

    def test_convert_camera_info(
        self, temp_bag_path: Path, sample_camera_info_msg: MagicMock
    ) -> None:
        """Converts CameraInfo message correctly."""
        parser = RosbagParser(temp_bag_path)
        timestamp_ns = 3_000_000_000_000

        result = parser._convert_camera_info(sample_camera_info_msg, timestamp_ns)

        assert result.timestamp == 3000.0
        assert result.frame_id == "camera_front"
        assert result.width == 640
        assert result.height == 480
        assert result.distortion_model == "plumb_bob"
        assert result.K.shape == (3, 3)
        assert result.D.shape == (5,)
        assert result.R.shape == (3, 3)
        assert result.P.shape == (3, 4)

    def test_intrinsic_matrix(
        self, temp_bag_path: Path, sample_camera_info_msg: MagicMock
    ) -> None:
        """Intrinsic matrix K is parsed correctly."""
        parser = RosbagParser(temp_bag_path)
        result = parser._convert_camera_info(sample_camera_info_msg, 0)

        # Check focal lengths and principal point
        assert result.K[0, 0] == 500.0  # fx
        assert result.K[1, 1] == 500.0  # fy
        assert result.K[0, 2] == 320.0  # cx
        assert result.K[1, 2] == 240.0  # cy


class TestTopicFiltering:
    """Tests for topic type filtering."""

    def test_pointcloud_types(self, temp_bag_path: Path) -> None:
        """POINTCLOUD2_TYPES contains expected message types."""
        assert "sensor_msgs/msg/PointCloud2" in RosbagParser.POINTCLOUD2_TYPES
        assert "sensor_msgs/PointCloud2" in RosbagParser.POINTCLOUD2_TYPES

    def test_image_types(self, temp_bag_path: Path) -> None:
        """IMAGE_TYPES contains expected message types."""
        assert "sensor_msgs/msg/Image" in RosbagParser.IMAGE_TYPES
        assert "sensor_msgs/msg/CompressedImage" in RosbagParser.IMAGE_TYPES

    def test_camera_info_types(self, temp_bag_path: Path) -> None:
        """CAMERA_INFO_TYPES contains expected message types."""
        assert "sensor_msgs/msg/CameraInfo" in RosbagParser.CAMERA_INFO_TYPES


class TestGetInfo:
    """Tests for bag info retrieval."""

    @pytest.mark.skipif(
        not _has_rosbags(),
        reason="rosbags library not installed",
    )
    def test_get_info_caching(self, temp_bag_path: Path) -> None:
        """get_info caches result."""
        parser = RosbagParser(temp_bag_path)

        with (
            patch.object(parser, "_info_cache", None),
            patch("rosbags.highlevel.AnyReader") as mock_reader,
        ):
            mock_ctx = MagicMock()
            mock_reader.return_value.__enter__.return_value = mock_ctx
            mock_ctx.connections = []
            mock_ctx.duration = 1_000_000_000
            mock_ctx.start_time = 0
            mock_ctx.end_time = 1_000_000_000

            # First call
            info1 = parser.get_info()

            # Second call should use cache
            info2 = parser.get_info()

            # Should only open bag once
            assert mock_reader.call_count == 1
            assert info1 is info2

    @pytest.mark.skipif(
        not _has_rosbags(),
        reason="rosbags library not installed",
    )
    def test_get_info_force_refresh(self, temp_bag_path: Path) -> None:
        """get_info with force_refresh re-reads bag."""
        parser = RosbagParser(temp_bag_path)

        with patch("rosbags.highlevel.AnyReader") as mock_reader:
            mock_ctx = MagicMock()
            mock_reader.return_value.__enter__.return_value = mock_ctx
            mock_ctx.connections = []
            mock_ctx.duration = 1_000_000_000
            mock_ctx.start_time = 0
            mock_ctx.end_time = 1_000_000_000

            # First call
            parser.get_info()

            # Force refresh
            parser.get_info(force_refresh=True)

            # Should open bag twice
            assert mock_reader.call_count == 2


class TestFactoryFunction:
    """Tests for get_parser factory function."""

    def test_get_parser(self, temp_bag_path: Path) -> None:
        """get_parser creates RosbagParser instance."""
        from backend.data.rosbag_parser import get_parser

        parser = get_parser(temp_bag_path)
        assert isinstance(parser, RosbagParser)
        assert parser.bag_path == temp_bag_path.resolve()


class TestExceptions:
    """Tests for exception handling."""

    def test_rosbag_parse_error(self) -> None:
        """RosbagParseError can be raised with message."""
        with pytest.raises(RosbagParseError, match="Test error"):
            raise RosbagParseError("Test error")

    def test_parse_error_chain(self) -> None:
        """RosbagParseError preserves exception chain."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise RosbagParseError("Wrapped error") from e
        except RosbagParseError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
