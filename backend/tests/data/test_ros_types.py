"""
Unit tests for ROS type definitions.

Tests dataclass creation, validation, and Pydantic model serialization.
"""

from __future__ import annotations

import numpy as np

from backend.data.ros_types import (
    BagFormat,
    BagInfo,
    BagInfoResponse,
    CameraInfoMessage,
    ExtractImagesRequest,
    ExtractionResult,
    ExtractPointcloudRequest,
    ExtractSyncedRequest,
    ImageMessage,
    PointCloud2Message,
    SyncedExtractionResult,
    SyncedFrame,
    TopicInfo,
    TopicInfoResponse,
)


class TestBagFormat:
    """Tests for BagFormat enum."""

    def test_ros1_value(self) -> None:
        """ROS1 format has correct string value."""
        assert BagFormat.ROS1.value == "ros1"

    def test_ros2_value(self) -> None:
        """ROS2 format has correct string value."""
        assert BagFormat.ROS2.value == "ros2"

    def test_is_string_enum(self) -> None:
        """BagFormat is a string enum."""
        assert isinstance(BagFormat.ROS1, str)
        assert BagFormat.ROS1 == "ros1"


class TestTopicInfo:
    """Tests for TopicInfo dataclass."""

    def test_creation(self) -> None:
        """TopicInfo can be created with required fields."""
        topic = TopicInfo(
            name="/velodyne_points",
            msg_type="sensor_msgs/msg/PointCloud2",
            message_count=100,
        )
        assert topic.name == "/velodyne_points"
        assert topic.msg_type == "sensor_msgs/msg/PointCloud2"
        assert topic.message_count == 100
        assert topic.frequency_hz is None

    def test_with_frequency(self) -> None:
        """TopicInfo can include frequency."""
        topic = TopicInfo(
            name="/camera/image_raw",
            msg_type="sensor_msgs/msg/Image",
            message_count=300,
            frequency_hz=30.0,
        )
        assert topic.frequency_hz == 30.0


class TestBagInfo:
    """Tests for BagInfo dataclass."""

    def test_creation(self) -> None:
        """BagInfo can be created with all fields."""
        topics = [
            TopicInfo("/topic1", "std_msgs/msg/String", 10),
            TopicInfo("/topic2", "std_msgs/msg/Int32", 20),
        ]
        info = BagInfo(
            path="/data/recording.bag",
            format=BagFormat.ROS1,
            duration_sec=10.5,
            start_time=1000.0,
            end_time=1010.5,
            topics=topics,
            message_count=30,
        )
        assert info.path == "/data/recording.bag"
        assert info.format == BagFormat.ROS1
        assert info.duration_sec == 10.5
        assert len(info.topics) == 2


class TestPointCloud2Message:
    """Tests for PointCloud2Message dataclass."""

    def test_creation(self) -> None:
        """PointCloud2Message can be created with required fields."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        msg = PointCloud2Message(
            timestamp=1000.0,
            frame_id="velodyne",
            points=points,
        )
        assert msg.timestamp == 1000.0
        assert msg.frame_id == "velodyne"
        assert msg.points.shape == (2, 3)
        assert msg.intensity is None
        assert msg.rgb is None

    def test_point_count_property(self) -> None:
        """point_count property returns correct value."""
        points = np.zeros((100, 3))
        msg = PointCloud2Message(
            timestamp=0.0,
            frame_id="lidar",
            points=points,
        )
        assert msg.point_count == 100

    def test_with_intensity(self) -> None:
        """PointCloud2Message can include intensity."""
        points = np.array([[1.0, 2.0, 3.0]])
        intensity = np.array([0.5], dtype=np.float32)
        msg = PointCloud2Message(
            timestamp=0.0,
            frame_id="lidar",
            points=points,
            intensity=intensity,
        )
        assert msg.intensity is not None
        assert msg.intensity[0] == 0.5

    def test_with_rgb(self) -> None:
        """PointCloud2Message can include RGB colors."""
        points = np.array([[1.0, 2.0, 3.0]])
        rgb = np.array([[255, 128, 64]], dtype=np.uint8)
        msg = PointCloud2Message(
            timestamp=0.0,
            frame_id="lidar",
            points=points,
            rgb=rgb,
        )
        assert msg.rgb is not None
        assert msg.rgb[0, 0] == 255


class TestImageMessage:
    """Tests for ImageMessage dataclass."""

    def test_creation(self) -> None:
        """ImageMessage can be created with required fields."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        msg = ImageMessage(
            timestamp=1000.0,
            frame_id="camera_front",
            image=image,
            encoding="rgb8",
            width=640,
            height=480,
        )
        assert msg.timestamp == 1000.0
        assert msg.encoding == "rgb8"
        assert msg.width == 640
        assert msg.height == 480


class TestCameraInfoMessage:
    """Tests for CameraInfoMessage dataclass."""

    def test_creation(self) -> None:
        """CameraInfoMessage can be created with matrices."""
        K = np.eye(3)
        D = np.zeros(5)
        R = np.eye(3)
        P = np.zeros((3, 4))
        msg = CameraInfoMessage(
            timestamp=1000.0,
            frame_id="camera",
            width=640,
            height=480,
            K=K,
            D=D,
            R=R,
            P=P,
        )
        assert msg.timestamp == 1000.0
        assert msg.K.shape == (3, 3)
        assert msg.D.shape == (5,)
        assert msg.distortion_model == "plumb_bob"


class TestSyncedFrame:
    """Tests for SyncedFrame dataclass."""

    def test_creation_empty(self) -> None:
        """SyncedFrame can be created with no sensor data."""
        frame = SyncedFrame(
            timestamp=1000.0,
            pointcloud=None,
            image=None,
            camera_info=None,
            sync_error_ms=0.0,
        )
        assert frame.timestamp == 1000.0
        assert frame.pointcloud is None
        assert frame.frame_index == 0

    def test_with_sensor_data(self) -> None:
        """SyncedFrame can contain sensor data."""
        points = np.zeros((10, 3))
        pc = PointCloud2Message(timestamp=1000.0, frame_id="lidar", points=points)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        img = ImageMessage(
            timestamp=1000.005,
            frame_id="camera",
            image=image,
            encoding="rgb8",
            width=640,
            height=480,
        )
        frame = SyncedFrame(
            timestamp=1000.0,
            pointcloud=pc,
            image=img,
            camera_info=None,
            sync_error_ms=5.0,
            frame_index=42,
        )
        assert frame.pointcloud is not None
        assert frame.image is not None
        assert frame.sync_error_ms == 5.0
        assert frame.frame_index == 42


class TestPydanticModels:
    """Tests for Pydantic API models."""

    def test_topic_info_response(self) -> None:
        """TopicInfoResponse serializes correctly."""
        response = TopicInfoResponse(
            name="/topic",
            msg_type="std_msgs/msg/String",
            message_count=100,
            frequency_hz=10.0,
        )
        data = response.model_dump()
        assert data["name"] == "/topic"
        assert data["frequency_hz"] == 10.0

    def test_bag_info_response(self) -> None:
        """BagInfoResponse serializes correctly."""
        topics = [
            TopicInfoResponse(name="/t1", msg_type="type1", message_count=10),
        ]
        response = BagInfoResponse(
            path="/data/test.bag",
            format="ros1",
            duration_sec=10.0,
            start_time=1000.0,
            end_time=1010.0,
            message_count=10,
            topics=topics,
        )
        data = response.model_dump()
        assert data["format"] == "ros1"
        assert len(data["topics"]) == 1

    def test_extract_pointcloud_request_validation(self) -> None:
        """ExtractPointcloudRequest validates input."""
        request = ExtractPointcloudRequest(
            bag_path="/data/test.bag",
            topic="/velodyne_points",
            output_dir="/output",
            max_frames=100,
        )
        assert request.output_format == "pcd"
        assert request.skip_existing is False

    def test_extract_images_request_defaults(self) -> None:
        """ExtractImagesRequest has correct defaults."""
        request = ExtractImagesRequest(
            bag_path="/data/test.bag",
            topic="/camera/image_raw",
            output_dir="/output",
        )
        assert request.output_format == "png"
        assert request.max_frames is None

    def test_extract_synced_request(self) -> None:
        """ExtractSyncedRequest validates input."""
        request = ExtractSyncedRequest(
            bag_path="/data/test.bag",
            pointcloud_topic="/velodyne_points",
            image_topic="/camera/image_raw",
            output_dir="/output",
            max_sync_error_ms=25.0,
        )
        assert request.max_sync_error_ms == 25.0
        assert request.camera_info_topic is None

    def test_extraction_result(self) -> None:
        """ExtractionResult serializes correctly."""
        result = ExtractionResult(
            extracted_count=50,
            output_dir="/output",
            processing_time_sec=2.5,
            files=["/output/frame_000000.pcd"],
        )
        data = result.model_dump()
        assert data["extracted_count"] == 50
        assert len(data["files"]) == 1

    def test_synced_extraction_result(self) -> None:
        """SyncedExtractionResult serializes correctly."""
        result = SyncedExtractionResult(
            extracted_count=25,
            output_dir="/output",
            processing_time_sec=5.0,
            average_sync_error_ms=10.5,
            pointcloud_files=["/output/pc_000000.pcd"],
            image_files=["/output/img_000000.png"],
        )
        data = result.model_dump()
        assert data["average_sync_error_ms"] == 10.5
        assert len(data["pointcloud_files"]) == 1
