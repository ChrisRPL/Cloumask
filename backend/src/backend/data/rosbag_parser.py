"""
ROS bag parser for extracting sensor data.

Supports ROS1 (.bag) and ROS2 (.db3, .mcap) formats using the rosbags library.
Provides lazy loading, memory-efficient streaming, and timestamp synchronization.

Implements spec: 05-point-cloud/03-rosbag-extraction
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.data.ros_types import (
    BagFormat,
    BagInfo,
    CameraInfoMessage,
    ImageMessage,
    PointCloud2Message,
    SyncedFrame,
    TopicInfo,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RosbagParseError(Exception):
    """Raised when bag parsing fails."""


class RosbagParser:
    """
    Parse ROS1 and ROS2 bag files.

    Provides methods for:
    - Inspecting bag metadata and topics
    - Extracting PointCloud2 messages
    - Extracting Image messages
    - Extracting CameraInfo messages
    - Synchronizing multi-sensor data

    Uses lazy loading for rosbags library to avoid import overhead.

    Example:
        parser = RosbagParser("/path/to/recording.bag")
        info = parser.get_info()
        print(f"Duration: {info.duration_sec}s, Topics: {len(info.topics)}")

        for pc in parser.iter_pointcloud2("/velodyne_points", max_messages=100):
            print(f"Frame at {pc.timestamp}: {pc.point_count} points")
    """

    POINTCLOUD2_TYPES = {
        "sensor_msgs/msg/PointCloud2",
        "sensor_msgs/PointCloud2",
    }
    IMAGE_TYPES = {
        "sensor_msgs/msg/Image",
        "sensor_msgs/Image",
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/CompressedImage",
    }
    CAMERA_INFO_TYPES = {
        "sensor_msgs/msg/CameraInfo",
        "sensor_msgs/CameraInfo",
    }

    def __init__(self, bag_path: str | Path) -> None:
        """
        Initialize parser with bag file path.

        Args:
            bag_path: Path to ROS bag file (.bag, .db3, or .mcap).

        Raises:
            FileNotFoundError: If bag file does not exist.
            ValueError: If file format is not supported.
        """
        self.bag_path = Path(bag_path).resolve()

        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag file not found: {self.bag_path}")

        self._format = self._detect_format()
        self._info_cache: BagInfo | None = None

        logger.debug(
            "Initialized RosbagParser for %s (format: %s)",
            self.bag_path,
            self._format.value,
        )

    def _detect_format(self) -> BagFormat:
        """Detect bag format from file extension and magic bytes."""
        suffix = self.bag_path.suffix.lower()

        if suffix == ".bag":
            return BagFormat.ROS1
        elif suffix in {".db3", ".mcap"}:
            return BagFormat.ROS2
        elif self.bag_path.is_dir():
            # ROS2 bags can be directories containing metadata.yaml
            metadata = self.bag_path / "metadata.yaml"
            if metadata.exists():
                return BagFormat.ROS2

        # Try to detect from content
        try:
            with self.bag_path.open("rb") as f:
                magic = f.read(13)
                if magic == b"#ROSBAG V2.0\n":
                    return BagFormat.ROS1
        except Exception:
            pass

        raise ValueError(
            f"Unsupported bag format: {suffix}. "
            "Supported: .bag (ROS1), .db3/.mcap (ROS2)"
        )

    @property
    def format(self) -> BagFormat:
        """Get the detected bag format."""
        return self._format

    def get_info(self, *, force_refresh: bool = False) -> BagInfo:
        """
        Get bag metadata and topic list.

        Caches result for subsequent calls unless force_refresh is True.

        Args:
            force_refresh: If True, re-read metadata from bag file.

        Returns:
            BagInfo with duration, timestamps, and topic details.

        Raises:
            RosbagParseError: If bag cannot be read.
        """
        if self._info_cache is not None and not force_refresh:
            return self._info_cache

        from rosbags.highlevel import AnyReader

        try:
            with AnyReader([self.bag_path]) as reader:
                topics = []
                total_count = 0

                for conn in reader.connections:
                    # Count messages for this connection
                    count = sum(1 for _ in reader.messages([conn]))
                    total_count += count

                    # Estimate frequency if we have duration
                    freq_hz = None
                    if reader.duration > 0 and count > 1:
                        freq_hz = count / (reader.duration / 1e9)

                    topics.append(
                        TopicInfo(
                            name=conn.topic,
                            msg_type=conn.msgtype,
                            message_count=count,
                            frequency_hz=freq_hz,
                        )
                    )

                self._info_cache = BagInfo(
                    path=str(self.bag_path),
                    format=self._format,
                    duration_sec=reader.duration / 1e9,
                    start_time=reader.start_time / 1e9,
                    end_time=reader.end_time / 1e9,
                    topics=topics,
                    message_count=total_count,
                )

                logger.info(
                    "Bag info: %s, %.1fs duration, %d topics, %d messages",
                    self.bag_path.name,
                    self._info_cache.duration_sec,
                    len(topics),
                    total_count,
                )

                return self._info_cache

        except Exception as e:
            raise RosbagParseError(f"Failed to read bag info: {e}") from e

    def get_topics_by_type(self, msg_types: set[str]) -> list[TopicInfo]:
        """
        Get topics that match any of the given message types.

        Args:
            msg_types: Set of message type strings to match.

        Returns:
            List of TopicInfo for matching topics.
        """
        info = self.get_info()
        return [t for t in info.topics if t.msg_type in msg_types]

    def get_pointcloud_topics(self) -> list[TopicInfo]:
        """Get all PointCloud2 topics in the bag."""
        return self.get_topics_by_type(self.POINTCLOUD2_TYPES)

    def get_image_topics(self) -> list[TopicInfo]:
        """Get all Image topics in the bag."""
        return self.get_topics_by_type(self.IMAGE_TYPES)

    def get_camera_info_topics(self) -> list[TopicInfo]:
        """Get all CameraInfo topics in the bag."""
        return self.get_topics_by_type(self.CAMERA_INFO_TYPES)

    def iter_pointcloud2(
        self,
        topic: str,
        *,
        max_messages: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> Generator[PointCloud2Message, None, None]:
        """
        Iterate over PointCloud2 messages from a topic.

        Memory-efficient generator that yields one message at a time.

        Args:
            topic: Topic name to extract from.
            max_messages: Maximum number of messages to yield.
            start_time: Start timestamp filter (epoch seconds).
            end_time: End timestamp filter (epoch seconds).

        Yields:
            PointCloud2Message for each message.

        Raises:
            RosbagParseError: If extraction fails.
            ValueError: If topic is not a PointCloud2 topic.
        """
        from rosbags.highlevel import AnyReader

        try:
            with AnyReader([self.bag_path]) as reader:
                connections = [c for c in reader.connections if c.topic == topic]

                if not connections:
                    raise ValueError(f"Topic not found: {topic}")

                conn = connections[0]
                if conn.msgtype not in self.POINTCLOUD2_TYPES:
                    raise ValueError(
                        f"Topic {topic} is not PointCloud2, got {conn.msgtype}"
                    )

                count = 0
                for conn, timestamp, rawdata in reader.messages(connections):
                    timestamp_sec = timestamp / 1e9

                    # Apply time filters
                    if start_time and timestamp_sec < start_time:
                        continue
                    if end_time and timestamp_sec > end_time:
                        break

                    msg = reader.deserialize(rawdata, conn.msgtype)
                    yield self._convert_pointcloud2(msg, timestamp)

                    count += 1
                    if max_messages and count >= max_messages:
                        break

        except Exception as e:
            if isinstance(e, (ValueError, RosbagParseError)):
                raise
            raise RosbagParseError(f"Failed to extract PointCloud2: {e}") from e

    def extract_pointcloud2(
        self,
        topic: str,
        max_messages: int | None = None,
    ) -> list[PointCloud2Message]:
        """
        Extract all PointCloud2 messages from a topic.

        Convenience method that collects all messages into a list.
        For large bags, prefer iter_pointcloud2 to avoid memory issues.

        Args:
            topic: Topic name to extract from.
            max_messages: Maximum number of messages to extract.

        Returns:
            List of PointCloud2Message.
        """
        return list(self.iter_pointcloud2(topic, max_messages=max_messages))

    def iter_images(
        self,
        topic: str,
        *,
        max_messages: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> Generator[ImageMessage, None, None]:
        """
        Iterate over Image messages from a topic.

        Handles both raw and compressed image messages.

        Args:
            topic: Topic name to extract from.
            max_messages: Maximum number of messages to yield.
            start_time: Start timestamp filter (epoch seconds).
            end_time: End timestamp filter (epoch seconds).

        Yields:
            ImageMessage for each message.
        """
        from rosbags.highlevel import AnyReader

        try:
            with AnyReader([self.bag_path]) as reader:
                connections = [c for c in reader.connections if c.topic == topic]

                if not connections:
                    raise ValueError(f"Topic not found: {topic}")

                conn = connections[0]
                is_compressed = "Compressed" in conn.msgtype

                count = 0
                for conn, timestamp, rawdata in reader.messages(connections):
                    timestamp_sec = timestamp / 1e9

                    if start_time and timestamp_sec < start_time:
                        continue
                    if end_time and timestamp_sec > end_time:
                        break

                    msg = reader.deserialize(rawdata, conn.msgtype)

                    if is_compressed:
                        yield self._convert_compressed_image(msg, timestamp)
                    else:
                        yield self._convert_image(msg, timestamp)

                    count += 1
                    if max_messages and count >= max_messages:
                        break

        except Exception as e:
            if isinstance(e, (ValueError, RosbagParseError)):
                raise
            raise RosbagParseError(f"Failed to extract Image: {e}") from e

    def extract_images(
        self,
        topic: str,
        max_messages: int | None = None,
    ) -> list[ImageMessage]:
        """Extract all Image messages from a topic."""
        return list(self.iter_images(topic, max_messages=max_messages))

    def iter_camera_info(
        self,
        topic: str,
        *,
        max_messages: int | None = None,
    ) -> Generator[CameraInfoMessage, None, None]:
        """
        Iterate over CameraInfo messages from a topic.

        Args:
            topic: Topic name to extract from.
            max_messages: Maximum number of messages to yield.

        Yields:
            CameraInfoMessage for each message.
        """
        from rosbags.highlevel import AnyReader

        try:
            with AnyReader([self.bag_path]) as reader:
                connections = [c for c in reader.connections if c.topic == topic]

                if not connections:
                    raise ValueError(f"Topic not found: {topic}")

                for count, (conn, timestamp, rawdata) in enumerate(
                    reader.messages(connections), start=1
                ):
                    msg = reader.deserialize(rawdata, conn.msgtype)
                    yield self._convert_camera_info(msg, timestamp)

                    if max_messages and count >= max_messages:
                        break

        except Exception as e:
            if isinstance(e, (ValueError, RosbagParseError)):
                raise
            raise RosbagParseError(f"Failed to extract CameraInfo: {e}") from e

    def extract_camera_info(
        self,
        topic: str,
        max_messages: int | None = None,
    ) -> list[CameraInfoMessage]:
        """Extract all CameraInfo messages from a topic."""
        return list(self.iter_camera_info(topic, max_messages=max_messages))

    def sync_frames(
        self,
        pointcloud_topic: str,
        image_topic: str,
        camera_info_topic: str | None = None,
        *,
        max_sync_error_ms: float = 50.0,
        max_frames: int | None = None,
    ) -> Generator[SyncedFrame, None, None]:
        """
        Generate synchronized frames from multiple topics.

        Uses point cloud timestamps as reference and finds nearest
        matching images within the sync error threshold.

        Args:
            pointcloud_topic: PointCloud2 topic name.
            image_topic: Image topic name.
            camera_info_topic: Optional CameraInfo topic name.
            max_sync_error_ms: Maximum allowed timestamp difference in ms.
            max_frames: Maximum number of frames to generate.

        Yields:
            SyncedFrame with aligned sensor data.
        """
        # Extract all messages first for efficient synchronization
        logger.info(
            "Loading messages for synchronization: pc=%s, img=%s",
            pointcloud_topic,
            image_topic,
        )

        pointclouds = self.extract_pointcloud2(pointcloud_topic, max_messages=max_frames)
        images = self.extract_images(image_topic)
        camera_infos = (
            self.extract_camera_info(camera_info_topic) if camera_info_topic else []
        )

        if not pointclouds:
            logger.warning("No point clouds found in topic %s", pointcloud_topic)
            return

        if not images:
            logger.warning("No images found in topic %s", image_topic)
            return

        logger.info(
            "Synchronizing %d point clouds with %d images",
            len(pointclouds),
            len(images),
        )

        # Build timestamp index for fast lookup
        image_times = np.array([img.timestamp for img in images])
        camera_info_times = (
            np.array([ci.timestamp for ci in camera_infos]) if camera_infos else None
        )

        frame_idx = 0
        for pc in pointclouds:
            # Find nearest image
            img_diffs = np.abs(image_times - pc.timestamp)
            nearest_img_idx = int(np.argmin(img_diffs))
            img_error_ms = float(img_diffs[nearest_img_idx] * 1000)

            if img_error_ms > max_sync_error_ms:
                continue  # Skip if no image within threshold

            # Find nearest camera info
            camera_info = None
            ci_error_ms = 0.0
            if camera_infos and camera_info_times is not None:
                ci_diffs = np.abs(camera_info_times - pc.timestamp)
                nearest_ci_idx = int(np.argmin(ci_diffs))
                ci_error_ms = float(ci_diffs[nearest_ci_idx] * 1000)
                if ci_error_ms <= max_sync_error_ms:
                    camera_info = camera_infos[nearest_ci_idx]

            sync_error = max(img_error_ms, ci_error_ms)

            yield SyncedFrame(
                timestamp=pc.timestamp,
                pointcloud=pc,
                image=images[nearest_img_idx],
                camera_info=camera_info,
                sync_error_ms=sync_error,
                frame_index=frame_idx,
            )

            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break

        logger.info("Generated %d synchronized frames", frame_idx)

    def _convert_pointcloud2(self, msg: Any, timestamp: int) -> PointCloud2Message:
        """Convert ROS PointCloud2 message to numpy arrays."""
        # Get header info
        frame_id = msg.header.frame_id if hasattr(msg.header, "frame_id") else ""

        # Parse field definitions
        fields = {f.name: (f.offset, f.datatype, f.count) for f in msg.fields}
        field_names = list(fields.keys())

        # Calculate point count
        point_step = msg.point_step
        height = msg.height
        width = msg.width
        point_count = width if height == 1 else height * width

        # Parse binary data
        data = bytes(msg.data)
        points = np.zeros((point_count, 3), dtype=np.float64)
        intensity: np.ndarray | None = None
        rgb: np.ndarray | None = None

        for i in range(point_count):
            offset = i * point_step

            # Extract XYZ
            if "x" in fields:
                x_off, _, _ = fields["x"]
                points[i, 0] = struct.unpack_from("<f", data, offset + x_off)[0]
            if "y" in fields:
                y_off, _, _ = fields["y"]
                points[i, 1] = struct.unpack_from("<f", data, offset + y_off)[0]
            if "z" in fields:
                z_off, _, _ = fields["z"]
                points[i, 2] = struct.unpack_from("<f", data, offset + z_off)[0]

        # Extract intensity if available
        if "intensity" in fields:
            intensity = np.zeros(point_count, dtype=np.float32)
            i_off, i_dt, _ = fields["intensity"]
            fmt = "<f" if i_dt == 7 else "<B"
            for i in range(point_count):
                offset = i * point_step + i_off
                intensity[i] = struct.unpack_from(fmt, data, offset)[0]

        # Extract RGB if available (packed as uint32)
        if "rgb" in fields:
            rgb = np.zeros((point_count, 3), dtype=np.uint8)
            rgb_off, _, _ = fields["rgb"]
            for i in range(point_count):
                offset = i * point_step + rgb_off
                packed = struct.unpack_from("<I", data, offset)[0]
                rgb[i, 0] = (packed >> 16) & 0xFF  # R
                rgb[i, 1] = (packed >> 8) & 0xFF  # G
                rgb[i, 2] = packed & 0xFF  # B

        return PointCloud2Message(
            timestamp=timestamp / 1e9,
            frame_id=frame_id,
            points=points,
            intensity=intensity,
            rgb=rgb,
            fields=field_names,
        )

    def _convert_image(self, msg: Any, timestamp: int) -> ImageMessage:
        """Convert ROS Image message to numpy array."""
        frame_id = msg.header.frame_id if hasattr(msg.header, "frame_id") else ""
        encoding = msg.encoding
        width = msg.width
        height = msg.height

        # Convert based on encoding
        data = bytes(msg.data)

        # Validate data size for known encodings
        expected_sizes = {
            "rgb8": height * width * 3,
            "bgr8": height * width * 3,
            "mono8": height * width,
            "16UC1": height * width * 2,
        }
        if encoding in expected_sizes and len(data) != expected_sizes[encoding]:
            raise RosbagParseError(
                f"Image data size mismatch for {encoding}: "
                f"expected {expected_sizes[encoding]}, got {len(data)}"
            )

        if encoding in ("rgb8", "bgr8"):
            image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            if encoding == "bgr8":
                image = image[:, :, ::-1].copy()  # BGR to RGB
        elif encoding == "mono8":
            image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 1)
            image = np.repeat(image, 3, axis=2)  # Convert to 3-channel
        elif encoding == "16UC1":
            image = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
            image = (image / 256).astype(np.uint8)
            image = np.stack([image] * 3, axis=-1)
        else:
            # Default: try to interpret as 8-bit
            image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, -1)
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)

        return ImageMessage(
            timestamp=timestamp / 1e9,
            frame_id=frame_id,
            image=image,
            encoding=encoding,
            width=width,
            height=height,
        )

    def _convert_compressed_image(self, msg: Any, timestamp: int) -> ImageMessage:
        """Convert ROS CompressedImage message to numpy array."""
        import cv2

        frame_id = msg.header.frame_id if hasattr(msg.header, "frame_id") else ""

        # Decode compressed data
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if image is None:
            raise RosbagParseError("Failed to decode compressed image")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        return ImageMessage(
            timestamp=timestamp / 1e9,
            frame_id=frame_id,
            image=image,
            encoding="rgb8",
            width=width,
            height=height,
        )

    def _convert_camera_info(self, msg: Any, timestamp: int) -> CameraInfoMessage:
        """Convert ROS CameraInfo message."""
        frame_id = msg.header.frame_id if hasattr(msg.header, "frame_id") else ""

        # Extract matrices
        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d)
        R = np.array(msg.r).reshape(3, 3)
        P = np.array(msg.p).reshape(3, 4)

        distortion_model = getattr(msg, "distortion_model", "plumb_bob")

        return CameraInfoMessage(
            timestamp=timestamp / 1e9,
            frame_id=frame_id,
            width=msg.width,
            height=msg.height,
            K=K,
            D=D,
            R=R,
            P=P,
            distortion_model=distortion_model,
        )


def get_parser(bag_path: str | Path) -> RosbagParser:
    """
    Factory function to create a RosbagParser.

    Args:
        bag_path: Path to ROS bag file.

    Returns:
        Configured RosbagParser instance.
    """
    return RosbagParser(bag_path)
