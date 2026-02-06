"""ROS bag extraction tool for converting ROS recordings to standard formats.

Extracts point clouds and images from ROS1 (.bag) and ROS2 (.db3, .mcap) files.
Supports auto-discovery of sensor topics and timestamp synchronization between
LiDAR and camera data.

Implements spec: 05-point-cloud/03-rosbag-extraction, 05-point-cloud/08-agent-tools
Integration point: backend/data/rosbag_parser.py
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

# Supported ROS bag file extensions
ROSBAG_EXTENSIONS: frozenset[str] = frozenset({".bag", ".db3", ".mcap"})


@register_tool
class ExtractRosbagTool(BaseTool):
    """Extract point clouds and images from ROS bag files."""

    name = "extract_rosbag"
    description = """Extract point clouds and images from a ROS bag file.

Supports:
- ROS1 (.bag) and ROS2 (.db3, .mcap) formats
- Automatic topic discovery for sensor data
- Timestamp synchronization between LiDAR and camera
- Memory-efficient streaming extraction

Extraction Modes:
- Synchronized: Extracts paired LiDAR + camera frames (default when both present)
- Separate: Extracts point clouds or images independently

Output Formats:
- Point clouds: PCD, PLY, NPY
- Images: PNG, JPG

Examples:
- extract_rosbag(bag_path, output_dir)  # Auto-detect topics, extract synced frames
- extract_rosbag(bag_path, output_dir, max_frames=50)  # Limit to 50 frames
- extract_rosbag(bag_path, output_dir, pointcloud_topic="/velodyne_points")
- extract_rosbag(bag_path, output_dir, sync_sensors=False)  # Extract separately"""
    category = ToolCategory.UTILITY

    parameters = [
        ToolParameter(
            name="bag_path",
            type=str,
            description="Path to ROS bag file (.bag, .db3, .mcap)",
            required=True,
        ),
        ToolParameter(
            name="output_dir",
            type=str,
            description="Directory to save extracted data",
            required=True,
        ),
        ToolParameter(
            name="pointcloud_topic",
            type=str,
            description="PointCloud2 topic name (auto-detected if not specified)",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="image_topic",
            type=str,
            description="Image topic name (auto-detected if not specified)",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="max_frames",
            type=int,
            description="Maximum number of frames to extract",
            required=False,
            default=100,
        ),
        ToolParameter(
            name="sync_sensors",
            type=bool,
            description="Synchronize point clouds with camera images by timestamp",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="max_sync_error_ms",
            type=float,
            description="Maximum allowed timestamp difference in ms for synchronization",
            required=False,
            default=50.0,
        ),
    ]

    async def execute(
        self,
        bag_path: str,
        output_dir: str,
        pointcloud_topic: str | None = None,
        image_topic: str | None = None,
        max_frames: int = 100,
        sync_sensors: bool = True,
        max_sync_error_ms: float = 50.0,
    ) -> ToolResult:
        """
        Execute ROS bag extraction.

        Args:
            bag_path: Path to ROS bag file.
            output_dir: Directory for extracted output.
            pointcloud_topic: PointCloud2 topic (auto-detected if None).
            image_topic: Image topic (auto-detected if None).
            max_frames: Maximum frames to extract.
            sync_sensors: Whether to synchronize LiDAR and camera.
            max_sync_error_ms: Max sync error threshold in milliseconds.

        Returns:
            ToolResult with extraction statistics.
        """
        from backend.data.rosbag_parser import RosbagParser

        bag_p = Path(bag_path)
        out_dir = Path(output_dir)

        # Validate bag file
        if not bag_p.exists():
            return error_result(f"Bag file not found: {bag_path}")

        if bag_p.is_file() and bag_p.suffix.lower() not in ROSBAG_EXTENSIONS:
            return error_result(
                f"Unsupported bag format: {bag_p.suffix}. "
                f"Supported: {', '.join(sorted(ROSBAG_EXTENSIONS))}"
            )

        # Validate parameters
        if max_frames < 1:
            return error_result("max_frames must be at least 1")

        if max_sync_error_ms < 0:
            return error_result("max_sync_error_ms must be non-negative")

        try:
            start_time = time.perf_counter()

            # Parse bag info
            self.report_progress(0, 4, "Reading bag metadata...")
            parser = RosbagParser(bag_path)
            bag_info = parser.get_info()

            # Auto-detect topics
            pc_topic = pointcloud_topic
            img_topic = image_topic

            if pc_topic is None:
                pc_topics = parser.get_pointcloud_topics()
                if pc_topics:
                    # Select the topic with most messages
                    pc_topic = max(pc_topics, key=lambda t: t.message_count).name
                    logger.info("Auto-detected PointCloud2 topic: %s", pc_topic)

            if img_topic is None:
                img_topics = parser.get_image_topics()
                if img_topics:
                    img_topic = max(img_topics, key=lambda t: t.message_count).name
                    logger.info("Auto-detected Image topic: %s", img_topic)

            if pc_topic is None and img_topic is None:
                return error_result(
                    "No PointCloud2 or Image topics found in bag. "
                    f"Available topics: {[t.name for t in bag_info.topics]}"
                )

            # Create output directory
            out_dir.mkdir(parents=True, exist_ok=True)

            pointcloud_files: list[str] = []
            image_files: list[str] = []
            sync_errors: list[float] = []

            if sync_sensors and pc_topic and img_topic:
                # Synchronized extraction
                pointcloud_files, image_files, sync_errors = (
                    self._extract_synced(
                        parser, out_dir, pc_topic, img_topic,
                        max_frames, max_sync_error_ms,
                    )
                )
            else:
                # Separate extraction
                if pc_topic:
                    self.report_progress(2, 4, f"Extracting point clouds from {pc_topic}...")
                    pointcloud_files = self._extract_pointclouds(
                        parser, out_dir, pc_topic, max_frames
                    )

                if img_topic:
                    self.report_progress(3, 4, f"Extracting images from {img_topic}...")
                    image_files = self._extract_images(
                        parser, out_dir, img_topic, max_frames
                    )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            avg_sync_error = (
                sum(sync_errors) / len(sync_errors) if sync_errors else 0.0
            )

            self.report_progress(4, 4, "Extraction complete")

            return success_result(
                {
                    "bag_path": bag_path,
                    "bag_format": bag_info.format.value,
                    "bag_duration_sec": round(bag_info.duration_sec, 2),
                    "output_dir": str(out_dir),
                    "frames_extracted": max(len(pointcloud_files), len(image_files)),
                    "pointcloud_topic": pc_topic,
                    "image_topic": img_topic,
                    "pointcloud_files_count": len(pointcloud_files),
                    "image_files_count": len(image_files),
                    "sync_enabled": sync_sensors and bool(pc_topic and img_topic),
                    "avg_sync_error_ms": round(avg_sync_error, 2),
                    "processing_time_ms": round(elapsed_ms, 2),
                }
            )

        except ImportError as e:
            logger.exception("rosbags library not installed")
            return error_result(
                f"rosbags library not installed: {e}. "
                "Install with: pip install rosbags"
            )
        except FileNotFoundError as e:
            return error_result(str(e))
        except ValueError as e:
            return error_result(f"Invalid bag file or parameter: {e}")
        except Exception as e:
            logger.exception("ROS bag extraction failed")
            return error_result(f"ROS bag extraction failed: {e}")

    def _extract_synced(
        self,
        parser: Any,
        out_dir: Path,
        pc_topic: str,
        img_topic: str,
        max_frames: int,
        max_sync_error_ms: float,
    ) -> tuple[list[str], list[str], list[float]]:
        """Extract synchronized point cloud and image frames."""
        import cv2
        import open3d as o3d

        self.report_progress(1, 4, "Synchronizing sensor data...")

        pointcloud_files: list[str] = []
        image_files: list[str] = []
        sync_errors: list[float] = []

        pc_dir = out_dir / "pointclouds"
        img_dir = out_dir / "images"
        pc_dir.mkdir(exist_ok=True)
        img_dir.mkdir(exist_ok=True)

        frames = parser.sync_frames(
            pointcloud_topic=pc_topic,
            image_topic=img_topic,
            max_sync_error_ms=max_sync_error_ms,
            max_frames=max_frames,
        )

        self.report_progress(2, 4, "Saving synchronized frames...")

        for frame in frames:
            idx = frame.frame_index

            # Save point cloud as PCD
            if frame.pointcloud is not None:
                pc_path = str(pc_dir / f"frame_{idx:06d}.pcd")
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(frame.pointcloud.points)
                o3d.io.write_point_cloud(pc_path, pcd)
                pointcloud_files.append(pc_path)

            # Save image as PNG
            if frame.image is not None:
                img_path = str(img_dir / f"frame_{idx:06d}.png")
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(frame.image.image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, img_bgr)
                image_files.append(img_path)

            sync_errors.append(frame.sync_error_ms)

        return pointcloud_files, image_files, sync_errors

    def _extract_pointclouds(
        self,
        parser: Any,
        out_dir: Path,
        topic: str,
        max_frames: int,
    ) -> list[str]:
        """Extract point clouds from a topic."""
        import open3d as o3d

        pc_dir = out_dir / "pointclouds"
        pc_dir.mkdir(exist_ok=True)

        files: list[str] = []
        for i, msg in enumerate(parser.iter_pointcloud2(topic, max_messages=max_frames)):
            pc_path = str(pc_dir / f"pc_{i:06d}.pcd")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(msg.points)
            o3d.io.write_point_cloud(pc_path, pcd)
            files.append(pc_path)

        return files

    def _extract_images(
        self,
        parser: Any,
        out_dir: Path,
        topic: str,
        max_frames: int,
    ) -> list[str]:
        """Extract images from a topic."""
        import cv2

        img_dir = out_dir / "images"
        img_dir.mkdir(exist_ok=True)

        files: list[str] = []
        for i, msg in enumerate(parser.iter_images(topic, max_messages=max_frames)):
            img_path = str(img_dir / f"img_{i:06d}.png")
            img_bgr = cv2.cvtColor(msg.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img_bgr)
            files.append(img_path)

        return files
