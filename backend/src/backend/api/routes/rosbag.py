"""
FastAPI routes for ROS bag extraction.

Provides REST endpoints for bag inspection and sensor data extraction.

Implements spec: 05-point-cloud/03-rosbag-extraction
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.data.ros_types import (
    BagInfoResponse,
    ExtractImagesRequest,
    ExtractionResult,
    ExtractPointcloudRequest,
    ExtractSyncedRequest,
    SyncedExtractionResult,
    TopicInfoResponse,
)
from backend.data.rosbag_parser import RosbagParseError, RosbagParser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rosbag", tags=["ROS Bag"])


@router.get("/info", response_model=BagInfoResponse)
async def get_bag_info(
    path: str = Query(..., description="Path to ROS bag file"),
) -> BagInfoResponse:
    """
    Get metadata about a ROS bag file.

    Returns duration, timestamps, and topic information.
    """
    try:
        bag_path = Path(path)
        if not bag_path.exists():
            raise HTTPException(status_code=404, detail=f"Bag file not found: {path}")

        parser = RosbagParser(bag_path)
        info = parser.get_info()

        topics = [
            TopicInfoResponse(
                name=t.name,
                msg_type=t.msg_type,
                message_count=t.message_count,
                frequency_hz=t.frequency_hz,
            )
            for t in info.topics
        ]

        logger.info("Retrieved info for %s: %d topics", path, len(topics))

        return BagInfoResponse(
            path=info.path,
            format=info.format.value,
            duration_sec=info.duration_sec,
            start_time=info.start_time,
            end_time=info.end_time,
            message_count=info.message_count,
            topics=topics,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RosbagParseError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error getting bag info: %s", path)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/topics", response_model=list[TopicInfoResponse])
async def get_topics(
    path: str = Query(..., description="Path to ROS bag file"),
    msg_type: str | None = Query(None, description="Filter by message type"),
) -> list[TopicInfoResponse]:
    """
    List all topics in a ROS bag file.

    Optionally filter by message type substring.
    """
    try:
        bag_path = Path(path)
        if not bag_path.exists():
            raise HTTPException(status_code=404, detail=f"Bag file not found: {path}")

        parser = RosbagParser(bag_path)
        info = parser.get_info()

        topics = info.topics
        if msg_type:
            topics = [t for t in topics if msg_type.lower() in t.msg_type.lower()]

        return [
            TopicInfoResponse(
                name=t.name,
                msg_type=t.msg_type,
                message_count=t.message_count,
                frequency_hz=t.frequency_hz,
            )
            for t in topics
        ]

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error listing topics: %s", path)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/extract/pointcloud", response_model=ExtractionResult)
async def extract_pointcloud(request: ExtractPointcloudRequest) -> ExtractionResult:
    """
    Extract point clouds from a ROS bag to files.

    Extracts PointCloud2 messages and saves them in the specified format.
    """
    try:
        bag_path = Path(request.bag_path)
        if not bag_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Bag not found: {request.bag_path}"
            )

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        parser = RosbagParser(bag_path)

        start_time = time.perf_counter()
        files: list[str] = []
        count = 0

        for pc in parser.iter_pointcloud2(request.topic, max_messages=request.max_frames):
            # Generate output filename
            timestamp_str = f"{pc.timestamp:.6f}".replace(".", "_")
            filename = f"frame_{count:06d}_{timestamp_str}.{request.output_format}"
            output_path = output_dir / filename

            if request.skip_existing and output_path.exists():
                count += 1
                files.append(str(output_path))
                continue

            # Save based on format
            if request.output_format == "npy":
                np.save(str(output_path), pc.points)
            else:
                # Use Open3D for PCD/PLY formats
                import open3d as o3d

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc.points)

                if pc.rgb is not None:
                    colors = pc.rgb.astype(np.float64) / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                elif pc.intensity is not None:
                    max_intensity = pc.intensity.max()
                    intensity_norm = (
                        pc.intensity / max_intensity if max_intensity > 0 else pc.intensity
                    )
                    colors = np.stack([intensity_norm] * 3, axis=1)
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                o3d.io.write_point_cloud(str(output_path), pcd)

            files.append(str(output_path))
            count += 1

            if count % 100 == 0:
                logger.info("Extracted %d point clouds", count)

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Extracted %d point clouds from %s in %.1fs",
            count,
            request.topic,
            elapsed,
        )

        return ExtractionResult(
            extracted_count=count,
            output_dir=str(output_dir),
            processing_time_sec=elapsed,
            files=files,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RosbagParseError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error extracting point clouds")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/extract/images", response_model=ExtractionResult)
async def extract_images(request: ExtractImagesRequest) -> ExtractionResult:
    """
    Extract images from a ROS bag to files.

    Extracts Image messages and saves them in the specified format.
    """
    import cv2

    try:
        bag_path = Path(request.bag_path)
        if not bag_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Bag not found: {request.bag_path}"
            )

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        parser = RosbagParser(bag_path)

        start_time = time.perf_counter()
        files: list[str] = []
        count = 0

        for img in parser.iter_images(request.topic, max_messages=request.max_frames):
            timestamp_str = f"{img.timestamp:.6f}".replace(".", "_")
            filename = f"frame_{count:06d}_{timestamp_str}.{request.output_format}"
            output_path = output_dir / filename

            if request.skip_existing and output_path.exists():
                count += 1
                files.append(str(output_path))
                continue

            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(img.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), image_bgr)

            files.append(str(output_path))
            count += 1

            if count % 100 == 0:
                logger.info("Extracted %d images", count)

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Extracted %d images from %s in %.1fs",
            count,
            request.topic,
            elapsed,
        )

        return ExtractionResult(
            extracted_count=count,
            output_dir=str(output_dir),
            processing_time_sec=elapsed,
            files=files,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error extracting images")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/extract/synced", response_model=SyncedExtractionResult)
async def extract_synced_frames(
    request: ExtractSyncedRequest,
) -> SyncedExtractionResult:
    """
    Extract synchronized point cloud and image frames.

    Aligns timestamps across sensors and extracts matched pairs.
    """
    import cv2

    try:
        bag_path = Path(request.bag_path)
        if not bag_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Bag not found: {request.bag_path}"
            )

        output_dir = Path(request.output_dir)
        pc_dir = output_dir / "pointclouds"
        img_dir = output_dir / "images"
        pc_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)

        parser = RosbagParser(bag_path)

        start_time = time.perf_counter()
        pc_files: list[str] = []
        img_files: list[str] = []
        sync_errors: list[float] = []

        for frame in parser.sync_frames(
            request.pointcloud_topic,
            request.image_topic,
            request.camera_info_topic,
            max_sync_error_ms=request.max_sync_error_ms,
            max_frames=request.max_frames,
        ):
            idx = frame.frame_index
            timestamp_str = f"{frame.timestamp:.6f}".replace(".", "_")

            # Save point cloud
            if frame.pointcloud:
                pc_filename = (
                    f"frame_{idx:06d}_{timestamp_str}.{request.pointcloud_format}"
                )
                pc_path = pc_dir / pc_filename

                if request.pointcloud_format == "npy":
                    np.save(str(pc_path), frame.pointcloud.points)
                else:
                    import open3d as o3d

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(frame.pointcloud.points)

                    if frame.pointcloud.intensity is not None:
                        max_intensity = max(frame.pointcloud.intensity.max(), 1)
                        intensity_norm = frame.pointcloud.intensity / max_intensity
                        colors = np.stack([intensity_norm] * 3, axis=1)
                        pcd.colors = o3d.utility.Vector3dVector(colors)

                    o3d.io.write_point_cloud(str(pc_path), pcd)

                pc_files.append(str(pc_path))

            # Save image
            if frame.image:
                img_filename = f"frame_{idx:06d}_{timestamp_str}.{request.image_format}"
                img_path = img_dir / img_filename

                image_bgr = cv2.cvtColor(frame.image.image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(img_path), image_bgr)
                img_files.append(str(img_path))

            sync_errors.append(frame.sync_error_ms)

            if (idx + 1) % 100 == 0:
                logger.info("Extracted %d synced frames", idx + 1)

        elapsed = time.perf_counter() - start_time
        avg_sync_error = sum(sync_errors) / len(sync_errors) if sync_errors else 0.0

        logger.info(
            "Extracted %d synced frames in %.1fs (avg sync error: %.2fms)",
            len(pc_files),
            elapsed,
            avg_sync_error,
        )

        return SyncedExtractionResult(
            extracted_count=len(pc_files),
            output_dir=str(output_dir),
            processing_time_sec=elapsed,
            average_sync_error_ms=avg_sync_error,
            pointcloud_files=pc_files,
            image_files=img_files,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RosbagParseError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error extracting synced frames")
        raise HTTPException(status_code=500, detail=str(e)) from e
