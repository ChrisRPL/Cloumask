"""3D object detection tool for point clouds using PV-RCNN++ or CenterPoint.

Detects vehicles, pedestrians, and cyclists in LiDAR or other 3D point cloud data.
Supports multiple coordinate systems (KITTI, nuScenes, Waymo) and file formats.

Implements spec: 03-cv-models/07-3d-detection, 03-cv-models/08-cv-tools
Integration point: backend/cv/detection_3d.py
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.constants import DETECTION_3D_CLASSES, POINTCLOUD_EXTENSIONS
from backend.agent.tools.registry import register_tool

logger = logging.getLogger(__name__)


def _heuristic_fallback_enabled() -> bool:
    return os.getenv("CLOUMASK_ENABLE_3D_HEURISTIC_FALLBACK", "0") == "1"


def _heuristic_pointcloud_detections(
    points: np.ndarray,
    confidence_threshold: float,
    classes: list[str] | None,
) -> list[dict[str, Any]]:
    """Build coarse 3D boxes from voxel clusters as a runtime fallback."""
    if points.size == 0:
        return []

    xyz = points[:, :3].astype(np.float32, copy=False)
    finite_mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite_mask]
    if xyz.shape[0] == 0:
        return []

    # Remove most ground points to keep coarse object-like clusters.
    ground_z = float(np.percentile(xyz[:, 2], 12))
    non_ground = xyz[xyz[:, 2] > ground_z + 0.12]
    if non_ground.shape[0] >= 60:
        xyz = non_ground

    voxel_size = np.array([1.6, 1.6, 1.2], dtype=np.float32)
    voxel_keys = np.floor(xyz / voxel_size).astype(np.int32)
    clusters: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    for key, point in zip(voxel_keys, xyz, strict=False):
        clusters[tuple(int(v) for v in key)].append(point)

    preferred_class = classes[0] if classes else "Car"
    detections: list[dict[str, Any]] = []

    for _, cluster_points in sorted(
        clusters.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        cluster_count = len(cluster_points)
        if cluster_count < 90:
            continue

        cluster = np.asarray(cluster_points, dtype=np.float32)
        mins = cluster.min(axis=0)
        maxs = cluster.max(axis=0)
        size = maxs - mins
        if np.any(size < np.array([0.2, 0.2, 0.2], dtype=np.float32)):
            continue

        confidence = min(0.95, 0.5 + cluster_count / 6000.0)
        if confidence < confidence_threshold:
            continue

        center = ((mins + maxs) / 2.0).tolist()
        dimensions = np.maximum(size, np.array([0.4, 0.4, 0.4], dtype=np.float32)).tolist()
        detections.append(
            {
                "class": preferred_class,
                "confidence": round(float(confidence), 3),
                "center": {
                    "x": round(float(center[0]), 3),
                    "y": round(float(center[1]), 3),
                    "z": round(float(center[2]), 3),
                },
                "dimensions": {
                    "length": round(float(dimensions[0]), 3),
                    "width": round(float(dimensions[1]), 3),
                    "height": round(float(dimensions[2]), 3),
                },
                "rotation": 0.0,
            }
        )

        if len(detections) >= 10:
            break

    # Ensure at least one reviewable annotation for dense clouds.
    if not detections and xyz.shape[0] >= 200:
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        size = np.maximum(maxs - mins, np.array([0.4, 0.4, 0.4], dtype=np.float32))
        center = (mins + maxs) / 2.0
        fallback_confidence = max(confidence_threshold, 0.6)
        detections.append(
            {
                "class": preferred_class,
                "confidence": round(float(fallback_confidence), 3),
                "center": {
                    "x": round(float(center[0]), 3),
                    "y": round(float(center[1]), 3),
                    "z": round(float(center[2]), 3),
                },
                "dimensions": {
                    "length": round(float(size[0]), 3),
                    "width": round(float(size[1]), 3),
                    "height": round(float(size[2]), 3),
                },
                "rotation": 0.0,
            }
        )

    return detections


@register_tool
class Detect3DTool(BaseTool):
    """Detect 3D objects in point clouds using PV-RCNN++ or CenterPoint."""

    name = "detect_3d"
    description = """Detect 3D objects (vehicles, pedestrians, cyclists) in point clouds.

Model Selection:
- PV-RCNN++ (default, prefer_accuracy=True): 84% 3D AP, ~4GB VRAM, 150-200ms
- CenterPoint (prefer_accuracy=False): 79% 3D AP, ~3GB VRAM, 80-100ms

Supported Formats:
- KITTI binary (.bin), PCD, PLY, LAS/LAZ

Coordinate Systems:
- kitti: x=forward, y=left, z=up (default)
- nuscenes: x=right, y=forward, z=up
- waymo: same as kitti

Examples:
- detect_3d(path)  # PV-RCNN++ with all classes
- detect_3d(path, classes=["Car"])  # Vehicles only
- detect_3d(path, prefer_accuracy=False)  # Faster CenterPoint"""
    category = ToolCategory.DETECTION

    parameters = [
        ToolParameter(
            name="input_path",
            type=str,
            description="Path to point cloud file (.pcd, .ply, .las, .laz, .bin)",
            required=True,
        ),
        ToolParameter(
            name="classes",
            type=list,
            description="Classes to detect: Car, Pedestrian, Cyclist. None for all.",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="confidence",
            type=float,
            description="Minimum confidence threshold (0-1)",
            required=False,
            default=0.3,
        ),
        ToolParameter(
            name="coordinate_system",
            type=str,
            description="Input coordinate system: kitti, nuscenes, or waymo",
            required=False,
            default="kitti",
            enum_values=["kitti", "nuscenes", "waymo"],
        ),
        ToolParameter(
            name="prefer_accuracy",
            type=bool,
            description="Use PV-RCNN++ (slower, more accurate) instead of CenterPoint",
            required=False,
            default=True,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        classes: list[str] | None = None,
        confidence: float = 0.3,
        coordinate_system: str = "kitti",
        prefer_accuracy: bool = True,
    ) -> ToolResult:
        """
        Execute 3D object detection on a point cloud.

        Args:
            input_path: Path to point cloud file.
            classes: Classes to detect (Car, Pedestrian, Cyclist).
            confidence: Minimum confidence threshold (0-1).
            coordinate_system: Input coordinate system.
            prefer_accuracy: Use PV-RCNN++ (True) or CenterPoint (False).

        Returns:
            ToolResult with 3D detection statistics and bounding boxes.
        """
        from backend.cv.detection_3d import CoordinateSystem, get_3d_detector

        input_p = Path(input_path)

        # Validate input
        if not input_p.exists():
            return error_result(f"Input not found: {input_path}")

        if not input_p.is_file():
            return error_result("3D detection requires a single point cloud file")

        if input_p.suffix.lower() not in POINTCLOUD_EXTENSIONS:
            return error_result(
                f"Unsupported point cloud format: {input_p.suffix}. "
                f"Supported: {', '.join(sorted(POINTCLOUD_EXTENSIONS))}"
            )

        if confidence < 0 or confidence > 1:
            return error_result(
                f"Invalid confidence: {confidence}. Must be between 0 and 1."
            )

        # Validate classes
        if classes:
            invalid = [c for c in classes if c not in DETECTION_3D_CLASSES]
            if invalid:
                return error_result(
                    f"Invalid classes: {invalid}. "
                    f"Valid classes: {DETECTION_3D_CLASSES}"
                )

        # Map coordinate system string to enum
        coord_map = {
            "kitti": CoordinateSystem.KITTI,
            "nuscenes": CoordinateSystem.NUSCENES,
            "waymo": CoordinateSystem.WAYMO,
        }
        coord_system = coord_map.get(coordinate_system.lower())
        if coord_system is None:
            return error_result(
                f"Invalid coordinate_system: {coordinate_system}. "
                "Valid options: kitti, nuscenes, waymo"
            )

        try:
            # Get and load 3D detector
            detector = get_3d_detector(prefer_accuracy=prefer_accuracy)
            detector.load()

            try:
                self.report_progress(1, 1, f"Detecting 3D objects with {detector.info.name}...")

                result = detector.predict(
                    str(input_p),
                    classes=classes,
                    confidence=confidence,
                    coordinate_system=coord_system,
                )

                # Aggregate statistics
                class_counts: dict[str, int] = {}
                detections_data: list[dict[str, Any]] = []

                for det in result.detections:
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

                    detections_data.append({
                        "class": det.class_name,
                        "confidence": round(det.confidence, 3),
                        "center": {
                            "x": round(det.center[0], 3),
                            "y": round(det.center[1], 3),
                            "z": round(det.center[2], 3),
                        },
                        "dimensions": {
                            "length": round(det.dimensions[0], 3),
                            "width": round(det.dimensions[1], 3),
                            "height": round(det.dimensions[2], 3),
                        },
                        "rotation": round(det.rotation, 4),
                    })

                return success_result(
                    {
                        "pointcloud_path": str(input_p),
                        "count": len(result.detections),
                        "classes_found": class_counts,
                        "detections": detections_data,
                        "model": result.model_name,
                        "coordinate_system": coordinate_system,
                        "processing_time_ms": round(result.processing_time_ms, 2),
                    }
                )

            finally:
                detector.unload()

        except ImportError as e:
            if _heuristic_fallback_enabled():
                logger.warning(
                    "OpenPCDet unavailable (%s). Falling back to heuristic 3D clustering.",
                    e,
                )
                return self._run_heuristic_fallback(
                    input_p=input_p,
                    source_path=source_path,
                    source_was_directory=source_was_directory,
                    source_file_count=source_file_count,
                    confidence=confidence,
                    classes=classes,
                    coordinate_system=coordinate_system,
                )

            logger.exception("CV dependencies not installed")
            return error_result(
                f"CV dependencies not installed: {e}. "
                "3D detection requires OpenPCDet. Install with: pip install -r requirements-cv.txt"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory: %s", e)
                return error_result(
                    f"GPU out of memory. PV-RCNN++ needs ~4GB, CenterPoint needs ~3GB. "
                    f"Try: prefer_accuracy=False for lower VRAM usage. Error: {e}"
                )

            if _heuristic_fallback_enabled():
                logger.warning(
                    "3D detector runtime unavailable (%s). Falling back to heuristic clustering.",
                    e,
                )
                return self._run_heuristic_fallback(
                    input_p=input_p,
                    source_path=source_path,
                    source_was_directory=source_was_directory,
                    source_file_count=source_file_count,
                    confidence=confidence,
                    classes=classes,
                    coordinate_system=coordinate_system,
                )

            logger.exception("3D detection failed")
            return error_result(f"3D detection failed: {e}")
        except Exception as e:
            logger.exception("3D detection failed")
            return error_result(f"3D detection failed: {e}")

    def _run_heuristic_fallback(
        self,
        *,
        input_p: Path,
        source_path: Path,
        source_was_directory: bool,
        source_file_count: int,
        confidence: float,
        classes: list[str] | None,
        coordinate_system: str,
    ) -> ToolResult:
        """Fallback 3D detection when OpenPCDet models are unavailable."""
        from backend.cv.detection_3d import PointCloudLoader

        start = time.perf_counter()
        points = PointCloudLoader.load(str(input_p))
        detections_data = _heuristic_pointcloud_detections(
            points,
            confidence_threshold=confidence,
            classes=classes,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        class_counts: dict[str, int] = {}
        for det in detections_data:
            class_name = str(det["class"])
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return success_result(
            {
                "pointcloud_path": str(input_p),
                "source_path": str(source_path),
                "source_was_directory": source_was_directory,
                "source_file_count": source_file_count,
                "count": len(detections_data),
                "classes_found": class_counts,
                "detections": detections_data,
                "model": "heuristic_fallback",
                "coordinate_system": coordinate_system,
                "processing_time_ms": round(elapsed_ms, 2),
            }
        )
