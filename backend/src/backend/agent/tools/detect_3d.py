"""3D object detection tool for point clouds using PV-RCNN++ or CenterPoint.

Detects vehicles, pedestrians, and cyclists in LiDAR or other 3D point cloud data.
Supports multiple coordinate systems (KITTI, nuScenes, Waymo) and file formats.

Implements spec: 03-cv-models/07-3d-detection, 03-cv-models/08-cv-tools
Integration point: backend/cv/detection_3d.py
"""

from __future__ import annotations

import logging
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
from backend.agent.tools.constants import DETECTION_3D_CLASSES, POINTCLOUD_EXTENSIONS
from backend.agent.tools.registry import register_tool

logger = logging.getLogger(__name__)


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

        source_path = Path(input_path)
        input_p = source_path
        source_was_directory = False
        source_file_count = 1

        # Validate input
        if not source_path.exists():
            return error_result(f"Input not found: {input_path}")

        if source_path.is_dir():
            source_was_directory = True
            candidates = sorted({
                path
                for ext in POINTCLOUD_EXTENSIONS
                for path in source_path.rglob(f"*{ext}")
                if path.is_file()
            })
            source_file_count = len(candidates)
            if not candidates:
                return error_result(
                    f"No supported pointcloud files found in directory: {input_path}"
                )
            input_p = candidates[0]
            logger.info(
                "Resolved pointcloud directory %s to %s (%d candidate files)",
                source_path,
                input_p,
                source_file_count,
            )
        elif not source_path.is_file():
            return error_result("3D detection requires a point cloud file or directory")

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
                        "source_path": str(source_path),
                        "source_was_directory": source_was_directory,
                        "source_file_count": source_file_count,
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
            logger.exception("3D detection failed")
            return error_result(f"3D detection failed: {e}")
        except Exception as e:
            logger.exception("3D detection failed")
            return error_result(f"3D detection failed: {e}")
