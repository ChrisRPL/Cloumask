"""2D-3D projection and fusion tools for sensor fusion operations.

Projects 3D detections to 2D image coordinates using calibration data.
Useful for visualizing 3D detections on camera images.

Implements spec: 05-point-cloud/05-2d-3d-fusion, 05-point-cloud/08-agent-tools
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

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


@register_tool
class Project3DTo2DTool(BaseTool):
    """Project 3D detections to 2D image coordinates."""

    name = "project_3d_to_2d"
    description = """Project 3D bounding boxes to 2D image coordinates.

Requires:
- 3D detections (JSON file or point cloud for detection)
- Camera calibration file (KITTI, nuScenes, ROS, or JSON format)

Outputs:
- 2D bounding boxes in pixel coordinates
- Visibility status for each detection

Use Cases:
- Visualize 3D detections on camera images
- Verify sensor alignment
- Create fused 2D-3D visualizations

Calibration Formats:
- kitti: P0-P3, R0_rect, Tr_velo_to_cam (.txt)
- nuscenes: camera_intrinsic, lidar_to_camera (.json)
- json: K, D, width, height, T_cam_lidar (.json)

Examples:
- project_3d_to_2d(calibration_path="calib.txt", detections_path="dets.json")
- project_3d_to_2d(calibration_path="calib.txt", pointcloud_path="scan.bin", run_detection=True)"""

    category = ToolCategory.DETECTION

    parameters = [
        ToolParameter(
            name="calibration_path",
            type=str,
            description="Path to calibration file",
            required=True,
        ),
        ToolParameter(
            name="detections_path",
            type=str,
            description="Path to 3D detections JSON file",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="pointcloud_path",
            type=str,
            description="Path to point cloud for detection",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="calibration_format",
            type=str,
            description="Calibration format: kitti, nuscenes, ros, json",
            required=False,
            default="kitti",
            enum_values=["kitti", "nuscenes", "ros", "json"],
        ),
        ToolParameter(
            name="camera_id",
            type=int,
            description="Camera ID for KITTI format (0-3)",
            required=False,
            default=2,
        ),
        ToolParameter(
            name="run_detection",
            type=bool,
            description="Run 3D detection on pointcloud first",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="confidence",
            type=float,
            description="Confidence threshold for detection (if run_detection=True)",
            required=False,
            default=0.3,
        ),
    ]

    async def execute(
        self,
        calibration_path: str,
        detections_path: str | None = None,
        pointcloud_path: str | None = None,
        calibration_format: str = "kitti",
        camera_id: int = 2,
        run_detection: bool = False,
        confidence: float = 0.3,
    ) -> ToolResult:
        """Execute 3D to 2D projection."""
        from backend.cv.types import Detection3D
        from backend.data.calibration import CameraCalibration
        from backend.data.projection import project_detections_3d_to_2d

        # Validate inputs
        calib_p = Path(calibration_path)
        if not calib_p.exists():
            return error_result(f"Calibration file not found: {calibration_path}")

        if detections_path is None and pointcloud_path is None:
            return error_result("Either detections_path or pointcloud_path is required")

        try:
            # Load calibration
            self.report_progress(0, 3, "Loading calibration...")

            if calibration_format == "kitti":
                calib = CameraCalibration.from_kitti(str(calib_p), camera_id)
            elif calibration_format == "nuscenes":
                with open(calib_p) as f:
                    data = json.load(f)
                calib = CameraCalibration.from_nuscenes(data)
            elif calibration_format == "json":
                calib = CameraCalibration.from_json(str(calib_p))
            elif calibration_format == "ros":
                from backend.data.ros_types import CameraInfoMessage

                with open(calib_p) as f:
                    data = json.load(f)
                msg = CameraInfoMessage(**data)
                calib = CameraCalibration.from_ros(msg)
            else:
                return error_result(f"Unsupported calibration format: {calibration_format}")

            # Get 3D detections
            detections: list[Detection3D] = []

            if detections_path:
                self.report_progress(1, 3, "Loading detections...")
                det_p = Path(detections_path)
                if not det_p.exists():
                    return error_result(f"Detections file not found: {detections_path}")

                with open(det_p) as f:
                    data = json.load(f)

                for d in data.get("detections", []):
                    detections.append(Detection3D(**d))

            elif pointcloud_path and run_detection:
                # Run 3D detection
                pc_p = Path(pointcloud_path)
                if not pc_p.exists():
                    return error_result(f"Point cloud not found: {pointcloud_path}")

                self.report_progress(1, 3, "Running 3D detection...")

                from backend.cv.detection_3d import get_3d_detector

                detector = get_3d_detector(prefer_accuracy=True)
                detector.load()
                try:
                    result = detector.predict(str(pc_p), confidence=confidence)
                    detections = result.detections
                finally:
                    detector.unload()
            else:
                return error_result(
                    "Provide detections_path or set run_detection=True with pointcloud_path"
                )

            if not detections:
                return success_result(
                    {
                        "message": "No 3D detections to project",
                        "count": 0,
                        "visible_count": 0,
                        "projections": [],
                        "image_size": {"width": calib.width, "height": calib.height},
                    }
                )

            # Project to 2D
            self.report_progress(2, 3, "Projecting to 2D...")
            results = project_detections_3d_to_2d(detections, calib)

            projections = []
            visible_count = 0
            for det, box_2d in results:
                visible = box_2d is not None
                if visible:
                    visible_count += 1
                projections.append(
                    {
                        "class_name": det.class_name,
                        "confidence": round(det.confidence, 3),
                        "visible": visible,
                        "box_2d": box_2d,
                        "center_3d": det.center,
                        "dimensions": det.dimensions,
                    }
                )

            self.report_progress(3, 3, "Done")

            return success_result(
                {
                    "count": len(detections),
                    "visible_count": visible_count,
                    "image_size": {"width": calib.width, "height": calib.height},
                    "calibration_format": calib.source_format,
                    "has_extrinsic": calib.has_extrinsic,
                    "projections": projections,
                }
            )

        except Exception as e:
            logger.exception("3D to 2D projection failed")
            return error_result(f"Projection failed: {e}")
