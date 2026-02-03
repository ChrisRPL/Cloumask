"""
2D-3D detection fusion pipeline.

Matches 2D image detections with 3D LiDAR detections using
projected IoU and creates fused annotations.

Implements spec: 05-point-cloud/05-2d-3d-fusion
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.cv.types import BBox, Detection, Detection3D
    from backend.data.calibration import CameraCalibration

from backend.data.formats.fused_annotation import FusedAnnotation
from backend.data.projection import project_bbox3d_to_2d

logger = logging.getLogger(__name__)


def compute_iou_2d(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """Compute 2D IoU between two axis-aligned boxes.

    Args:
        box1: (x_min, y_min, x_max, y_max) first box.
        box2: (x_min, y_min, x_max, y_max) second box.

    Returns:
        IoU value in [0, 1].
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def fuse_detections(
    detections_2d: list["Detection"],
    detections_3d: list["Detection3D"],
    calib: "CameraCalibration",
    *,
    iou_threshold: float = 0.3,
    class_match_required: bool = False,
    img_w: int | None = None,
    img_h: int | None = None,
) -> list[FusedAnnotation]:
    """Match 2D and 3D detections using projected IoU.

    Uses greedy matching: pairs are sorted by IoU descending and
    matched one at a time. Unmatched detections are preserved.

    Args:
        detections_2d: List of 2D detections with normalized bboxes.
        detections_3d: List of 3D detections.
        calib: Camera calibration for projection.
        iou_threshold: Minimum IoU for a valid match.
        class_match_required: If True, only match same class names.
        img_w: Image width (defaults to calib.width).
        img_h: Image height (defaults to calib.height).

    Returns:
        List of FusedAnnotation objects (matched + unmatched).
    """
    from backend.cv.types import BBox

    img_w = img_w or calib.width
    img_h = img_h or calib.height

    # Project all 3D boxes to 2D
    projected_3d: list[tuple["Detection3D", tuple[float, float, float, float] | None]] = []
    for det3d in detections_3d:
        bbox_2d = project_bbox3d_to_2d(det3d, calib)
        projected_3d.append((det3d, bbox_2d))

    # Track which detections have been matched
    matched_2d: set[int] = set()
    matched_3d: set[int] = set()
    fused: list[FusedAnnotation] = []

    # Build candidate pairs with IoU
    pairs: list[tuple[float, int, int, "Detection", "Detection3D", tuple[float, float, float, float]]] = []

    for i, det2d in enumerate(detections_2d):
        # Convert normalized bbox to pixel coordinates
        box2d_pixels = det2d.bbox.to_xyxy(img_w, img_h)
        box2d_tuple = (
            float(box2d_pixels[0]),
            float(box2d_pixels[1]),
            float(box2d_pixels[2]),
            float(box2d_pixels[3]),
        )

        for j, (det3d, proj_box) in enumerate(projected_3d):
            if proj_box is None:
                continue

            # Check class match if required
            if class_match_required:
                if det2d.class_name.lower() != det3d.class_name.lower():
                    continue

            iou = compute_iou_2d(box2d_tuple, proj_box)
            if iou >= iou_threshold:
                pairs.append((iou, i, j, det2d, det3d, proj_box))

    # Sort by IoU descending for greedy matching
    pairs.sort(key=lambda x: -x[0])

    # Greedy matching
    for iou, i, j, det2d, det3d, _proj_box in pairs:
        if i in matched_2d or j in matched_3d:
            continue

        matched_2d.add(i)
        matched_3d.add(j)

        # Compute depth from 3D center in camera frame
        depth = None
        if calib.T_cam_lidar is not None:
            center_lidar = np.array([*det3d.center, 1.0], dtype=np.float64)
            center_cam = calib.T_cam_lidar @ center_lidar
            depth = float(center_cam[2])
        else:
            depth = float(det3d.center[2])

        fused.append(
            FusedAnnotation(
                bbox_2d=det2d.bbox,
                detection_3d=det3d,
                class_id=det2d.class_id,
                class_name=det2d.class_name,
                confidence_2d=det2d.confidence,
                confidence_3d=det3d.confidence,
                iou_2d_3d=iou,
                depth_meters=depth,
            )
        )

    # Add unmatched 2D detections
    for i, det2d in enumerate(detections_2d):
        if i not in matched_2d:
            fused.append(
                FusedAnnotation(
                    bbox_2d=det2d.bbox,
                    detection_3d=None,
                    class_id=det2d.class_id,
                    class_name=det2d.class_name,
                    confidence_2d=det2d.confidence,
                    confidence_3d=None,
                )
            )

    # Add unmatched 3D detections (create 2D bbox from projection)
    for j, (det3d, proj_box) in enumerate(projected_3d):
        if j not in matched_3d and proj_box is not None:
            # Create normalized bbox from projected pixel coordinates
            x_min, y_min, x_max, y_max = proj_box
            bbox_2d = BBox.from_xyxy(
                int(x_min), int(y_min), int(x_max), int(y_max), img_w, img_h
            )

            # Compute depth
            depth = None
            if calib.T_cam_lidar is not None:
                center_lidar = np.array([*det3d.center, 1.0], dtype=np.float64)
                center_cam = calib.T_cam_lidar @ center_lidar
                depth = float(center_cam[2])
            else:
                depth = float(det3d.center[2])

            fused.append(
                FusedAnnotation(
                    bbox_2d=bbox_2d,
                    detection_3d=det3d,
                    class_id=det3d.class_id,
                    class_name=det3d.class_name,
                    confidence_2d=det3d.confidence,  # Use 3D confidence as 2D proxy
                    confidence_3d=det3d.confidence,
                    depth_meters=depth,
                )
            )

    return fused
