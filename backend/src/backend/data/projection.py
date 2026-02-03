"""
3D-2D projection and 2D-3D lifting functions for sensor fusion.

Provides vectorized operations for:
- Projecting 3D points to 2D image coordinates
- Projecting 3D bounding boxes to 2D image boxes
- Lifting 2D detections to 3D using point cloud depth

Uses OpenCV for distortion handling and numpy for performance.

Implements spec: 05-point-cloud/05-2d-3d-fusion
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from backend.cv.types import Detection3D
    from backend.data.calibration import CameraCalibration

logger = logging.getLogger(__name__)


def transform_points(
    points: "NDArray[np.float64]",
    T: "NDArray[np.float64]",
) -> "NDArray[np.float64]":
    """Apply 4x4 homogeneous transformation to 3D points.

    Args:
        points: (N, 3) array of 3D points.
        T: (4, 4) transformation matrix.

    Returns:
        (N, 3) array of transformed points.
    """
    N = points.shape[0]
    if N == 0:
        return points.copy()

    # Convert to homogeneous coordinates
    ones = np.ones((N, 1), dtype=np.float64)
    points_h = np.hstack([points, ones])  # (N, 4)

    # Apply transform
    transformed = (T @ points_h.T).T  # (N, 4)

    return transformed[:, :3]


def project_points_to_image(
    points_3d: "NDArray[np.float64]",
    calib: "CameraCalibration",
    *,
    filter_behind: bool = True,
    filter_outside: bool = True,
) -> tuple["NDArray[np.float64]", "NDArray[np.bool_]"]:
    """Project 3D points to 2D image coordinates.

    Uses vectorized numpy operations for performance. Achieves <10ms
    for 100K points on modern hardware.

    Args:
        points_3d: (N, 3) array of 3D points in LiDAR frame.
        calib: Camera calibration data.
        filter_behind: Exclude points behind camera (z <= 0).
        filter_outside: Exclude points outside image bounds.

    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates (u, v).
        valid_mask: (N,) boolean mask indicating valid projections.
    """
    N = points_3d.shape[0]
    if N == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)

    # Start with all valid
    valid_mask = np.ones(N, dtype=bool)

    # Transform to camera frame if extrinsic available
    if calib.T_cam_lidar is not None:
        points_cam = transform_points(points_3d, calib.T_cam_lidar)
    else:
        points_cam = points_3d.copy()

    # Filter points behind camera (z <= 0)
    if filter_behind:
        valid_mask &= points_cam[:, 2] > 0

    # Initialize output
    points_2d = np.full((N, 2), np.nan, dtype=np.float64)

    # Get indices of valid points for projection
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return points_2d, valid_mask

    valid_points = points_cam[valid_indices]

    # Project using OpenCV if distortion present
    if calib.has_distortion:
        # cv2.projectPoints expects (N, 1, 3) or (N, 3)
        rvec = np.zeros(3, dtype=np.float64)  # No rotation (already in camera frame)
        tvec = np.zeros(3, dtype=np.float64)  # No translation

        projected, _ = cv2.projectPoints(
            valid_points.astype(np.float64),
            rvec,
            tvec,
            calib.K.astype(np.float64),
            calib.D.astype(np.float64),
        )
        points_2d[valid_indices] = projected.squeeze()
    else:
        # Simple pinhole projection (vectorized)
        z = valid_points[:, 2:3]  # (M, 1)
        xy_normalized = valid_points[:, :2] / z  # (M, 2)

        # Apply intrinsic matrix: u = fx * x' + cx, v = fy * y' + cy
        fx = calib.K[0, 0]
        fy = calib.K[1, 1]
        cx = calib.K[0, 2]
        cy = calib.K[1, 2]

        points_2d[valid_indices, 0] = fx * xy_normalized[:, 0] + cx
        points_2d[valid_indices, 1] = fy * xy_normalized[:, 1] + cy

    # Filter points outside image bounds
    if filter_outside:
        in_image = (
            (points_2d[:, 0] >= 0)
            & (points_2d[:, 0] < calib.width)
            & (points_2d[:, 1] >= 0)
            & (points_2d[:, 1] < calib.height)
        )
        valid_mask &= in_image

    return points_2d, valid_mask


def project_bbox3d_to_2d(
    detection: "Detection3D",
    calib: "CameraCalibration",
) -> tuple[float, float, float, float] | None:
    """Project 3D bounding box to 2D image box.

    Projects all 8 corners of the 3D box and returns the bounding
    rectangle of valid projections.

    Args:
        detection: 3D detection with center, dimensions, rotation.
        calib: Camera calibration data.

    Returns:
        (x_min, y_min, x_max, y_max) in pixels, or None if not visible.
    """
    # Get 8 corners of the 3D box
    corners_3d = detection.to_corners()  # (8, 3)

    # Project corners to image
    corners_2d, valid = project_points_to_image(
        corners_3d,
        calib,
        filter_behind=True,
        filter_outside=False,  # Allow corners outside for bounding rect
    )

    # Need at least 1 valid corner
    if not np.any(valid):
        return None

    valid_corners = corners_2d[valid]

    # Compute bounding rectangle and clip to image
    x_min = float(np.clip(valid_corners[:, 0].min(), 0, calib.width))
    x_max = float(np.clip(valid_corners[:, 0].max(), 0, calib.width))
    y_min = float(np.clip(valid_corners[:, 1].min(), 0, calib.height))
    y_max = float(np.clip(valid_corners[:, 1].max(), 0, calib.height))

    # Check for degenerate boxes (zero area)
    if x_max <= x_min or y_max <= y_min:
        return None

    return (x_min, y_min, x_max, y_max)


def project_detections_3d_to_2d(
    detections: list["Detection3D"],
    calib: "CameraCalibration",
) -> list[tuple["Detection3D", tuple[float, float, float, float] | None]]:
    """Batch project multiple 3D detections to 2D boxes.

    Args:
        detections: List of 3D detections.
        calib: Camera calibration data.

    Returns:
        List of (detection, bbox_2d) tuples where bbox_2d may be None.
    """
    results = []
    for det in detections:
        bbox_2d = project_bbox3d_to_2d(det, calib)
        results.append((det, bbox_2d))
    return results


def lift_2d_to_3d(
    bbox_2d: tuple[float, float, float, float],
    points_3d: "NDArray[np.float64]",
    calib: "CameraCalibration",
    *,
    class_name: str | None = None,
    class_priors: dict[str, dict] | None = None,
    min_points: int = 3,
) -> "Detection3D | None":
    """Lift 2D bounding box to 3D using point cloud depth.

    Projects all points to image, finds points inside the 2D box
    (frustum query), and estimates 3D position using median.

    Args:
        bbox_2d: (x_min, y_min, x_max, y_max) in pixels.
        points_3d: (N, 3+) point cloud in LiDAR frame.
        calib: Camera calibration data.
        class_name: Optional class name for size priors.
        class_priors: Dict mapping class names to {"dimensions": (l, w, h)}.
        min_points: Minimum points required in frustum.

    Returns:
        Detection3D with estimated position, or None if insufficient points.
    """
    from backend.cv.types import Detection3D

    # Project all points to image
    points_2d, valid = project_points_to_image(
        points_3d[:, :3],
        calib,
        filter_behind=True,
        filter_outside=True,
    )

    # Find points inside 2D box (frustum query)
    x_min, y_min, x_max, y_max = bbox_2d
    in_box = (
        valid
        & (points_2d[:, 0] >= x_min)
        & (points_2d[:, 0] <= x_max)
        & (points_2d[:, 1] >= y_min)
        & (points_2d[:, 1] <= y_max)
    )

    frustum_points = points_3d[in_box, :3]

    if len(frustum_points) < min_points:
        logger.debug(
            "Insufficient points in frustum: %d < %d",
            len(frustum_points),
            min_points,
        )
        return None

    # Estimate center using robust median
    center = np.median(frustum_points, axis=0)

    # Get dimensions from class prior or point spread
    if class_priors and class_name and class_name in class_priors:
        dimensions = tuple(class_priors[class_name].get("dimensions", (4.0, 1.8, 1.5)))
    else:
        # Use point spread with some padding
        point_range = frustum_points.max(axis=0) - frustum_points.min(axis=0)
        # Ensure minimum dimensions
        dimensions = tuple(np.maximum(point_range * 1.1, [0.5, 0.5, 0.5]))

    # Estimate rotation using PCA on XY plane
    rotation = _estimate_rotation_pca(frustum_points)

    return Detection3D(
        class_id=0,
        class_name=class_name or "unknown",
        center=(float(center[0]), float(center[1]), float(center[2])),
        dimensions=dimensions,  # type: ignore[arg-type]
        rotation=rotation,
        confidence=0.5,  # Lower confidence for lifted detections
    )


def _estimate_rotation_pca(points: "NDArray[np.float64]") -> float:
    """Estimate yaw rotation using PCA on XY plane.

    Finds the principal direction in the XY plane and returns
    the angle as the estimated yaw.

    Args:
        points: (N, 3) point cloud.

    Returns:
        Yaw angle in radians.
    """
    if len(points) < 3:
        return 0.0

    # Use XY coordinates only
    xy = points[:, :2]
    centered = xy - xy.mean(axis=0)

    # PCA via SVD
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        principal_dir = Vt[0]  # First principal component
        yaw = float(np.arctan2(principal_dir[1], principal_dir[0]))
        return yaw
    except np.linalg.LinAlgError:
        logger.warning("SVD failed in PCA rotation estimation")
        return 0.0
