"""
2D-3D face region mapping for point cloud anonymization.

Projects 3D points into virtual camera views, renders depth/intensity
images for face detection, and maps detected 2D face boxes back to
3D point indices via frustum queries.

Implements spec: 05-point-cloud/07-anonymization-3d (projection section)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.cv.virtual_camera import VirtualCamera

logger = logging.getLogger(__name__)


def project_points_to_camera(
    points: np.ndarray,
    camera: VirtualCamera,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points into a virtual camera's image plane.

    Args:
        points: (N, 3) array of 3D points in world coordinates.
        camera: Virtual camera with intrinsics and pose.

    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates (u, v).
        depths: (N,) array of depth values in camera frame.
        valid: (N,) boolean mask for points in front of camera and
               inside image bounds.
    """
    N = points.shape[0]
    if N == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=bool),
        )

    # Transform world points to camera frame
    inv_view = camera.inv_view_matrix
    R = inv_view[:3, :3]
    t = inv_view[:3, 3]
    points_cam = (R @ points.T).T + t  # (N, 3)

    # Depth along camera Z axis (OpenGL: camera looks along -Z)
    depths = -points_cam[:, 2]

    # Start with points in front of camera
    valid = depths > 0

    # Project to 2D using intrinsics
    K = camera.K
    # Avoid division by zero for points behind camera
    safe_z = np.where(valid, -points_cam[:, 2], 1.0)

    u = K[0, 0] * points_cam[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / safe_z + K[1, 2]

    points_2d = np.stack([u, v], axis=1)

    # Filter to image bounds
    W, H = camera.resolution
    valid &= (u >= 0) & (u < W) & (v >= 0) & (v < H)

    return points_2d, depths, valid


def render_depth_image(
    points: np.ndarray,
    camera: VirtualCamera,
    *,
    point_radius: int = 2,
) -> np.ndarray:
    """
    Render a depth/intensity image by splatting 3D points.

    Produces a 3-channel uint8 image suitable for face detection models.
    Closer points appear brighter. Point splatting uses a small kernel
    to fill gaps.

    Args:
        points: (N, 3) array of 3D points in world coordinates.
        camera: Virtual camera definition.
        point_radius: Splat radius in pixels for gap filling.

    Returns:
        (H, W, 3) uint8 RGB image with depth-encoded intensity.
    """
    W, H = camera.resolution
    depth_buffer = np.zeros((H, W), dtype=np.float32)

    points_2d, depths, valid = project_points_to_camera(points, camera)

    if not np.any(valid):
        return np.zeros((H, W, 3), dtype=np.uint8)

    # Use only valid points
    u = points_2d[valid, 0].astype(np.int32)
    v = points_2d[valid, 1].astype(np.int32)
    d = depths[valid]

    # Inverse depth for rendering (closer = brighter)
    inv_depth = 1.0 / np.maximum(d, 0.01)

    # Splat points with small radius for better coverage
    if point_radius <= 1:
        # Single pixel splatting (fastest)
        # Use maximum to handle overlapping points (closest wins)
        np.maximum.at(depth_buffer, (v, u), inv_depth)
    else:
        # Multi-pixel splatting
        for du in range(-point_radius, point_radius + 1):
            for dv in range(-point_radius, point_radius + 1):
                uu = np.clip(u + du, 0, W - 1)
                vv = np.clip(v + dv, 0, H - 1)
                np.maximum.at(depth_buffer, (vv, uu), inv_depth)

    # Normalise to 0-255
    max_val = depth_buffer.max()
    if max_val > 0:
        normalized = (depth_buffer / max_val * 255).astype(np.uint8)
    else:
        normalized = np.zeros((H, W), dtype=np.uint8)

    # Convert to 3-channel RGB (face detectors expect colour images)
    return np.stack([normalized, normalized, normalized], axis=-1)


def find_points_in_2d_box(
    points_2d: np.ndarray,
    valid: np.ndarray,
    box: tuple[float, float, float, float],
    margin: float = 1.0,
) -> np.ndarray:
    """
    Find 3D point indices whose projections fall inside a 2D box.

    Args:
        points_2d: (N, 2) projected pixel coordinates.
        valid: (N,) boolean mask of valid projections.
        box: (x_min, y_min, x_max, y_max) in pixel coordinates.
        margin: Factor to expand the box (1.0 = no expansion, 1.2 = 20%).

    Returns:
        1D array of point indices inside the (optionally expanded) box.
    """
    x_min, y_min, x_max, y_max = box

    if margin != 1.0:
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        hw = (x_max - x_min) / 2 * margin
        hh = (y_max - y_min) / 2 * margin
        x_min, y_min = cx - hw, cy - hh
        x_max, y_max = cx + hw, cy + hh

    in_box = (
        valid
        & (points_2d[:, 0] >= x_min)
        & (points_2d[:, 0] <= x_max)
        & (points_2d[:, 1] >= y_min)
        & (points_2d[:, 1] <= y_max)
    )

    return np.where(in_box)[0]


def expand_box(
    box: tuple[float, float, float, float],
    margin: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """
    Expand a 2D bounding box by a margin factor, clamped to image bounds.

    Args:
        box: (x_min, y_min, x_max, y_max) in pixels.
        margin: Expansion factor (1.0 = no change, 1.2 = 20% larger).
        img_w: Image width for clamping.
        img_h: Image height for clamping.

    Returns:
        Expanded and clamped (x_min, y_min, x_max, y_max).
    """
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    hw = (x_max - x_min) / 2 * margin
    hh = (y_max - y_min) / 2 * margin

    return (
        max(0.0, cx - hw),
        max(0.0, cy - hh),
        min(float(img_w), cx + hw),
        min(float(img_h), cy + hh),
    )
