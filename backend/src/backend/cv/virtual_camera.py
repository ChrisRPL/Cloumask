"""
Virtual camera generation for 3D point cloud anonymization.

Generates synthetic pinhole camera poses distributed around a point cloud
scene to enable multi-view face detection. Each camera produces a view
matrix and intrinsic parameters suitable for projecting 3D points to 2D.

Implements spec: 05-point-cloud/07-anonymization-3d (virtual camera section)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VirtualCamera:
    """
    Virtual pinhole camera with intrinsic and extrinsic parameters.

    Attributes:
        K: 3x3 intrinsic matrix.
        view_matrix: 4x4 camera-to-world matrix (camera pose).
        position: Camera position in world coordinates.
        resolution: (width, height) of the output image.
    """

    K: np.ndarray
    view_matrix: np.ndarray
    position: np.ndarray
    resolution: tuple[int, int]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.resolution[0]

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.resolution[1]

    @property
    def inv_view_matrix(self) -> np.ndarray:
        """World-to-camera transform (inverse of view_matrix)."""
        return np.linalg.inv(self.view_matrix)


def generate_virtual_cameras(
    bounds: tuple[np.ndarray, np.ndarray],
    num_views: int = 8,
    resolution: tuple[int, int] = (640, 480),
) -> list[VirtualCamera]:
    """
    Generate virtual camera poses distributed around a point cloud.

    Cameras are placed in a ring around the scene center at varying
    heights, all looking inward. This ensures 360-degree coverage
    for face detection in projected depth views.

    Args:
        bounds: (min_xyz, max_xyz) arrays defining the point cloud AABB.
        num_views: Number of viewpoints to generate (>=1).
        resolution: Output image resolution as (width, height).

    Returns:
        List of VirtualCamera objects.

    Raises:
        ValueError: If num_views < 1 or resolution dimensions are invalid.
    """
    if num_views < 1:
        raise ValueError(f"num_views must be >= 1, got {num_views}")
    if resolution[0] < 1 or resolution[1] < 1:
        raise ValueError(f"resolution must be positive, got {resolution}")

    min_xyz, max_xyz = np.asarray(bounds[0], dtype=np.float64), np.asarray(
        bounds[1], dtype=np.float64
    )
    center = (min_xyz + max_xyz) / 2
    size = float(np.max(max_xyz - min_xyz))

    # Place cameras at 1.5x the scene extent
    radius = size * 1.5
    if radius < 0.1:
        radius = 1.0  # Minimum radius for degenerate scenes

    cameras: list[VirtualCamera] = []

    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        # Alternate heights: odd cameras slightly above, even slightly below center
        height_offset = size * 0.3 * (i % 2 - 0.5)

        cam_pos = np.array(
            [
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                center[2] + height_offset,
            ],
            dtype=np.float64,
        )

        # Camera looks at scene center
        view_matrix = _build_look_at(cam_pos, center, up=np.array([0.0, 0.0, 1.0]))

        # Simple pinhole intrinsics (~90° horizontal FOV)
        fx = fy = float(resolution[0])
        cx, cy = resolution[0] / 2.0, resolution[1] / 2.0
        K = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=np.float64,
        )

        cameras.append(
            VirtualCamera(
                K=K,
                view_matrix=view_matrix,
                position=cam_pos,
                resolution=resolution,
            )
        )

    logger.debug(
        "Generated %d virtual cameras around center (%.1f, %.1f, %.1f), radius=%.1f",
        num_views,
        center[0],
        center[1],
        center[2],
        radius,
    )

    return cameras


def _build_look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """
    Build a 4x4 camera-to-world (view) matrix using look-at convention.

    The resulting matrix transforms from camera space to world space.
    Camera convention: -Z looks at target, +Y is up, +X is right.

    Args:
        eye: Camera position in world coordinates.
        target: Point the camera looks at.
        up: World up vector.

    Returns:
        4x4 camera-to-world transformation matrix.
    """
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        # Degenerate: camera at target, return identity
        return np.eye(4, dtype=np.float64)

    forward = forward / forward_norm

    # Right = forward x up (then normalise)
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        # Forward is parallel to up; pick an arbitrary perpendicular
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm

    # Recompute up to ensure orthogonality
    true_up = np.cross(right, forward)

    # Build camera-to-world matrix
    # Columns: right, true_up, -forward (OpenGL convention: camera looks along -Z)
    view = np.eye(4, dtype=np.float64)
    view[:3, 0] = right
    view[:3, 1] = true_up
    view[:3, 2] = -forward
    view[:3, 3] = eye

    return view
