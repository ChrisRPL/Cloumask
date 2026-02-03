"""
Camera calibration data structures and format parsers.

Supports loading calibration data from:
- KITTI format (.txt files with P0-P3, R0_rect, Tr_velo_to_cam)
- nuScenes format (JSON with camera_intrinsic, lidar_to_camera)
- ROS CameraInfo messages (K, D, R, P matrices)
- Custom JSON format

Implements spec: 05-point-cloud/05-2d-3d-fusion
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from backend.data.ros_types import CameraInfoMessage

logger = logging.getLogger(__name__)


class CameraCalibration(BaseModel):
    """Camera intrinsic and extrinsic calibration parameters.

    Stores camera calibration data in a unified format with support for
    different source formats (KITTI, nuScenes, ROS).

    Attributes:
        K: Intrinsic camera matrix (3x3).
        D: Distortion coefficients [k1, k2, p1, p2, k3, ...].
        width: Image width in pixels.
        height: Image height in pixels.
        T_cam_lidar: Camera-to-LiDAR extrinsic transform (4x4), optional.
        R: Rectification matrix (3x3), optional (for stereo).
        P: Projection matrix (3x4), optional (for stereo).
        distortion_model: Type of distortion model.
        source_format: Original calibration format for debugging.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Intrinsic matrix (3x3)
    K: Any = Field(..., description="Camera intrinsic matrix (3x3)")

    # Distortion coefficients
    D: Any = Field(
        default_factory=lambda: np.zeros(5, dtype=np.float64),
        description="Distortion coefficients",
    )

    # Image dimensions
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")

    # Camera to LiDAR extrinsic transform (4x4 homogeneous)
    T_cam_lidar: Any | None = Field(
        None, description="Camera-to-LiDAR transform (4x4)"
    )

    # Rectification matrix for stereo (3x3)
    R: Any | None = Field(None, description="Rectification matrix (3x3)")

    # Projection matrix for stereo (3x4)
    P: Any | None = Field(None, description="Projection matrix (3x4)")

    # Distortion model name
    distortion_model: Literal[
        "plumb_bob", "rational_polynomial", "equidistant", "none"
    ] = Field("plumb_bob", description="Distortion model type")

    # Source format for debugging
    source_format: str = Field("unknown", description="Original calibration format")

    @field_validator("K", mode="before")
    @classmethod
    def validate_K(cls, v: Any) -> "NDArray[np.float64]":
        """Validate and reshape intrinsic matrix to (3, 3)."""
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape == (9,):
            arr = arr.reshape(3, 3)
        if arr.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got shape {arr.shape}")
        return arr

    @field_validator("D", mode="before")
    @classmethod
    def validate_D(cls, v: Any) -> "NDArray[np.float64]":
        """Validate distortion coefficients."""
        if v is None:
            return np.zeros(5, dtype=np.float64)
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"D must be 1D, got shape {arr.shape}")
        return arr

    @field_validator("T_cam_lidar", mode="before")
    @classmethod
    def validate_T_cam_lidar(cls, v: Any) -> "NDArray[np.float64] | None":
        """Validate and reshape extrinsic transform to (4, 4)."""
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape == (12,):
            # 3x4 -> 4x4 with [0,0,0,1] bottom row
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = arr.reshape(3, 4)
            return T
        if arr.shape == (3, 4):
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = arr
            return T
        if arr.shape == (16,):
            arr = arr.reshape(4, 4)
        if arr.shape != (4, 4):
            raise ValueError(f"T_cam_lidar must be 4x4, got shape {arr.shape}")
        return arr

    @field_validator("R", mode="before")
    @classmethod
    def validate_R(cls, v: Any) -> "NDArray[np.float64] | None":
        """Validate rectification matrix."""
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape == (9,):
            arr = arr.reshape(3, 3)
        if arr.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got shape {arr.shape}")
        return arr

    @field_validator("P", mode="before")
    @classmethod
    def validate_P(cls, v: Any) -> "NDArray[np.float64] | None":
        """Validate projection matrix."""
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape == (12,):
            arr = arr.reshape(3, 4)
        if arr.shape != (3, 4):
            raise ValueError(f"P must be 3x4, got shape {arr.shape}")
        return arr

    @property
    def has_distortion(self) -> bool:
        """Check if non-zero distortion coefficients exist."""
        return self.D is not None and np.any(self.D != 0)

    @property
    def has_extrinsic(self) -> bool:
        """Check if extrinsic transform is available."""
        return self.T_cam_lidar is not None

    @classmethod
    def from_kitti(cls, calib_path: str, camera_id: int = 2) -> Self:
        """Load calibration from KITTI format file.

        KITTI calibration files contain:
        - P0-P3: Projection matrices for cameras 0-3
        - R0_rect: Rectification matrix
        - Tr_velo_to_cam: Velodyne to camera transform

        Args:
            calib_path: Path to KITTI calibration .txt file.
            camera_id: Camera to use (0-3), default 2 (left color).

        Returns:
            CameraCalibration instance.

        Raises:
            FileNotFoundError: If calibration file doesn't exist.
            ValueError: If required keys are missing.
        """
        path = Path(calib_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")

        # Parse KITTI format: "Key: val1 val2 val3 ..."
        calib: dict[str, np.ndarray] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, values = line.split(":", 1)
                calib[key.strip()] = np.array(
                    [float(v) for v in values.split()], dtype=np.float64
                )

        # Get projection matrix for specified camera
        p_key = f"P{camera_id}"
        if p_key not in calib:
            raise ValueError(f"Missing {p_key} in calibration file")

        P = calib[p_key].reshape(3, 4)
        K = P[:3, :3]

        # Build extrinsic: T_cam_lidar = R0_rect @ Tr_velo_to_cam
        T_cam_lidar = None
        if "R0_rect" in calib and "Tr_velo_to_cam" in calib:
            R0_rect = calib["R0_rect"].reshape(3, 3)
            Tr_velo = calib["Tr_velo_to_cam"].reshape(3, 4)

            # Build 4x4 transform
            T_cam_lidar = np.eye(4, dtype=np.float64)
            T_cam_lidar[:3, :3] = R0_rect @ Tr_velo[:3, :3]
            T_cam_lidar[:3, 3] = R0_rect @ Tr_velo[:3, 3]

        # KITTI images are typically already rectified/undistorted
        return cls(
            K=K,
            D=np.zeros(5, dtype=np.float64),
            width=1242,  # KITTI standard width
            height=375,  # KITTI standard height
            T_cam_lidar=T_cam_lidar,
            P=P,
            distortion_model="none",
            source_format="kitti",
        )

    @classmethod
    def from_nuscenes(cls, sensor_data: dict[str, Any]) -> Self:
        """Load calibration from nuScenes format.

        nuScenes provides calibration as JSON with:
        - camera_intrinsic: 3x3 intrinsic matrix
        - lidar_to_camera: 4x4 transform (optional)

        Args:
            sensor_data: Dict with nuScenes calibration data.

        Returns:
            CameraCalibration instance.
        """
        K = np.array(sensor_data["camera_intrinsic"], dtype=np.float64)
        if K.shape == (9,):
            K = K.reshape(3, 3)

        T_cam_lidar = None
        if "lidar_to_camera" in sensor_data:
            T_cam_lidar = np.array(
                sensor_data["lidar_to_camera"], dtype=np.float64
            )

        # nuScenes image dimensions
        width = sensor_data.get("width", 1600)
        height = sensor_data.get("height", 900)

        return cls(
            K=K,
            D=np.zeros(5, dtype=np.float64),
            width=width,
            height=height,
            T_cam_lidar=T_cam_lidar,
            distortion_model="none",
            source_format="nuscenes",
        )

    @classmethod
    def from_ros(cls, msg: "CameraInfoMessage") -> Self:
        """Load calibration from ROS CameraInfo message.

        Args:
            msg: CameraInfoMessage dataclass from ros_types.

        Returns:
            CameraCalibration instance.
        """
        # Map ROS distortion model names
        distortion_map = {
            "plumb_bob": "plumb_bob",
            "rational_polynomial": "rational_polynomial",
            "equidistant": "equidistant",
            "fisheye": "equidistant",
        }
        distortion_model = distortion_map.get(
            msg.distortion_model.lower(), "plumb_bob"
        )

        return cls(
            K=msg.K,
            D=msg.D,
            width=msg.width,
            height=msg.height,
            R=msg.R if msg.R is not None and msg.R.size > 0 else None,
            P=msg.P if msg.P is not None and msg.P.size > 0 else None,
            distortion_model=distortion_model,  # type: ignore[arg-type]
            source_format="ros",
        )

    @classmethod
    def from_json(cls, json_path: str) -> Self:
        """Load calibration from custom JSON format.

        JSON should contain:
        - K: 3x3 or flat array of 9 intrinsic values
        - D: distortion coefficients (optional)
        - width, height: image dimensions
        - T_cam_lidar: 4x4 or 3x4 extrinsic (optional)
        - R, P: stereo matrices (optional)

        Args:
            json_path: Path to JSON calibration file.

        Returns:
            CameraCalibration instance.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {json_path}")

        with open(path) as f:
            data = json.load(f)

        return cls(
            K=data["K"],
            D=data.get("D", np.zeros(5)),
            width=data["width"],
            height=data["height"],
            T_cam_lidar=data.get("T_cam_lidar"),
            R=data.get("R"),
            P=data.get("P"),
            distortion_model=data.get("distortion_model", "plumb_bob"),
            source_format="json",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibration to JSON-compatible dictionary.

        Returns:
            Dict with all calibration data as lists.
        """
        result = {
            "K": self.K.tolist(),
            "D": self.D.tolist(),
            "width": self.width,
            "height": self.height,
            "distortion_model": self.distortion_model,
            "source_format": self.source_format,
        }

        if self.T_cam_lidar is not None:
            result["T_cam_lidar"] = self.T_cam_lidar.tolist()
        if self.R is not None:
            result["R"] = self.R.tolist()
        if self.P is not None:
            result["P"] = self.P.tolist()

        return result
