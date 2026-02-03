"""
Tests for camera calibration module.

Tests CameraCalibration Pydantic model and format parsers for
KITTI, nuScenes, ROS, and JSON formats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.data.calibration import CameraCalibration


# Sample KITTI calibration content
KITTI_CALIB_CONTENT = """P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01
"""


@pytest.fixture
def kitti_calib_file(tmp_path: Path) -> Path:
    """Create a temporary KITTI calibration file."""
    calib_file = tmp_path / "calib.txt"
    calib_file.write_text(KITTI_CALIB_CONTENT)
    return calib_file


@pytest.fixture
def nuscenes_calib_data() -> dict:
    """Sample nuScenes calibration data."""
    return {
        "camera_intrinsic": [[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1]],
        "lidar_to_camera": [
            [0.01, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -0.1],
            [1.0, 0.01, 0.0, -0.5],
            [0, 0, 0, 1],
        ],
        "width": 1600,
        "height": 900,
    }


@pytest.fixture
def json_calib_file(tmp_path: Path) -> Path:
    """Create a temporary JSON calibration file."""
    import json

    calib_data = {
        "K": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "D": [0.1, -0.2, 0.0, 0.0, 0.0],
        "width": 640,
        "height": 480,
        "T_cam_lidar": [
            [1, 0, 0, 0.1],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1],
        ],
    }

    calib_file = tmp_path / "calib.json"
    calib_file.write_text(json.dumps(calib_data))
    return calib_file


class TestCameraCalibrationKITTI:
    """Tests for KITTI format calibration."""

    def test_from_kitti_default_camera(self, kitti_calib_file: Path) -> None:
        """Test loading KITTI calibration with default camera (P2)."""
        calib = CameraCalibration.from_kitti(str(kitti_calib_file), camera_id=2)

        assert calib.K.shape == (3, 3)
        assert calib.width == 1242
        assert calib.height == 375
        assert calib.source_format == "kitti"

    def test_from_kitti_has_extrinsic(self, kitti_calib_file: Path) -> None:
        """Test that KITTI calibration includes extrinsic transform."""
        calib = CameraCalibration.from_kitti(str(kitti_calib_file))

        assert calib.has_extrinsic is True
        assert calib.T_cam_lidar is not None
        assert calib.T_cam_lidar.shape == (4, 4)

    def test_from_kitti_no_distortion(self, kitti_calib_file: Path) -> None:
        """KITTI images are pre-rectified, should have no distortion."""
        calib = CameraCalibration.from_kitti(str(kitti_calib_file))

        assert calib.has_distortion is False
        assert np.allclose(calib.D, 0)

    def test_from_kitti_different_cameras(self, kitti_calib_file: Path) -> None:
        """Test loading different cameras from KITTI."""
        for camera_id in [0, 1, 2, 3]:
            calib = CameraCalibration.from_kitti(str(kitti_calib_file), camera_id)
            assert calib.K.shape == (3, 3)

    def test_from_kitti_file_not_found(self) -> None:
        """Test error handling for missing calibration file."""
        with pytest.raises(FileNotFoundError):
            CameraCalibration.from_kitti("/nonexistent/path.txt")


class TestCameraCalibrationNuScenes:
    """Tests for nuScenes format calibration."""

    def test_from_nuscenes(self, nuscenes_calib_data: dict) -> None:
        """Test loading nuScenes calibration."""
        calib = CameraCalibration.from_nuscenes(nuscenes_calib_data)

        assert calib.K.shape == (3, 3)
        assert calib.width == 1600
        assert calib.height == 900
        assert calib.source_format == "nuscenes"

    def test_from_nuscenes_has_extrinsic(self, nuscenes_calib_data: dict) -> None:
        """Test nuScenes calibration includes lidar-to-camera transform."""
        calib = CameraCalibration.from_nuscenes(nuscenes_calib_data)

        assert calib.has_extrinsic is True
        assert calib.T_cam_lidar.shape == (4, 4)


class TestCameraCalibrationJSON:
    """Tests for custom JSON format calibration."""

    def test_from_json(self, json_calib_file: Path) -> None:
        """Test loading JSON calibration."""
        calib = CameraCalibration.from_json(str(json_calib_file))

        assert calib.K.shape == (3, 3)
        assert calib.width == 640
        assert calib.height == 480
        assert calib.source_format == "json"

    def test_from_json_with_distortion(self, json_calib_file: Path) -> None:
        """Test JSON calibration with distortion coefficients."""
        calib = CameraCalibration.from_json(str(json_calib_file))

        assert calib.has_distortion is True
        assert len(calib.D) == 5


class TestCameraCalibrationValidation:
    """Tests for calibration validation."""

    def test_K_matrix_validation_3x3(self) -> None:
        """Test K matrix accepts 3x3 array."""
        calib = CameraCalibration(
            K=np.eye(3),
            width=640,
            height=480,
        )
        assert calib.K.shape == (3, 3)

    def test_K_matrix_validation_flat(self) -> None:
        """Test K matrix accepts flat 9-element array."""
        calib = CameraCalibration(
            K=np.eye(3).flatten(),
            width=640,
            height=480,
        )
        assert calib.K.shape == (3, 3)

    def test_K_matrix_validation_invalid(self) -> None:
        """Test K matrix rejects invalid shapes."""
        with pytest.raises(ValueError, match="K must be 3x3"):
            CameraCalibration(
                K=np.eye(4),
                width=640,
                height=480,
            )

    def test_T_cam_lidar_validation_4x4(self) -> None:
        """Test T_cam_lidar accepts 4x4 array."""
        calib = CameraCalibration(
            K=np.eye(3),
            width=640,
            height=480,
            T_cam_lidar=np.eye(4),
        )
        assert calib.T_cam_lidar.shape == (4, 4)

    def test_T_cam_lidar_validation_3x4(self) -> None:
        """Test T_cam_lidar accepts 3x4 array and converts to 4x4."""
        T_3x4 = np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]])
        calib = CameraCalibration(
            K=np.eye(3),
            width=640,
            height=480,
            T_cam_lidar=T_3x4,
        )
        assert calib.T_cam_lidar.shape == (4, 4)
        assert np.allclose(calib.T_cam_lidar[3], [0, 0, 0, 1])

    def test_to_dict_serialization(self, kitti_calib_file: Path) -> None:
        """Test calibration serializes to dict correctly."""
        calib = CameraCalibration.from_kitti(str(kitti_calib_file))
        data = calib.to_dict()

        assert "K" in data
        assert "width" in data
        assert "height" in data
        assert data["source_format"] == "kitti"
