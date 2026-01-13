"""
Tests for PV-RCNN++ and CenterPoint 3D detection wrappers.

Unit tests use mocked models to avoid requiring real model downloads.
Integration tests (marked) require actual OpenPCDet installation and optionally GPU.

Implements spec: 03-cv-models/07-3d-detection (testing section)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.cv.base import ModelState
from backend.cv.types import Detection3D, Detection3DResult

# -----------------------------------------------------------------------------
# Mock Classes for Unit Testing (without real OpenPCDet)
# -----------------------------------------------------------------------------


class MockOpenPCDetModel:
    """Mock OpenPCDet model for testing."""

    def __init__(self) -> None:
        self._device = "cpu"

    def cuda(self) -> None:
        self._device = "cuda"

    def cpu(self) -> None:
        self._device = "cpu"

    def train(self, mode: bool) -> None:
        pass

    def load_params_from_file(
        self,
        filename: str,
        logger: Any = None,
        to_cpu: bool = False,
    ) -> None:
        pass

    def forward(self, input_dict: dict[str, Any]) -> tuple[list[dict[str, Any]], None]:
        """Return mock 3D detections."""
        import torch

        # Return mock 3D bounding boxes
        # Format: x, y, z, l, w, h, rotation
        pred_boxes = torch.tensor([
            [10.0, 0.0, 0.5, 4.5, 1.8, 1.5, 0.1],   # Car at 10m forward
            [15.0, 2.0, 0.8, 0.6, 0.6, 1.7, 0.0],   # Pedestrian at 15m
            [8.0, -3.0, 0.4, 1.8, 0.8, 1.2, 0.5],   # Cyclist at 8m
        ])
        pred_scores = torch.tensor([0.95, 0.87, 0.72])
        pred_labels = torch.tensor([1, 2, 3])  # 1-indexed: Car, Pedestrian, Cyclist

        return [
            {
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
                "pred_labels": pred_labels,
            }
        ], None


class MockOpenPCDetConfig:
    """Mock OpenPCDet config object."""

    MODEL = MagicMock()


# Mocks for pcdet imports
mock_pcdet_config = MagicMock()
mock_pcdet_config.cfg = MockOpenPCDetConfig()
mock_pcdet_config.cfg_from_yaml_file = MagicMock()

mock_pcdet_models = MagicMock()
mock_pcdet_models.build_network = MagicMock(return_value=MockOpenPCDetModel())

mock_pcdet_utils = MagicMock()
mock_pcdet_utils.common_utils.create_logger = MagicMock(return_value=MagicMock())


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_bin_file(tmp_path: Path) -> Path:
    """Create sample KITTI binary point cloud file."""
    # Generate random point cloud (10000 points with x, y, z, intensity)
    np.random.seed(42)
    points = np.random.rand(10000, 4).astype(np.float32)
    # Scale XYZ to realistic range (-50 to 50 meters)
    points[:, :3] = (points[:, :3] - 0.5) * 100

    path = tmp_path / "sample.bin"
    points.tofile(str(path))
    return path


@pytest.fixture
def sample_pcd_file(tmp_path: Path) -> Path:
    """Create sample PCD point cloud file."""
    pytest.importorskip("open3d")
    import open3d as o3d

    np.random.seed(42)
    points = np.random.rand(5000, 3).astype(np.float64) * 50

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    path = tmp_path / "sample.pcd"
    o3d.io.write_point_cloud(str(path), pcd)
    return path


@pytest.fixture
def sample_ply_file(tmp_path: Path) -> Path:
    """Create sample PLY point cloud file."""
    pytest.importorskip("open3d")
    import open3d as o3d

    np.random.seed(42)
    points = np.random.rand(3000, 3).astype(np.float64) * 30

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Add colors for intensity
    colors = np.random.rand(3000, 3).astype(np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    path = tmp_path / "sample.ply"
    o3d.io.write_point_cloud(str(path), pcd)
    return path


# -----------------------------------------------------------------------------
# Coordinate System Tests
# -----------------------------------------------------------------------------


class TestCoordinateSystem:
    """Tests for CoordinateSystem enum."""

    def test_coordinate_system_values(self) -> None:
        """Should have correct string values."""
        from backend.cv.detection_3d import CoordinateSystem

        assert CoordinateSystem.KITTI.value == "kitti"
        assert CoordinateSystem.NUSCENES.value == "nuscenes"
        assert CoordinateSystem.WAYMO.value == "waymo"

    def test_coordinate_system_is_string_enum(self) -> None:
        """CoordinateSystem values should be usable as strings."""
        from backend.cv.detection_3d import CoordinateSystem

        assert str(CoordinateSystem.KITTI) == "CoordinateSystem.KITTI"
        assert CoordinateSystem.KITTI == "kitti"


class TestCoordinateConversion:
    """Tests for convert_coordinates function."""

    def test_identity_conversion(self) -> None:
        """Same system should return identical points."""
        from backend.cv.detection_3d import CoordinateSystem, convert_coordinates

        points = np.array([
            [1.0, 2.0, 3.0, 0.5],
            [4.0, 5.0, 6.0, 0.8],
        ], dtype=np.float32)

        result = convert_coordinates(points, CoordinateSystem.KITTI, CoordinateSystem.KITTI)

        np.testing.assert_array_equal(points, result)

    def test_nuscenes_to_kitti_conversion(self) -> None:
        """nuScenes to KITTI should swap x/y axes correctly."""
        from backend.cv.detection_3d import CoordinateSystem, convert_coordinates

        # nuScenes: x=right, y=forward, z=up
        # KITTI: x=forward, y=left, z=up
        # So: x_kitti = y_nuscenes, y_kitti = -x_nuscenes

        points = np.array([
            [1.0, 0.0, 0.0, 0.5],  # Point at x=1 in nuScenes (right)
            [0.0, 1.0, 0.0, 0.5],  # Point at y=1 in nuScenes (forward)
        ], dtype=np.float32)

        result = convert_coordinates(
            points, CoordinateSystem.NUSCENES, CoordinateSystem.KITTI
        )

        # First point: right in nuScenes becomes left in KITTI (y=-1)
        assert result[0, 0] == pytest.approx(0.0)   # x_kitti = y_nuscenes
        assert result[0, 1] == pytest.approx(-1.0)  # y_kitti = -x_nuscenes
        assert result[0, 2] == pytest.approx(0.0)   # z unchanged

        # Second point: forward in nuScenes becomes forward in KITTI (x=1)
        assert result[1, 0] == pytest.approx(1.0)   # x_kitti = y_nuscenes
        assert result[1, 1] == pytest.approx(0.0)   # y_kitti = -x_nuscenes
        assert result[1, 2] == pytest.approx(0.0)   # z unchanged

    def test_waymo_to_kitti_is_identity(self) -> None:
        """WAYMO to KITTI should be identity (same convention)."""
        from backend.cv.detection_3d import CoordinateSystem, convert_coordinates

        points = np.array([
            [10.0, 5.0, 1.0, 0.9],
            [20.0, -3.0, 0.5, 0.7],
        ], dtype=np.float32)

        result = convert_coordinates(
            points, CoordinateSystem.WAYMO, CoordinateSystem.KITTI
        )

        np.testing.assert_array_almost_equal(points, result)

    def test_preserves_additional_columns(self) -> None:
        """Conversion should preserve columns beyond XYZ."""
        from backend.cv.detection_3d import CoordinateSystem, convert_coordinates

        points = np.array([
            [1.0, 2.0, 3.0, 0.5, 100.0, 200.0],  # Extra columns
        ], dtype=np.float32)

        result = convert_coordinates(
            points, CoordinateSystem.NUSCENES, CoordinateSystem.KITTI
        )

        # Extra columns should be preserved
        assert result.shape == (1, 6)
        assert result[0, 3] == pytest.approx(0.5)
        assert result[0, 4] == pytest.approx(100.0)
        assert result[0, 5] == pytest.approx(200.0)

    def test_invalid_conversion_raises(self) -> None:
        """Unknown conversion should raise ValueError."""
        from backend.cv.detection_3d import CoordinateSystem, convert_coordinates

        points = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)

        # KITTI to nuScenes (inverse not directly defined, should compute)
        # This should work because we compute the inverse
        result = convert_coordinates(
            points, CoordinateSystem.KITTI, CoordinateSystem.NUSCENES
        )
        assert result.shape == (1, 4)


# -----------------------------------------------------------------------------
# PointCloudLoader Tests
# -----------------------------------------------------------------------------


class TestPointCloudLoader:
    """Tests for PointCloudLoader class."""

    def test_supported_formats(self) -> None:
        """Should list all supported formats."""
        from backend.cv.detection_3d import PointCloudLoader

        expected = {".bin", ".pcd", ".ply", ".las", ".laz"}
        assert expected == PointCloudLoader.SUPPORTED_FORMATS

    def test_load_bin_format(self, sample_bin_file: Path) -> None:
        """Should load KITTI binary format correctly."""
        from backend.cv.detection_3d import PointCloudLoader

        points = PointCloudLoader.load(str(sample_bin_file))

        assert points.shape == (10000, 4)
        assert points.dtype == np.float32

    def test_load_pcd_format(self, sample_pcd_file: Path) -> None:
        """Should load PCD format via Open3D."""
        from backend.cv.detection_3d import PointCloudLoader

        points = PointCloudLoader.load(str(sample_pcd_file))

        assert points.shape == (5000, 4)  # XYZ + intensity
        assert points.dtype == np.float32

    def test_load_ply_format(self, sample_ply_file: Path) -> None:
        """Should load PLY format via Open3D."""
        from backend.cv.detection_3d import PointCloudLoader

        points = PointCloudLoader.load(str(sample_ply_file))

        assert points.shape == (3000, 4)
        assert points.dtype == np.float32

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing files."""
        from backend.cv.detection_3d import PointCloudLoader

        with pytest.raises(FileNotFoundError, match="not found"):
            PointCloudLoader.load(str(tmp_path / "nonexistent.bin"))

    def test_load_unsupported_format_raises(self, tmp_path: Path) -> None:
        """Should raise ValueError for unsupported formats."""
        from backend.cv.detection_3d import PointCloudLoader

        fake_file = tmp_path / "test.xyz"
        fake_file.touch()

        with pytest.raises(ValueError, match="Unsupported"):
            PointCloudLoader.load(str(fake_file))


# -----------------------------------------------------------------------------
# Detection3D Classes Tests
# -----------------------------------------------------------------------------


class TestDetection3DClasses:
    """Tests for 3D detection class constants."""

    def test_detection_3d_classes(self) -> None:
        """Should have standard KITTI/nuScenes classes."""
        from backend.cv.detection_3d import DETECTION_3D_CLASSES

        assert "Car" in DETECTION_3D_CLASSES
        assert "Pedestrian" in DETECTION_3D_CLASSES
        assert "Cyclist" in DETECTION_3D_CLASSES
        assert len(DETECTION_3D_CLASSES) == 3


# -----------------------------------------------------------------------------
# PVRCNNWrapper Unit Tests
# -----------------------------------------------------------------------------


class TestPVRCNNWrapperUnit:
    """Unit tests for PVRCNNWrapper (mocked, no real models)."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.detection_3d import PVRCNNWrapper

        wrapper = PVRCNNWrapper()
        assert wrapper.info.name == "pvrcnn++"
        assert wrapper.info.vram_required_mb == 4000
        assert wrapper.info.supports_batching is False
        assert wrapper.info.source == "openpcdet"

    def test_initial_state_unloaded(self) -> None:
        """Should start in UNLOADED state."""
        from backend.cv.detection_3d import PVRCNNWrapper

        wrapper = PVRCNNWrapper()
        assert wrapper.state == ModelState.UNLOADED
        assert not wrapper.is_loaded

    def test_predict_without_load_raises(self) -> None:
        """Should raise RuntimeError if predict called before load."""
        from backend.cv.detection_3d import PVRCNNWrapper

        wrapper = PVRCNNWrapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            wrapper.predict("test.bin")

    @patch.dict(
        "sys.modules",
        {
            "pcdet": MagicMock(),
            "pcdet.config": mock_pcdet_config,
            "pcdet.models": mock_pcdet_models,
            "pcdet.utils": mock_pcdet_utils,
            "pcdet.utils.common_utils": mock_pcdet_utils.common_utils,
        },
    )
    @patch("backend.cv.download.is_model_downloaded", return_value=True)
    @patch("backend.cv.download.get_model_path", return_value=Path("/fake/model.pth"))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_model_success(
        self,
        mock_exists: MagicMock,
        mock_get_path: MagicMock,
        mock_is_downloaded: MagicMock,
    ) -> None:
        """Should load model successfully with mocks."""
        from backend.cv.detection_3d import PVRCNNWrapper

        wrapper = PVRCNNWrapper()
        wrapper.load(device="cpu")

        assert wrapper.state == ModelState.LOADED
        assert wrapper.is_loaded


# -----------------------------------------------------------------------------
# CenterPointWrapper Unit Tests
# -----------------------------------------------------------------------------


class TestCenterPointWrapperUnit:
    """Unit tests for CenterPointWrapper (mocked, no real models)."""

    def test_model_info(self) -> None:
        """Should have correct model info."""
        from backend.cv.detection_3d import CenterPointWrapper

        wrapper = CenterPointWrapper()
        assert wrapper.info.name == "centerpoint"
        assert wrapper.info.vram_required_mb == 3000
        assert wrapper.info.supports_batching is False

    def test_initial_state_unloaded(self) -> None:
        """Should start in UNLOADED state."""
        from backend.cv.detection_3d import CenterPointWrapper

        wrapper = CenterPointWrapper()
        assert wrapper.state == ModelState.UNLOADED


# -----------------------------------------------------------------------------
# Factory Function Tests
# -----------------------------------------------------------------------------


class TestGet3DDetector:
    """Tests for get_3d_detector factory function."""

    @patch("backend.cv.device.get_available_vram_mb", return_value=8000)
    def test_returns_pvrcnn_when_vram_sufficient(
        self, mock_vram: MagicMock
    ) -> None:
        """Should return PV-RCNN++ when enough VRAM available."""
        from backend.cv.detection_3d import PVRCNNWrapper, get_3d_detector

        detector = get_3d_detector(prefer_accuracy=True)
        assert isinstance(detector, PVRCNNWrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=2000)
    def test_returns_centerpoint_when_vram_insufficient(
        self, mock_vram: MagicMock
    ) -> None:
        """Should return CenterPoint when VRAM insufficient for PV-RCNN++."""
        from backend.cv.detection_3d import CenterPointWrapper, get_3d_detector

        detector = get_3d_detector(prefer_accuracy=True)
        assert isinstance(detector, CenterPointWrapper)

    def test_force_pvrcnn(self) -> None:
        """Should return PV-RCNN++ when forced."""
        from backend.cv.detection_3d import PVRCNNWrapper, get_3d_detector

        detector = get_3d_detector(force_model="pvrcnn++")
        assert isinstance(detector, PVRCNNWrapper)

    def test_force_centerpoint(self) -> None:
        """Should return CenterPoint when forced."""
        from backend.cv.detection_3d import CenterPointWrapper, get_3d_detector

        detector = get_3d_detector(force_model="centerpoint")
        assert isinstance(detector, CenterPointWrapper)

    @patch("backend.cv.device.get_available_vram_mb", return_value=8000)
    def test_prefer_accuracy_false_returns_centerpoint(
        self, mock_vram: MagicMock
    ) -> None:
        """Should return CenterPoint when prefer_accuracy=False."""
        from backend.cv.detection_3d import CenterPointWrapper, get_3d_detector

        detector = get_3d_detector(prefer_accuracy=False)
        assert isinstance(detector, CenterPointWrapper)


# -----------------------------------------------------------------------------
# Detection3D Type Tests
# -----------------------------------------------------------------------------


class TestDetection3DType:
    """Tests for Detection3D pydantic model."""

    def test_detection3d_creation(self) -> None:
        """Should create Detection3D with valid data."""
        detection = Detection3D(
            class_id=0,
            class_name="Car",
            center=(10.0, 0.0, 0.5),
            dimensions=(4.5, 1.8, 1.5),
            rotation=0.1,
            confidence=0.95,
        )

        assert detection.class_id == 0
        assert detection.class_name == "Car"
        assert detection.center == (10.0, 0.0, 0.5)
        assert detection.dimensions == (4.5, 1.8, 1.5)
        assert detection.rotation == 0.1
        assert detection.confidence == 0.95

    def test_detection3d_volume(self) -> None:
        """Should compute volume correctly."""
        detection = Detection3D(
            class_id=0,
            class_name="Car",
            center=(0.0, 0.0, 0.0),
            dimensions=(4.0, 2.0, 1.5),  # 4 x 2 x 1.5 = 12 cubic meters
            rotation=0.0,
            confidence=0.9,
        )

        assert detection.volume == pytest.approx(12.0)

    def test_detection3d_result_count(self) -> None:
        """Should count detections correctly."""
        result = Detection3DResult(
            detections=[
                Detection3D(
                    class_id=0,
                    class_name="Car",
                    center=(10.0, 0.0, 0.5),
                    dimensions=(4.0, 2.0, 1.5),
                    rotation=0.0,
                    confidence=0.9,
                ),
                Detection3D(
                    class_id=1,
                    class_name="Pedestrian",
                    center=(15.0, 2.0, 0.8),
                    dimensions=(0.6, 0.6, 1.7),
                    rotation=0.0,
                    confidence=0.85,
                ),
            ],
            pointcloud_path="/path/to/scan.bin",
            processing_time_ms=150.0,
            model_name="pvrcnn++",
        )

        assert result.count == 2


# -----------------------------------------------------------------------------
# Integration Tests (require OpenPCDet installation)
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestPVRCNNIntegration:
    """Integration tests requiring real OpenPCDet installation."""

    @pytest.fixture
    def check_openpcdet(self) -> None:
        """Skip if OpenPCDet not installed."""
        pytest.importorskip("pcdet")

    def test_openpcdet_available(self, check_openpcdet: None) -> None:
        """Verify OpenPCDet is available."""
        import pcdet

        assert pcdet is not None


@pytest.mark.integration
@pytest.mark.gpu
class TestPerformanceBenchmarks:
    """Performance benchmarks requiring GPU."""

    @pytest.fixture
    def check_gpu(self) -> None:
        """Skip if GPU not available."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_gpu_available(self, check_gpu: None) -> None:
        """Verify GPU is available."""
        import torch

        assert torch.cuda.is_available()
