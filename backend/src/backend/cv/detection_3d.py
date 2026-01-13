"""
PV-RCNN++ and CenterPoint 3D object detection wrappers.

This module provides PV-RCNN++ as the primary 3D detector and CenterPoint as a
fallback for lower VRAM requirements. Both use the OpenPCDet framework.

Implements spec: 03-cv-models/07-3d-detection
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.cv.base import BaseModelWrapper, ModelInfo, register_model
from backend.cv.types import Detection3D, Detection3DResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Default classes for 3D detection (KITTI/nuScenes)
DETECTION_3D_CLASSES: list[str] = ["Car", "Pedestrian", "Cyclist"]


class CoordinateSystem(str, Enum):
    """
    Point cloud coordinate systems for 3D detection.

    Different datasets use different coordinate conventions. This enum
    helps convert between them for consistent processing.

    KITTI: x=forward, y=left, z=up (LiDAR-centric)
    nuScenes: x=right, y=forward, z=up
    WAYMO: x=forward, y=left, z=up (vehicle-centric, similar to KITTI)
    """

    KITTI = "kitti"
    NUSCENES = "nuscenes"
    WAYMO = "waymo"


# Transformation matrices between coordinate systems
# These are 4x4 homogeneous transformation matrices (rotation only, no translation)
_COORD_TRANSFORMS: dict[tuple[CoordinateSystem, CoordinateSystem], np.ndarray] = {
    # nuScenes (x=right, y=forward) -> KITTI (x=forward, y=left)
    # x_kitti = y_nuscenes, y_kitti = -x_nuscenes, z_kitti = z_nuscenes
    (CoordinateSystem.NUSCENES, CoordinateSystem.KITTI): np.array(
        [
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    ),
    # WAYMO is similar to KITTI (both x=forward, y=left, z=up)
    # Minor differences in exact conventions, but usually compatible
    (CoordinateSystem.WAYMO, CoordinateSystem.KITTI): np.eye(4, dtype=np.float32),
}


def convert_coordinates(
    points: np.ndarray,
    from_system: CoordinateSystem,
    to_system: CoordinateSystem,
) -> np.ndarray:
    """
    Convert point cloud between coordinate systems.

    Args:
        points: Point cloud array of shape (N, 3+) where first 3 columns are XYZ.
        from_system: Source coordinate system.
        to_system: Target coordinate system.

    Returns:
        Transformed point cloud with same shape as input.

    Raises:
        ValueError: If no transform is defined between the given systems.
    """
    if from_system == to_system:
        return points

    # Look up transform
    key = (from_system, to_system)
    if key in _COORD_TRANSFORMS:
        transform = _COORD_TRANSFORMS[key]
    else:
        # Try inverse
        inv_key = (to_system, from_system)
        if inv_key in _COORD_TRANSFORMS:
            transform = np.linalg.inv(_COORD_TRANSFORMS[inv_key])
        else:
            raise ValueError(
                f"No coordinate transform defined: {from_system.value} -> {to_system.value}"
            )

    # Apply rotation to XYZ columns only (3x3 submatrix)
    xyz = points[:, :3].astype(np.float32)
    rotated = xyz @ transform[:3, :3].T

    # Preserve additional columns (intensity, etc.)
    if points.shape[1] > 3:
        result: np.ndarray = np.hstack([rotated, points[:, 3:]])
        return result
    result_xyz: np.ndarray = rotated.astype(np.float32)
    return result_xyz


class PointCloudLoader:
    """
    Multi-format point cloud loader.

    Supports loading point clouds from various formats commonly used in
    autonomous driving and robotics applications.

    Supported formats:
        - .bin: KITTI binary format (N x 4: x, y, z, intensity)
        - .pcd: Point Cloud Data format (via Open3D)
        - .ply: Polygon File Format (via Open3D)
        - .las/.laz: LAS/LAZ LiDAR format (via laspy)
    """

    SUPPORTED_FORMATS: set[str] = {".bin", ".pcd", ".ply", ".las", ".laz"}

    @classmethod
    def load(cls, path: str) -> np.ndarray:
        """
        Load point cloud from various formats.

        Args:
            path: Path to point cloud file.

        Returns:
            np.ndarray of shape (N, 4+) with columns [x, y, z, intensity, ...].
            Intensity is normalized to [0, 1] where possible.

        Raises:
            ValueError: If format is not supported or file cannot be read.
            FileNotFoundError: If file does not exist.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {path}")

        ext = file_path.suffix.lower()

        if ext == ".bin":
            return cls._load_bin(path)
        elif ext == ".pcd":
            return cls._load_pcd(path)
        elif ext == ".ply":
            return cls._load_ply(path)
        elif ext in {".las", ".laz"}:
            return cls._load_las(path)
        else:
            raise ValueError(
                f"Unsupported point cloud format: {ext}. "
                f"Supported formats: {cls.SUPPORTED_FORMATS}"
            )

    @staticmethod
    def _load_bin(path: str) -> np.ndarray:
        """
        Load KITTI binary format.

        KITTI binary files store points as consecutive float32 values:
        [x0, y0, z0, i0, x1, y1, z1, i1, ...]

        Args:
            path: Path to .bin file.

        Returns:
            np.ndarray of shape (N, 4) with [x, y, z, intensity].
        """
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        logger.debug("Loaded %d points from BIN file: %s", len(points), path)
        return points

    @staticmethod
    def _load_pcd(path: str) -> np.ndarray:
        """
        Load PCD format via Open3D.

        Args:
            path: Path to .pcd file.

        Returns:
            np.ndarray of shape (N, 4) with [x, y, z, intensity].
        """
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError(f"Empty or invalid PCD file: {path}")

        points = np.asarray(pcd.points, dtype=np.float32)

        # Extract intensity from colors if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            # Use grayscale (R channel) as intensity, already in [0, 1]
            intensity = colors[:, 0:1].astype(np.float32)
        else:
            # Default intensity to 0.5 if not available
            intensity = np.full((len(points), 1), 0.5, dtype=np.float32)

        result = np.hstack([points, intensity])
        logger.debug("Loaded %d points from PCD file: %s", len(points), path)
        return result

    @staticmethod
    def _load_ply(path: str) -> np.ndarray:
        """
        Load PLY format via Open3D.

        Args:
            path: Path to .ply file.

        Returns:
            np.ndarray of shape (N, 4) with [x, y, z, intensity].
        """
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError(f"Empty or invalid PLY file: {path}")

        points = np.asarray(pcd.points, dtype=np.float32)

        # PLY files may have colors
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            intensity = colors[:, 0:1].astype(np.float32)
        else:
            intensity = np.full((len(points), 1), 0.5, dtype=np.float32)

        result = np.hstack([points, intensity])
        logger.debug("Loaded %d points from PLY file: %s", len(points), path)
        return result

    @staticmethod
    def _load_las(path: str) -> np.ndarray:
        """
        Load LAS/LAZ format via laspy.

        Args:
            path: Path to .las or .laz file.

        Returns:
            np.ndarray of shape (N, 4) with [x, y, z, intensity].
        """
        import laspy

        las = laspy.read(path)

        # Extract XYZ (laspy provides scaled coordinates)
        x = np.array(las.x, dtype=np.float32)
        y = np.array(las.y, dtype=np.float32)
        z = np.array(las.z, dtype=np.float32)

        # Extract intensity if available
        if hasattr(las, "intensity") and las.intensity is not None:
            intensity = np.array(las.intensity, dtype=np.float32)
            # Normalize intensity to [0, 1] (typically 16-bit unsigned)
            if intensity.max() > 1.0:
                intensity = intensity / 65535.0
        else:
            intensity = np.full_like(x, 0.5)

        result = np.column_stack([x, y, z, intensity])
        logger.debug("Loaded %d points from LAS file: %s", len(result), path)
        return result


class _Base3DDetector(BaseModelWrapper[Detection3DResult]):
    """
    Base class for 3D object detectors.

    Provides common functionality for OpenPCDet-based detectors including
    point cloud loading, coordinate conversion, and result parsing.
    """

    def __init__(self, config_path: str | None = None) -> None:
        """
        Initialize 3D detector.

        Args:
            config_path: Optional custom OpenPCDet config path.
        """
        super().__init__()
        self._config_path = config_path
        self._cfg: Any = None
        self._pcdet_model: Any = None

    def _validate_pointcloud(self, points: np.ndarray) -> None:
        """
        Validate point cloud before inference.

        Args:
            points: Point cloud array.

        Raises:
            ValueError: If point cloud is invalid.
        """
        if len(points) == 0:
            raise ValueError("Empty point cloud")

        if len(points) > 1_000_000:
            logger.warning(
                "Large point cloud (%d points) may cause OOM. Consider downsampling.",
                len(points),
            )

        # Check for NaN/Inf in XYZ coordinates
        if np.any(~np.isfinite(points[:, :3])):
            raise ValueError("Point cloud contains NaN or Inf values in coordinates")

    def _prepare_input(self, points: np.ndarray) -> dict[str, Any]:
        """
        Prepare input dict for OpenPCDet model.

        Args:
            points: Point cloud array of shape (N, 4+).

        Returns:
            Input dict expected by OpenPCDet model.
        """
        import torch

        # Ensure at least 4 columns (x, y, z, intensity)
        if points.shape[1] < 4:
            padding = np.zeros((len(points), 4 - points.shape[1]), dtype=np.float32)
            points = np.hstack([points, padding])

        # Add batch index column (required by OpenPCDet)
        batch_idx = np.zeros((len(points), 1), dtype=np.float32)
        points_with_batch = np.hstack([batch_idx, points[:, :4]])

        input_dict = {
            "points": torch.from_numpy(points_with_batch).float(),
            "frame_id": "inference",
            "batch_size": 1,
        }

        if self._device == "cuda":
            input_dict["points"] = input_dict["points"].cuda()  # type: ignore[attr-defined]

        return input_dict

    def _parse_predictions(
        self,
        pred_dict: dict[str, Any],
        confidence: float,
        classes: list[str] | None,
    ) -> list[Detection3D]:
        """
        Parse OpenPCDet predictions to Detection3D list.

        Args:
            pred_dict: Prediction dict from OpenPCDet model.
            confidence: Minimum confidence threshold.
            classes: Optional list of classes to filter.

        Returns:
            List of Detection3D objects sorted by confidence.
        """
        detections: list[Detection3D] = []

        pred_boxes = pred_dict["pred_boxes"].cpu().numpy()  # (N, 7)
        pred_scores = pred_dict["pred_scores"].cpu().numpy()
        pred_labels = pred_dict["pred_labels"].cpu().numpy()  # 1-indexed

        for i in range(len(pred_boxes)):
            score = float(pred_scores[i])
            if score < confidence:
                continue

            # Labels are 1-indexed in OpenPCDet
            cls_id = int(pred_labels[i]) - 1
            if cls_id < 0 or cls_id >= len(DETECTION_3D_CLASSES):
                logger.warning("Invalid class ID: %d, skipping", cls_id + 1)
                continue

            class_name = DETECTION_3D_CLASSES[cls_id]

            # Filter by class if specified
            if classes and class_name not in classes:
                continue

            box = pred_boxes[i]
            # OpenPCDet format: x, y, z, l, w, h, rotation (yaw)

            detections.append(
                Detection3D(
                    class_id=cls_id,
                    class_name=class_name,
                    center=(float(box[0]), float(box[1]), float(box[2])),
                    dimensions=(float(box[3]), float(box[4]), float(box[5])),
                    rotation=float(box[6]),
                    confidence=score,
                )
            )

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


@register_model
class PVRCNNWrapper(_Base3DDetector):
    """
    PV-RCNN++ 3D object detection wrapper using OpenPCDet.

    Primary 3D detector offering highest accuracy (84% 3D AP on KITTI)
    at the cost of higher VRAM (~4GB) and inference time (~150-200ms).

    Detects common road objects: Car, Pedestrian, Cyclist.

    Example:
        detector = PVRCNNWrapper()
        detector.load()
        result = detector.predict("scan.pcd", classes=["Car"])
        print(f"Found {result.count} cars")
        detector.unload()
    """

    info = ModelInfo(
        name="pvrcnn++",
        description="PV-RCNN++ 3D detector via OpenPCDet (KITTI classes)",
        vram_required_mb=4000,
        supports_batching=False,  # Point clouds vary in size
        supports_gpu=True,
        source="openpcdet",
        version="0.6.0",
        extra={"benchmark": "KITTI 84% 3D AP", "classes": DETECTION_3D_CLASSES},
    )

    def _get_config_path(self) -> str:
        """Get config path for PV-RCNN++."""
        if self._config_path:
            return self._config_path

        from backend.cv.download import get_models_dir

        return str(get_models_dir() / "pvrcnn" / "pv_rcnn_plusplus.yaml")

    def _get_checkpoint_path(self) -> str:
        """Get checkpoint path for PV-RCNN++."""
        from backend.cv.download import get_model_path

        return str(get_model_path("pvrcnn++"))

    def _load_model(self, device: str) -> None:
        """
        Load PV-RCNN++ model via OpenPCDet.

        Args:
            device: Target device ("cuda" or "cpu").

        Raises:
            RuntimeError: If model files not found or OpenPCDet not installed.
        """
        try:
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.models import build_network
            from pcdet.utils import common_utils
        except ImportError as e:
            raise RuntimeError(
                "OpenPCDet not installed. Install from source: "
                "https://github.com/open-mmlab/OpenPCDet"
            ) from e

        from backend.cv.download import is_model_downloaded

        # Check checkpoint exists
        if not is_model_downloaded("pvrcnn++"):
            raise RuntimeError(
                "PV-RCNN++ checkpoint not found. Download from OpenPCDet model zoo "
                "and place in models/pvrcnn/ directory. "
                "See models/README.md for instructions."
            )

        # Load configuration
        config_path = self._get_config_path()
        if not Path(config_path).exists():
            raise RuntimeError(
                f"PV-RCNN++ config not found: {config_path}. "
                "Copy from OpenPCDet tools/cfgs/ directory."
            )

        logger.info("Loading PV-RCNN++ config from %s", config_path)
        cfg_from_yaml_file(config_path, cfg)
        self._cfg = cfg

        # Build network
        self._pcdet_model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(DETECTION_3D_CLASSES),
            dataset=None,  # No dataset needed for inference
        )

        # Load checkpoint
        checkpoint_path = self._get_checkpoint_path()
        logger.info("Loading PV-RCNN++ checkpoint from %s", checkpoint_path)

        pcdet_logger = common_utils.create_logger()
        self._pcdet_model.load_params_from_file(
            filename=checkpoint_path,
            logger=pcdet_logger,
            to_cpu=(device == "cpu"),
        )

        # Move to device
        if device == "cuda":
            import torch

            if torch.cuda.is_available():
                self._pcdet_model.cuda()
            else:
                logger.warning("CUDA requested but not available, using CPU")
                device = "cpu"

        # Set model to inference mode
        self._pcdet_model.train(False)
        self._model = self._pcdet_model

        logger.info("PV-RCNN++ loaded on %s", device)

    def _unload_model(self) -> None:
        """Unload model and free memory."""
        if self._pcdet_model is not None:
            del self._pcdet_model
            self._pcdet_model = None
        self._cfg = None

    def predict(
        self,
        input_path: str,
        *,
        confidence: float = 0.3,
        classes: list[str] | None = None,
        coordinate_system: CoordinateSystem = CoordinateSystem.KITTI,
        **kwargs: Any,
    ) -> Detection3DResult:
        """
        Detect 3D objects in a point cloud.

        Args:
            input_path: Path to point cloud file (PCD, PLY, LAS, BIN).
            confidence: Minimum confidence threshold [0-1].
            classes: Classes to detect (None = all: Car, Pedestrian, Cyclist).
            coordinate_system: Input point cloud coordinate system.
            **kwargs: Additional arguments (ignored).

        Returns:
            Detection3DResult with 3D bounding boxes.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If point cloud cannot be loaded.
        """
        import torch

        if not self.is_loaded or self._pcdet_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Load point cloud
        points = PointCloudLoader.load(input_path)

        # Validate
        self._validate_pointcloud(points)

        # Convert coordinates if needed (OpenPCDet uses KITTI convention)
        if coordinate_system != CoordinateSystem.KITTI:
            points = convert_coordinates(points, coordinate_system, CoordinateSystem.KITTI)

        # Prepare input
        input_dict = self._prepare_input(points)

        start = time.perf_counter()

        with torch.no_grad():
            pred_dicts, _ = self._pcdet_model.forward(input_dict)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Parse predictions
        detections = self._parse_predictions(pred_dicts[0], confidence, classes)

        return Detection3DResult(
            detections=detections,
            pointcloud_path=input_path,
            processing_time_ms=elapsed_ms,
            model_name=self.info.name,
        )


@register_model
class CenterPointWrapper(_Base3DDetector):
    """
    CenterPoint 3D object detection wrapper using OpenPCDet.

    Fallback 3D detector offering faster inference (~80-100ms) with
    lower VRAM requirements (~3GB) at slightly reduced accuracy (79% 3D AP).

    Uses anchor-free center-based detection for efficient processing.

    Example:
        detector = CenterPointWrapper()
        detector.load()
        result = detector.predict("scan.bin", confidence=0.25)
        detector.unload()
    """

    info = ModelInfo(
        name="centerpoint",
        description="CenterPoint 3D detector via OpenPCDet (faster, lower VRAM)",
        vram_required_mb=3000,
        supports_batching=False,
        supports_gpu=True,
        source="openpcdet",
        version="0.6.0",
        extra={"benchmark": "KITTI 79% 3D AP", "classes": DETECTION_3D_CLASSES},
    )

    def _get_config_path(self) -> str:
        """Get config path for CenterPoint."""
        if self._config_path:
            return self._config_path

        from backend.cv.download import get_models_dir

        return str(get_models_dir() / "centerpoint" / "centerpoint.yaml")

    def _get_checkpoint_path(self) -> str:
        """Get checkpoint path for CenterPoint."""
        from backend.cv.download import get_model_path

        return str(get_model_path("centerpoint"))

    def _load_model(self, device: str) -> None:
        """
        Load CenterPoint model via OpenPCDet.

        Args:
            device: Target device ("cuda" or "cpu").

        Raises:
            RuntimeError: If model files not found or OpenPCDet not installed.
        """
        try:
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.models import build_network
            from pcdet.utils import common_utils
        except ImportError as e:
            raise RuntimeError(
                "OpenPCDet not installed. Install from source: "
                "https://github.com/open-mmlab/OpenPCDet"
            ) from e

        from backend.cv.download import is_model_downloaded

        # Check checkpoint exists
        if not is_model_downloaded("centerpoint"):
            raise RuntimeError(
                "CenterPoint checkpoint not found. Download from OpenPCDet model zoo "
                "and place in models/centerpoint/ directory. "
                "See models/README.md for instructions."
            )

        # Load configuration
        config_path = self._get_config_path()
        if not Path(config_path).exists():
            raise RuntimeError(
                f"CenterPoint config not found: {config_path}. "
                "Copy from OpenPCDet tools/cfgs/ directory."
            )

        logger.info("Loading CenterPoint config from %s", config_path)
        cfg_from_yaml_file(config_path, cfg)
        self._cfg = cfg

        # Build network
        self._pcdet_model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(DETECTION_3D_CLASSES),
            dataset=None,
        )

        # Load checkpoint
        checkpoint_path = self._get_checkpoint_path()
        logger.info("Loading CenterPoint checkpoint from %s", checkpoint_path)

        pcdet_logger = common_utils.create_logger()
        self._pcdet_model.load_params_from_file(
            filename=checkpoint_path,
            logger=pcdet_logger,
            to_cpu=(device == "cpu"),
        )

        # Move to device
        if device == "cuda":
            import torch

            if torch.cuda.is_available():
                self._pcdet_model.cuda()
            else:
                logger.warning("CUDA requested but not available, using CPU")
                device = "cpu"

        # Set model to inference mode
        self._pcdet_model.train(False)
        self._model = self._pcdet_model

        logger.info("CenterPoint loaded on %s", device)

    def _unload_model(self) -> None:
        """Unload model and free memory."""
        if self._pcdet_model is not None:
            del self._pcdet_model
            self._pcdet_model = None
        self._cfg = None

    def predict(
        self,
        input_path: str,
        *,
        confidence: float = 0.3,
        classes: list[str] | None = None,
        coordinate_system: CoordinateSystem = CoordinateSystem.KITTI,
        **kwargs: Any,
    ) -> Detection3DResult:
        """
        Detect 3D objects in a point cloud.

        Args:
            input_path: Path to point cloud file (PCD, PLY, LAS, BIN).
            confidence: Minimum confidence threshold [0-1].
            classes: Classes to detect (None = all: Car, Pedestrian, Cyclist).
            coordinate_system: Input point cloud coordinate system.
            **kwargs: Additional arguments (ignored).

        Returns:
            Detection3DResult with 3D bounding boxes.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If point cloud cannot be loaded.
        """
        import torch

        if not self.is_loaded or self._pcdet_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Load point cloud
        points = PointCloudLoader.load(input_path)

        # Validate
        self._validate_pointcloud(points)

        # Convert coordinates if needed
        if coordinate_system != CoordinateSystem.KITTI:
            points = convert_coordinates(points, coordinate_system, CoordinateSystem.KITTI)

        # Prepare input
        input_dict = self._prepare_input(points)

        start = time.perf_counter()

        with torch.no_grad():
            pred_dicts, _ = self._pcdet_model.forward(input_dict)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Parse predictions
        detections = self._parse_predictions(pred_dicts[0], confidence, classes)

        return Detection3DResult(
            detections=detections,
            pointcloud_path=input_path,
            processing_time_ms=elapsed_ms,
            model_name=self.info.name,
        )


def get_3d_detector(
    prefer_accuracy: bool = True,
    force_model: str | None = None,
) -> BaseModelWrapper[Detection3DResult]:
    """
    Get appropriate 3D object detector based on requirements and resources.

    Factory function that selects the best 3D detector for the given requirements.
    By default prefers PV-RCNN++ for accuracy. Falls back to CenterPoint if
    insufficient VRAM is available.

    Args:
        prefer_accuracy: If True, prefer PV-RCNN++ over CenterPoint.
        force_model: Force specific model ("pvrcnn++" or "centerpoint").

    Returns:
        Appropriate 3D detector wrapper (unloaded - call load() before use).

    Example:
        detector = get_3d_detector(prefer_accuracy=True)
        detector.load()
        result = detector.predict("scan.pcd", classes=["Car"])
        print(f"Found {result.count} vehicles")
        detector.unload()
    """
    from backend.cv.device import get_available_vram_mb

    # Force specific model if requested
    if force_model in {"centerpoint"}:
        logger.info("Returning CenterPoint 3D detector (forced)")
        return CenterPointWrapper()
    elif force_model in {"pvrcnn++", "pvrcnn"}:
        logger.info("Returning PV-RCNN++ 3D detector (forced)")
        return PVRCNNWrapper()

    # Select based on accuracy preference and available VRAM
    if prefer_accuracy:
        available = get_available_vram_mb()
        if available >= PVRCNNWrapper.info.vram_required_mb:
            logger.info(
                "Selecting PV-RCNN++ for accuracy (VRAM available: %dMB)",
                available,
            )
            return PVRCNNWrapper()
        logger.info(
            "PV-RCNN++ needs %dMB VRAM, only %dMB available, using CenterPoint",
            PVRCNNWrapper.info.vram_required_mb,
            available,
        )

    # Default to CenterPoint (faster, lower VRAM)
    logger.info("Returning CenterPoint 3D detector (default/fallback)")
    return CenterPointWrapper()
