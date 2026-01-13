# 3D Object Detection

> **Status:** Completed
> **Priority:** P1 (High)
> **Dependencies:** [00-infrastructure.md](./00-infrastructure.md)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

3D object detection on point clouds using PV-RCNN++ as the primary model with CenterPoint as a fallback. These models detect objects like vehicles, pedestrians, and cyclists in LiDAR data, returning 3D bounding boxes with position, dimensions, and orientation. Essential for autonomous driving and robotics datasets.

## Goals

- [ ] Integrate PV-RCNN++ from OpenPCDet framework
- [ ] Implement CenterPoint fallback (faster, lower accuracy)
- [ ] Support common point cloud formats (PCD, PLY, LAS, nuScenes)
- [ ] Handle varying point cloud densities
- [ ] Return 3D bounding boxes with orientation

## Technical Design

### Model Specifications

| Model | Size | VRAM | Inference | 3D AP (KITTI) |
|-------|------|------|-----------|---------------|
| PV-RCNN++ | ~150MB | ~4GB | 150-200ms | 84% |
| CenterPoint | ~100MB | ~3GB | 80-100ms | 79% |

### Supported Coordinate Systems

```python
from enum import Enum

class CoordinateSystem(Enum):
    """Point cloud coordinate systems."""
    KITTI = "kitti"      # x=forward, y=left, z=up (camera-centric)
    NUSCENES = "nuscenes"  # x=right, y=forward, z=up
    WAYMO = "waymo"       # x=forward, y=left, z=up (vehicle-centric)

def convert_coordinates(
    points: np.ndarray,
    from_system: CoordinateSystem,
    to_system: CoordinateSystem
) -> np.ndarray:
    """Convert point cloud between coordinate systems."""
    # Transformation matrices...
    pass
```

### PV-RCNN++ Wrapper

```python
from typing import Optional, List, Callable
from backend.cv.base import BaseModelWrapper
from backend.cv.types import Detection3D
import numpy as np
import torch

class PVRCNNWrapper(BaseModelWrapper):
    """PV-RCNN++ 3D object detection wrapper using OpenPCDet."""

    model_name = "pvrcnn++"
    vram_required_mb = 5000
    supports_batching = False  # Point clouds vary in size

    # Default classes (KITTI/nuScenes style)
    DEFAULT_CLASSES = ["Car", "Pedestrian", "Cyclist"]

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize PV-RCNN++ wrapper.

        Args:
            config_path: Path to OpenPCDet config (uses default if None)
        """
        self._model = None
        self._config = None
        self._device: str = "cpu"
        self._config_path = config_path

    def load(self, device: str = "cuda") -> None:
        """Load PV-RCNN++ model from OpenPCDet."""
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils

        # Load config
        config_path = self._config_path or self._get_default_config()
        cfg_from_yaml_file(config_path, cfg)
        self._config = cfg

        # Build model
        self._model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(self.DEFAULT_CLASSES),
            dataset=None
        )

        # Load weights
        checkpoint_path = self._get_checkpoint_path()
        self._model.load_params_from_file(
            filename=checkpoint_path,
            logger=common_utils.create_logger(),
            to_cpu=True
        )

        self._device = device
        if device == "cuda":
            self._model.cuda()

        self._model.eval()

    def _get_default_config(self) -> str:
        """Get default PV-RCNN++ config path."""
        from backend.cv.download import MODELS_DIR
        return str(MODELS_DIR / "pvrcnn" / "pv_rcnn_plusplus.yaml")

    def _get_checkpoint_path(self) -> str:
        """Get checkpoint path."""
        from backend.cv.download import MODELS_DIR
        return str(MODELS_DIR / "pvrcnn" / "pv_rcnn_plusplus_8369.pth")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        pointcloud_path: str,
        classes: Optional[List[str]] = None,
        confidence: float = 0.3,
        coordinate_system: CoordinateSystem = CoordinateSystem.KITTI,
    ) -> List[Detection3D]:
        """
        Detect 3D objects in a point cloud.

        Args:
            pointcloud_path: Path to point cloud file (PCD, PLY, LAS, BIN)
            classes: Classes to detect (None = all)
            confidence: Minimum confidence threshold
            coordinate_system: Input coordinate system

        Returns:
            List of Detection3D objects
        """
        import time

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Load point cloud
        points = self._load_pointcloud(pointcloud_path)

        # Convert coordinates if needed
        if coordinate_system != CoordinateSystem.KITTI:
            points = convert_coordinates(points, coordinate_system, CoordinateSystem.KITTI)

        start = time.perf_counter()

        # Prepare input
        input_dict = self._prepare_input(points)

        # Run inference
        with torch.no_grad():
            pred_dicts, _ = self._model.forward(input_dict)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Convert predictions to our types
        detections = []
        pred = pred_dicts[0]

        for i in range(len(pred["pred_boxes"])):
            box = pred["pred_boxes"][i].cpu().numpy()
            score = pred["pred_scores"][i].cpu().item()
            cls_id = int(pred["pred_labels"][i].cpu().item()) - 1  # 1-indexed

            if score < confidence:
                continue

            if classes and self.DEFAULT_CLASSES[cls_id] not in classes:
                continue

            # box format: x, y, z, l, w, h, rotation (KITTI)
            detections.append(Detection3D(
                class_id=cls_id,
                class_name=self.DEFAULT_CLASSES[cls_id],
                center=(float(box[0]), float(box[1]), float(box[2])),
                dimensions=(float(box[3]), float(box[4]), float(box[5])),
                rotation=float(box[6]),
                confidence=float(score),
            ))

        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections

    def _load_pointcloud(self, path: str) -> np.ndarray:
        """Load point cloud from various formats."""
        import os

        ext = os.path.splitext(path)[1].lower()

        if ext == ".bin":
            # KITTI binary format (N x 4: x, y, z, intensity)
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        elif ext == ".pcd":
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            points = np.asarray(pcd.points)
            # Add intensity if available
            if pcd.colors:
                intensity = np.asarray(pcd.colors)[:, 0:1]
                points = np.hstack([points, intensity])
            else:
                points = np.hstack([points, np.ones((len(points), 1))])
        elif ext == ".ply":
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            points = np.asarray(pcd.points)
            points = np.hstack([points, np.ones((len(points), 1))])
        elif ext in [".las", ".laz"]:
            import laspy
            las = laspy.read(path)
            points = np.vstack([las.x, las.y, las.z, las.intensity]).T
        else:
            raise ValueError(f"Unsupported format: {ext}")

        return points.astype(np.float32)

    def _prepare_input(self, points: np.ndarray) -> dict:
        """Prepare input dict for OpenPCDet model."""
        from pcdet.datasets import DatasetTemplate

        # Create dummy dataset for preprocessing
        input_dict = {
            "points": points,
            "frame_id": 0,
        }

        # Apply voxelization and other preprocessing
        # (OpenPCDet handles this internally)

        return input_dict
```

### CenterPoint Fallback Wrapper

```python
class CenterPointWrapper(BaseModelWrapper):
    """CenterPoint 3D object detection wrapper."""

    model_name = "centerpoint"
    vram_required_mb = 4000
    supports_batching = False

    DEFAULT_CLASSES = ["Car", "Pedestrian", "Cyclist"]

    def __init__(self):
        self._model = None
        self._device: str = "cpu"

    def load(self, device: str = "cuda") -> None:
        """Load CenterPoint model."""
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network

        config_path = self._get_config_path()
        cfg_from_yaml_file(config_path, cfg)

        self._model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(self.DEFAULT_CLASSES),
            dataset=None
        )

        checkpoint_path = self._get_checkpoint_path()
        self._model.load_params_from_file(filename=checkpoint_path)

        self._device = device
        if device == "cuda":
            self._model.cuda()

        self._model.eval()

    def _get_config_path(self) -> str:
        from backend.cv.download import MODELS_DIR
        return str(MODELS_DIR / "centerpoint" / "centerpoint.yaml")

    def _get_checkpoint_path(self) -> str:
        from backend.cv.download import MODELS_DIR
        return str(MODELS_DIR / "centerpoint" / "centerpoint.pth")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        pointcloud_path: str,
        classes: Optional[List[str]] = None,
        confidence: float = 0.3,
        **kwargs
    ) -> List[Detection3D]:
        """CenterPoint inference (similar interface to PV-RCNN++)."""
        # Similar implementation to PVRCNNWrapper
        ...
```

### 3D Detector Factory

```python
def get_3d_detector(
    prefer_accuracy: bool = True,
    force_model: Optional[str] = None
) -> BaseModelWrapper:
    """
    Get 3D object detector.

    Args:
        prefer_accuracy: If True, prefer PV-RCNN++ over CenterPoint
        force_model: Force specific model ("pvrcnn" or "centerpoint")

    Returns:
        3D detector wrapper
    """
    from backend.cv.device import get_available_vram_mb

    if force_model == "centerpoint":
        return CenterPointWrapper()
    elif force_model == "pvrcnn":
        return PVRCNNWrapper()

    if prefer_accuracy:
        available = get_available_vram_mb()
        if available >= 5000:
            return PVRCNNWrapper()

    return CenterPointWrapper()
```

## Implementation Tasks

- [ ] **PV-RCNN++ Integration**
  - [ ] Create `backend/cv/detection_3d.py`
  - [ ] Implement PVRCNNWrapper with OpenPCDet
  - [ ] Add point cloud loading (PCD, PLY, LAS, BIN)
  - [ ] Coordinate system conversion

- [ ] **CenterPoint Fallback**
  - [ ] Implement CenterPointWrapper
  - [ ] Verify OpenPCDet interface compatibility

- [ ] **Point Cloud Loading**
  - [ ] Support KITTI binary format
  - [ ] Support PCD format (Open3D)
  - [ ] Support PLY format
  - [ ] Support LAS/LAZ format (laspy)

- [ ] **Model Downloads**
  - [ ] Add to download registry
  - [ ] Document model placement
  - [ ] Config file setup

- [ ] **Testing**
  - [ ] Unit tests with mock
  - [ ] Integration tests (GPU)
  - [ ] Coordinate conversion tests
  - [ ] Performance benchmarks

## Acceptance Criteria

- [ ] `detect_3d(pointcloud_path)` returns 3D bounding boxes
- [ ] Supports nuScenes, KITTI coordinate systems
- [ ] Handles point clouds with 10K-1M points
- [ ] Fallback to CenterPoint when VRAM insufficient
- [ ] All supported formats load correctly (PCD, PLY, LAS, BIN)
- [ ] **VRAM Budget:** PV-RCNN++ <5GB, CenterPoint <4GB
- [ ] **Performance:** PV-RCNN++ <200ms/scan, CenterPoint <100ms/scan on GPU

## Files to Create

```
backend/cv/
└── detection_3d.py   # PVRCNNWrapper, CenterPointWrapper, get_3d_detector()
```

## Testing

```python
# test_detection_3d.py
import pytest
from backend.cv.detection_3d import PVRCNNWrapper, CenterPointWrapper, get_3d_detector

@pytest.fixture
def pvrcnn():
    p = PVRCNNWrapper()
    p.load("cuda" if torch.cuda.is_available() else "cpu")
    yield p
    p.unload()

@pytest.mark.gpu
def test_detect_vehicles(pvrcnn, kitti_pointcloud):
    detections = pvrcnn.predict(kitti_pointcloud, classes=["Car"])
    # KITTI sample should have vehicles
    assert len(detections) >= 1
    assert detections[0].class_name == "Car"

def test_3d_bbox_format(pvrcnn, kitti_pointcloud):
    detections = pvrcnn.predict(kitti_pointcloud)
    for det in detections:
        assert len(det.center) == 3
        assert len(det.dimensions) == 3
        assert isinstance(det.rotation, float)

def test_load_kitti_bin(pvrcnn, kitti_bin_file):
    detections = pvrcnn.predict(kitti_bin_file)
    assert isinstance(detections, list)

def test_load_pcd(pvrcnn, sample_pcd):
    detections = pvrcnn.predict(sample_pcd)
    assert isinstance(detections, list)

def test_load_las(pvrcnn, sample_las):
    detections = pvrcnn.predict(sample_las)
    assert isinstance(detections, list)

@pytest.mark.gpu
def test_vram_budget(pvrcnn):
    from backend.cv.device import get_vram_usage
    used, _ = get_vram_usage()
    assert used < 5000  # <5GB

@pytest.mark.gpu
def test_inference_speed(pvrcnn, kitti_pointcloud):
    import time
    times = []
    for _ in range(5):
        start = time.perf_counter()
        pvrcnn.predict(kitti_pointcloud)
        times.append((time.perf_counter() - start) * 1000)

    assert sum(times) / len(times) < 200  # <200ms average
```
