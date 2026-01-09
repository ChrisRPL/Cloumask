# 3D Object Detection (OpenPCDet)

> **Status:** 🔴 Not Started
> **Priority:** P1 (High)
> **Dependencies:** 02-python-open3d, 03-cv-models (ModelManager pattern)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Integrate OpenPCDet for 3D object detection in point clouds. Support PV-RCNN++ as the primary high-accuracy model and CenterPoint as a faster/lighter fallback. Return 3D bounding boxes with class labels and confidence scores.

## Goals

- [ ] Set up OpenPCDet environment with pretrained weights
- [ ] Implement PV-RCNN++ inference (primary model)
- [ ] Implement CenterPoint inference (fallback model)
- [ ] Define standardized 3D bounding box output schema
- [ ] Handle KITTI and nuScenes coordinate conventions
- [ ] Automatic GPU/CPU fallback based on VRAM availability

## Technical Design

### Dependencies

```txt
# requirements.txt
torch>=2.0.0
# OpenPCDet installed from source or wheel
# spconv-cu118 or spconv-cu121 for sparse convolutions
```

### 3D Bounding Box Schema

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class BBox3D:
    """3D bounding box representation."""
    # Center position (x, y, z) in LiDAR frame
    center: tuple[float, float, float]
    # Dimensions (length, width, height) in meters
    dimensions: tuple[float, float, float]
    # Rotation around Z-axis (yaw) in radians
    rotation: float
    # Object class (car, pedestrian, cyclist, etc.)
    class_name: str
    # Detection confidence score [0, 1]
    score: float
    # Optional track ID for multi-frame tracking
    track_id: int | None = None

    def to_corners(self) -> np.ndarray:
        """Convert to 8 corner points (8, 3)."""
        l, w, h = self.dimensions
        x, y, z = self.center

        # Box corners before rotation
        corners = np.array([
            [-l/2, -w/2, -h/2],
            [+l/2, -w/2, -h/2],
            [+l/2, +w/2, -h/2],
            [-l/2, +w/2, -h/2],
            [-l/2, -w/2, +h/2],
            [+l/2, -w/2, +h/2],
            [+l/2, +w/2, +h/2],
            [-l/2, +w/2, +h/2],
        ])

        # Apply rotation
        cos_yaw, sin_yaw = np.cos(self.rotation), np.sin(self.rotation)
        rot_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        corners = corners @ rot_matrix.T

        # Translate to center
        corners += np.array([x, y, z])
        return corners


@dataclass
class Detection3DResult:
    """Result of 3D object detection."""
    boxes: list[BBox3D]
    model_name: str
    inference_time_ms: float
    point_count: int
    coordinate_system: str  # "kitti" | "nuscenes" | "waymo"
```

### Detection3DModel Class

```python
from pathlib import Path
import torch
import numpy as np


class Detection3DModel:
    """Wrapper for 3D object detection models."""

    SUPPORTED_MODELS = {
        "pvrcnn++": {
            "config": "cfgs/kitti_models/pv_rcnn_plusplus.yaml",
            "checkpoint": "pv_rcnn_plusplus_8369.pth",
            "classes": ["Car", "Pedestrian", "Cyclist"],
        },
        "centerpoint": {
            "config": "cfgs/nuscenes_models/centerpoint.yaml",
            "checkpoint": "centerpoint_pillar_512.pth",
            "classes": ["car", "truck", "bus", "pedestrian", "cyclist"],
        },
    }

    def __init__(
        self,
        model_name: str = "pvrcnn++",
        device: str | None = None,
        model_dir: Path = Path("models/3d_detection")
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._loaded = False

    def load(self) -> None:
        """Load model weights (lazy loading)."""
        if self._loaded:
            return

        config = self.SUPPORTED_MODELS[self.model_name]
        # OpenPCDet model loading logic
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils

        cfg_from_yaml_file(self.model_dir / config["config"], cfg)
        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(config["classes"]),
            dataset=None
        )
        self.model.load_params_from_file(
            filename=self.model_dir / config["checkpoint"],
            to_cpu=(self.device == "cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._loaded = False
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def detect(
        self,
        points: np.ndarray,
        score_threshold: float = 0.3
    ) -> Detection3DResult:
        """
        Run 3D object detection on point cloud.

        Args:
            points: (N, 4+) array with [x, y, z, intensity, ...]
            score_threshold: Minimum confidence for detections

        Returns:
            Detection3DResult with 3D bounding boxes
        """
        import time
        self.load()

        start_time = time.time()

        # Prepare input
        input_dict = self._prepare_input(points)

        # Run inference
        with torch.no_grad():
            pred_dicts, _ = self.model(input_dict)

        # Parse outputs
        boxes = self._parse_predictions(pred_dicts, score_threshold)

        inference_time = (time.time() - start_time) * 1000

        return Detection3DResult(
            boxes=boxes,
            model_name=self.model_name,
            inference_time_ms=inference_time,
            point_count=len(points),
            coordinate_system="kitti" if "kitti" in self.model_name else "nuscenes",
        )

    def _prepare_input(self, points: np.ndarray) -> dict:
        """Prepare point cloud for model input."""
        # Ensure at least 4 columns (x, y, z, intensity)
        if points.shape[1] < 4:
            points = np.hstack([points, np.ones((len(points), 1))])

        # Create input dict for OpenPCDet
        input_dict = {
            "points": torch.from_numpy(points).float().to(self.device),
            "batch_size": 1,
        }
        return input_dict

    def _parse_predictions(
        self,
        pred_dicts: list,
        threshold: float
    ) -> list[BBox3D]:
        """Parse model predictions to BBox3D list."""
        boxes = []
        config = self.SUPPORTED_MODELS[self.model_name]

        for pred in pred_dicts:
            pred_boxes = pred["pred_boxes"].cpu().numpy()  # (N, 7)
            pred_scores = pred["pred_scores"].cpu().numpy()
            pred_labels = pred["pred_labels"].cpu().numpy()

            mask = pred_scores >= threshold
            for box, score, label in zip(
                pred_boxes[mask], pred_scores[mask], pred_labels[mask]
            ):
                boxes.append(BBox3D(
                    center=(float(box[0]), float(box[1]), float(box[2])),
                    dimensions=(float(box[3]), float(box[4]), float(box[5])),
                    rotation=float(box[6]),
                    class_name=config["classes"][int(label) - 1],
                    score=float(score),
                ))

        return boxes


def get_best_available_model() -> str:
    """Select model based on available VRAM."""
    if not torch.cuda.is_available():
        return "centerpoint"  # CPU-friendly

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 8:
        return "pvrcnn++"
    else:
        return "centerpoint"
```

## Implementation Tasks

- [ ] **Setup OpenPCDet environment**
  - [ ] Install OpenPCDet from source
  - [ ] Install spconv for sparse convolutions
  - [ ] Download pretrained weights (KITTI, nuScenes)
  - [ ] Verify CUDA/cuDNN compatibility

- [ ] **Implement Detection3DModel wrapper**
  - [ ] load() with lazy initialization
  - [ ] unload() for memory management
  - [ ] detect() inference pipeline

- [ ] **Implement PV-RCNN++ model**
  - [ ] KITTI pretrained weights
  - [ ] nuScenes pretrained weights (optional)
  - [ ] Point cloud preprocessing

- [ ] **Implement CenterPoint fallback**
  - [ ] Pillar-based variant (memory efficient)
  - [ ] Automatic selection based on VRAM

- [ ] **Define output formats**
  - [ ] BBox3D dataclass
  - [ ] KITTI format export
  - [ ] nuScenes format export
  - [ ] JSON serialization

- [ ] **FastAPI integration**
  - [ ] Create routes/detect3d.py
  - [ ] Streaming progress updates
  - [ ] Batch detection endpoint

## Files to Create/Modify

| Path | Action | Purpose |
|------|--------|---------|
| `backend/cv/detection_3d.py` | Create | Detection3DModel wrapper |
| `backend/cv/models/openpcdet_wrapper.py` | Create | OpenPCDet integration |
| `backend/cv/models/__init__.py` | Modify | Export detection_3d |
| `backend/api/routes/detect3d.py` | Create | FastAPI endpoints |
| `backend/api/main.py` | Modify | Register detect3d router |
| `backend/tests/cv/test_detection_3d.py` | Create | Unit tests |
| `models/3d_detection/README.md` | Create | Model download instructions |
| `models/3d_detection/.gitkeep` | Create | Placeholder for weights |

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/detect3d/infer` | Run 3D detection on point cloud |
| GET | `/detect3d/models` | List available models |
| POST | `/detect3d/load` | Preload specific model |
| POST | `/detect3d/unload` | Unload model from memory |

### Request/Response Models

```python
class Detect3DRequest(BaseModel):
    input_path: str  # Path to PCD/PLY/NPY file
    model: str = "auto"  # "pvrcnn++" | "centerpoint" | "auto"
    score_threshold: float = 0.3
    output_format: str = "json"  # "json" | "kitti" | "nuscenes"

class Detect3DResponse(BaseModel):
    boxes: list[dict]  # BBox3D as dicts
    model_name: str
    inference_time_ms: float
    point_count: int
```

## Acceptance Criteria

- [ ] PV-RCNN++ detects cars in KITTI sample with mAP >80%
- [ ] Inference on 100K point cloud completes in <500ms on GPU
- [ ] 3D boxes include: center (x,y,z), dimensions (l,w,h), rotation (yaw), class, score
- [ ] Automatic fallback to CenterPoint when VRAM <4GB available
- [ ] Model loading <10 seconds on first inference
- [ ] Model unloading frees GPU memory (verified with nvidia-smi)
- [ ] `pytest tests/cv/test_detection_3d.py -v` passes

## Testing Strategy

```python
import pytest
import numpy as np
from cv.detection_3d import Detection3DModel, BBox3D


@pytest.fixture
def sample_points():
    """Create sample point cloud for testing."""
    # Random points in vehicle-like region
    points = np.random.rand(50000, 4)
    points[:, 0] *= 70  # x range
    points[:, 1] = points[:, 1] * 40 - 20  # y range
    points[:, 2] = points[:, 2] * 4 - 2  # z range
    return points.astype(np.float32)


def test_bbox3d_corners():
    bbox = BBox3D(
        center=(0, 0, 0),
        dimensions=(4, 2, 1.5),
        rotation=0,
        class_name="Car",
        score=0.9
    )
    corners = bbox.to_corners()
    assert corners.shape == (8, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_pvrcnn_inference(sample_points):
    model = Detection3DModel("pvrcnn++")
    result = model.detect(sample_points)
    assert result.inference_time_ms > 0
    assert result.model_name == "pvrcnn++"
    model.unload()


def test_model_selection():
    model_name = get_best_available_model()
    assert model_name in ["pvrcnn++", "centerpoint"]
```

## Model Download Instructions

```bash
# Create models directory
mkdir -p models/3d_detection

# Download PV-RCNN++ weights (KITTI)
wget -O models/3d_detection/pv_rcnn_plusplus_8369.pth \
  https://drive.google.com/...  # OpenPCDet model zoo

# Download CenterPoint weights (nuScenes)
wget -O models/3d_detection/centerpoint_pillar_512.pth \
  https://drive.google.com/...  # OpenPCDet model zoo

# Download configs
git clone --depth 1 https://github.com/open-mmlab/OpenPCDet.git /tmp/openpcdet
cp -r /tmp/openpcdet/tools/cfgs models/3d_detection/cfgs
```

## Performance Considerations

- PV-RCNN++ requires ~6GB VRAM, CenterPoint ~3GB
- First inference is slow (model loading), subsequent calls are fast
- Consider model caching via ModelManager singleton
- Use half precision (FP16) for faster inference if accuracy allows
- Profile with `torch.profiler` for bottleneck identification

## Coordinate System Notes

| System | X | Y | Z | Origin |
|--------|---|---|---|--------|
| KITTI | Forward | Left | Up | LiDAR center |
| nuScenes | Right | Forward | Up | Vehicle center |
| Waymo | Forward | Left | Up | LiDAR center |

Conversion functions needed when mixing datasets.

## Related Sub-Specs

- [02-python-open3d.md](./02-python-open3d.md) - Input point cloud format
- [05-2d-3d-fusion.md](./05-2d-3d-fusion.md) - Project 3D boxes to 2D
- [06-threejs-viewer.md](./06-threejs-viewer.md) - Visualize 3D boxes
- [08-agent-tools.md](./08-agent-tools.md) - detect_3d agent tool
