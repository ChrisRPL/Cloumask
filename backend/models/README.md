# Model Weights

This directory contains model weights for Cloumask's CV models. Most models auto-download on first use, but some require manual setup.

## Directory Structure

```
models/
├── yolo11m.pt              # YOLO11m object detection (auto-download)
├── sam2.1_b.pt             # SAM2 segmentation (auto-download)
├── mobile_sam.pt           # MobileSAM (auto-download)
├── yolov8l-worldv2.pt      # YOLO-World open-vocabulary (auto-download)
├── yunet/                  # YuNet face detection
├── pvrcnn/                 # PV-RCNN++ 3D detection (manual)
│   ├── pv_rcnn_plusplus.yaml
│   └── pv_rcnn_plusplus_8369.pth
└── centerpoint/            # CenterPoint 3D detection (manual)
    ├── centerpoint.yaml
    └── centerpoint_pillar_512.pth
```

## Auto-Download Models

These models download automatically via ultralytics or HuggingFace:

| Model | Size | Source |
|-------|------|--------|
| yolo11m.pt | 40MB | Ultralytics |
| sam2.1_b.pt | 400MB | Ultralytics |
| mobile_sam.pt | 40MB | Ultralytics |
| yolov8l-worldv2.pt | 170MB | Ultralytics |

## Manual Download: 3D Detection Models

PV-RCNN++ and CenterPoint require OpenPCDet installation and manual model download.

### 1. Install OpenPCDet

```bash
# Install spconv (match your CUDA version)
pip install spconv-cu118  # For CUDA 11.8
# or
pip install spconv-cu121  # For CUDA 12.1

# Clone and install OpenPCDet
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
python setup.py develop
```

### 2. Download Model Weights

Download from [OpenPCDet Model Zoo](https://github.com/open-mmlab/OpenPCDet):

**PV-RCNN++ (KITTI):**
- Config: `tools/cfgs/kitti_models/pv_rcnn_plusplus.yaml`
- Checkpoint: `pv_rcnn_plusplus_8369.pth` (~150MB)

**CenterPoint (KITTI):**
- Config: `tools/cfgs/kitti_models/centerpoint.yaml`
- Checkpoint: `centerpoint_pillar_512.pth` (~100MB)

### 3. Place Files

```bash
mkdir -p models/pvrcnn models/centerpoint

# Copy config files from OpenPCDet
cp OpenPCDet/tools/cfgs/kitti_models/pv_rcnn_plusplus.yaml models/pvrcnn/
cp OpenPCDet/tools/cfgs/kitti_models/centerpoint.yaml models/centerpoint/

# Move downloaded checkpoints
mv pv_rcnn_plusplus_8369.pth models/pvrcnn/
mv centerpoint_pillar_512.pth models/centerpoint/
```

### 4. Verify Installation

```python
from backend.cv import get_3d_detector

detector = get_3d_detector()
detector.load()
result = detector.predict("path/to/pointcloud.bin")
print(f"Found {result.count} objects")
detector.unload()
```

## Environment Variables

Override model directory location:

```bash
export CLOUMASK_MODELS_DIR=/path/to/custom/models
```

## Disk Space

Approximate total size for all models:

| Category | Size |
|----------|------|
| 2D Detection (YOLO11, RT-DETR) | ~200MB |
| Segmentation (SAM2, MobileSAM) | ~450MB |
| Face Detection (SCRFD, YuNet) | ~50MB |
| Open-Vocabulary (YOLO-World) | ~170MB |
| 3D Detection (PV-RCNN++, CenterPoint) | ~250MB |
| **Total** | **~1.1GB** |
