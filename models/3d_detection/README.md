# 3D Object Detection Models

This directory contains model weights and configurations for 3D object detection using OpenPCDet.

## Supported Models

| Model | VRAM | Benchmark | Use Case |
|-------|------|-----------|----------|
| **PV-RCNN++** | ~4GB | 84% 3D AP (KITTI) | Highest accuracy, production |
| **CenterPoint** | ~3GB | 79% 3D AP (KITTI) | Faster inference, lower VRAM |

Both models detect: **Car**, **Pedestrian**, **Cyclist**

## Prerequisites

### 1. Install OpenPCDet from Source

OpenPCDet requires source installation with CUDA compilation:

```bash
# Clone OpenPCDet
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet

# Install dependencies
pip install -r requirements.txt

# Build (requires CUDA toolkit matching your PyTorch version)
python setup.py develop

# Verify installation
python -c "import pcdet; print('OpenPCDet installed successfully')"
```

### 2. Install spconv (Sparse Convolutions)

spconv is required for efficient 3D convolutions:

```bash
# For CUDA 11.8
pip install spconv-cu118

# For CUDA 12.1
pip install spconv-cu121

# Verify
python -c "import spconv; print(f'spconv version: {spconv.__version__}')"
```

## Model Downloads

### PV-RCNN++ (Recommended)

Download from the [OpenPCDet Model Zoo](https://github.com/open-mmlab/OpenPCDet#model-zoo):

```bash
# Create directory
mkdir -p models/3d_detection/pvrcnn

# Download checkpoint (KITTI trained)
# Get the link from OpenPCDet README -> Model Zoo -> KITTI 3D Detection
wget -O models/3d_detection/pvrcnn/pv_rcnn_plusplus_8369.pth \
  "https://drive.google.com/uc?export=download&id=<FILE_ID>"

# Download config
wget -O models/3d_detection/pvrcnn/pv_rcnn_plusplus.yaml \
  "https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/tools/cfgs/kitti_models/pv_rcnn_plusplus.yaml"
```

### CenterPoint (Fallback)

```bash
# Create directory
mkdir -p models/3d_detection/centerpoint

# Download checkpoint
wget -O models/3d_detection/centerpoint/centerpoint_pillar.pth \
  "https://drive.google.com/uc?export=download&id=<FILE_ID>"

# Download config
wget -O models/3d_detection/centerpoint/centerpoint.yaml \
  "https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/tools/cfgs/kitti_models/centerpoint.yaml"
```

## Directory Structure

After setup, the directory should look like:

```
models/3d_detection/
├── README.md
├── pvrcnn/
│   ├── pv_rcnn_plusplus.yaml      # Model config
│   └── pv_rcnn_plusplus_8369.pth  # Checkpoint (~150MB)
└── centerpoint/
    ├── centerpoint.yaml           # Model config
    └── centerpoint_pillar.pth     # Checkpoint (~100MB)
```

## Usage

```python
from backend.cv.detection_3d import get_3d_detector

# Auto-select based on VRAM
detector = get_3d_detector(prefer_accuracy=True)
detector.load()

# Run inference
result = detector.predict(
    "scan.pcd",
    confidence=0.3,
    classes=["Car", "Pedestrian"],
)

print(f"Found {result.count} objects")
for det in result.detections:
    print(f"  {det.class_name}: {det.confidence:.2f} at {det.center}")

detector.unload()
```

## API Endpoints

The backend provides REST endpoints for 3D detection:

```bash
# List available models
curl http://localhost:8765/detect3d/models

# Run detection
curl -X POST http://localhost:8765/detect3d/infer \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/path/to/scan.bin", "model": "auto", "confidence": 0.3}'

# Preload model
curl -X POST http://localhost:8765/detect3d/load \
  -H "Content-Type: application/json" \
  -d '{"model": "pvrcnn++", "device": "cuda"}'
```

## Supported Input Formats

- `.bin` - KITTI binary format (N x 4: x, y, z, intensity)
- `.pcd` - Point Cloud Data format
- `.ply` - Polygon File Format
- `.las`/`.laz` - LAS/LAZ LiDAR format

## Coordinate Systems

Models are trained on KITTI coordinates (x=forward, y=left, z=up).
The API automatically converts from:
- `nuscenes` (x=right, y=forward, z=up)
- `waymo` (same as KITTI)

## Troubleshooting

### OpenPCDet import error
```
RuntimeError: OpenPCDet not installed
```
Solution: Follow the installation steps above. Ensure CUDA toolkit version matches PyTorch.

### spconv build failure
```
error: command 'nvcc' failed
```
Solution: Install CUDA toolkit and ensure `nvcc` is in PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
Solution: Use CenterPoint (lower VRAM) or reduce point cloud size:
```python
detector = get_3d_detector(force_model="centerpoint")
```

### Model checkpoint not found
```
RuntimeError: PV-RCNN++ checkpoint not found
```
Solution: Download the checkpoint as described above and place in correct directory.
