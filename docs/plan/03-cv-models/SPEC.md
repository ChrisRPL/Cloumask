# CV Models Module

> **Status:** 🔴 Not Started
> **Priority:** P0 (Critical)
> **Dependencies:** 01-foundation, 02-agent-system

## Overview

Integrate computer vision models for detection, segmentation, and anonymization. Models run locally via PyTorch with GPU acceleration. Each model is wrapped as an agent tool.

## Goals

- [ ] YOLO11 object detection with configurable classes
- [ ] SAM3 text-prompted segmentation
- [ ] SCRFD face detection + anonymization (blur/blackbox)
- [ ] License plate detection and anonymization
- [ ] Lazy model loading to manage VRAM
- [ ] Batch processing with progress callbacks

## Technical Design

### Model Selection (January 2026)

| Task | Primary | Fallback | VRAM |
|------|---------|----------|------|
| Detection | YOLO11m | YOLO26, RF-DETR | ~2GB |
| Segmentation | SAM3 | SAM2, MobileSAM | ~8GB |
| Open-Vocab | YOLO-World | GroundingDINO | ~4GB |
| Faces | SCRFD-10G | YuNet | ~1GB |
| 3D Detection | PV-RCNN++ | CenterPoint | ~4GB |

### Model Loading Pattern
```python
class ModelManager:
    _models: dict[str, nn.Module] = {}

    def get_model(self, name: str) -> nn.Module:
        if name not in self._models:
            self._models[name] = self._load_model(name)
        return self._models[name]

    def unload_model(self, name: str):
        if name in self._models:
            del self._models[name]
            torch.cuda.empty_cache()
```

### Anonymization Modes
- **Blur:** Gaussian blur (configurable kernel)
- **Blackbox:** Solid color fill
- **Pixelate:** Mosaic effect
- **Mask:** SAM3 precise boundary

## Implementation Tasks

- [ ] **Model Infrastructure**
  - [ ] Create `ModelManager` singleton
  - [ ] Implement lazy loading
  - [ ] Add VRAM monitoring
  - [ ] GPU/CPU fallback logic

- [ ] **Object Detection**
  - [ ] Integrate YOLO11m via ultralytics
  - [ ] Add confidence threshold config
  - [ ] Implement class filtering
  - [ ] Add batch processing

- [ ] **Segmentation**
  - [ ] Integrate SAM3 from HuggingFace
  - [ ] Implement text prompt interface
  - [ ] Add point/box prompt fallback
  - [ ] Mask refinement options

- [ ] **Anonymization**
  - [ ] Integrate SCRFD for faces
  - [ ] Add license plate detector
  - [ ] Implement blur/blackbox/pixelate
  - [ ] SAM3 precise masking option

- [ ] **Agent Tools**
  - [ ] `detect_objects` tool
  - [ ] `segment_sam3` tool
  - [ ] `anonymize` tool
  - [ ] `detect_faces` tool
  - [ ] `detect_3d` tool

## Acceptance Criteria

- [ ] `detect_objects("car, person", image_path)` returns bounding boxes
- [ ] `segment_sam3("red car", image_path)` returns masks
- [ ] `anonymize(image_path, faces=True, plates=True)` produces anonymized image
- [ ] Models load on first use, not at startup
- [ ] VRAM usage stays under 10GB for typical workflow
- [ ] Batch of 100 images processes with progress updates

## Files to Create/Modify

```
backend/
├── cv/
│   ├── __init__.py
│   ├── types.py            # Pydantic models (Detection, Mask, etc.)
│   ├── manager.py          # ModelManager singleton
│   ├── device.py           # GPU/CPU detection, VRAM monitoring
│   ├── download.py         # Model download utilities
│   ├── base.py             # BaseModelWrapper abstract class
│   ├── detection.py        # YOLO11, RT-DETR
│   ├── segmentation.py     # SAM3, SAM2, MobileSAM
│   ├── faces.py            # SCRFD, YuNet
│   ├── openvocab.py        # YOLO-World, GroundingDINO
│   ├── plates.py           # License plate detection
│   ├── anonymization.py    # Face/plate blur pipeline
│   └── detection_3d.py     # PV-RCNN++, CenterPoint
├── agent/
│   └── tools/
│       ├── __init__.py     # CV_TOOLS registry
│       ├── detect.py       # detect_objects tool
│       ├── segment.py      # segment_sam3 tool
│       ├── anonymize.py    # anonymize tool
│       ├── faces.py        # detect_faces tool
│       └── detect_3d.py    # detect_3d tool

models/                      # Model weights (gitignored)
├── yolo11m.pt
├── sam3/
├── scrfd/
├── pvrcnn/
├── centerpoint/
└── README.md               # Download instructions
```

## Sub-Specs

Detailed implementation specs (implement in order):

| # | Spec | Description | Complexity |
|---|------|-------------|------------|
| 00 | [00-infrastructure.md](./00-infrastructure.md) | Base types, ModelManager, VRAM monitoring | M |
| 01 | [01-yolo11-detection.md](./01-yolo11-detection.md) | YOLO11m + RT-DETR fallback | S |
| 02 | [02-sam3-segmentation.md](./02-sam3-segmentation.md) | SAM3 + SAM2/MobileSAM fallbacks | L |
| 03 | [03-scrfd-faces.md](./03-scrfd-faces.md) | SCRFD-10G + YuNet fallback | M |
| 04 | [04-yolo-world-openvocab.md](./04-yolo-world-openvocab.md) | YOLO-World + GroundingDINO fallback | M |
| 05 | [05-plate-detection.md](./05-plate-detection.md) | License plate detection | S |
| 06 | [06-anonymization.md](./06-anonymization.md) | Blur/blackbox/pixelate/mask pipeline | M |
| 07 | [07-3d-detection.md](./07-3d-detection.md) | PV-RCNN++ + CenterPoint fallback | L |
| 08 | [08-cv-tools.md](./08-cv-tools.md) | LangGraph agent tools | M |

**Complexity:** S = Small (~2 days), M = Medium (~4 days), L = Large (~6 days)

### Recommended Implementation Order

1. **00-infrastructure** - Foundation for everything
2. **01-yolo11-detection** - Simplest model, establishes patterns
3. **03-scrfd-faces** - Needed for anonymization
4. **05-plate-detection** - Quick, builds on patterns
5. **06-anonymization** - First demo-able feature
6. **02-sam3-segmentation** - Complex, parallel development
7. **04-yolo-world-openvocab** - Extends detection
8. **07-3d-detection** - Point cloud support
9. **08-cv-tools** - Integrates everything
