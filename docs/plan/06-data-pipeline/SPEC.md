# Data Pipeline Module

> **Status:** 🔴 Not Started
> **Priority:** P1 (High)
> **Dependencies:** 01-foundation, 03-cv-models

## Overview

Handle dataset operations: format import/export, duplicate detection, label QA, dataset splitting, and augmentation. Support all major annotation formats (YOLO, COCO, KITTI, etc.).

## Goals

- [ ] Import labels from YOLO, COCO, KITTI, Pascal VOC, CVAT
- [ ] Export to all supported formats
- [ ] Duplicate/similar image detection
- [ ] Label quality assurance checks
- [ ] Train/val/test splitting with stratification
- [ ] Augmentation via Albumentations

## Technical Design

### Supported Label Formats

| Format | Import | Export | Notes |
|--------|--------|--------|-------|
| YOLO v5/v8/v11 | ✓ | ✓ | txt per image |
| COCO JSON | ✓ | ✓ | Single JSON |
| KITTI | ✓ | ✓ | txt per image |
| Pascal VOC | ✓ | ✓ | XML per image |
| CVAT XML | ✓ | ✓ | Single XML |
| nuScenes | ✓ | ✓ | JSON + sensor data |
| OpenLABEL | ✓ | ✓ | ASAM standard |

### Internal Label Format
```python
@dataclass
class Label:
    class_name: str
    class_id: int
    bbox: BBox  # x, y, w, h (normalized)
    mask: Optional[np.ndarray]
    confidence: float
    attributes: dict

@dataclass
class Sample:
    image_path: Path
    labels: list[Label]
    metadata: dict
```

### Duplicate Detection
- **Perceptual hashing:** pHash, dHash
- **Embedding similarity:** CLIP embeddings
- **Threshold:** Configurable similarity cutoff

### Label QA Checks
- Missing labels (images with no annotations)
- Overlapping boxes (IoU > threshold)
- Out-of-bounds boxes
- Class imbalance warnings
- Tiny/huge box detection

## Implementation Tasks

- [ ] **Format Loaders**
  - [ ] YOLO format loader
  - [ ] COCO JSON loader
  - [ ] KITTI format loader
  - [ ] Pascal VOC loader
  - [ ] CVAT XML loader

- [ ] **Format Exporters**
  - [ ] YOLO format exporter
  - [ ] COCO JSON exporter
  - [ ] KITTI format exporter
  - [ ] Pascal VOC exporter

- [ ] **Duplicate Detection**
  - [ ] Perceptual hash computation
  - [ ] CLIP embedding extraction
  - [ ] Similarity index building
  - [ ] Duplicate clustering

- [ ] **Label QA**
  - [ ] Missing label detection
  - [ ] Overlap analysis
  - [ ] Bounds checking
  - [ ] Class distribution stats
  - [ ] HTML report generation

- [ ] **Dataset Operations**
  - [ ] Train/val/test splitting
  - [ ] Stratified splitting
  - [ ] Cross-validation folds
  - [ ] Dataset merging

- [ ] **Augmentation**
  - [ ] Albumentations integration
  - [ ] Preset augmentation pipelines
  - [ ] Preview augmentations

- [ ] **Agent Tools**
  - [ ] `convert_format` tool
  - [ ] `find_duplicates` tool
  - [ ] `label_qa` tool
  - [ ] `split_dataset` tool
  - [ ] `export` tool (full impl)

## Acceptance Criteria

- [ ] Convert COCO dataset to YOLO format correctly
- [ ] Find 95%+ of near-duplicate images
- [ ] QA report identifies all overlap issues
- [ ] Stratified split maintains class ratios
- [ ] Export produces valid format (parseable by tools)
- [ ] Augmentation preview shows transformations

## Files to Create/Modify

```
backend/
├── data/
│   ├── __init__.py
│   ├── formats/
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract loader/exporter
│   │   ├── yolo.py         # YOLO format
│   │   ├── coco.py         # COCO JSON
│   │   ├── kitti.py        # KITTI format
│   │   ├── voc.py          # Pascal VOC
│   │   └── cvat.py         # CVAT XML
│   ├── models.py           # Label, Sample dataclasses
│   ├── duplicates.py       # Duplicate detection
│   ├── qa.py               # Label QA checks
│   ├── splitting.py        # Dataset splitting
│   ├── augmentation.py     # Albumentations wrapper
│   └── report.py           # HTML report generation
├── agent/
│   └── tools/
│       ├── convert.py      # convert_format
│       ├── duplicates.py   # find_duplicates
│       ├── qa.py           # label_qa
│       ├── split.py        # split_dataset
│       └── export.py       # export (full)
```

## Sub-Specs (expand later)

- `format-loaders.md` - Import implementation details
- `format-exporters.md` - Export implementation details
- `duplicate-detection.md` - Hashing and similarity
- `label-qa.md` - QA check implementations
- `dataset-operations.md` - Split and merge logic
- `augmentation.md` - Albumentations integration
