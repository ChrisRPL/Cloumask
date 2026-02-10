# Data Pipeline Module

> **Status:** 🟢 Complete (Implemented + Tested)
> **Priority:** P1 (High)
> **Dependencies:** 01-foundation, 03-cv-models

## Overview

Handle dataset operations: format import/export, duplicate detection, label QA, dataset splitting, and augmentation. Support all major annotation formats (YOLO, COCO, KITTI, etc.).

## Goals

- [x] Import labels from YOLO, COCO, KITTI, Pascal VOC, CVAT
- [x] Export to all supported formats
- [x] Duplicate/similar image detection
- [x] Label quality assurance checks
- [x] Train/val/test splitting with stratification
- [x] Augmentation via Albumentations

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

- [x] **Format Loaders**
  - [x] YOLO format loader
  - [x] COCO JSON loader
  - [x] KITTI format loader
  - [x] Pascal VOC loader
  - [x] CVAT XML loader

- [x] **Format Exporters**
  - [x] YOLO format exporter
  - [x] COCO JSON exporter
  - [x] KITTI format exporter
  - [x] Pascal VOC exporter

- [x] **Duplicate Detection**
  - [x] Perceptual hash computation
  - [x] CLIP embedding extraction
  - [x] Similarity index building
  - [x] Duplicate clustering

- [x] **Label QA**
  - [x] Missing label detection
  - [x] Overlap analysis
  - [x] Bounds checking
  - [x] Class distribution stats
  - [x] HTML report generation

- [x] **Dataset Operations**
  - [x] Train/val/test splitting
  - [x] Stratified splitting
  - [x] Cross-validation folds
  - [x] Dataset merging

- [x] **Augmentation**
  - [x] Albumentations integration
  - [x] Preset augmentation pipelines
  - [x] Preview augmentations

- [x] **Agent Tools**
  - [x] `convert_format` tool
  - [x] `find_duplicates` tool
  - [x] `label_qa` tool
  - [x] `split_dataset` tool
  - [x] `export` tool (full impl)

## Acceptance Criteria

- [x] Convert COCO dataset to YOLO format correctly
- [x] Find 95%+ of near-duplicate images
- [x] QA report identifies all overlap issues
- [x] Stratified split maintains class ratios
- [x] Export produces valid format (parseable by tools)
- [x] Augmentation preview shows transformations

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
