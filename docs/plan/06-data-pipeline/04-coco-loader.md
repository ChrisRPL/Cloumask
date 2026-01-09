# COCO Format Loader

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base
> **Blocks:** 11-coco-exporter, 22-convert-format-tool

## Objective

Implement a loader for COCO (Common Objects in Context) format. COCO uses a single JSON file containing all annotations, with support for detection, segmentation, and keypoints.

## Acceptance Criteria

- [ ] Parse COCO JSON structure (images, annotations, categories)
- [ ] Load bounding boxes and convert to normalized format
- [ ] Support RLE and polygon segmentation masks
- [ ] Handle multiple annotation files (instances, keypoints, etc.)
- [ ] Lazy iteration via `iter_samples()`
- [ ] Validate structure and report warnings
- [ ] Unit tests with sample COCO dataset

## COCO Format Specification

### Directory Structure
```
dataset/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── captions_train.json    # Optional
├── train/                      # or train2017/
│   ├── 000000000001.jpg
│   └── 000000000002.jpg
└── val/
    └── ...
```

### JSON Structure
```json
{
  "info": {...},
  "licenses": [...],
  "images": [
    {
      "id": 1,
      "file_name": "000000000001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],  // Absolute pixels, top-left
      "area": 1234.5,
      "segmentation": [...],          // Polygon or RLE
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "human"
    }
  ]
}
```

## Implementation Steps

### 1. Create coco.py

Create `backend/data/formats/coco.py`:

```python
"""COCO format loader and exporter.

Supports COCO detection and segmentation format.
- Single JSON annotation file
- Absolute pixel bounding boxes (x, y, w, h)
- Polygon or RLE segmentation masks
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from backend.data.formats.base import FormatLoader, FormatRegistry, ProgressCallback
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)


def decode_rle(rle: dict, height: int, width: int) -> np.ndarray:
    """Decode COCO RLE segmentation to binary mask.

    Args:
        rle: RLE dict with 'counts' and 'size'
        height: Image height
        width: Image width

    Returns:
        Binary mask array (H, W)
    """
    counts = rle.get("counts", [])
    if isinstance(counts, str):
        # Compressed RLE - use pycocotools if available
        try:
            from pycocotools import mask as mask_utils
            return mask_utils.decode(rle).astype(np.uint8)
        except ImportError:
            logger.warning("pycocotools not installed, cannot decode compressed RLE")
            return np.zeros((height, width), dtype=np.uint8)

    # Uncompressed RLE
    mask = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    val = 0
    for count in counts:
        mask[pos:pos + count] = val
        pos += count
        val = 1 - val
    return mask.reshape((height, width), order="F")


def polygon_to_mask(polygon: list[list[float]], height: int, width: int) -> np.ndarray:
    """Convert COCO polygon to binary mask.

    Args:
        polygon: List of polygon coordinates [[x1,y1,x2,y2,...], ...]
        height: Image height
        width: Image width

    Returns:
        Binary mask array (H, W)
    """
    try:
        import cv2
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygon:
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask
    except ImportError:
        logger.warning("cv2 not installed, cannot convert polygon to mask")
        return np.zeros((height, width), dtype=np.uint8)


@FormatRegistry.register_loader
class CocoLoader(FormatLoader):
    """Load COCO format datasets.

    Expects:
    - annotations/ directory with JSON files
    - Image directories (train/, val/, etc.)

    Example:
        loader = CocoLoader(Path("/data/coco"))
        dataset = loader.load()
    """

    format_name = "coco"
    description = "COCO JSON format (detection/segmentation)"
    extensions = [".json"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: Optional[list[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        annotation_file: Optional[str] = None,
        load_masks: bool = True,
    ) -> None:
        """Initialize COCO loader.

        Args:
            root_path: Dataset root directory
            class_names: Override class names from JSON
            progress_callback: Progress callback
            annotation_file: Specific annotation file to load
            load_masks: Whether to load segmentation masks
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.annotation_file = annotation_file
        self.load_masks = load_masks
        self._coco_data: Optional[dict] = None
        self._categories: dict[int, dict] = {}
        self._image_info: dict[int, dict] = {}
        self._annotations: dict[int, list[dict]] = defaultdict(list)

    def _find_annotation_file(self) -> Optional[Path]:
        """Find the annotation JSON file."""
        if self.annotation_file:
            path = self.root_path / self.annotation_file
            if path.exists():
                return path
            path = self.root_path / "annotations" / self.annotation_file
            if path.exists():
                return path
            return None

        # Look for common patterns
        ann_dir = self.root_path / "annotations"
        if ann_dir.exists():
            # Prefer instances_* files
            for pattern in ["instances_*.json", "*.json"]:
                files = list(ann_dir.glob(pattern))
                if files:
                    return sorted(files)[0]

        # Check root for JSON files
        for json_file in self.root_path.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if "images" in data and "annotations" in data:
                    return json_file
            except (json.JSONDecodeError, KeyError):
                continue

        return None

    def _load_json(self) -> None:
        """Load and parse the COCO JSON file."""
        if self._coco_data is not None:
            return

        ann_file = self._find_annotation_file()
        if ann_file is None:
            raise FileNotFoundError(f"No COCO annotation file found in {self.root_path}")

        logger.info(f"Loading COCO annotations from {ann_file}")
        with open(ann_file) as f:
            self._coco_data = json.load(f)

        # Index categories
        for cat in self._coco_data.get("categories", []):
            self._categories[cat["id"]] = cat

        # Index images
        for img in self._coco_data.get("images", []):
            self._image_info[img["id"]] = img

        # Group annotations by image
        for ann in self._coco_data.get("annotations", []):
            self._annotations[ann["image_id"]].append(ann)

    def _infer_class_names(self) -> list[str]:
        """Get class names from COCO categories."""
        self._load_json()
        # Sort by category ID to get ordered list
        sorted_cats = sorted(self._categories.values(), key=lambda x: x["id"])
        return [cat["name"] for cat in sorted_cats]

    def _find_image(self, image_info: dict) -> Optional[Path]:
        """Find the actual image file.

        Args:
            image_info: COCO image dict with file_name

        Returns:
            Path to image or None
        """
        file_name = image_info["file_name"]

        # Try various locations
        candidates = [
            self.root_path / file_name,
            self.root_path / "images" / file_name,
            self.root_path / "train" / file_name,
            self.root_path / "train2017" / file_name,
            self.root_path / "val" / file_name,
            self.root_path / "val2017" / file_name,
        ]

        # Also try with path components from file_name
        if "/" in file_name:
            candidates.append(self.root_path / Path(file_name))

        for path in candidates:
            if path.exists():
                return path

        return None

    def _parse_annotation(
        self,
        ann: dict,
        img_width: int,
        img_height: int,
    ) -> Optional[Label]:
        """Parse a single COCO annotation.

        Args:
            ann: COCO annotation dict
            img_width: Image width for normalization
            img_height: Image height for normalization

        Returns:
            Label object or None if invalid
        """
        cat_id = ann.get("category_id")
        if cat_id not in self._categories:
            logger.warning(f"Unknown category_id: {cat_id}")
            return None

        category = self._categories[cat_id]
        class_name = category["name"]

        # Get class_id as 0-based index
        sorted_cat_ids = sorted(self._categories.keys())
        class_id = sorted_cat_ids.index(cat_id)

        # Parse bbox (COCO: x, y, width, height in pixels)
        bbox_data = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox_data) != 4:
            return None

        x, y, w, h = bbox_data
        bbox = BBox.from_absolute(
            (x, y, w, h),
            img_width,
            img_height,
            fmt=BBox.format.XYWH if hasattr(BBox, 'format') else None,
        )
        # Manual normalization since BBox.from_absolute uses XYXY by default
        bbox = BBox.from_xywh(
            x / img_width,
            y / img_height,
            w / img_width,
            h / img_height,
        )

        # Parse segmentation mask
        mask = None
        if self.load_masks and "segmentation" in ann:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                # RLE format
                mask = decode_rle(seg, img_height, img_width)
            elif isinstance(seg, list) and len(seg) > 0:
                # Polygon format
                mask = polygon_to_mask(seg, img_height, img_width)

        # Additional attributes
        attributes = {}
        if "iscrowd" in ann:
            attributes["iscrowd"] = bool(ann["iscrowd"])
        if "area" in ann:
            attributes["area"] = ann["area"]

        return Label(
            class_name=class_name,
            class_id=class_id,
            bbox=bbox,
            mask=mask,
            attributes=attributes,
        )

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all samples in the dataset.

        Yields:
            Sample objects with labels
        """
        self._load_json()

        total = len(self._image_info)
        for idx, (image_id, img_info) in enumerate(self._image_info.items()):
            # Find image file
            image_path = self._find_image(img_info)
            if image_path is None:
                logger.warning(f"Image not found: {img_info['file_name']}")
                continue

            # Parse annotations for this image
            img_width = img_info.get("width", 0)
            img_height = img_info.get("height", 0)

            labels = []
            for ann in self._annotations.get(image_id, []):
                label = self._parse_annotation(ann, img_width, img_height)
                if label:
                    labels.append(label)

            yield Sample(
                image_path=image_path,
                labels=labels,
                image_width=img_width,
                image_height=img_height,
                metadata={
                    "coco_id": image_id,
                    "coco_info": {k: v for k, v in img_info.items() if k != "file_name"},
                },
            )

            self._report_progress(idx + 1, total, "Loading COCO")

    def load(self) -> Dataset:
        """Load the full dataset.

        Returns:
            Dataset with all samples
        """
        samples = list(self.iter_samples())
        return Dataset(
            samples,
            name=self.root_path.name,
            class_names=self.get_class_names(),
        )

    def validate(self) -> list[str]:
        """Validate dataset structure.

        Returns:
            List of warning messages
        """
        warnings: list[str] = []

        # Check for annotation file
        ann_file = self._find_annotation_file()
        if ann_file is None:
            warnings.append("No COCO annotation file found")
            return warnings

        # Load and validate JSON structure
        try:
            self._load_json()
        except json.JSONDecodeError as e:
            warnings.append(f"Invalid JSON: {e}")
            return warnings

        # Check required fields
        if not self._coco_data.get("images"):
            warnings.append("No images in annotation file")
        if not self._coco_data.get("annotations"):
            warnings.append("No annotations in annotation file")
        if not self._coco_data.get("categories"):
            warnings.append("No categories in annotation file")

        # Check for missing images
        missing = 0
        for img_info in self._image_info.values():
            if self._find_image(img_info) is None:
                missing += 1
        if missing > 0:
            warnings.append(f"{missing} images not found on disk")

        # Check for orphan annotations
        image_ids = set(self._image_info.keys())
        orphan_anns = sum(
            1 for ann in self._coco_data.get("annotations", [])
            if ann["image_id"] not in image_ids
        )
        if orphan_anns > 0:
            warnings.append(f"{orphan_anns} annotations reference missing images")

        return warnings

    def summary(self) -> dict:
        """Get summary information."""
        self._load_json()
        base = super().summary()
        base["annotation_file"] = str(self._find_annotation_file())
        base["num_images"] = len(self._image_info)
        base["num_annotations"] = sum(len(anns) for anns in self._annotations.values())
        base["num_categories"] = len(self._categories)
        return base
```

### 2. Register in formats __init__.py

Update `backend/data/formats/__init__.py`:

```python
from backend.data.formats import yolo  # noqa: F401
from backend.data.formats import coco  # noqa: F401
```

### 3. Create unit tests

Create `backend/tests/data/test_coco_loader.py`:

```python
"""Tests for COCO format loader."""

import json
from pathlib import Path

import pytest

from backend.data.formats.coco import CocoLoader
from backend.data.models import Dataset


@pytest.fixture
def coco_dataset(tmp_path):
    """Create a sample COCO dataset."""
    # Create structure
    ann_dir = tmp_path / "annotations"
    train_dir = tmp_path / "train"
    ann_dir.mkdir()
    train_dir.mkdir()

    # Create annotation file
    coco_data = {
        "info": {"description": "Test dataset"},
        "images": [
            {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img002.jpg", "width": 800, "height": 600},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "area": 30000,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [300, 200, 100, 80],
                "area": 8000,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [50, 50, 300, 250],
                "area": 75000,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
        ],
    }

    (ann_dir / "instances_train.json").write_text(json.dumps(coco_data))

    # Create image files
    (train_dir / "img001.jpg").touch()
    (train_dir / "img002.jpg").touch()

    return tmp_path


class TestCocoLoader:
    """Tests for CocoLoader."""

    def test_load_dataset(self, coco_dataset):
        """Test loading a COCO dataset."""
        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        assert isinstance(ds, Dataset)
        assert len(ds) == 2

    def test_class_names(self, coco_dataset):
        """Test class names from categories."""
        loader = CocoLoader(coco_dataset)
        names = loader.get_class_names()
        assert names == ["person", "car"]

    def test_parse_bbox(self, coco_dataset):
        """Test bounding box parsing."""
        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        # Find img001
        sample = next(s for s in ds if s.image_path.stem == "img001")
        assert len(sample.labels) == 2

        # Check bbox normalization
        lbl = sample.labels[0]
        assert lbl.class_name == "person"
        # Original: x=100, y=100, w=200, h=150 on 640x480 image
        assert lbl.bbox.cx == pytest.approx((100 + 100) / 640, rel=0.01)
        assert lbl.bbox.cy == pytest.approx((100 + 75) / 480, rel=0.01)

    def test_image_dimensions(self, coco_dataset):
        """Test image dimensions are stored."""
        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        sample = ds[0]
        assert sample.image_width == 640
        assert sample.image_height == 480

    def test_metadata(self, coco_dataset):
        """Test COCO metadata is preserved."""
        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        sample = ds[0]
        assert "coco_id" in sample.metadata
        assert sample.metadata["coco_id"] == 1

    def test_iter_samples(self, coco_dataset):
        """Test lazy iteration."""
        loader = CocoLoader(coco_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 2

    def test_validate_valid_dataset(self, coco_dataset):
        """Test validation on valid dataset."""
        loader = CocoLoader(coco_dataset)
        warnings = loader.validate()
        assert len(warnings) == 0

    def test_validate_missing_images(self, coco_dataset):
        """Test validation detects missing images."""
        # Remove an image
        (coco_dataset / "train" / "img002.jpg").unlink()

        loader = CocoLoader(coco_dataset)
        warnings = loader.validate()
        assert any("not found" in w for w in warnings)

    def test_summary(self, coco_dataset):
        """Test summary method."""
        loader = CocoLoader(coco_dataset)
        summary = loader.summary()

        assert summary["format"] == "coco"
        assert summary["num_images"] == 2
        assert summary["num_annotations"] == 3
        assert summary["num_categories"] == 2

    def test_polygon_segmentation(self, tmp_path):
        """Test loading polygon segmentation."""
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()

        coco_data = {
            "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [{
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 30, 30],
                "segmentation": [[10, 10, 40, 10, 40, 40, 10, 40]],
                "area": 900,
                "iscrowd": 0,
            }],
            "categories": [{"id": 1, "name": "obj"}],
        }
        (ann_dir / "instances.json").write_text(json.dumps(coco_data))
        (tmp_path / "img.jpg").touch()

        loader = CocoLoader(tmp_path, load_masks=True)
        ds = loader.load()
        # Mask loading depends on cv2 being available
        # Just verify label is parsed
        assert len(ds[0].labels) == 1

    def test_specific_annotation_file(self, coco_dataset):
        """Test loading specific annotation file."""
        loader = CocoLoader(
            coco_dataset,
            annotation_file="instances_train.json"
        )
        ds = loader.load()
        assert len(ds) == 2
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/coco.py` | Create | COCO loader implementation |
| `backend/data/formats/__init__.py` | Modify | Register COCO loader |
| `backend/tests/data/test_coco_loader.py` | Create | Unit tests |

## Verification

```bash
# Run tests
cd backend
pytest tests/data/test_coco_loader.py -v

# Test with real COCO dataset
python -c "
from pathlib import Path
from backend.data.formats import get_loader

path = Path('/path/to/coco')
if path.exists():
    loader = get_loader(path, format_name='coco')
    print(f'Categories: {loader.get_class_names()[:10]}...')
    ds = loader.load()
    print(f'Loaded {len(ds)} samples')
"
```

## Notes

- COCO category IDs can be non-contiguous (e.g., 1, 2, 3, 5, 7)
- We map to 0-indexed class_id based on sorted category IDs
- Segmentation can be polygon (list of lists) or RLE (dict with counts)
- Compressed RLE requires pycocotools for decoding
- Area in annotations is in pixels squared
- iscrowd flag indicates crowd regions (single annotation, many instances)
