# YOLO Format Loader

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base
> **Blocks:** 10-yolo-exporter, 22-convert-format-tool

## Objective

Implement a loader for YOLO format datasets (v5, v8, v11). YOLO uses one `.txt` file per image with normalized bounding boxes, plus a `data.yaml` config file.

## Acceptance Criteria

- [ ] Parse YOLO `data.yaml` for class names, paths
- [ ] Load labels from `.txt` files (class cx cy w h format)
- [ ] Handle both absolute and relative image paths
- [ ] Support segmentation masks (polygon format)
- [ ] Validate structure and report warnings
- [ ] Lazy iteration via `iter_samples()`
- [ ] Unit tests with sample YOLO dataset

## YOLO Format Specification

### Directory Structure
```
dataset/
├── data.yaml           # Config with paths and class names
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── labels/
│       ├── img001.txt
│       └── img002.txt
├── val/
│   ├── images/
│   └── labels/
└── test/              # Optional
    ├── images/
    └── labels/
```

### data.yaml Format
```yaml
path: /path/to/dataset  # Optional root path
train: train/images     # Relative to path
val: val/images
test: test/images       # Optional

names:
  0: person
  1: car
  2: bicycle
# OR
names: [person, car, bicycle]
```

### Label File Format
```
# class_id center_x center_y width height
0 0.5 0.5 0.2 0.3
1 0.25 0.75 0.1 0.15

# For segmentation (polygon points):
# class_id x1 y1 x2 y2 x3 y3 ... (normalized)
```

## Implementation Steps

### 1. Create yolo.py

Create `backend/data/formats/yolo.py`:

```python
"""YOLO format loader and exporter.

Supports YOLO v5, v8, and v11 formats.
- One .txt label file per image
- data.yaml config with class names
- Normalized bounding boxes (cx, cy, w, h)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import yaml

from backend.data.formats.base import FormatLoader, FormatRegistry, ProgressCallback
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)


@FormatRegistry.register_loader
class YoloLoader(FormatLoader):
    """Load YOLO format datasets.

    Expects:
    - data.yaml with 'names' (class names) and 'train'/'val'/'test' paths
    - images/ and labels/ subdirectories
    - One .txt file per image with labels

    Example:
        loader = YoloLoader(Path("/data/yolo_dataset"))
        dataset = loader.load()
    """

    format_name = "yolo"
    description = "YOLO v5/v8/v11 format (txt per image)"
    extensions = [".txt"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: Optional[list[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        splits: Optional[list[str]] = None,
    ) -> None:
        """Initialize YOLO loader.

        Args:
            root_path: Dataset root directory (contains data.yaml)
            class_names: Override class names from data.yaml
            progress_callback: Progress callback
            splits: Which splits to load (default: all available)
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.splits = splits or ["train", "val", "test"]
        self._config: Optional[dict] = None
        self._class_names_from_yaml: list[str] = []
        self._load_config()

    def _load_config(self) -> None:
        """Load data.yaml configuration."""
        yaml_path = self.root_path / "data.yaml"
        if not yaml_path.exists():
            # Try finding any yaml file
            yaml_files = list(self.root_path.glob("*.yaml")) + list(self.root_path.glob("*.yml"))
            if yaml_files:
                yaml_path = yaml_files[0]
            else:
                logger.warning(f"No data.yaml found in {self.root_path}")
                return

        with open(yaml_path) as f:
            self._config = yaml.safe_load(f)

        # Parse class names
        if "names" in self._config:
            names = self._config["names"]
            if isinstance(names, dict):
                # {0: 'person', 1: 'car'}
                max_id = max(names.keys())
                self._class_names_from_yaml = [""] * (max_id + 1)
                for idx, name in names.items():
                    self._class_names_from_yaml[idx] = name
            elif isinstance(names, list):
                self._class_names_from_yaml = names

    def _infer_class_names(self) -> list[str]:
        """Get class names from data.yaml."""
        return self._class_names_from_yaml

    def _get_split_paths(self) -> dict[str, Path]:
        """Get image directories for each split."""
        splits: dict[str, Path] = {}

        if self._config:
            base_path = self.root_path
            if "path" in self._config:
                config_path = Path(self._config["path"])
                if config_path.is_absolute():
                    base_path = config_path
                else:
                    base_path = self.root_path / config_path

            for split in self.splits:
                if split in self._config:
                    split_path = self._config[split]
                    if isinstance(split_path, str):
                        full_path = base_path / split_path
                        if full_path.exists():
                            splits[split] = full_path
        else:
            # No config, try standard structure
            for split in self.splits:
                for pattern in [f"{split}/images", f"{split}", f"images/{split}"]:
                    check_path = self.root_path / pattern
                    if check_path.exists():
                        splits[split] = check_path
                        break

        return splits

    def _find_label_file(self, image_path: Path) -> Optional[Path]:
        """Find corresponding label file for an image."""
        # Standard: images/ -> labels/
        if "images" in image_path.parts:
            label_path = Path(str(image_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"))
            label_path = label_path.with_suffix(".txt")
            if label_path.exists():
                return label_path

        # Same directory
        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            return label_path

        # labels/ subdirectory next to image
        label_path = image_path.parent / "labels" / (image_path.stem + ".txt")
        if label_path.exists():
            return label_path

        return None

    def _parse_label_file(self, label_path: Path, class_names: list[str]) -> list[Label]:
        """Parse a YOLO label file.

        Args:
            label_path: Path to .txt file
            class_names: Ordered list of class names

        Returns:
            List of Label objects
        """
        labels: list[Label] = []

        with open(label_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 5:
                    logger.warning(f"{label_path}:{line_num}: Invalid line (need 5+ values): {line}")
                    continue

                try:
                    class_id = int(parts[0])
                    values = [float(x) for x in parts[1:]]
                except ValueError as e:
                    logger.warning(f"{label_path}:{line_num}: Parse error: {e}")
                    continue

                # Get class name
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"class_{class_id}"
                    logger.warning(f"{label_path}: Unknown class_id {class_id}")

                if len(values) == 4:
                    # Bounding box: cx cy w h
                    cx, cy, w, h = values
                    bbox = BBox.from_cxcywh(cx, cy, w, h)
                    labels.append(Label(
                        class_name=class_name,
                        class_id=class_id,
                        bbox=bbox,
                    ))
                elif len(values) >= 6 and len(values) % 2 == 0:
                    # Segmentation polygon: x1 y1 x2 y2 ...
                    # Compute bounding box from polygon
                    xs = values[0::2]
                    ys = values[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox = BBox.from_xyxy(x_min, y_min, x_max, y_max)

                    # Store polygon in attributes
                    labels.append(Label(
                        class_name=class_name,
                        class_id=class_id,
                        bbox=bbox,
                        attributes={"polygon": values},
                    ))
                else:
                    logger.warning(f"{label_path}:{line_num}: Unexpected value count: {len(values)}")

        return labels

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all samples in the dataset.

        Yields:
            Sample objects with labels
        """
        class_names = self.get_class_names()
        split_paths = self._get_split_paths()
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Count total images for progress
        total = 0
        for split_path in split_paths.values():
            total += sum(1 for ext in image_extensions for _ in split_path.glob(f"*{ext}"))

        processed = 0
        for split, images_dir in split_paths.items():
            for ext in image_extensions:
                for image_path in sorted(images_dir.glob(f"*{ext}")):
                    # Find label file
                    label_path = self._find_label_file(image_path)
                    labels = []
                    if label_path:
                        labels = self._parse_label_file(label_path, class_names)

                    yield Sample(
                        image_path=image_path,
                        labels=labels,
                        metadata={"split": split},
                    )

                    processed += 1
                    self._report_progress(processed, total, f"Loading {split}")

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

        # Check for data.yaml
        if not (self.root_path / "data.yaml").exists():
            yaml_files = list(self.root_path.glob("*.yaml"))
            if not yaml_files:
                warnings.append("No data.yaml found")
            else:
                warnings.append(f"Using {yaml_files[0].name} instead of data.yaml")

        # Check class names
        if not self._class_names_from_yaml:
            warnings.append("No class names defined in config")

        # Check split directories
        split_paths = self._get_split_paths()
        if not split_paths:
            warnings.append("No train/val/test directories found")
        else:
            for split, path in split_paths.items():
                image_count = sum(
                    1 for ext in [".jpg", ".jpeg", ".png"]
                    for _ in path.glob(f"*{ext}")
                )
                if image_count == 0:
                    warnings.append(f"No images in {split} split")

        # Check for orphan labels (labels without images)
        for split, images_dir in split_paths.items():
            labels_dir = Path(str(images_dir).replace("images", "labels"))
            if labels_dir.exists():
                for label_file in labels_dir.glob("*.txt"):
                    image_stem = label_file.stem
                    has_image = any(
                        (images_dir / f"{image_stem}{ext}").exists()
                        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
                    )
                    if not has_image:
                        warnings.append(f"Orphan label: {label_file.name}")

        return warnings

    def summary(self) -> dict:
        """Get summary information."""
        base = super().summary()
        split_paths = self._get_split_paths()
        base["splits"] = list(split_paths.keys())
        base["config_file"] = "data.yaml" if (self.root_path / "data.yaml").exists() else None
        return base
```

### 2. Register in formats __init__.py

Update `backend/data/formats/__init__.py`:

```python
# Import format modules to trigger registration
from backend.data.formats import yolo  # noqa: F401
```

### 3. Create test fixtures

Create `backend/tests/data/fixtures/yolo_sample/`:

```bash
mkdir -p backend/tests/data/fixtures/yolo_sample/train/images
mkdir -p backend/tests/data/fixtures/yolo_sample/train/labels
```

Create `backend/tests/data/fixtures/yolo_sample/data.yaml`:
```yaml
train: train/images
val: val/images

names:
  0: person
  1: car
  2: bicycle
```

Create `backend/tests/data/fixtures/yolo_sample/train/labels/img001.txt`:
```
0 0.5 0.5 0.2 0.3
1 0.25 0.75 0.1 0.15
```

### 4. Create unit tests

Create `backend/tests/data/test_yolo_loader.py`:

```python
"""Tests for YOLO format loader."""

from pathlib import Path

import pytest

from backend.data.formats.yolo import YoloLoader
from backend.data.models import Dataset


@pytest.fixture
def yolo_dataset(tmp_path):
    """Create a sample YOLO dataset."""
    # Create structure
    train_images = tmp_path / "train" / "images"
    train_labels = tmp_path / "train" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)

    val_images = tmp_path / "val" / "images"
    val_labels = tmp_path / "val" / "labels"
    val_images.mkdir(parents=True)
    val_labels.mkdir(parents=True)

    # Create data.yaml
    (tmp_path / "data.yaml").write_text("""
train: train/images
val: val/images
names:
  0: person
  1: car
  2: bicycle
""")

    # Create sample images (empty files for testing)
    (train_images / "img001.jpg").touch()
    (train_images / "img002.jpg").touch()
    (val_images / "img003.jpg").touch()

    # Create labels
    (train_labels / "img001.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.25 0.75 0.1 0.15\n")
    (train_labels / "img002.txt").write_text("0 0.3 0.4 0.15 0.2\n")
    (val_labels / "img003.txt").write_text("2 0.6 0.6 0.1 0.1\n")

    return tmp_path


class TestYoloLoader:
    """Tests for YoloLoader."""

    def test_load_dataset(self, yolo_dataset):
        """Test loading a YOLO dataset."""
        loader = YoloLoader(yolo_dataset)
        ds = loader.load()

        assert isinstance(ds, Dataset)
        assert len(ds) == 3
        assert ds.class_names == ["person", "car", "bicycle"]

    def test_load_specific_splits(self, yolo_dataset):
        """Test loading specific splits only."""
        loader = YoloLoader(yolo_dataset, splits=["train"])
        ds = loader.load()

        assert len(ds) == 2
        assert all(s.metadata.get("split") == "train" for s in ds)

    def test_parse_labels(self, yolo_dataset):
        """Test that labels are parsed correctly."""
        loader = YoloLoader(yolo_dataset)
        ds = loader.load()

        # Find img001
        img001 = next(s for s in ds if s.image_path.stem == "img001")
        assert len(img001.labels) == 2

        # Check first label
        lbl = img001.labels[0]
        assert lbl.class_name == "person"
        assert lbl.class_id == 0
        assert lbl.bbox.cx == pytest.approx(0.5)
        assert lbl.bbox.cy == pytest.approx(0.5)

    def test_class_names_from_yaml(self, yolo_dataset):
        """Test class names are read from data.yaml."""
        loader = YoloLoader(yolo_dataset)
        assert loader.get_class_names() == ["person", "car", "bicycle"]

    def test_class_names_override(self, yolo_dataset):
        """Test class names can be overridden."""
        loader = YoloLoader(yolo_dataset, class_names=["a", "b", "c"])
        assert loader.get_class_names() == ["a", "b", "c"]

    def test_iter_samples(self, yolo_dataset):
        """Test lazy iteration."""
        loader = YoloLoader(yolo_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 3

    def test_validate(self, yolo_dataset):
        """Test validation returns no warnings for valid dataset."""
        loader = YoloLoader(yolo_dataset)
        warnings = loader.validate()
        assert len(warnings) == 0

    def test_validate_missing_yaml(self, tmp_path):
        """Test validation warns on missing data.yaml."""
        (tmp_path / "train" / "images").mkdir(parents=True)
        loader = YoloLoader(tmp_path)
        warnings = loader.validate()
        assert any("data.yaml" in w for w in warnings)

    def test_summary(self, yolo_dataset):
        """Test summary method."""
        loader = YoloLoader(yolo_dataset)
        summary = loader.summary()
        assert summary["format"] == "yolo"
        assert "train" in summary["splits"]
        assert "val" in summary["splits"]

    def test_segmentation_polygon(self, tmp_path):
        """Test parsing segmentation polygons."""
        images = tmp_path / "train" / "images"
        labels = tmp_path / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)

        (tmp_path / "data.yaml").write_text("train: train/images\nnames: [obj]")
        (images / "seg.jpg").touch()
        # Polygon format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        (labels / "seg.txt").write_text("0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")

        loader = YoloLoader(tmp_path)
        ds = loader.load()
        sample = ds[0]

        assert len(sample.labels) == 1
        assert "polygon" in sample.labels[0].attributes
        assert len(sample.labels[0].attributes["polygon"]) == 8

    def test_progress_callback(self, yolo_dataset):
        """Test progress callback is called."""
        progress_calls = []

        def callback(current, total, msg):
            progress_calls.append((current, total, msg))

        loader = YoloLoader(yolo_dataset, progress_callback=callback)
        loader.load()

        assert len(progress_calls) == 3  # 3 images
        assert progress_calls[-1][0] == progress_calls[-1][1]  # Last call: current == total
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/yolo.py` | Create | YOLO loader implementation |
| `backend/data/formats/__init__.py` | Modify | Register YOLO loader |
| `backend/tests/data/test_yolo_loader.py` | Create | Unit tests |
| `backend/tests/data/fixtures/yolo_sample/` | Create | Test fixtures |

## Verification

```bash
# Run tests
cd backend
pytest tests/data/test_yolo_loader.py -v

# Test with real YOLO dataset (if available)
python -c "
from pathlib import Path
from backend.data.formats import get_loader, detect_format

# Auto-detect and load
path = Path('/path/to/yolo/dataset')
if path.exists():
    fmt = detect_format(path)
    print(f'Detected: {fmt}')

    loader = get_loader(path)
    ds = loader.load()
    print(f'Loaded {len(ds)} samples')
    print(f'Classes: {ds.class_names}')
    print(f'Stats: {ds.stats()}')
"
```

## Notes

- YOLO format evolved through v5, v8, v11 - all use same basic structure
- Class IDs are 0-indexed in label files
- Segmentation polygons are stored as flat list in label file
- Empty label files = image with no objects (valid)
- Missing label file = unlabeled image (handled gracefully)
- Paths in data.yaml can be absolute or relative to yaml location
