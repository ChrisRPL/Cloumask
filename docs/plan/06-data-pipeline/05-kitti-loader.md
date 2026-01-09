# KITTI Format Loader

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base
> **Blocks:** 12-kitti-exporter, 22-convert-format-tool

## Objective

Implement a loader for KITTI format, commonly used in autonomous driving datasets. KITTI uses one `.txt` file per image with 15 fields per object including 3D information.

## Acceptance Criteria

- [ ] Parse KITTI 2D detection format (15 fields per line)
- [ ] Support 3D bbox and pose information (optional fields)
- [ ] Handle camera calibration files
- [ ] Validate structure and report warnings
- [ ] Unit tests with sample KITTI dataset

## KITTI Format Specification

### Directory Structure
```
dataset/
├── training/
│   ├── image_2/           # Left color camera
│   │   ├── 000000.png
│   │   └── 000001.png
│   ├── label_2/           # Labels for image_2
│   │   ├── 000000.txt
│   │   └── 000001.txt
│   └── calib/             # Calibration (optional)
│       ├── 000000.txt
│       └── 000001.txt
└── testing/
    ├── image_2/
    └── calib/
```

### Label Format (15 fields)
```
# type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom h w l x y z rotation_y
Car 0.00 0 -1.82 517.0 174.0 636.0 224.0 1.47 1.60 3.69 1.04 1.82 9.64 -1.57
Pedestrian 0.00 1 0.21 397.0 181.0 434.0 268.0 1.72 0.50 0.80 -5.52 1.77 10.85 0.17
```

Fields:
1. type: Object class (Car, Pedestrian, Cyclist, etc.)
2. truncated: 0.0-1.0 (how much object is outside image)
3. occluded: 0=visible, 1=partly, 2=largely, 3=unknown
4. alpha: Observation angle (-pi to pi)
5-8. bbox: 2D bbox in pixels (left, top, right, bottom)
9-11. dimensions: 3D object dimensions (height, width, length in meters)
12-14. location: 3D object location (x, y, z in meters)
15. rotation_y: Rotation around Y-axis (-pi to pi)

## Implementation Steps

### 1. Create kitti.py

Create `backend/data/formats/kitti.py`:

```python
"""KITTI format loader and exporter.

Supports KITTI 2D/3D detection format used in autonomous driving.
- One .txt label file per image
- 15 fields per object line
- Pixel coordinates (absolute, not normalized)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from backend.data.formats.base import FormatLoader, FormatRegistry, ProgressCallback
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)


# KITTI class names
KITTI_CLASSES = [
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
    "DontCare",
]


@FormatRegistry.register_loader
class KittiLoader(FormatLoader):
    """Load KITTI format datasets.

    Expects:
    - training/ or testing/ directory
    - image_2/ for images, label_2/ for labels
    - One .txt file per image with 15-field format

    Example:
        loader = KittiLoader(Path("/data/kitti"))
        dataset = loader.load()
    """

    format_name = "kitti"
    description = "KITTI format (autonomous driving)"
    extensions = [".txt"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: Optional[list[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        splits: Optional[list[str]] = None,
        include_dontcare: bool = False,
        image_dir: str = "image_2",
        label_dir: str = "label_2",
    ) -> None:
        """Initialize KITTI loader.

        Args:
            root_path: Dataset root directory
            class_names: Override class names
            progress_callback: Progress callback
            splits: Which splits to load (default: ['training'])
            include_dontcare: Include DontCare labels
            image_dir: Image subdirectory name
            label_dir: Label subdirectory name
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.splits = splits or ["training"]
        self.include_dontcare = include_dontcare
        self.image_dir = image_dir
        self.label_dir = label_dir

    def _infer_class_names(self) -> list[str]:
        """Get KITTI class names."""
        if self.include_dontcare:
            return KITTI_CLASSES
        return [c for c in KITTI_CLASSES if c != "DontCare"]

    def _get_split_paths(self) -> dict[str, tuple[Path, Path]]:
        """Get image and label directories for each split.

        Returns:
            Dict of split -> (image_dir, label_dir)
        """
        paths: dict[str, tuple[Path, Path]] = {}

        for split in self.splits:
            # Try split/image_2 structure
            img_dir = self.root_path / split / self.image_dir
            lbl_dir = self.root_path / split / self.label_dir

            if not img_dir.exists():
                # Try direct structure (no split subdirectory)
                img_dir = self.root_path / self.image_dir
                lbl_dir = self.root_path / self.label_dir

            if img_dir.exists():
                paths[split] = (img_dir, lbl_dir if lbl_dir.exists() else None)

        return paths

    def _parse_label_line(
        self,
        line: str,
        img_width: int,
        img_height: int,
        class_names: list[str],
    ) -> Optional[Label]:
        """Parse a single KITTI label line.

        Args:
            line: Label line (15 space-separated fields)
            img_width: Image width for normalization
            img_height: Image height for normalization
            class_names: List of valid class names

        Returns:
            Label object or None if invalid/skipped
        """
        parts = line.strip().split()
        if len(parts) < 15:
            logger.warning(f"Invalid KITTI line (need 15 fields): {line[:50]}...")
            return None

        type_name = parts[0]

        # Skip DontCare if not wanted
        if type_name == "DontCare" and not self.include_dontcare:
            return None

        try:
            truncated = float(parts[1])
            occluded = int(parts[2])
            alpha = float(parts[3])
            left, top, right, bottom = [float(x) for x in parts[4:8]]
            height_3d, width_3d, length_3d = [float(x) for x in parts[8:11]]
            loc_x, loc_y, loc_z = [float(x) for x in parts[11:14]]
            rotation_y = float(parts[14])
        except (ValueError, IndexError) as e:
            logger.warning(f"Parse error in KITTI line: {e}")
            return None

        # Get class ID
        if type_name in class_names:
            class_id = class_names.index(type_name)
        else:
            logger.warning(f"Unknown KITTI class: {type_name}")
            class_id = len(class_names)  # Unknown class

        # Normalize bbox
        bbox = BBox.from_xyxy(
            left / img_width,
            top / img_height,
            right / img_width,
            bottom / img_height,
        )

        # Store 3D info in attributes
        attributes = {
            "truncated": truncated,
            "occluded": occluded,
            "alpha": alpha,
            "dimensions_3d": {"height": height_3d, "width": width_3d, "length": length_3d},
            "location_3d": {"x": loc_x, "y": loc_y, "z": loc_z},
            "rotation_y": rotation_y,
        }

        return Label(
            class_name=type_name,
            class_id=class_id,
            bbox=bbox,
            attributes=attributes,
        )

    def _get_image_dimensions(self, image_path: Path) -> tuple[int, int]:
        """Get image dimensions.

        Returns:
            (width, height) tuple
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except ImportError:
            # Fallback: assume standard KITTI size
            return (1242, 375)

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all samples in the dataset.

        Yields:
            Sample objects with labels
        """
        class_names = self.get_class_names()
        split_paths = self._get_split_paths()

        # Count total
        total = 0
        for img_dir, _ in split_paths.values():
            total += len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))

        processed = 0
        for split, (img_dir, lbl_dir) in split_paths.items():
            # Get all images
            images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))

            for image_path in images:
                # Get image dimensions
                img_width, img_height = self._get_image_dimensions(image_path)

                # Find label file
                labels = []
                if lbl_dir:
                    label_path = lbl_dir / (image_path.stem + ".txt")
                    if label_path.exists():
                        with open(label_path) as f:
                            for line in f:
                                if line.strip():
                                    label = self._parse_label_line(
                                        line, img_width, img_height, class_names
                                    )
                                    if label:
                                        labels.append(label)

                yield Sample(
                    image_path=image_path,
                    labels=labels,
                    image_width=img_width,
                    image_height=img_height,
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

        split_paths = self._get_split_paths()
        if not split_paths:
            warnings.append("No valid splits found (expected training/ or testing/)")
            return warnings

        for split, (img_dir, lbl_dir) in split_paths.items():
            # Count images
            img_count = len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))
            if img_count == 0:
                warnings.append(f"No images in {split}/{self.image_dir}")

            # Check for labels
            if lbl_dir is None:
                warnings.append(f"No label directory for {split}")
            else:
                lbl_count = len(list(lbl_dir.glob("*.txt")))
                if lbl_count == 0:
                    warnings.append(f"No labels in {split}/{self.label_dir}")
                elif lbl_count < img_count:
                    warnings.append(f"{split}: {img_count - lbl_count} images without labels")

        return warnings

    def summary(self) -> dict:
        """Get summary information."""
        base = super().summary()
        base["splits"] = list(self._get_split_paths().keys())
        base["include_dontcare"] = self.include_dontcare
        return base
```

### 2. Create unit tests

Create `backend/tests/data/test_kitti_loader.py`:

```python
"""Tests for KITTI format loader."""

from pathlib import Path

import pytest

from backend.data.formats.kitti import KittiLoader, KITTI_CLASSES
from backend.data.models import Dataset


@pytest.fixture
def kitti_dataset(tmp_path):
    """Create a sample KITTI dataset."""
    # Create structure
    train_img = tmp_path / "training" / "image_2"
    train_lbl = tmp_path / "training" / "label_2"
    train_img.mkdir(parents=True)
    train_lbl.mkdir(parents=True)

    # Create dummy images (touch files)
    (train_img / "000000.png").touch()
    (train_img / "000001.png").touch()

    # Create labels
    (train_lbl / "000000.txt").write_text(
        "Car 0.00 0 -1.82 517.0 174.0 636.0 224.0 1.47 1.60 3.69 1.04 1.82 9.64 -1.57\n"
        "Pedestrian 0.00 1 0.21 397.0 181.0 434.0 268.0 1.72 0.50 0.80 -5.52 1.77 10.85 0.17\n"
    )
    (train_lbl / "000001.txt").write_text(
        "Cyclist 0.50 2 1.23 100.0 150.0 200.0 300.0 1.80 0.60 1.90 2.50 1.60 15.20 0.85\n"
    )

    return tmp_path


class TestKittiLoader:
    """Tests for KittiLoader."""

    def test_load_dataset(self, kitti_dataset):
        """Test loading a KITTI dataset."""
        loader = KittiLoader(kitti_dataset)
        ds = loader.load()

        assert isinstance(ds, Dataset)
        assert len(ds) == 2

    def test_class_names(self, kitti_dataset):
        """Test class names."""
        loader = KittiLoader(kitti_dataset)
        names = loader.get_class_names()
        # Should not include DontCare by default
        assert "DontCare" not in names
        assert "Car" in names

    def test_include_dontcare(self, kitti_dataset):
        """Test including DontCare class."""
        loader = KittiLoader(kitti_dataset, include_dontcare=True)
        names = loader.get_class_names()
        assert "DontCare" in names

    def test_parse_labels(self, kitti_dataset):
        """Test label parsing."""
        loader = KittiLoader(kitti_dataset)
        ds = loader.load()

        sample = next(s for s in ds if s.image_path.stem == "000000")
        assert len(sample.labels) == 2

        car_label = next(l for l in sample.labels if l.class_name == "Car")
        assert car_label.class_name == "Car"
        # Check 3D attributes preserved
        assert "dimensions_3d" in car_label.attributes
        assert "location_3d" in car_label.attributes
        assert car_label.attributes["truncated"] == 0.0
        assert car_label.attributes["occluded"] == 0

    def test_bbox_normalization(self, kitti_dataset):
        """Test bbox is normalized."""
        loader = KittiLoader(kitti_dataset)
        ds = loader.load()

        sample = ds[0]
        for label in sample.labels:
            assert 0 <= label.bbox.cx <= 1
            assert 0 <= label.bbox.cy <= 1
            assert 0 <= label.bbox.w <= 1
            assert 0 <= label.bbox.h <= 1

    def test_3d_attributes(self, kitti_dataset):
        """Test 3D attributes are preserved."""
        loader = KittiLoader(kitti_dataset)
        ds = loader.load()

        sample = ds[0]
        label = sample.labels[0]

        assert "dimensions_3d" in label.attributes
        dims = label.attributes["dimensions_3d"]
        assert "height" in dims
        assert "width" in dims
        assert "length" in dims

        assert "location_3d" in label.attributes
        loc = label.attributes["location_3d"]
        assert "x" in loc
        assert "y" in loc
        assert "z" in loc

    def test_iter_samples(self, kitti_dataset):
        """Test lazy iteration."""
        loader = KittiLoader(kitti_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 2

    def test_validate(self, kitti_dataset):
        """Test validation."""
        loader = KittiLoader(kitti_dataset)
        warnings = loader.validate()
        assert len(warnings) == 0

    def test_validate_missing_labels(self, kitti_dataset):
        """Test validation detects missing labels."""
        # Remove a label file
        (kitti_dataset / "training" / "label_2" / "000001.txt").unlink()

        loader = KittiLoader(kitti_dataset)
        warnings = loader.validate()
        assert any("without labels" in w for w in warnings)

    def test_summary(self, kitti_dataset):
        """Test summary method."""
        loader = KittiLoader(kitti_dataset)
        summary = loader.summary()

        assert summary["format"] == "kitti"
        assert "training" in summary["splits"]
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/kitti.py` | Create | KITTI loader implementation |
| `backend/data/formats/__init__.py` | Modify | Register KITTI loader |
| `backend/tests/data/test_kitti_loader.py` | Create | Unit tests |

## Verification

```bash
# Run tests
cd backend
pytest tests/data/test_kitti_loader.py -v
```

## Notes

- KITTI uses 0-indexed file names (000000.png, 000001.png)
- Standard image size is 1242x375 pixels
- DontCare regions mark areas to ignore during evaluation
- 3D coordinates are in camera coordinate system
- truncated=1 means object is completely outside image bounds
- occluded=3 means occlusion level unknown
