# Pascal VOC Format Loader

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base
> **Blocks:** 13-voc-exporter, 22-convert-format-tool

## Objective

Implement a loader for Pascal VOC (Visual Object Classes) format. VOC uses XML annotation files per image with bounding boxes and object attributes.

## Acceptance Criteria

- [ ] Parse VOC XML annotation format
- [ ] Support bndbox, difficult, truncated, occluded attributes
- [ ] Handle segmentation class/object indices
- [ ] Validate structure and report warnings
- [ ] Unit tests with sample VOC dataset

## Pascal VOC Format Specification

### Directory Structure
```
VOC2012/
├── Annotations/          # XML annotation files
│   ├── 2007_000001.xml
│   └── 2007_000002.xml
├── ImageSets/            # Train/val/test splits
│   └── Main/
│       ├── train.txt
│       ├── val.txt
│       └── trainval.txt
├── JPEGImages/           # Image files
│   ├── 2007_000001.jpg
│   └── 2007_000002.jpg
└── SegmentationClass/    # Segmentation masks (optional)
    └── 2007_000001.png
```

### XML Annotation Format
```xml
<annotation>
  <folder>VOC2012</folder>
  <filename>2007_000001.jpg</filename>
  <size>
    <width>500</width>
    <height>375</height>
    <depth>3</depth>
  </size>
  <object>
    <name>person</name>
    <pose>Frontal</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>174</xmin>
      <ymin>101</ymin>
      <xmax>349</xmax>
      <ymax>351</ymax>
    </bndbox>
  </object>
</annotation>
```

## Implementation Steps

### 1. Create voc.py

Create `backend/data/formats/voc.py`:

```python
"""Pascal VOC format loader and exporter.

Supports VOC 2007/2012 detection and segmentation format.
- One XML annotation file per image
- Bounding boxes in pixel coordinates
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional

from backend.data.formats.base import FormatLoader, FormatRegistry, ProgressCallback
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)


# Standard VOC classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


@FormatRegistry.register_loader
class VocLoader(FormatLoader):
    """Load Pascal VOC format datasets.

    Expects:
    - Annotations/ directory with XML files
    - JPEGImages/ directory with images

    Example:
        loader = VocLoader(Path("/data/VOC2012"))
        dataset = loader.load()
    """

    format_name = "voc"
    description = "Pascal VOC XML format"
    extensions = [".xml"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: Optional[list[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        split: Optional[str] = None,
        include_difficult: bool = True,
        annotations_dir: str = "Annotations",
        images_dir: str = "JPEGImages",
    ) -> None:
        """Initialize VOC loader.

        Args:
            root_path: Dataset root directory
            class_names: Override class names
            progress_callback: Progress callback
            split: Specific split to load (train, val, trainval, test)
            include_difficult: Include objects marked as difficult
            annotations_dir: Annotations subdirectory name
            images_dir: Images subdirectory name
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.split = split
        self.include_difficult = include_difficult
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self._discovered_classes: set[str] = set()

    def _infer_class_names(self) -> list[str]:
        """Get class names (use VOC standard or discovered)."""
        if self._discovered_classes:
            return sorted(self._discovered_classes)
        return VOC_CLASSES

    def _get_annotation_dir(self) -> Optional[Path]:
        """Get annotations directory."""
        ann_dir = self.root_path / self.annotations_dir
        if ann_dir.exists():
            return ann_dir

        # Try common alternatives
        for alt in ["annotations", "Annotation", "labels"]:
            alt_dir = self.root_path / alt
            if alt_dir.exists():
                return alt_dir

        return None

    def _get_images_dir(self) -> Optional[Path]:
        """Get images directory."""
        img_dir = self.root_path / self.images_dir
        if img_dir.exists():
            return img_dir

        # Try common alternatives
        for alt in ["images", "JPEGs", "imgs"]:
            alt_dir = self.root_path / alt
            if alt_dir.exists():
                return alt_dir

        return None

    def _get_split_files(self) -> Optional[list[str]]:
        """Get list of files for the specified split."""
        if not self.split:
            return None

        # Look for ImageSets/Main/*.txt
        split_file = self.root_path / "ImageSets" / "Main" / f"{self.split}.txt"
        if not split_file.exists():
            split_file = self.root_path / "ImageSets" / f"{self.split}.txt"

        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            return None

        files = []
        with open(split_file) as f:
            for line in f:
                # Each line: filename or "filename class_flag"
                parts = line.strip().split()
                if parts:
                    files.append(parts[0])
        return files

    def _parse_annotation(self, xml_path: Path, class_names: list[str]) -> tuple[Sample, list[str]]:
        """Parse a VOC XML annotation file.

        Args:
            xml_path: Path to XML file
            class_names: List of class names for ID mapping

        Returns:
            (Sample object, list of new class names found)
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get filename
        filename_elem = root.find("filename")
        if filename_elem is None or not filename_elem.text:
            raise ValueError(f"No filename in {xml_path}")
        filename = filename_elem.text

        # Get size
        size_elem = root.find("size")
        img_width = int(size_elem.find("width").text) if size_elem is not None else 0
        img_height = int(size_elem.find("height").text) if size_elem is not None else 0

        # Find image file
        images_dir = self._get_images_dir()
        image_path = None
        if images_dir:
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = images_dir / filename
                if candidate.exists():
                    image_path = candidate
                    break
                # Try without extension
                stem = Path(filename).stem
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

        if image_path is None:
            image_path = self.root_path / filename  # Fallback

        # Parse objects
        labels = []
        new_classes: list[str] = []

        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                continue
            class_name = name_elem.text.strip()

            # Track discovered classes
            self._discovered_classes.add(class_name)

            # Check difficult flag
            difficult_elem = obj.find("difficult")
            is_difficult = difficult_elem is not None and difficult_elem.text == "1"
            if is_difficult and not self.include_difficult:
                continue

            # Get class ID
            if class_name in class_names:
                class_id = class_names.index(class_name)
            else:
                new_classes.append(class_name)
                class_id = len(class_names) + len(new_classes) - 1

            # Parse bndbox
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            try:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"Invalid bndbox in {xml_path}: {e}")
                continue

            # Normalize to 0-1
            if img_width > 0 and img_height > 0:
                bbox = BBox.from_xyxy(
                    xmin / img_width,
                    ymin / img_height,
                    xmax / img_width,
                    ymax / img_height,
                )
            else:
                # Store absolute if no size info
                bbox = BBox.from_xyxy(xmin, ymin, xmax, ymax)
                logger.warning(f"No size info in {xml_path}, using absolute coords")

            # Additional attributes
            attributes = {"difficult": is_difficult}

            truncated_elem = obj.find("truncated")
            if truncated_elem is not None:
                attributes["truncated"] = truncated_elem.text == "1"

            pose_elem = obj.find("pose")
            if pose_elem is not None and pose_elem.text:
                attributes["pose"] = pose_elem.text

            occluded_elem = obj.find("occluded")
            if occluded_elem is not None:
                attributes["occluded"] = occluded_elem.text == "1"

            labels.append(Label(
                class_name=class_name,
                class_id=class_id,
                bbox=bbox,
                attributes=attributes,
            ))

        sample = Sample(
            image_path=image_path,
            labels=labels,
            image_width=img_width,
            image_height=img_height,
            metadata={"annotation_file": str(xml_path)},
        )

        return sample, new_classes

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all samples in the dataset.

        Yields:
            Sample objects with labels
        """
        class_names = self._class_names or VOC_CLASSES
        ann_dir = self._get_annotation_dir()

        if ann_dir is None:
            logger.error(f"No annotations directory found in {self.root_path}")
            return

        # Get files to process
        split_files = self._get_split_files()

        if split_files:
            xml_files = [ann_dir / f"{f}.xml" for f in split_files if (ann_dir / f"{f}.xml").exists()]
        else:
            xml_files = sorted(ann_dir.glob("*.xml"))

        total = len(xml_files)
        for idx, xml_path in enumerate(xml_files):
            try:
                sample, _ = self._parse_annotation(xml_path, list(class_names))
                yield sample
            except Exception as e:
                logger.warning(f"Error parsing {xml_path}: {e}")
                continue

            self._report_progress(idx + 1, total, "Loading VOC")

    def load(self) -> Dataset:
        """Load the full dataset.

        Returns:
            Dataset with all samples
        """
        samples = list(self.iter_samples())

        # Use discovered classes if no explicit class names
        if not self._class_names:
            class_names = sorted(self._discovered_classes) if self._discovered_classes else VOC_CLASSES
        else:
            class_names = self._class_names

        return Dataset(
            samples,
            name=self.root_path.name,
            class_names=class_names,
        )

    def validate(self) -> list[str]:
        """Validate dataset structure.

        Returns:
            List of warning messages
        """
        warnings: list[str] = []

        ann_dir = self._get_annotation_dir()
        img_dir = self._get_images_dir()

        if ann_dir is None:
            warnings.append(f"Annotations directory not found ({self.annotations_dir})")
        else:
            xml_count = len(list(ann_dir.glob("*.xml")))
            if xml_count == 0:
                warnings.append("No XML annotation files found")

        if img_dir is None:
            warnings.append(f"Images directory not found ({self.images_dir})")
        else:
            img_count = sum(
                len(list(img_dir.glob(f"*{ext}")))
                for ext in [".jpg", ".jpeg", ".png"]
            )
            if img_count == 0:
                warnings.append("No images found")

        # Check for split file if specified
        if self.split:
            split_file = self.root_path / "ImageSets" / "Main" / f"{self.split}.txt"
            if not split_file.exists():
                warnings.append(f"Split file not found: {self.split}.txt")

        return warnings

    def summary(self) -> dict:
        """Get summary information."""
        base = super().summary()
        base["split"] = self.split
        base["include_difficult"] = self.include_difficult
        return base
```

### 2. Create unit tests

Create `backend/tests/data/test_voc_loader.py`:

```python
"""Tests for Pascal VOC format loader."""

from pathlib import Path

import pytest

from backend.data.formats.voc import VocLoader, VOC_CLASSES
from backend.data.models import Dataset


@pytest.fixture
def voc_dataset(tmp_path):
    """Create a sample VOC dataset."""
    # Create structure
    ann_dir = tmp_path / "Annotations"
    img_dir = tmp_path / "JPEGImages"
    ann_dir.mkdir()
    img_dir.mkdir()

    # Create annotation XML
    xml_content = '''<?xml version="1.0"?>
<annotation>
  <filename>img001.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <object>
    <name>person</name>
    <pose>Frontal</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>100</xmin>
      <ymin>100</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
  <object>
    <name>car</name>
    <truncated>1</truncated>
    <difficult>1</difficult>
    <bndbox>
      <xmin>400</xmin>
      <ymin>200</ymin>
      <xmax>600</xmax>
      <ymax>350</ymax>
    </bndbox>
  </object>
</annotation>'''

    (ann_dir / "img001.xml").write_text(xml_content)

    xml_content2 = '''<?xml version="1.0"?>
<annotation>
  <filename>img002.jpg</filename>
  <size><width>800</width><height>600</height><depth>3</depth></size>
  <object>
    <name>dog</name>
    <bndbox>
      <xmin>50</xmin><ymin>50</ymin><xmax>200</xmax><ymax>300</ymax>
    </bndbox>
  </object>
</annotation>'''
    (ann_dir / "img002.xml").write_text(xml_content2)

    # Create images
    (img_dir / "img001.jpg").touch()
    (img_dir / "img002.jpg").touch()

    return tmp_path


class TestVocLoader:
    """Tests for VocLoader."""

    def test_load_dataset(self, voc_dataset):
        """Test loading a VOC dataset."""
        loader = VocLoader(voc_dataset)
        ds = loader.load()

        assert isinstance(ds, Dataset)
        assert len(ds) == 2

    def test_parse_labels(self, voc_dataset):
        """Test label parsing."""
        loader = VocLoader(voc_dataset)
        ds = loader.load()

        sample = next(s for s in ds if s.image_path.stem == "img001")
        assert len(sample.labels) == 2

        person = next(l for l in sample.labels if l.class_name == "person")
        assert person.bbox.cx == pytest.approx(0.3125, rel=0.01)  # (100+300)/2 / 640
        assert person.attributes["difficult"] is False

    def test_difficult_flag(self, voc_dataset):
        """Test difficult object handling."""
        loader = VocLoader(voc_dataset)
        ds = loader.load()

        sample = next(s for s in ds if s.image_path.stem == "img001")
        car = next(l for l in sample.labels if l.class_name == "car")
        assert car.attributes["difficult"] is True

    def test_exclude_difficult(self, voc_dataset):
        """Test excluding difficult objects."""
        loader = VocLoader(voc_dataset, include_difficult=False)
        ds = loader.load()

        sample = next(s for s in ds if s.image_path.stem == "img001")
        assert len(sample.labels) == 1
        assert sample.labels[0].class_name == "person"

    def test_truncated_attribute(self, voc_dataset):
        """Test truncated attribute."""
        loader = VocLoader(voc_dataset)
        ds = loader.load()

        sample = next(s for s in ds if s.image_path.stem == "img001")
        car = next(l for l in sample.labels if l.class_name == "car")
        assert car.attributes["truncated"] is True

    def test_discovers_classes(self, voc_dataset):
        """Test class discovery from data."""
        loader = VocLoader(voc_dataset)
        ds = loader.load()

        # Should include discovered classes
        assert "person" in ds.class_names
        assert "car" in ds.class_names
        assert "dog" in ds.class_names

    def test_iter_samples(self, voc_dataset):
        """Test lazy iteration."""
        loader = VocLoader(voc_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 2

    def test_validate(self, voc_dataset):
        """Test validation."""
        loader = VocLoader(voc_dataset)
        warnings = loader.validate()
        assert len(warnings) == 0

    def test_validate_missing_dirs(self, tmp_path):
        """Test validation detects missing directories."""
        loader = VocLoader(tmp_path)
        warnings = loader.validate()
        assert any("Annotations" in w for w in warnings)
        assert any("Images" in w or "JPEGImages" in w for w in warnings)

    def test_split_loading(self, voc_dataset):
        """Test loading specific split."""
        # Create split file
        imagesets = voc_dataset / "ImageSets" / "Main"
        imagesets.mkdir(parents=True)
        (imagesets / "train.txt").write_text("img001\n")

        loader = VocLoader(voc_dataset, split="train")
        ds = loader.load()
        assert len(ds) == 1
        assert ds[0].image_path.stem == "img001"
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/voc.py` | Create | VOC loader implementation |
| `backend/data/formats/__init__.py` | Modify | Register VOC loader |
| `backend/tests/data/test_voc_loader.py` | Create | Unit tests |

## Verification

```bash
# Run tests
cd backend
pytest tests/data/test_voc_loader.py -v
```

## Notes

- VOC uses 1-based pixel coordinates (min inclusive, max inclusive)
- Standard VOC has 20 classes for detection, more for segmentation
- `difficult` flag marks hard-to-recognize objects (often excluded from evaluation)
- `truncated` indicates object extends beyond image boundary
- Segmentation masks stored as PNG with indexed colors
