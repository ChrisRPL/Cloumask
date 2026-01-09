# CVAT Format Loader

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base
> **Blocks:** 14-cvat-exporter, 22-convert-format-tool

## Objective

Implement a loader for CVAT (Computer Vision Annotation Tool) XML export format. CVAT exports a single XML file containing all images and annotations.

## Acceptance Criteria

- [ ] Parse CVAT for images 1.1 XML format
- [ ] Support box, polygon, polyline, points annotations
- [ ] Handle track annotations for video
- [ ] Support label attributes
- [ ] Validate structure and report warnings
- [ ] Unit tests with sample CVAT export

## CVAT Format Specification

### Export Structure
```
cvat_export/
├── annotations.xml    # All annotations
└── images/            # Optional, may be external
    ├── frame_0000.jpg
    └── frame_0001.jpg
```

### XML Structure
```xml
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <id>1</id>
      <name>My Task</name>
      <size>1000</size>
      <labels>
        <label>
          <name>car</name>
          <attributes>
            <attribute>
              <name>color</name>
              <values>red,blue,green</values>
            </attribute>
          </attributes>
        </label>
      </labels>
    </task>
  </meta>
  <image id="0" name="frame_0000.jpg" width="1920" height="1080">
    <box label="car" xtl="100" ytl="200" xbr="300" ybr="400" occluded="0">
      <attribute name="color">red</attribute>
    </box>
    <polygon label="person" points="100,200;150,200;150,300;100,300" />
  </image>
  <track id="0" label="car">
    <box frame="0" xtl="100" ytl="200" xbr="300" ybr="400" outside="0" occluded="0" />
    <box frame="1" xtl="105" ytl="205" xbr="305" ybr="405" outside="0" occluded="0" />
  </track>
</annotations>
```

## Implementation Steps

### 1. Create cvat.py

Create `backend/data/formats/cvat.py`:

```python
"""CVAT XML format loader and exporter.

Supports CVAT for images 1.1 export format.
- Single XML file with all annotations
- Box, polygon, polyline, points shapes
- Track annotations for video sequences
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

from backend.data.formats.base import FormatLoader, FormatRegistry, ProgressCallback
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)


@FormatRegistry.register_loader
class CvatLoader(FormatLoader):
    """Load CVAT XML export format.

    Expects:
    - annotations.xml or any .xml file with <annotations> root
    - Optional images/ directory

    Example:
        loader = CvatLoader(Path("/data/cvat_export"))
        dataset = loader.load()
    """

    format_name = "cvat"
    description = "CVAT XML format (annotations tool)"
    extensions = [".xml"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: Optional[list[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        xml_file: Optional[str] = None,
        load_tracks: bool = True,
    ) -> None:
        """Initialize CVAT loader.

        Args:
            root_path: Dataset root directory
            class_names: Override class names from XML
            progress_callback: Progress callback
            xml_file: Specific XML file to load
            load_tracks: Whether to expand tracks to per-frame annotations
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.xml_file = xml_file
        self.load_tracks = load_tracks
        self._labels: list[dict] = []
        self._images: dict[int, dict] = {}
        self._tracks: list[dict] = []

    def _find_xml_file(self) -> Optional[Path]:
        """Find the CVAT XML file."""
        if self.xml_file:
            path = self.root_path / self.xml_file
            if path.exists():
                return path
            return None

        # Look for annotations.xml
        if (self.root_path / "annotations.xml").exists():
            return self.root_path / "annotations.xml"

        # Find any XML with <annotations> root
        for xml_path in self.root_path.glob("*.xml"):
            try:
                tree = ET.parse(xml_path)
                if tree.getroot().tag == "annotations":
                    return xml_path
            except ET.ParseError:
                continue

        return None

    def _parse_xml(self) -> None:
        """Parse the CVAT XML file."""
        xml_path = self._find_xml_file()
        if xml_path is None:
            raise FileNotFoundError(f"No CVAT XML file found in {self.root_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Parse labels from meta
        meta = root.find("meta")
        if meta is not None:
            task = meta.find("task") or meta.find("project")
            if task is not None:
                labels_elem = task.find("labels")
                if labels_elem is not None:
                    for label in labels_elem.findall("label"):
                        name = label.find("name")
                        if name is not None and name.text:
                            label_info = {"name": name.text, "attributes": []}
                            attrs = label.find("attributes")
                            if attrs is not None:
                                for attr in attrs.findall("attribute"):
                                    attr_name = attr.find("name")
                                    if attr_name is not None:
                                        label_info["attributes"].append(attr_name.text)
                            self._labels.append(label_info)

        # Parse images
        for image in root.findall("image"):
            img_id = int(image.get("id", 0))
            self._images[img_id] = {
                "id": img_id,
                "name": image.get("name", ""),
                "width": int(image.get("width", 0)),
                "height": int(image.get("height", 0)),
                "annotations": [],
            }

            # Parse shapes within image
            for shape_type in ["box", "polygon", "polyline", "points", "cuboid"]:
                for shape in image.findall(shape_type):
                    ann = self._parse_shape(shape, shape_type)
                    if ann:
                        self._images[img_id]["annotations"].append(ann)

        # Parse tracks (video annotations)
        if self.load_tracks:
            for track in root.findall("track"):
                track_id = int(track.get("id", 0))
                track_label = track.get("label", "")

                for shape in track.findall("box"):
                    frame = int(shape.get("frame", 0))
                    outside = shape.get("outside", "0") == "1"

                    if outside:
                        continue  # Object not visible in this frame

                    ann = self._parse_shape(shape, "box")
                    if ann:
                        ann["track_id"] = track_id
                        ann["label"] = track_label  # Use track label

                        # Add to frame (create if needed)
                        if frame not in self._images:
                            self._images[frame] = {
                                "id": frame,
                                "name": f"frame_{frame:06d}",
                                "width": 0,
                                "height": 0,
                                "annotations": [],
                            }
                        self._images[frame]["annotations"].append(ann)

    def _parse_shape(self, elem: ET.Element, shape_type: str) -> Optional[dict]:
        """Parse a shape element.

        Args:
            elem: XML element
            shape_type: Type of shape (box, polygon, etc.)

        Returns:
            Annotation dict or None
        """
        ann: dict = {
            "type": shape_type,
            "label": elem.get("label", ""),
            "occluded": elem.get("occluded", "0") == "1",
            "attributes": {},
        }

        # Parse attributes
        for attr in elem.findall("attribute"):
            attr_name = attr.get("name", "")
            if attr_name and attr.text:
                ann["attributes"][attr_name] = attr.text

        if shape_type == "box":
            try:
                ann["xtl"] = float(elem.get("xtl", 0))
                ann["ytl"] = float(elem.get("ytl", 0))
                ann["xbr"] = float(elem.get("xbr", 0))
                ann["ybr"] = float(elem.get("ybr", 0))
            except (ValueError, TypeError):
                return None

        elif shape_type == "polygon" or shape_type == "polyline":
            points_str = elem.get("points", "")
            if points_str:
                points = []
                for pt in points_str.split(";"):
                    try:
                        x, y = pt.split(",")
                        points.extend([float(x), float(y)])
                    except (ValueError, IndexError):
                        continue
                ann["points"] = points

        elif shape_type == "points":
            points_str = elem.get("points", "")
            if points_str:
                points = []
                for pt in points_str.split(";"):
                    try:
                        x, y = pt.split(",")
                        points.extend([float(x), float(y)])
                    except (ValueError, IndexError):
                        continue
                ann["points"] = points

        return ann

    def _infer_class_names(self) -> list[str]:
        """Get class names from parsed labels."""
        if not self._labels:
            self._parse_xml()
        return [lbl["name"] for lbl in self._labels]

    def _find_image(self, name: str) -> Optional[Path]:
        """Find image file.

        Args:
            name: Image filename from XML

        Returns:
            Path to image or None
        """
        # Try various locations
        candidates = [
            self.root_path / name,
            self.root_path / "images" / name,
            self.root_path / "data" / name,
        ]

        for path in candidates:
            if path.exists():
                return path

        # Try just the filename without any directory
        basename = Path(name).name
        for subdir in [self.root_path, self.root_path / "images"]:
            if subdir.exists():
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    candidate = subdir / f"{Path(basename).stem}{ext}"
                    if candidate.exists():
                        return candidate

        return None

    def _annotation_to_label(
        self,
        ann: dict,
        img_width: int,
        img_height: int,
        class_names: list[str],
    ) -> Optional[Label]:
        """Convert CVAT annotation to Label.

        Args:
            ann: Annotation dict from parsing
            img_width: Image width for normalization
            img_height: Image height for normalization
            class_names: List of class names

        Returns:
            Label object or None
        """
        class_name = ann.get("label", "")
        if not class_name:
            return None

        # Get class ID
        if class_name in class_names:
            class_id = class_names.index(class_name)
        else:
            class_id = len(class_names)  # Unknown class

        # Parse bbox based on shape type
        shape_type = ann.get("type", "box")

        if shape_type == "box":
            xtl = ann.get("xtl", 0)
            ytl = ann.get("ytl", 0)
            xbr = ann.get("xbr", 0)
            ybr = ann.get("ybr", 0)

            if img_width > 0 and img_height > 0:
                bbox = BBox.from_xyxy(
                    xtl / img_width,
                    ytl / img_height,
                    xbr / img_width,
                    ybr / img_height,
                )
            else:
                bbox = BBox.from_xyxy(xtl, ytl, xbr, ybr)

        elif shape_type in ["polygon", "polyline", "points"]:
            points = ann.get("points", [])
            if len(points) < 4:
                return None

            # Compute bounding box from points
            xs = points[0::2]
            ys = points[1::2]
            xtl, xbr = min(xs), max(xs)
            ytl, ybr = min(ys), max(ys)

            if img_width > 0 and img_height > 0:
                bbox = BBox.from_xyxy(
                    xtl / img_width,
                    ytl / img_height,
                    xbr / img_width,
                    ybr / img_height,
                )
            else:
                bbox = BBox.from_xyxy(xtl, ytl, xbr, ybr)
        else:
            return None

        # Build attributes
        attributes = dict(ann.get("attributes", {}))
        attributes["occluded"] = ann.get("occluded", False)
        attributes["shape_type"] = shape_type

        if shape_type in ["polygon", "polyline"]:
            attributes["points"] = ann.get("points", [])

        return Label(
            class_name=class_name,
            class_id=class_id,
            bbox=bbox,
            attributes=attributes,
            track_id=ann.get("track_id"),
        )

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all samples in the dataset.

        Yields:
            Sample objects with labels
        """
        if not self._images:
            self._parse_xml()

        class_names = self.get_class_names()
        total = len(self._images)

        for idx, (img_id, img_data) in enumerate(sorted(self._images.items())):
            # Find image file
            image_path = self._find_image(img_data["name"])
            if image_path is None:
                logger.warning(f"Image not found: {img_data['name']}")
                image_path = self.root_path / img_data["name"]

            img_width = img_data["width"]
            img_height = img_data["height"]

            # Convert annotations to labels
            labels = []
            for ann in img_data["annotations"]:
                label = self._annotation_to_label(ann, img_width, img_height, class_names)
                if label:
                    labels.append(label)

            yield Sample(
                image_path=image_path,
                labels=labels,
                image_width=img_width if img_width > 0 else None,
                image_height=img_height if img_height > 0 else None,
                metadata={"cvat_id": img_id},
            )

            self._report_progress(idx + 1, total, "Loading CVAT")

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

        xml_path = self._find_xml_file()
        if xml_path is None:
            warnings.append("No CVAT XML file found")
            return warnings

        try:
            self._parse_xml()
        except ET.ParseError as e:
            warnings.append(f"XML parse error: {e}")
            return warnings

        if not self._labels:
            warnings.append("No labels defined in XML")

        if not self._images:
            warnings.append("No images found in XML")

        # Check for missing images
        missing = 0
        for img_data in self._images.values():
            if self._find_image(img_data["name"]) is None:
                missing += 1
        if missing > 0:
            warnings.append(f"{missing} images not found on disk")

        return warnings

    def summary(self) -> dict:
        """Get summary information."""
        if not self._images:
            self._parse_xml()

        base = super().summary()
        base["xml_file"] = str(self._find_xml_file())
        base["num_images"] = len(self._images)
        base["num_labels"] = len(self._labels)
        base["has_tracks"] = any("track_id" in ann for img in self._images.values() for ann in img["annotations"])
        return base
```

### 2. Create unit tests

Create `backend/tests/data/test_cvat_loader.py`:

```python
"""Tests for CVAT format loader."""

from pathlib import Path

import pytest

from backend.data.formats.cvat import CvatLoader
from backend.data.models import Dataset


@pytest.fixture
def cvat_dataset(tmp_path):
    """Create a sample CVAT dataset."""
    xml_content = '''<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label><name>car</name></label>
        <label><name>person</name></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="frame_000000.jpg" width="1920" height="1080">
    <box label="car" xtl="100" ytl="200" xbr="400" ybr="500" occluded="0">
      <attribute name="color">red</attribute>
    </box>
    <polygon label="person" points="500,300;600,300;600,500;500,500" occluded="1" />
  </image>
  <image id="1" name="frame_000001.jpg" width="1920" height="1080">
    <box label="person" xtl="200" ytl="100" xbr="350" ybr="400" occluded="0" />
  </image>
</annotations>'''

    (tmp_path / "annotations.xml").write_text(xml_content)

    # Create images
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "frame_000000.jpg").touch()
    (images_dir / "frame_000001.jpg").touch()

    return tmp_path


class TestCvatLoader:
    """Tests for CvatLoader."""

    def test_load_dataset(self, cvat_dataset):
        """Test loading a CVAT dataset."""
        loader = CvatLoader(cvat_dataset)
        ds = loader.load()

        assert isinstance(ds, Dataset)
        assert len(ds) == 2

    def test_class_names(self, cvat_dataset):
        """Test class names from labels."""
        loader = CvatLoader(cvat_dataset)
        names = loader.get_class_names()
        assert names == ["car", "person"]

    def test_parse_box(self, cvat_dataset):
        """Test box annotation parsing."""
        loader = CvatLoader(cvat_dataset)
        ds = loader.load()

        sample = next(s for s in ds if "000000" in str(s.image_path))
        car = next(l for l in sample.labels if l.class_name == "car")

        assert car.bbox.cx == pytest.approx(250 / 1920, rel=0.01)
        assert car.attributes.get("color") == "red"

    def test_parse_polygon(self, cvat_dataset):
        """Test polygon annotation parsing."""
        loader = CvatLoader(cvat_dataset)
        ds = loader.load()

        sample = next(s for s in ds if "000000" in str(s.image_path))
        person = next(l for l in sample.labels if l.class_name == "person")

        assert person.attributes["shape_type"] == "polygon"
        assert "points" in person.attributes
        assert person.attributes["occluded"] is True

    def test_occluded_attribute(self, cvat_dataset):
        """Test occluded attribute."""
        loader = CvatLoader(cvat_dataset)
        ds = loader.load()

        sample = ds[0]
        car = next(l for l in sample.labels if l.class_name == "car")
        assert car.attributes["occluded"] is False

    def test_iter_samples(self, cvat_dataset):
        """Test lazy iteration."""
        loader = CvatLoader(cvat_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 2

    def test_validate(self, cvat_dataset):
        """Test validation."""
        loader = CvatLoader(cvat_dataset)
        warnings = loader.validate()
        assert len(warnings) == 0

    def test_validate_missing_xml(self, tmp_path):
        """Test validation detects missing XML."""
        loader = CvatLoader(tmp_path)
        warnings = loader.validate()
        assert any("XML" in w for w in warnings)

    def test_track_annotations(self, tmp_path):
        """Test loading track annotations."""
        xml_content = '''<?xml version="1.0"?>
<annotations>
  <meta>
    <task><labels><label><name>car</name></label></labels></task>
  </meta>
  <track id="0" label="car">
    <box frame="0" xtl="100" ytl="100" xbr="200" ybr="200" outside="0" occluded="0" />
    <box frame="1" xtl="110" ytl="100" xbr="210" ybr="200" outside="0" occluded="0" />
    <box frame="2" xtl="120" ytl="100" xbr="220" ybr="200" outside="1" occluded="0" />
  </track>
</annotations>'''
        (tmp_path / "annotations.xml").write_text(xml_content)

        loader = CvatLoader(tmp_path, load_tracks=True)
        ds = loader.load()

        # Should have 2 frames (frame 2 is outside)
        assert len(ds) == 2

        # Check track_id preserved
        for sample in ds:
            for label in sample.labels:
                assert label.track_id == 0
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/cvat.py` | Create | CVAT loader implementation |
| `backend/data/formats/__init__.py` | Modify | Register CVAT loader |
| `backend/tests/data/test_cvat_loader.py` | Create | Unit tests |

## Verification

```bash
cd backend
pytest tests/data/test_cvat_loader.py -v
```

## Notes

- CVAT supports multiple annotation types: box, polygon, polyline, points, cuboid
- Track annotations link objects across video frames
- `outside="1"` means object is not visible in that frame
- Polygons are converted to bounding boxes but original points are preserved
- CVAT exports can be from tasks or projects (slightly different meta structure)
