# OpenLABEL Format Loader

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base
> **Blocks:** 16-openlabel-exporter, 22-convert-format-tool

## Objective

Implement a loader for OpenLABEL (ASAM) format, an open standard for labeling sensor data in automotive applications. Supports multi-sensor, multi-frame annotations with rich semantic structure.

## Acceptance Criteria

- [ ] Parse OpenLABEL JSON structure
- [ ] Support 2D and 3D bounding boxes
- [ ] Handle object streams (tracks)
- [ ] Support actions, events, and relations
- [ ] Parse coordinate systems and transforms
- [ ] Unit tests with sample OpenLABEL file

## OpenLABEL Format Specification

### JSON Structure
```json
{
  "openlabel": {
    "metadata": {
      "schema_version": "1.0.0"
    },
    "coordinate_systems": {...},
    "streams": {...},
    "objects": {
      "uid1": {
        "name": "car_001",
        "type": "car",
        "object_data": {...}
      }
    },
    "frames": {
      "0": {
        "objects": {
          "uid1": {
            "object_data": {
              "bbox": [{
                "name": "bbox2d",
                "val": [x, y, w, h]
              }],
              "cuboid": [{
                "name": "cuboid3d",
                "val": [x, y, z, qx, qy, qz, qw, sx, sy, sz]
              }]
            }
          }
        }
      }
    }
  }
}
```

### Key Concepts
- **Objects**: Persistent entities with unique IDs
- **Frames**: Temporal snapshots of annotations
- **Streams**: Sensor data streams
- **Object Data**: Per-object, per-frame annotations

## Implementation Steps

### 1. Create openlabel.py

Create `backend/data/formats/openlabel.py`:

```python
"""OpenLABEL (ASAM) format loader and exporter.

Supports OpenLABEL 1.0 format for automotive sensor data.
- Multi-sensor support
- 2D and 3D annotations
- Object tracking
- Rich semantic structure
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

from backend.data.formats.base import FormatLoader, FormatRegistry, ProgressCallback
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)


@FormatRegistry.register_loader
class OpenlabelLoader(FormatLoader):
    """Load OpenLABEL format datasets.

    Expects:
    - JSON file with openlabel schema
    - Optional image/sensor files referenced in streams

    Example:
        loader = OpenlabelLoader(Path("/data/openlabel"))
        dataset = loader.load()
    """

    format_name = "openlabel"
    description = "OpenLABEL (ASAM) format"
    extensions = [".json"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: Optional[list[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        json_file: Optional[str] = None,
        stream_filter: Optional[list[str]] = None,
    ) -> None:
        """Initialize OpenLABEL loader.

        Args:
            root_path: Dataset root directory
            class_names: Override class names
            progress_callback: Progress callback
            json_file: Specific JSON file to load
            stream_filter: Only load annotations for these streams
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.json_file = json_file
        self.stream_filter = stream_filter
        self._data: Optional[dict] = None
        self._objects: dict[str, dict] = {}
        self._frames: dict[str, dict] = {}
        self._streams: dict[str, dict] = {}

    def _find_json_file(self) -> Optional[Path]:
        """Find the OpenLABEL JSON file."""
        if self.json_file:
            path = self.root_path / self.json_file
            if path.exists():
                return path
            return None

        # Look for JSON with openlabel key
        for json_path in self.root_path.glob("*.json"):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                if "openlabel" in data:
                    return json_path
            except (json.JSONDecodeError, KeyError):
                continue

        return None

    def _load_json(self) -> None:
        """Load and parse the OpenLABEL JSON file."""
        if self._data is not None:
            return

        json_path = self._find_json_file()
        if json_path is None:
            raise FileNotFoundError(f"No OpenLABEL JSON file found in {self.root_path}")

        logger.info(f"Loading OpenLABEL from {json_path}")
        with open(json_path) as f:
            raw_data = json.load(f)

        self._data = raw_data.get("openlabel", raw_data)

        # Parse objects
        self._objects = self._data.get("objects", {})

        # Parse frames
        self._frames = self._data.get("frames", {})

        # Parse streams
        self._streams = self._data.get("streams", {})

    def _infer_class_names(self) -> list[str]:
        """Get class names from objects."""
        self._load_json()
        types = set()
        for obj in self._objects.values():
            obj_type = obj.get("type", "")
            if obj_type:
                types.add(obj_type)
        return sorted(types)

    def _get_frame_image(self, frame_data: dict, frame_id: str) -> Optional[Path]:
        """Find image file for a frame.

        Args:
            frame_data: Frame data dict
            frame_id: Frame identifier

        Returns:
            Path to image or None
        """
        # Check frame_properties for image reference
        frame_props = frame_data.get("frame_properties", {})

        # Check streams for image data
        streams = frame_props.get("streams", {})
        for stream_name, stream_data in streams.items():
            if self.stream_filter and stream_name not in self.stream_filter:
                continue

            # Look for uri or file reference
            uri = stream_data.get("uri", "")
            if uri:
                image_path = self.root_path / uri
                if image_path.exists():
                    return image_path

        # Try finding by frame number
        for ext in [".jpg", ".jpeg", ".png"]:
            for pattern in [
                f"frame_{frame_id:>06}{ext}",
                f"{frame_id}{ext}",
                f"image_{frame_id}{ext}",
            ]:
                # Handle both string and int frame IDs
                try:
                    frame_num = int(frame_id)
                    pattern = f"frame_{frame_num:06d}{ext}"
                except ValueError:
                    pattern = f"{frame_id}{ext}"

                for subdir in [self.root_path, self.root_path / "images"]:
                    candidate = subdir / pattern
                    if candidate.exists():
                        return candidate

        return None

    def _parse_bbox_2d(
        self,
        bbox_data: list,
        img_width: int,
        img_height: int,
    ) -> Optional[BBox]:
        """Parse 2D bounding box.

        Args:
            bbox_data: List of bbox dicts
            img_width: Image width for normalization
            img_height: Image height for normalization

        Returns:
            BBox or None
        """
        if not bbox_data:
            return None

        # Find first valid bbox
        for bbox in bbox_data:
            val = bbox.get("val", [])
            if len(val) >= 4:
                x, y, w, h = val[:4]

                # Normalize if pixel coordinates
                if x > 1 or y > 1:
                    if img_width > 0 and img_height > 0:
                        return BBox.from_xywh(
                            x / img_width,
                            y / img_height,
                            w / img_width,
                            h / img_height,
                        )
                else:
                    return BBox.from_xywh(x, y, w, h)

        return None

    def _parse_cuboid_3d(self, cuboid_data: list) -> Optional[dict]:
        """Parse 3D cuboid.

        Args:
            cuboid_data: List of cuboid dicts

        Returns:
            3D box dict or None
        """
        if not cuboid_data:
            return None

        for cuboid in cuboid_data:
            val = cuboid.get("val", [])
            if len(val) >= 10:
                # [x, y, z, qx, qy, qz, qw, sx, sy, sz]
                return {
                    "position": val[:3],
                    "rotation": val[3:7],
                    "size": val[7:10],
                }

        return None

    def _get_frame_annotations(
        self,
        frame_data: dict,
        img_width: int,
        img_height: int,
    ) -> list[Label]:
        """Get annotations for a frame.

        Args:
            frame_data: Frame data dict
            img_width: Image width
            img_height: Image height

        Returns:
            List of Label objects
        """
        class_names = self.get_class_names()
        labels = []

        frame_objects = frame_data.get("objects", {})

        for obj_uid, obj_frame_data in frame_objects.items():
            # Get base object info
            base_obj = self._objects.get(obj_uid, {})
            obj_type = base_obj.get("type", "unknown")
            obj_name = base_obj.get("name", obj_uid)

            # Get class ID
            if obj_type in class_names:
                class_id = class_names.index(obj_type)
            else:
                class_id = len(class_names)

            # Parse object_data
            object_data = obj_frame_data.get("object_data", {})

            # Get 2D bbox
            bbox_data = object_data.get("bbox", [])
            bbox = self._parse_bbox_2d(bbox_data, img_width, img_height)

            if bbox is None:
                # Try to compute from poly2d
                poly2d = object_data.get("poly2d", [])
                if poly2d:
                    for poly in poly2d:
                        val = poly.get("val", [])
                        if len(val) >= 4:
                            xs = val[0::2]
                            ys = val[1::2]
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            if img_width > 0 and img_height > 0:
                                bbox = BBox.from_xyxy(
                                    x_min / img_width,
                                    y_min / img_height,
                                    x_max / img_width,
                                    y_max / img_height,
                                )
                            break

            if bbox is None:
                # Create placeholder
                bbox = BBox(cx=0.5, cy=0.5, w=0.1, h=0.1)

            # Build attributes
            attributes: dict[str, Any] = {
                "object_uid": obj_uid,
                "object_name": obj_name,
            }

            # Add 3D info
            cuboid_data = object_data.get("cuboid", [])
            cuboid_3d = self._parse_cuboid_3d(cuboid_data)
            if cuboid_3d:
                attributes["cuboid_3d"] = cuboid_3d

            # Add other object_data fields
            for key in ["text", "num", "boolean"]:
                if key in object_data:
                    for item in object_data[key]:
                        attr_name = item.get("name", key)
                        attr_val = item.get("val")
                        if attr_name and attr_val is not None:
                            attributes[attr_name] = attr_val

            # Track ID from object UID
            track_id = hash(obj_uid) % 100000

            labels.append(Label(
                class_name=obj_type,
                class_id=class_id,
                bbox=bbox,
                attributes=attributes,
                track_id=track_id,
            ))

        return labels

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all samples in the dataset.

        Yields:
            Sample objects with labels
        """
        self._load_json()

        total = len(self._frames)
        for idx, (frame_id, frame_data) in enumerate(sorted(self._frames.items())):
            # Find image
            image_path = self._get_frame_image(frame_data, frame_id)
            if image_path is None:
                image_path = self.root_path / f"frame_{frame_id}"

            # Get image dimensions from frame or default
            frame_props = frame_data.get("frame_properties", {})
            img_width = frame_props.get("width", 1920)
            img_height = frame_props.get("height", 1080)

            # Get annotations
            labels = self._get_frame_annotations(frame_data, img_width, img_height)

            yield Sample(
                image_path=image_path,
                labels=labels,
                image_width=img_width,
                image_height=img_height,
                metadata={
                    "frame_id": frame_id,
                    "timestamp": frame_props.get("timestamp", None),
                },
            )

            self._report_progress(idx + 1, total, "Loading OpenLABEL")

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

        json_path = self._find_json_file()
        if json_path is None:
            warnings.append("No OpenLABEL JSON file found")
            return warnings

        try:
            self._load_json()
        except json.JSONDecodeError as e:
            warnings.append(f"JSON parse error: {e}")
            return warnings

        # Check for required structure
        if not self._frames:
            warnings.append("No frames found in OpenLABEL file")

        if not self._objects:
            warnings.append("No objects defined in OpenLABEL file")

        # Check schema version
        metadata = self._data.get("metadata", {})
        version = metadata.get("schema_version", "")
        if not version.startswith("1."):
            warnings.append(f"Untested schema version: {version}")

        return warnings

    def summary(self) -> dict:
        """Get summary information."""
        self._load_json()
        base = super().summary()
        base["json_file"] = str(self._find_json_file())
        base["num_frames"] = len(self._frames)
        base["num_objects"] = len(self._objects)
        base["num_streams"] = len(self._streams)

        metadata = self._data.get("metadata", {})
        base["schema_version"] = metadata.get("schema_version", "unknown")

        return base
```

### 2. Create unit tests

Create `backend/tests/data/test_openlabel_loader.py`:

```python
"""Tests for OpenLABEL format loader."""

import json
from pathlib import Path

import pytest

from backend.data.formats.openlabel import OpenlabelLoader
from backend.data.models import Dataset


@pytest.fixture
def openlabel_dataset(tmp_path):
    """Create a sample OpenLABEL dataset."""
    openlabel_data = {
        "openlabel": {
            "metadata": {
                "schema_version": "1.0.0"
            },
            "objects": {
                "obj1": {
                    "name": "car_001",
                    "type": "car"
                },
                "obj2": {
                    "name": "ped_001",
                    "type": "pedestrian"
                }
            },
            "frames": {
                "0": {
                    "frame_properties": {
                        "width": 1920,
                        "height": 1080
                    },
                    "objects": {
                        "obj1": {
                            "object_data": {
                                "bbox": [{
                                    "name": "bbox2d",
                                    "val": [100, 200, 300, 150]
                                }],
                                "cuboid": [{
                                    "name": "cuboid3d",
                                    "val": [10.0, 5.0, 0.5, 0, 0, 0, 1, 4.0, 1.8, 1.5]
                                }]
                            }
                        }
                    }
                },
                "1": {
                    "frame_properties": {
                        "width": 1920,
                        "height": 1080
                    },
                    "objects": {
                        "obj1": {
                            "object_data": {
                                "bbox": [{
                                    "name": "bbox2d",
                                    "val": [110, 200, 300, 150]
                                }]
                            }
                        },
                        "obj2": {
                            "object_data": {
                                "bbox": [{
                                    "name": "bbox2d",
                                    "val": [500, 300, 80, 200]
                                }]
                            }
                        }
                    }
                }
            }
        }
    }

    (tmp_path / "annotations.json").write_text(json.dumps(openlabel_data))

    # Create frame images
    (tmp_path / "frame_000000.jpg").touch()
    (tmp_path / "frame_000001.jpg").touch()

    return tmp_path


class TestOpenlabelLoader:
    """Tests for OpenlabelLoader."""

    def test_load_dataset(self, openlabel_dataset):
        """Test loading an OpenLABEL dataset."""
        loader = OpenlabelLoader(openlabel_dataset)
        ds = loader.load()

        assert isinstance(ds, Dataset)
        assert len(ds) == 2

    def test_class_names(self, openlabel_dataset):
        """Test class names from objects."""
        loader = OpenlabelLoader(openlabel_dataset)
        names = loader.get_class_names()
        assert "car" in names
        assert "pedestrian" in names

    def test_parse_bbox(self, openlabel_dataset):
        """Test bbox parsing."""
        loader = OpenlabelLoader(openlabel_dataset)
        ds = loader.load()

        # First frame should have car
        sample = ds[0]
        assert len(sample.labels) == 1
        assert sample.labels[0].class_name == "car"

        # Verify normalization
        label = sample.labels[0]
        assert label.bbox.w == pytest.approx(300 / 1920, rel=0.01)

    def test_3d_cuboid(self, openlabel_dataset):
        """Test 3D cuboid parsing."""
        loader = OpenlabelLoader(openlabel_dataset)
        ds = loader.load()

        label = ds[0].labels[0]
        assert "cuboid_3d" in label.attributes
        cuboid = label.attributes["cuboid_3d"]
        assert "position" in cuboid
        assert "rotation" in cuboid
        assert "size" in cuboid

    def test_object_tracking(self, openlabel_dataset):
        """Test object UID preservation."""
        loader = OpenlabelLoader(openlabel_dataset)
        ds = loader.load()

        # Both frames should have car with same track_id
        car_frame0 = ds[0].labels[0]
        car_frame1 = next(l for l in ds[1].labels if l.class_name == "car")

        assert car_frame0.track_id == car_frame1.track_id
        assert car_frame0.attributes["object_uid"] == car_frame1.attributes["object_uid"]

    def test_multiple_objects_per_frame(self, openlabel_dataset):
        """Test multiple objects in single frame."""
        loader = OpenlabelLoader(openlabel_dataset)
        ds = loader.load()

        # Second frame should have car and pedestrian
        sample = ds[1]
        assert len(sample.labels) == 2
        types = {l.class_name for l in sample.labels}
        assert types == {"car", "pedestrian"}

    def test_iter_samples(self, openlabel_dataset):
        """Test lazy iteration."""
        loader = OpenlabelLoader(openlabel_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 2

    def test_validate(self, openlabel_dataset):
        """Test validation."""
        loader = OpenlabelLoader(openlabel_dataset)
        warnings = loader.validate()
        assert len(warnings) == 0

    def test_validate_missing_json(self, tmp_path):
        """Test validation detects missing JSON."""
        loader = OpenlabelLoader(tmp_path)
        warnings = loader.validate()
        assert any("JSON" in w for w in warnings)

    def test_summary(self, openlabel_dataset):
        """Test summary method."""
        loader = OpenlabelLoader(openlabel_dataset)
        summary = loader.summary()

        assert summary["format"] == "openlabel"
        assert summary["num_frames"] == 2
        assert summary["num_objects"] == 2
        assert summary["schema_version"] == "1.0.0"
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/openlabel.py` | Create | OpenLABEL loader implementation |
| `backend/data/formats/__init__.py` | Modify | Register OpenLABEL loader |
| `backend/tests/data/test_openlabel_loader.py` | Create | Unit tests |

## Verification

```bash
cd backend
pytest tests/data/test_openlabel_loader.py -v
```

## Notes

- OpenLABEL is an ASAM standard for automotive data labeling
- Schema version 1.0.0 is the current stable version
- Supports complex annotation types: bbox, cuboid, poly2d, poly3d
- Objects persist across frames with unique IDs
- Coordinate systems can be defined for multi-sensor setups
- Actions, events, and relations can describe scene dynamics
