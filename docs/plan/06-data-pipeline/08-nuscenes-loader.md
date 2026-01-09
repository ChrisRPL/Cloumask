# nuScenes Format Loader

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base
> **Blocks:** 15-nuscenes-exporter, 22-convert-format-tool

## Objective

Implement a loader for nuScenes format, a large-scale autonomous driving dataset with multi-sensor annotations including 3D bounding boxes, attributes, and sensor calibration.

## Acceptance Criteria

- [ ] Parse nuScenes table structure (JSON files)
- [ ] Load 2D and 3D bounding box annotations
- [ ] Support multi-camera setup (front, front-left, front-right, etc.)
- [ ] Handle object attributes (visibility, activity, etc.)
- [ ] Link annotations across sensors and timestamps
- [ ] Unit tests with sample nuScenes structure

## nuScenes Format Specification

### Directory Structure
```
nuscenes/
├── v1.0-mini/              # or v1.0-trainval
│   ├── sample.json         # Keyframe samples
│   ├── sample_data.json    # Sensor data references
│   ├── sample_annotation.json  # 3D annotations
│   ├── instance.json       # Object instances (tracks)
│   ├── category.json       # Object categories
│   ├── attribute.json      # Object attributes
│   ├── sensor.json         # Sensor definitions
│   ├── calibrated_sensor.json
│   ├── ego_pose.json       # Vehicle pose
│   └── scene.json          # Scene metadata
├── samples/                # Keyframe sensor data
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── LIDAR_TOP/
│   └── ...
└── sweeps/                 # Non-keyframe data
    └── ...
```

### Key Tables
- **sample**: Keyframe timestamps
- **sample_data**: Sensor readings at timestamps
- **sample_annotation**: 3D box annotations (token links to sample)
- **instance**: Object tracking across samples
- **category**: Object class definitions

## Implementation Steps

### 1. Create nuscenes.py

Create `backend/data/formats/nuscenes.py`:

```python
"""nuScenes format loader and exporter.

Supports nuScenes autonomous driving dataset format.
- Multi-sensor setup (cameras, LiDAR, radar)
- 3D bounding box annotations
- Object tracking and attributes
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from backend.data.formats.base import FormatLoader, FormatRegistry, ProgressCallback
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)


# nuScenes camera names
NUSCENES_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


@FormatRegistry.register_loader
class NuscenesLoader(FormatLoader):
    """Load nuScenes format datasets.

    Expects:
    - v1.0-*/ directory with JSON table files
    - samples/ directory with sensor data

    Example:
        loader = NuscenesLoader(Path("/data/nuscenes"))
        dataset = loader.load()
    """

    format_name = "nuscenes"
    description = "nuScenes format (autonomous driving)"
    extensions = [".json"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: Optional[list[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        version: str = "v1.0-trainval",
        cameras: Optional[list[str]] = None,
        with_3d: bool = True,
    ) -> None:
        """Initialize nuScenes loader.

        Args:
            root_path: Dataset root directory
            class_names: Override class names
            progress_callback: Progress callback
            version: Dataset version (v1.0-mini, v1.0-trainval, etc.)
            cameras: Which cameras to load (default: all)
            with_3d: Include 3D box information in attributes
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.version = version
        self.cameras = cameras or NUSCENES_CAMERAS
        self.with_3d = with_3d

        # Tables loaded on demand
        self._tables: dict[str, list] = {}
        self._category_map: dict[str, dict] = {}
        self._attribute_map: dict[str, dict] = {}
        self._instance_map: dict[str, dict] = {}
        self._sample_map: dict[str, dict] = {}
        self._sample_data_map: dict[str, dict] = {}
        self._calibration_map: dict[str, dict] = {}

    def _get_version_dir(self) -> Optional[Path]:
        """Find the version directory."""
        # Try exact version
        version_dir = self.root_path / self.version
        if version_dir.exists():
            return version_dir

        # Try any v1.0-* directory
        for subdir in self.root_path.iterdir():
            if subdir.is_dir() and subdir.name.startswith("v1.0"):
                return subdir

        # Tables might be in root
        if (self.root_path / "sample.json").exists():
            return self.root_path

        return None

    def _load_table(self, name: str) -> list:
        """Load a nuScenes table.

        Args:
            name: Table name (e.g., "sample", "category")

        Returns:
            List of table records
        """
        if name in self._tables:
            return self._tables[name]

        version_dir = self._get_version_dir()
        if version_dir is None:
            return []

        table_path = version_dir / f"{name}.json"
        if not table_path.exists():
            logger.warning(f"Table not found: {table_path}")
            return []

        with open(table_path) as f:
            self._tables[name] = json.load(f)

        return self._tables[name]

    def _build_indexes(self) -> None:
        """Build lookup indexes from tables."""
        # Categories
        for cat in self._load_table("category"):
            self._category_map[cat["token"]] = cat

        # Attributes
        for attr in self._load_table("attribute"):
            self._attribute_map[attr["token"]] = attr

        # Instances
        for inst in self._load_table("instance"):
            self._instance_map[inst["token"]] = inst

        # Samples
        for sample in self._load_table("sample"):
            self._sample_map[sample["token"]] = sample

        # Sample data
        for sd in self._load_table("sample_data"):
            self._sample_data_map[sd["token"]] = sd

        # Calibration
        for cal in self._load_table("calibrated_sensor"):
            self._calibration_map[cal["token"]] = cal

    def _infer_class_names(self) -> list[str]:
        """Get class names from category table."""
        categories = self._load_table("category")
        return sorted(set(cat["name"] for cat in categories))

    def _get_camera_sample_data(self, sample_token: str) -> dict[str, dict]:
        """Get sample_data records for each camera.

        Args:
            sample_token: Sample token

        Returns:
            Dict of camera_name -> sample_data record
        """
        result = {}
        for sd in self._load_table("sample_data"):
            if sd["sample_token"] == sample_token:
                # Get sensor name
                cal = self._calibration_map.get(sd["calibrated_sensor_token"], {})
                sensor_token = cal.get("sensor_token", "")

                for sensor in self._load_table("sensor"):
                    if sensor["token"] == sensor_token:
                        if sensor["channel"] in self.cameras:
                            result[sensor["channel"]] = sd
                        break

        return result

    def _project_3d_to_2d(
        self,
        box_3d: dict,
        calibration: dict,
        img_width: int,
        img_height: int,
    ) -> Optional[BBox]:
        """Project 3D box to 2D bounding box.

        This is a simplified projection - real nuScenes uses more complex transforms.

        Args:
            box_3d: 3D box dict with translation, size, rotation
            calibration: Camera calibration
            img_width: Image width
            img_height: Image height

        Returns:
            2D BBox or None if not visible
        """
        # Simplified: use the provided 2D box if available, otherwise estimate
        # In production, would use full camera projection matrix
        return None  # Return None to indicate 2D box not computed

    def _get_annotations_for_sample(
        self,
        sample_token: str,
        camera: str,
        img_width: int,
        img_height: int,
    ) -> list[Label]:
        """Get annotations for a sample and camera.

        Args:
            sample_token: Sample token
            camera: Camera name
            img_width: Image width
            img_height: Image height

        Returns:
            List of Label objects
        """
        class_names = self.get_class_names()
        labels = []

        for ann in self._load_table("sample_annotation"):
            if ann["sample_token"] != sample_token:
                continue

            # Get category
            cat = self._category_map.get(ann["category_token"], {})
            class_name = cat.get("name", "unknown")

            if class_name in class_names:
                class_id = class_names.index(class_name)
            else:
                class_id = len(class_names)

            # Get instance for tracking
            instance = self._instance_map.get(ann["instance_token"], {})
            track_id = hash(ann["instance_token"]) % 100000  # Numeric ID

            # Get attributes
            attributes = {}
            for attr_token in ann.get("attribute_tokens", []):
                attr = self._attribute_map.get(attr_token, {})
                if attr:
                    attr_name = attr.get("name", "")
                    attributes[attr_name] = True

            # Add visibility
            visibility = ann.get("visibility_token", "")
            for vis in self._load_table("visibility"):
                if vis["token"] == visibility:
                    attributes["visibility"] = vis["level"]
                    break

            # 3D box info
            if self.with_3d:
                attributes["translation_3d"] = ann.get("translation", [0, 0, 0])
                attributes["size_3d"] = ann.get("size", [0, 0, 0])  # wlh
                attributes["rotation_3d"] = ann.get("rotation", [1, 0, 0, 0])  # quaternion

            # For 2D, we need to project 3D box to image
            # This is complex and requires camera calibration
            # For now, create a placeholder bbox at image center
            # Real implementation would use nuscenes-devkit
            bbox = BBox(cx=0.5, cy=0.5, w=0.1, h=0.1)

            # Try to use visibility region if available
            if "bbox" in ann:
                # Some nuScenes extensions include 2D boxes
                x, y, w, h = ann["bbox"]
                bbox = BBox.from_xywh(
                    x / img_width,
                    y / img_height,
                    w / img_width,
                    h / img_height,
                )

            labels.append(Label(
                class_name=class_name,
                class_id=class_id,
                bbox=bbox,
                attributes=attributes,
                track_id=track_id,
            ))

        return labels

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all samples in the dataset.

        Yields:
            Sample objects with labels (one per camera per keyframe)
        """
        self._build_indexes()
        samples = self._load_table("sample")
        total = len(samples) * len(self.cameras)
        processed = 0

        for sample in samples:
            sample_token = sample["token"]
            camera_data = self._get_camera_sample_data(sample_token)

            for camera in self.cameras:
                if camera not in camera_data:
                    continue

                sd = camera_data[camera]

                # Get image path
                filename = sd["filename"]
                image_path = self.root_path / filename

                if not image_path.exists():
                    # Try samples/ subdirectory
                    image_path = self.root_path / "samples" / camera / Path(filename).name

                # Get image dimensions
                img_width = sd.get("width", 1600)
                img_height = sd.get("height", 900)

                # Get annotations
                labels = self._get_annotations_for_sample(
                    sample_token, camera, img_width, img_height
                )

                yield Sample(
                    image_path=image_path,
                    labels=labels,
                    image_width=img_width,
                    image_height=img_height,
                    metadata={
                        "sample_token": sample_token,
                        "camera": camera,
                        "timestamp": sample.get("timestamp", 0),
                    },
                )

                processed += 1
                self._report_progress(processed, total, f"Loading {camera}")

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

        version_dir = self._get_version_dir()
        if version_dir is None:
            warnings.append(f"No version directory found (expected {self.version})")
            return warnings

        # Check required tables
        required_tables = ["sample", "sample_data", "sample_annotation", "category"]
        for table in required_tables:
            if not (version_dir / f"{table}.json").exists():
                warnings.append(f"Missing table: {table}.json")

        # Check samples directory
        samples_dir = self.root_path / "samples"
        if not samples_dir.exists():
            warnings.append("No samples/ directory found")
        else:
            for camera in self.cameras:
                if not (samples_dir / camera).exists():
                    warnings.append(f"Camera directory not found: {camera}")

        return warnings

    def summary(self) -> dict:
        """Get summary information."""
        self._build_indexes()
        base = super().summary()
        base["version"] = self.version
        base["cameras"] = self.cameras
        base["num_samples"] = len(self._load_table("sample"))
        base["num_annotations"] = len(self._load_table("sample_annotation"))
        base["num_categories"] = len(self._category_map)
        return base
```

### 2. Create unit tests

Create `backend/tests/data/test_nuscenes_loader.py`:

```python
"""Tests for nuScenes format loader."""

import json
from pathlib import Path

import pytest

from backend.data.formats.nuscenes import NuscenesLoader
from backend.data.models import Dataset


@pytest.fixture
def nuscenes_dataset(tmp_path):
    """Create a minimal nuScenes dataset structure."""
    # Create version directory
    version_dir = tmp_path / "v1.0-mini"
    version_dir.mkdir()

    # Create tables
    categories = [
        {"token": "cat1", "name": "car"},
        {"token": "cat2", "name": "pedestrian"},
    ]
    (version_dir / "category.json").write_text(json.dumps(categories))

    samples = [
        {"token": "sample1", "timestamp": 1000000},
        {"token": "sample2", "timestamp": 1000100},
    ]
    (version_dir / "sample.json").write_text(json.dumps(samples))

    sensors = [
        {"token": "sensor1", "channel": "CAM_FRONT", "modality": "camera"},
    ]
    (version_dir / "sensor.json").write_text(json.dumps(sensors))

    calibrated_sensors = [
        {"token": "cal1", "sensor_token": "sensor1"},
    ]
    (version_dir / "calibrated_sensor.json").write_text(json.dumps(calibrated_sensors))

    sample_data = [
        {
            "token": "sd1",
            "sample_token": "sample1",
            "calibrated_sensor_token": "cal1",
            "filename": "samples/CAM_FRONT/img1.jpg",
            "width": 1600,
            "height": 900,
        },
        {
            "token": "sd2",
            "sample_token": "sample2",
            "calibrated_sensor_token": "cal1",
            "filename": "samples/CAM_FRONT/img2.jpg",
            "width": 1600,
            "height": 900,
        },
    ]
    (version_dir / "sample_data.json").write_text(json.dumps(sample_data))

    instances = [
        {"token": "inst1", "category_token": "cat1"},
    ]
    (version_dir / "instance.json").write_text(json.dumps(instances))

    annotations = [
        {
            "token": "ann1",
            "sample_token": "sample1",
            "instance_token": "inst1",
            "category_token": "cat1",
            "attribute_tokens": [],
            "translation": [10.0, 5.0, 0.5],
            "size": [4.0, 1.8, 1.5],
            "rotation": [1, 0, 0, 0],
        },
    ]
    (version_dir / "sample_annotation.json").write_text(json.dumps(annotations))

    (version_dir / "attribute.json").write_text("[]")
    (version_dir / "visibility.json").write_text("[]")

    # Create sample images
    samples_dir = tmp_path / "samples" / "CAM_FRONT"
    samples_dir.mkdir(parents=True)
    (samples_dir / "img1.jpg").touch()
    (samples_dir / "img2.jpg").touch()

    return tmp_path


class TestNuscenesLoader:
    """Tests for NuscenesLoader."""

    def test_load_dataset(self, nuscenes_dataset):
        """Test loading a nuScenes dataset."""
        loader = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"])
        ds = loader.load()

        assert isinstance(ds, Dataset)
        assert len(ds) == 2

    def test_class_names(self, nuscenes_dataset):
        """Test class names from categories."""
        loader = NuscenesLoader(nuscenes_dataset)
        names = loader.get_class_names()
        assert "car" in names
        assert "pedestrian" in names

    def test_annotations(self, nuscenes_dataset):
        """Test annotation loading."""
        loader = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"])
        ds = loader.load()

        # First sample should have annotation
        sample = ds[0]
        assert len(sample.labels) == 1
        assert sample.labels[0].class_name == "car"

    def test_3d_attributes(self, nuscenes_dataset):
        """Test 3D box attributes."""
        loader = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"], with_3d=True)
        ds = loader.load()

        label = ds[0].labels[0]
        assert "translation_3d" in label.attributes
        assert "size_3d" in label.attributes
        assert "rotation_3d" in label.attributes

    def test_metadata(self, nuscenes_dataset):
        """Test sample metadata."""
        loader = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"])
        ds = loader.load()

        sample = ds[0]
        assert "sample_token" in sample.metadata
        assert "camera" in sample.metadata
        assert sample.metadata["camera"] == "CAM_FRONT"

    def test_validate(self, nuscenes_dataset):
        """Test validation."""
        loader = NuscenesLoader(nuscenes_dataset)
        warnings = loader.validate()
        # May have warnings about missing cameras
        assert not any("sample.json" in w for w in warnings)

    def test_summary(self, nuscenes_dataset):
        """Test summary method."""
        loader = NuscenesLoader(nuscenes_dataset)
        summary = loader.summary()

        assert summary["format"] == "nuscenes"
        assert summary["num_samples"] == 2
        assert summary["num_annotations"] == 1
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/nuscenes.py` | Create | nuScenes loader implementation |
| `backend/data/formats/__init__.py` | Modify | Register nuScenes loader |
| `backend/tests/data/test_nuscenes_loader.py` | Create | Unit tests |

## Verification

```bash
cd backend
pytest tests/data/test_nuscenes_loader.py -v
```

## Notes

- nuScenes uses token-based linking between tables
- 3D boxes are in ego vehicle coordinate frame
- Full 2D projection requires camera calibration matrices
- For production, consider using official nuscenes-devkit
- Dataset versions: v1.0-mini (10 scenes), v1.0-trainval (850 scenes)
- Each keyframe has data from 6 cameras, 1 LiDAR, 5 radars
