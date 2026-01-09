# Format Base Classes

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models
> **Blocks:** All format loaders (03-09), all format exporters (10-16)

## Objective

Define abstract base classes for format loaders and exporters, plus a registry for auto-detection and format discovery.

## Acceptance Criteria

- [ ] `FormatLoader` ABC with `load()` method returning `Dataset`
- [ ] `FormatExporter` ABC with `export()` method taking `Dataset`
- [ ] Format registry with auto-detection from directory structure
- [ ] `get_loader()` and `get_exporter()` factory functions
- [ ] Lazy loading: loaders yield samples, don't load all at once
- [ ] Progress callback support for large datasets
- [ ] Unit tests for registry and auto-detection

## Implementation Steps

### 1. Create formats directory

```bash
mkdir -p backend/data/formats
touch backend/data/formats/__init__.py
```

### 2. Implement base.py

Create `backend/data/formats/base.py`:

```python
"""Abstract base classes for format loaders and exporters.

All format implementations inherit from these base classes.
Provides common interface for loading/exporting datasets.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional

if TYPE_CHECKING:
    from backend.data.models import Dataset, Sample

logger = logging.getLogger(__name__)


# Type alias for progress callbacks
ProgressCallback = Callable[[int, int, str], None]


class FormatLoader(ABC):
    """Abstract base for loading datasets from various formats.

    Subclasses implement format-specific parsing logic.
    All loaders convert to internal Sample/Label format.

    Example:
        loader = YoloLoader(Path("/data/yolo_dataset"))
        dataset = loader.load()
        for sample in dataset:
            print(sample.image_path, len(sample.labels))
    """

    # Format identifier (e.g., "yolo", "coco")
    format_name: str = "unknown"

    # File extensions this format typically uses
    extensions: list[str] = []

    # Description for UI/logs
    description: str = "Unknown format"

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: Optional[list[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize loader.

        Args:
            root_path: Root directory of the dataset
            class_names: Optional ordered list of class names
            progress_callback: Optional callback(current, total, message)
        """
        self.root_path = Path(root_path)
        self._class_names = class_names
        self._progress_callback = progress_callback

        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.root_path}")

    def _report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(current, total, message)

    @abstractmethod
    def load(self) -> "Dataset":
        """Load the dataset and return as Dataset object.

        Returns:
            Dataset containing all samples and metadata
        """
        pass

    @abstractmethod
    def iter_samples(self) -> Iterator["Sample"]:
        """Iterate over samples without loading all into memory.

        Yields:
            Sample objects one at a time
        """
        pass

    @abstractmethod
    def validate(self) -> list[str]:
        """Validate dataset structure and return warnings.

        Returns:
            List of warning/error messages (empty if valid)
        """
        pass

    def get_class_names(self) -> list[str]:
        """Get class names from dataset config or infer from data.

        Returns:
            Ordered list of class names
        """
        if self._class_names:
            return self._class_names
        return self._infer_class_names()

    def _infer_class_names(self) -> list[str]:
        """Infer class names from data. Override in subclasses."""
        return []

    def summary(self) -> dict:
        """Get summary information without loading full dataset."""
        return {
            "format": self.format_name,
            "root_path": str(self.root_path),
            "class_names": self.get_class_names(),
        }


class FormatExporter(ABC):
    """Abstract base for exporting datasets to various formats.

    Subclasses implement format-specific writing logic.
    All exporters read from internal Sample/Label format.

    Example:
        exporter = CocoExporter(Path("/output/coco"))
        exporter.export(dataset)
    """

    # Format identifier
    format_name: str = "unknown"

    # Description for UI/logs
    description: str = "Unknown format"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize exporter.

        Args:
            output_path: Output directory for exported dataset
            overwrite: Whether to overwrite existing files
            progress_callback: Optional callback(current, total, message)
        """
        self.output_path = Path(output_path)
        self.overwrite = overwrite
        self._progress_callback = progress_callback

    def _report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(current, total, message)

    def _ensure_output_dir(self) -> None:
        """Create output directory if needed."""
        if self.output_path.exists() and not self.overwrite:
            # Check if empty
            if any(self.output_path.iterdir()):
                raise FileExistsError(
                    f"Output directory not empty: {self.output_path}. "
                    "Use overwrite=True to overwrite."
                )
        self.output_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def export(
        self,
        dataset: "Dataset",
        *,
        copy_images: bool = True,
        image_subdir: str = "images",
    ) -> Path:
        """Export dataset to target format.

        Args:
            dataset: Dataset to export
            copy_images: Whether to copy images to output (vs. reference paths)
            image_subdir: Subdirectory name for images

        Returns:
            Path to exported dataset root
        """
        pass

    @abstractmethod
    def validate_export(self) -> list[str]:
        """Validate exported dataset and return warnings.

        Returns:
            List of warning/error messages (empty if valid)
        """
        pass


class FormatRegistry:
    """Registry for format loaders and exporters.

    Supports auto-detection and factory methods.

    Example:
        registry = FormatRegistry()
        loader = registry.get_loader(Path("/data/dataset"))  # auto-detect
        loader = registry.get_loader(Path("/data"), format_name="coco")  # explicit
    """

    _instance: Optional["FormatRegistry"] = None
    _loaders: dict[str, type[FormatLoader]] = {}
    _exporters: dict[str, type[FormatExporter]] = {}

    def __new__(cls) -> "FormatRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_loader(cls, loader_cls: type[FormatLoader]) -> type[FormatLoader]:
        """Register a loader class. Can be used as decorator.

        Args:
            loader_cls: FormatLoader subclass

        Returns:
            The registered class (for decorator use)
        """
        cls._loaders[loader_cls.format_name] = loader_cls
        logger.debug(f"Registered loader: {loader_cls.format_name}")
        return loader_cls

    @classmethod
    def register_exporter(cls, exporter_cls: type[FormatExporter]) -> type[FormatExporter]:
        """Register an exporter class. Can be used as decorator.

        Args:
            exporter_cls: FormatExporter subclass

        Returns:
            The registered class (for decorator use)
        """
        cls._exporters[exporter_cls.format_name] = exporter_cls
        logger.debug(f"Registered exporter: {exporter_cls.format_name}")
        return exporter_cls

    @classmethod
    def get_loader(
        cls,
        path: Path,
        *,
        format_name: Optional[str] = None,
        **kwargs,
    ) -> FormatLoader:
        """Get loader for a dataset.

        Args:
            path: Dataset root directory
            format_name: Explicit format name, or None for auto-detect
            **kwargs: Additional arguments for loader constructor

        Returns:
            Configured FormatLoader instance

        Raises:
            ValueError: If format cannot be detected or is unknown
        """
        if format_name is None:
            format_name = cls.detect_format(path)
            if format_name is None:
                raise ValueError(
                    f"Cannot detect format for: {path}. "
                    f"Available formats: {list(cls._loaders.keys())}"
                )

        if format_name not in cls._loaders:
            raise ValueError(
                f"Unknown format: {format_name}. "
                f"Available: {list(cls._loaders.keys())}"
            )

        return cls._loaders[format_name](path, **kwargs)

    @classmethod
    def get_exporter(
        cls,
        path: Path,
        format_name: str,
        **kwargs,
    ) -> FormatExporter:
        """Get exporter for a format.

        Args:
            path: Output directory
            format_name: Target format name
            **kwargs: Additional arguments for exporter constructor

        Returns:
            Configured FormatExporter instance

        Raises:
            ValueError: If format is unknown
        """
        if format_name not in cls._exporters:
            raise ValueError(
                f"Unknown export format: {format_name}. "
                f"Available: {list(cls._exporters.keys())}"
            )

        return cls._exporters[format_name](path, **kwargs)

    @classmethod
    def detect_format(cls, path: Path) -> Optional[str]:
        """Auto-detect dataset format from directory structure.

        Args:
            path: Dataset root directory

        Returns:
            Format name or None if cannot detect
        """
        path = Path(path)

        # YOLO: has data.yaml or *.yaml with train/val paths
        if (path / "data.yaml").exists():
            return "yolo"
        for yaml_file in path.glob("*.yaml"):
            try:
                import yaml
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and ("train" in data or "val" in data):
                    return "yolo"
            except Exception:
                pass

        # COCO: has annotations/*.json or instances_*.json
        if (path / "annotations").is_dir():
            if any((path / "annotations").glob("*.json")):
                return "coco"
        for json_file in path.glob("*.json"):
            try:
                import json
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "images" in data and "annotations" in data:
                    return "coco"
            except Exception:
                pass

        # Pascal VOC: has Annotations/*.xml
        if (path / "Annotations").is_dir():
            if any((path / "Annotations").glob("*.xml")):
                return "voc"

        # KITTI: has label_2/ directory
        if (path / "label_2").is_dir():
            return "kitti"

        # CVAT: single XML file with <annotations> root
        for xml_file in path.glob("*.xml"):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_file)
                if tree.getroot().tag == "annotations":
                    return "cvat"
            except Exception:
                pass

        # nuScenes: has v1.0-* directory with tables
        for subdir in path.iterdir():
            if subdir.is_dir() and subdir.name.startswith("v1.0"):
                if (subdir / "sample_annotation.json").exists():
                    return "nuscenes"

        # OpenLABEL: JSON with openlabel schema
        for json_file in path.glob("*.json"):
            try:
                import json
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "openlabel" in data:
                    return "openlabel"
            except Exception:
                pass

        return None

    @classmethod
    def list_formats(cls) -> dict[str, dict]:
        """List all registered formats.

        Returns:
            Dict of format info {name: {loader: bool, exporter: bool, desc: str}}
        """
        formats: dict[str, dict] = {}

        for name, loader in cls._loaders.items():
            formats[name] = {
                "loader": True,
                "exporter": name in cls._exporters,
                "description": loader.description,
            }

        for name, exporter in cls._exporters.items():
            if name not in formats:
                formats[name] = {
                    "loader": False,
                    "exporter": True,
                    "description": exporter.description,
                }

        return formats


# Module-level convenience functions

def get_loader(path: Path, format_name: Optional[str] = None, **kwargs) -> FormatLoader:
    """Get a format loader. See FormatRegistry.get_loader."""
    return FormatRegistry.get_loader(path, format_name=format_name, **kwargs)


def get_exporter(path: Path, format_name: str, **kwargs) -> FormatExporter:
    """Get a format exporter. See FormatRegistry.get_exporter."""
    return FormatRegistry.get_exporter(path, format_name, **kwargs)


def detect_format(path: Path) -> Optional[str]:
    """Detect dataset format. See FormatRegistry.detect_format."""
    return FormatRegistry.detect_format(path)


def list_formats() -> dict[str, dict]:
    """List available formats. See FormatRegistry.list_formats."""
    return FormatRegistry.list_formats()


def convert(
    input_path: Path,
    output_path: Path,
    target_format: str,
    *,
    source_format: Optional[str] = None,
    copy_images: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Convert dataset between formats.

    Args:
        input_path: Source dataset directory
        output_path: Target output directory
        target_format: Target format name
        source_format: Source format (auto-detect if None)
        copy_images: Whether to copy images
        progress_callback: Optional progress callback

    Returns:
        Path to converted dataset
    """
    # Load source
    loader = get_loader(input_path, format_name=source_format, progress_callback=progress_callback)
    dataset = loader.load()

    # Export to target
    exporter = get_exporter(output_path, target_format, progress_callback=progress_callback)
    return exporter.export(dataset, copy_images=copy_images)
```

### 3. Update formats __init__.py

Create `backend/data/formats/__init__.py`:

```python
"""Format loaders and exporters.

Supports YOLO, COCO, KITTI, Pascal VOC, CVAT, nuScenes, OpenLABEL.
"""

from backend.data.formats.base import (
    FormatExporter,
    FormatLoader,
    FormatRegistry,
    ProgressCallback,
    convert,
    detect_format,
    get_exporter,
    get_loader,
    list_formats,
)

# Import format modules to trigger registration
# (uncomment as they are implemented)
# from backend.data.formats import yolo
# from backend.data.formats import coco
# from backend.data.formats import kitti
# from backend.data.formats import voc
# from backend.data.formats import cvat
# from backend.data.formats import nuscenes
# from backend.data.formats import openlabel

__all__ = [
    # Base classes
    "FormatLoader",
    "FormatExporter",
    "FormatRegistry",
    "ProgressCallback",
    # Convenience functions
    "get_loader",
    "get_exporter",
    "detect_format",
    "list_formats",
    "convert",
]
```

### 4. Update data module __init__.py

Update `backend/data/__init__.py` to include formats:

```python
"""Data pipeline module.

Handles dataset operations: format import/export, duplicate detection,
label QA, dataset splitting, and augmentation.
"""

from backend.data.models import (
    BBox,
    BBoxFormat,
    Dataset,
    Label,
    Sample,
    # Pydantic schemas
    BBoxSchema,
    DatasetStatsSchema,
    LabelSchema,
    SampleSchema,
)

from backend.data.formats import (
    FormatLoader,
    FormatExporter,
    FormatRegistry,
    get_loader,
    get_exporter,
    detect_format,
    list_formats,
    convert,
)

__all__ = [
    # Core types
    "BBox",
    "BBoxFormat",
    "Label",
    "Sample",
    "Dataset",
    # API schemas
    "BBoxSchema",
    "LabelSchema",
    "SampleSchema",
    "DatasetStatsSchema",
    # Format handling
    "FormatLoader",
    "FormatExporter",
    "FormatRegistry",
    "get_loader",
    "get_exporter",
    "detect_format",
    "list_formats",
    "convert",
]
```

### 5. Create unit tests

Create `backend/tests/data/test_formats_base.py`:

```python
"""Tests for format base classes and registry."""

from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from backend.data.formats.base import (
    FormatExporter,
    FormatLoader,
    FormatRegistry,
    detect_format,
    get_exporter,
    get_loader,
)
from backend.data.models import Dataset, Sample


class MockLoader(FormatLoader):
    """Test loader implementation."""

    format_name = "mock"
    description = "Mock format for testing"
    extensions = [".mock"]

    def load(self) -> Dataset:
        return Dataset(list(self.iter_samples()), name="mock")

    def iter_samples(self) -> Iterator[Sample]:
        for i in range(3):
            yield Sample(image_path=self.root_path / f"img{i}.jpg")

    def validate(self) -> list[str]:
        return []


class MockExporter(FormatExporter):
    """Test exporter implementation."""

    format_name = "mock"
    description = "Mock format for testing"

    def export(self, dataset: Dataset, *, copy_images: bool = True, image_subdir: str = "images") -> Path:
        self._ensure_output_dir()
        return self.output_path

    def validate_export(self) -> list[str]:
        return []


class TestFormatLoader:
    """Tests for FormatLoader base class."""

    def test_requires_existing_path(self, tmp_path):
        """Test that non-existent path raises error."""
        with pytest.raises(FileNotFoundError):
            MockLoader(tmp_path / "nonexistent")

    def test_load_returns_dataset(self, tmp_path):
        """Test that load returns a Dataset."""
        tmp_path.mkdir(exist_ok=True)
        loader = MockLoader(tmp_path)
        ds = loader.load()
        assert isinstance(ds, Dataset)
        assert len(ds) == 3

    def test_iter_samples_yields(self, tmp_path):
        """Test that iter_samples yields Sample objects."""
        tmp_path.mkdir(exist_ok=True)
        loader = MockLoader(tmp_path)
        samples = list(loader.iter_samples())
        assert len(samples) == 3
        assert all(isinstance(s, Sample) for s in samples)

    def test_progress_callback(self, tmp_path):
        """Test that progress callback is called."""
        tmp_path.mkdir(exist_ok=True)
        callback = MagicMock()
        loader = MockLoader(tmp_path, progress_callback=callback)
        loader._report_progress(1, 10, "test")
        callback.assert_called_once_with(1, 10, "test")


class TestFormatExporter:
    """Tests for FormatExporter base class."""

    def test_creates_output_dir(self, tmp_path):
        """Test that export creates output directory."""
        output = tmp_path / "output"
        exporter = MockExporter(output)
        ds = Dataset([])
        exporter.export(ds)
        assert output.exists()

    def test_fails_on_non_empty_without_overwrite(self, tmp_path):
        """Test that non-empty dir raises without overwrite."""
        output = tmp_path / "output"
        output.mkdir()
        (output / "existing.txt").touch()
        exporter = MockExporter(output, overwrite=False)
        with pytest.raises(FileExistsError):
            exporter.export(Dataset([]))

    def test_allows_overwrite(self, tmp_path):
        """Test that overwrite=True allows non-empty dir."""
        output = tmp_path / "output"
        output.mkdir()
        (output / "existing.txt").touch()
        exporter = MockExporter(output, overwrite=True)
        exporter.export(Dataset([]))  # Should not raise


class TestFormatRegistry:
    """Tests for FormatRegistry."""

    def setup_method(self):
        """Register mock format for tests."""
        FormatRegistry.register_loader(MockLoader)
        FormatRegistry.register_exporter(MockExporter)

    def test_register_loader(self):
        """Test loader registration."""
        assert "mock" in FormatRegistry._loaders

    def test_register_exporter(self):
        """Test exporter registration."""
        assert "mock" in FormatRegistry._exporters

    def test_get_loader_explicit_format(self, tmp_path):
        """Test getting loader with explicit format."""
        tmp_path.mkdir(exist_ok=True)
        loader = get_loader(tmp_path, format_name="mock")
        assert isinstance(loader, MockLoader)

    def test_get_exporter(self, tmp_path):
        """Test getting exporter."""
        exporter = get_exporter(tmp_path, "mock")
        assert isinstance(exporter, MockExporter)

    def test_get_loader_unknown_format(self, tmp_path):
        """Test that unknown format raises ValueError."""
        tmp_path.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="Unknown format"):
            get_loader(tmp_path, format_name="unknown_format_xyz")

    def test_list_formats(self):
        """Test listing available formats."""
        formats = FormatRegistry.list_formats()
        assert "mock" in formats
        assert formats["mock"]["loader"] is True
        assert formats["mock"]["exporter"] is True


class TestDetectFormat:
    """Tests for format auto-detection."""

    def test_detect_yolo_by_data_yaml(self, tmp_path):
        """Test YOLO detection by data.yaml."""
        (tmp_path / "data.yaml").touch()
        assert detect_format(tmp_path) == "yolo"

    def test_detect_coco_by_annotations_dir(self, tmp_path):
        """Test COCO detection by annotations directory."""
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        (ann_dir / "instances_train.json").write_text('{"images": [], "annotations": []}')
        # Need valid JSON content
        import json
        (tmp_path / "instances.json").write_text(json.dumps({
            "images": [],
            "annotations": [],
            "categories": []
        }))
        assert detect_format(tmp_path) == "coco"

    def test_detect_voc_by_annotations_xml(self, tmp_path):
        """Test VOC detection by Annotations directory."""
        ann_dir = tmp_path / "Annotations"
        ann_dir.mkdir()
        (ann_dir / "img001.xml").touch()
        assert detect_format(tmp_path) == "voc"

    def test_detect_kitti_by_label_2(self, tmp_path):
        """Test KITTI detection by label_2 directory."""
        (tmp_path / "label_2").mkdir()
        assert detect_format(tmp_path) == "kitti"

    def test_detect_cvat_by_xml(self, tmp_path):
        """Test CVAT detection by XML with annotations root."""
        xml_content = '<?xml version="1.0"?><annotations></annotations>'
        (tmp_path / "annotations.xml").write_text(xml_content)
        assert detect_format(tmp_path) == "cvat"

    def test_returns_none_for_unknown(self, tmp_path):
        """Test that unknown format returns None."""
        (tmp_path / "random_file.txt").touch()
        assert detect_format(tmp_path) is None
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/__init__.py` | Create | Format module exports |
| `backend/data/formats/base.py` | Create | Abstract base classes + registry |
| `backend/data/__init__.py` | Modify | Add format exports |
| `backend/tests/data/test_formats_base.py` | Create | Unit tests |

## Verification

```bash
# Run tests
cd backend
pytest tests/data/test_formats_base.py -v

# Verify imports
python -c "
from backend.data.formats import FormatLoader, FormatExporter, FormatRegistry
from backend.data.formats import get_loader, get_exporter, detect_format, list_formats
print('Imports OK')
print('Available formats:', list_formats())
"

# Test auto-detection
python -c "
from pathlib import Path
from backend.data.formats import detect_format
import tempfile
import os

with tempfile.TemporaryDirectory() as tmp:
    # Create YOLO-like structure
    Path(tmp, 'data.yaml').touch()
    fmt = detect_format(Path(tmp))
    print(f'Detected format: {fmt}')
    assert fmt == 'yolo', 'Should detect YOLO'
"
```

Expected output:
```
Imports OK
Available formats: {'mock': {'loader': True, 'exporter': True, 'description': '...'}}
Detected format: yolo
```

## Notes

- Registry is a singleton - all format modules register on import
- Auto-detection checks in priority order: YOLO → COCO → VOC → KITTI → CVAT → nuScenes → OpenLABEL
- Progress callbacks enable UI updates during long operations
- `convert()` is a high-level convenience function for format conversion
- Loaders should implement `iter_samples()` for memory-efficient iteration
- All format modules will use `@FormatRegistry.register_loader` decorator
