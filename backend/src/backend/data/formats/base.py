"""Abstract base classes for format loaders and exporters.

All format implementations inherit from these base classes.
Provides common interface for loading/exporting datasets.

Implements spec: 06-data-pipeline/02-format-base
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

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
        class_names: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
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
    def load(self) -> Dataset:
        """Load the dataset and return as Dataset object.

        Returns:
            Dataset containing all samples and metadata
        """
        pass

    @abstractmethod
    def iter_samples(self) -> Iterator[Sample]:
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
        progress_callback: ProgressCallback | None = None,
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
        if (
            self.output_path.exists()
            and not self.overwrite
            and any(self.output_path.iterdir())
        ):
            raise FileExistsError(
                f"Output directory not empty: {self.output_path}. "
                "Use overwrite=True to overwrite."
            )
        self.output_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def export(
        self,
        dataset: Dataset,
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

    _instance: FormatRegistry | None = None
    _loaders: dict[str, type[FormatLoader]] = {}
    _exporters: dict[str, type[FormatExporter]] = {}

    def __new__(cls) -> FormatRegistry:
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
        format_name: str | None = None,
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
    def detect_format(cls, path: Path) -> str | None:
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

                with yaml_file.open() as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and ("train" in data or "val" in data):
                    return "yolo"
            except Exception:
                pass

        # COCO: has annotations/*.json or instances_*.json
        if (path / "annotations").is_dir() and any((path / "annotations").glob("*.json")):
            return "coco"
        for json_file in path.glob("*.json"):
            try:
                import json

                with json_file.open() as f:
                    data = json.load(f)
                if isinstance(data, dict) and "images" in data and "annotations" in data:
                    return "coco"
            except Exception:
                pass

        # Pascal VOC: has Annotations/*.xml
        if (path / "Annotations").is_dir() and any((path / "Annotations").glob("*.xml")):
            return "voc"

        # KITTI: has label_2/ directory (root or split subdirectory)
        if (path / "label_2").is_dir():
            return "kitti"
        if any(
            (path / split / "label_2").is_dir()
            for split in ("training", "testing", "train", "val", "test")
        ):
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
            if (
                subdir.is_dir()
                and subdir.name.startswith("v1.0")
                and (subdir / "sample_annotation.json").exists()
            ):
                return "nuscenes"

        # OpenLABEL: JSON with openlabel schema
        for json_file in path.glob("*.json"):
            try:
                import json

                with json_file.open() as f:
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


def get_loader(path: Path, format_name: str | None = None, **kwargs) -> FormatLoader:
    """Get a format loader. See FormatRegistry.get_loader."""
    return FormatRegistry.get_loader(path, format_name=format_name, **kwargs)


def get_exporter(path: Path, format_name: str, **kwargs) -> FormatExporter:
    """Get a format exporter. See FormatRegistry.get_exporter."""
    return FormatRegistry.get_exporter(path, format_name, **kwargs)


def detect_format(path: Path) -> str | None:
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
    source_format: str | None = None,
    copy_images: bool = True,
    augment: bool = False,
    augmentation_preset: str = "medium",
    augmentation_copies: int = 1,
    include_original: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Convert dataset between formats.

    Args:
        input_path: Source dataset directory
        output_path: Target output directory
        target_format: Target format name
        source_format: Source format (auto-detect if None)
        copy_images: Whether to copy images
        augment: Whether to augment samples before export
        augmentation_preset: Augmentation preset name
        augmentation_copies: Number of augmented copies per source sample
        include_original: Keep original samples when augmentation is enabled
        progress_callback: Optional progress callback

    Returns:
        Path to converted dataset
    """
    # Load source
    loader = get_loader(input_path, format_name=source_format, progress_callback=progress_callback)
    dataset = loader.load()

    if augment:
        if not copy_images:
            raise ValueError("Augmentation during export requires copy_images=True")

        from tempfile import TemporaryDirectory

        from backend.data.augmentation import augment_dataset

        with TemporaryDirectory(prefix="cloumask-augment-") as temp_dir:
            dataset = augment_dataset(
                dataset,
                output_dir=Path(temp_dir),
                preset=augmentation_preset,
                copies_per_sample=augmentation_copies,
                include_original=include_original,
            )
            exporter = get_exporter(output_path, target_format, progress_callback=progress_callback)
            return exporter.export(dataset, copy_images=copy_images)

    # Export to target
    exporter = get_exporter(output_path, target_format, progress_callback=progress_callback)
    return exporter.export(dataset, copy_images=copy_images)
