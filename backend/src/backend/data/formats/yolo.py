"""YOLO format loader and exporter.

Supports YOLO v5, v8, and v11 formats.
- One .txt label file per image
- data.yaml config with class names
- Normalized bounding boxes (cx, cy, w, h)

Implements spec: 06-data-pipeline/03-yolo-loader
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
        if self._config and "names" in self._config:
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
            label_path = Path(
                str(image_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
            )
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
                    labels.append(
                        Label(
                            class_name=class_name,
                            class_id=class_id,
                            bbox=bbox,
                        )
                    )
                elif len(values) >= 6 and len(values) % 2 == 0:
                    # Segmentation polygon: x1 y1 x2 y2 ...
                    # Compute bounding box from polygon
                    xs = values[0::2]
                    ys = values[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox = BBox.from_xyxy(x_min, y_min, x_max, y_max)

                    # Store polygon in attributes
                    labels.append(
                        Label(
                            class_name=class_name,
                            class_id=class_id,
                            bbox=bbox,
                            attributes={"polygon": values},
                        )
                    )
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
                    1 for ext in [".jpg", ".jpeg", ".png"] for _ in path.glob(f"*{ext}")
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
