"""YOLO format loader and exporter.

Supports YOLO v5, v8, and v11 formats.
- One .txt label file per image
- data.yaml config with class names
- Normalized bounding boxes (cx, cy, w, h)

Implements specs:
- 06-data-pipeline/03-yolo-loader
- 06-data-pipeline/10-yolo-exporter
"""

from __future__ import annotations

import logging
import random
import shutil
from collections.abc import Iterator
from pathlib import Path

import yaml

from backend.data.formats.base import (
    FormatExporter,
    FormatLoader,
    FormatRegistry,
    ProgressCallback,
)
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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
        class_names: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        splits: list[str] | None = None,
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
        self._config: dict | None = None
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

        with yaml_path.open() as f:
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

    def _find_label_file(self, image_path: Path) -> Path | None:
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

        with label_path.open() as f:
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

        # Count total images for progress
        total = 0
        for split_path in split_paths.values():
            total += sum(1 for ext in _IMAGE_EXTENSIONS for _ in split_path.glob(f"*{ext}"))

        processed = 0
        for split, images_dir in split_paths.items():
            for ext in _IMAGE_EXTENSIONS:
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
        for _split, images_dir in split_paths.items():
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


@FormatRegistry.register_exporter
class YoloExporter(FormatExporter):
    """Export datasets to YOLO format.

    Creates:
    - data.yaml with class names and split paths
    - <split>/images/ with image files (copied or linked)
    - <split>/labels/ with .txt label files
    """

    format_name = "yolo"
    description = "YOLO v5/v8/v11 format"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: ProgressCallback | None = None,
        split_ratios: dict[str, float] | None = None,
        split_seed: int | None = 42,
    ) -> None:
        """Initialize YOLO exporter.

        Args:
            output_path: Output directory
            overwrite: Whether to overwrite existing files
            progress_callback: Progress callback
            split_ratios: Optional split ratios, e.g. {"train": 0.8, "val": 0.2}
            split_seed: Random seed used when split_ratios is provided
        """
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.split_ratios = split_ratios
        self.split_seed = split_seed

    def _resolve_class_names(self, dataset: Dataset) -> list[str]:
        """Resolve class names from dataset metadata or labels."""
        if dataset.class_names and any(name for name in dataset.class_names):
            return list(dataset.class_names)

        class_by_id: dict[int, str] = {}
        for sample in dataset:
            for label in sample.labels:
                if label.class_id not in class_by_id or not class_by_id[label.class_id]:
                    class_by_id[label.class_id] = label.class_name or f"class_{label.class_id}"

        if not class_by_id:
            return []

        max_id = max(class_by_id.keys())
        return [class_by_id.get(class_id, f"class_{class_id}") for class_id in range(max_id + 1)]

    def _normalized_split_ratios(self) -> dict[str, float]:
        """Validate and normalize split ratios."""
        if not self.split_ratios:
            return {}

        normalized_input: dict[str, float] = {}
        for split_name, ratio in self.split_ratios.items():
            key = str(split_name).strip()
            if not key:
                continue
            if ratio > 0:
                normalized_input[key] = ratio

        if not normalized_input:
            raise ValueError("split_ratios must contain at least one positive ratio")

        total_ratio = sum(normalized_input.values())
        return {name: ratio / total_ratio for name, ratio in normalized_input.items()}

    def _split_indices(self, total: int) -> dict[str, list[int]]:
        """Create index splits from ratios."""
        ratios = self._normalized_split_ratios()
        if not ratios:
            return {"train": list(range(total))}

        rng = random.Random(self.split_seed)
        indices = list(range(total))
        rng.shuffle(indices)

        split_names = list(ratios.keys())
        split_map: dict[str, list[int]] = {}
        start = 0

        for split_name in split_names[:-1]:
            count = int(total * ratios[split_name])
            split_map[split_name] = indices[start : start + count]
            start += count

        split_map[split_names[-1]] = indices[start:]
        return split_map

    def _resolve_split_map(self, samples: list[Sample]) -> dict[str, list[int]]:
        """Resolve split assignment either from ratios or sample metadata."""
        if self.split_ratios:
            return self._split_indices(len(samples))

        split_map: dict[str, list[int]] = {}
        for idx, sample in enumerate(samples):
            split_name = str(sample.metadata.get("split", "train")).strip() or "train"
            if split_name not in split_map:
                split_map[split_name] = []
            split_map[split_name].append(idx)

        return split_map or {"train": list(range(len(samples)))}

    def _write_label_file(self, sample: Sample, label_path: Path) -> None:
        """Write labels to YOLO format file."""
        lines: list[str] = []

        for label in sample.labels:
            polygon = label.attributes.get("polygon")
            if isinstance(polygon, (list, tuple)) and len(polygon) >= 6 and len(polygon) % 2 == 0:
                try:
                    points = [float(value) for value in polygon]
                    points_str = " ".join(f"{point:.6f}" for point in points)
                    lines.append(f"{label.class_id} {points_str}")
                    continue
                except (TypeError, ValueError):
                    logger.warning("Invalid polygon in label for %s", sample.image_path)

            cx, cy, w, h = label.bbox.to_cxcywh()
            lines.append(f"{label.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _copy_or_link_image(
        self,
        source_path: Path,
        destination_path: Path,
        *,
        copy_images: bool,
        link_images: bool,
    ) -> None:
        """Copy or link an image to destination."""
        if destination_path.exists():
            return

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if copy_images:
            shutil.copy2(source_path, destination_path)
            return

        if not link_images:
            return

        try:
            destination_path.symlink_to(source_path.resolve())
        except OSError:
            shutil.copy2(source_path, destination_path)

    def _create_data_yaml(self, class_names: list[str], split_paths: dict[str, str]) -> None:
        """Create YOLO data.yaml."""
        yaml_data: dict[str, object] = {
            "path": ".",
            "nc": len(class_names),
            "names": class_names,
        }

        for split_name in ("train", "val", "test"):
            if split_name in split_paths:
                yaml_data[split_name] = split_paths[split_name]

        for split_name in split_paths:
            if split_name not in yaml_data:
                yaml_data[split_name] = split_paths[split_name]

        yaml_path = self.output_path / "data.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    def export(
        self,
        dataset: Dataset,
        *,
        copy_images: bool = True,
        image_subdir: str = "images",
        link_images: bool = False,
    ) -> Path:
        """Export dataset to YOLO format.

        Args:
            dataset: Dataset to export
            copy_images: Whether to copy images into output split folders
            image_subdir: Name of the image subdirectory per split
            link_images: Create symlinks instead of copies when copy_images=False

        Returns:
            Path to exported dataset root
        """
        self._ensure_output_dir()

        class_names = self._resolve_class_names(dataset)
        samples = list(dataset)
        total = len(samples)
        split_map = self._resolve_split_map(samples)

        split_paths: dict[str, str] = {}
        processed = 0

        for split_name, indices in split_map.items():
            images_dir = self.output_path / split_name / image_subdir
            labels_dir = self.output_path / split_name / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            split_paths[split_name] = f"{split_name}/{image_subdir}"

            for idx in indices:
                sample = samples[idx]
                image_name = sample.image_path.name
                destination_image = images_dir / image_name

                # Avoid collisions when multiple samples share file names.
                if destination_image.exists():
                    image_name = f"{sample.image_path.stem}_{idx}{sample.image_path.suffix}"
                    destination_image = images_dir / image_name

                if sample.image_path.exists():
                    self._copy_or_link_image(
                        sample.image_path,
                        destination_image,
                        copy_images=copy_images,
                        link_images=link_images,
                    )
                else:
                    logger.warning("Image not found during YOLO export: %s", sample.image_path)

                label_name = f"{Path(image_name).stem}.txt"
                label_path = labels_dir / label_name
                if label_path.exists():
                    label_path = labels_dir / f"{sample.image_path.stem}_{idx}.txt"
                self._write_label_file(sample, label_path)

                processed += 1
                self._report_progress(processed, total, f"Exporting {split_name}")

        self._create_data_yaml(class_names, split_paths)
        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported YOLO dataset and return warnings."""
        warnings: list[str] = []
        yaml_path = self.output_path / "data.yaml"
        if not yaml_path.exists():
            warnings.append("data.yaml not found")
            return warnings

        try:
            with yaml_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            warnings.append(f"Failed to parse data.yaml: {exc}")
            return warnings

        if not isinstance(data, dict):
            warnings.append("data.yaml must contain a mapping")
            return warnings

        names = data.get("names")
        if not isinstance(names, (list, dict)) or len(names) == 0:
            warnings.append("data.yaml 'names' must be a non-empty list or dict")

        base_path = self.output_path
        path_value = data.get("path")
        if isinstance(path_value, str) and path_value:
            configured_path = Path(path_value)
            base_path = (
                configured_path
                if configured_path.is_absolute()
                else (self.output_path / configured_path)
            )

        reserved_keys = {"path", "names", "nc", "download"}
        split_paths = {
            key: value
            for key, value in data.items()
            if key not in reserved_keys and isinstance(value, str)
        }
        if not split_paths:
            warnings.append("No split paths found in data.yaml")
            return warnings

        for split_name, split_value in split_paths.items():
            split_path = Path(split_value)
            images_dir = split_path if split_path.is_absolute() else (base_path / split_path)

            if not images_dir.exists():
                warnings.append(f"Split directory not found: {split_value}")
                continue
            if not images_dir.is_dir():
                warnings.append(f"Split path is not a directory: {split_value}")
                continue

            labels_dir = images_dir.parent / "labels"
            if not labels_dir.exists():
                warnings.append(f"Missing labels directory for split '{split_name}'")
                continue

            for image_path in sorted(images_dir.iterdir()):
                if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue

                label_path = labels_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    warnings.append(f"Missing label file for image '{split_name}/{image_path.name}'")
                    continue

                for line_num, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue

                    parts = stripped.split()
                    if len(parts) < 5:
                        warnings.append(
                            f"Invalid label line in {label_path.name}:{line_num} (need 5+ values)"
                        )
                        continue

                    try:
                        int(parts[0])
                        coordinates = [float(value) for value in parts[1:]]
                    except ValueError:
                        warnings.append(f"Invalid numeric values in {label_path.name}:{line_num}")
                        continue

                    is_bbox = len(coordinates) == 4
                    is_polygon = len(coordinates) >= 6 and len(coordinates) % 2 == 0
                    if not is_bbox and not is_polygon:
                        warnings.append(
                            f"Invalid coordinate count in {label_path.name}:{line_num}"
                        )

        return warnings
