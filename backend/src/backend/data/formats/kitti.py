"""KITTI format loader.

Supports KITTI 2D/3D detection format used in autonomous driving.
- One .txt label file per image
- 15 fields per object line
- Pixel coordinates (absolute, not normalized)

Implements spec: 06-data-pipeline/05-kitti-loader
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

from backend.data.formats.base import (
    FormatExporter,
    FormatLoader,
    FormatRegistry,
    ProgressCallback,
)
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
        class_names: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        splits: list[str] | None = None,
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
            return KITTI_CLASSES.copy()
        return [c for c in KITTI_CLASSES if c != "DontCare"]

    def _get_split_paths(self) -> dict[str, tuple[Path, Path | None]]:
        """Get image and label directories for each split.

        Returns:
            Dict of split -> (image_dir, label_dir)
        """
        paths: dict[str, tuple[Path, Path | None]] = {}

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
    ) -> Label | None:
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
        except Exception:
            # Fallback for invalid images
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
                labels: list[Label] = []
                if lbl_dir:
                    label_path = lbl_dir / (image_path.stem + ".txt")
                    if label_path.exists():
                        with label_path.open() as f:
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


@FormatRegistry.register_exporter
class KittiExporter(FormatExporter):
    """Export datasets to KITTI format.

    Creates:
    - <split>/image_2/ with images
    - <split>/label_2/ with labels

    Example:
        exporter = KittiExporter(Path("/output/kitti"))
        exporter.export(dataset)
    """

    format_name = "kitti"
    description = "KITTI format (autonomous driving)"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: ProgressCallback | None = None,
        split: str = "training",
    ) -> None:
        """Initialize KITTI exporter.

        Args:
            output_path: Output directory
            overwrite: Whether to overwrite existing files
            progress_callback: Progress callback
            split: Split name (training, testing)
        """
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.split = split

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        """Convert value to float, falling back to default."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: object, default: int) -> int:
        """Convert value to int, falling back to default."""
        if isinstance(value, bool):
            return int(value)

        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _resolve_image_dimensions(self, sample: Sample) -> tuple[int, int]:
        """Resolve image dimensions for export."""
        if sample.image_width and sample.image_height:
            return sample.image_width, sample.image_height

        try:
            from PIL import Image

            with Image.open(sample.image_path) as image:
                width, height = image.size
                return int(width), int(height)
        except Exception:
            logger.warning(
                "Unable to resolve image dimensions for %s. Using KITTI defaults.",
                sample.image_path,
            )
            return 1242, 375

    def _format_label_line(self, label: Label, img_width: int, img_height: int) -> str:
        """Format a label as a 15-field KITTI line."""
        x1, y1, x2, y2 = label.bbox.to_xyxy()
        left = x1 * img_width
        top = y1 * img_height
        right = x2 * img_width
        bottom = y2 * img_height

        attrs = label.attributes if isinstance(label.attributes, dict) else {}

        truncated = self._safe_float(attrs.get("truncated", 0.0), 0.0)
        occluded = self._safe_int(attrs.get("occluded", 0), 0)
        occluded = min(max(occluded, 0), 3)
        alpha = self._safe_float(attrs.get("alpha", -10.0), -10.0)

        dims_3d = attrs.get("dimensions_3d", {})
        if not isinstance(dims_3d, dict):
            dims_3d = {}
        height_3d = self._safe_float(dims_3d.get("height", -1.0), -1.0)
        width_3d = self._safe_float(dims_3d.get("width", -1.0), -1.0)
        length_3d = self._safe_float(dims_3d.get("length", -1.0), -1.0)

        loc_3d = attrs.get("location_3d", {})
        if not isinstance(loc_3d, dict):
            loc_3d = {}
        loc_x = self._safe_float(loc_3d.get("x", -1000.0), -1000.0)
        loc_y = self._safe_float(loc_3d.get("y", -1000.0), -1000.0)
        loc_z = self._safe_float(loc_3d.get("z", -1000.0), -1000.0)

        rotation_y = self._safe_float(attrs.get("rotation_y", -10.0), -10.0)
        class_name = label.class_name or f"class_{label.class_id}"

        return (
            f"{class_name} "
            f"{truncated:.2f} "
            f"{occluded} "
            f"{alpha:.2f} "
            f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f} "
            f"{height_3d:.2f} {width_3d:.2f} {length_3d:.2f} "
            f"{loc_x:.2f} {loc_y:.2f} {loc_z:.2f} "
            f"{rotation_y:.2f}"
        )

    def export(
        self,
        dataset: Dataset,
        *,
        copy_images: bool = True,
        image_subdir: str = "image_2",
    ) -> Path:
        """Export dataset to KITTI format."""
        self._ensure_output_dir()

        images_dir = self.output_path / self.split / image_subdir
        labels_dir = self.output_path / self.split / "label_2"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        samples = list(dataset)
        total = len(samples)

        for idx, sample in enumerate(samples):
            file_num = f"{idx:06d}"
            img_width, img_height = self._resolve_image_dimensions(sample)

            if copy_images:
                if sample.image_path.exists():
                    ext = sample.image_path.suffix or ".png"
                    destination = images_dir / f"{file_num}{ext}"
                    if not destination.exists():
                        shutil.copy2(sample.image_path, destination)
                else:
                    logger.warning(
                        "Image not found during KITTI export: %s",
                        sample.image_path,
                    )

            lines = [self._format_label_line(label, img_width, img_height) for label in sample.labels]
            label_path = labels_dir / f"{file_num}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            self._report_progress(idx + 1, total, f"Exporting {self.split}")

        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported KITTI dataset."""
        warnings: list[str] = []

        images_dir = self.output_path / self.split / "image_2"
        labels_dir = self.output_path / self.split / "label_2"

        if not images_dir.exists():
            warnings.append(f"No image_2 directory in {self.split}")
        if not labels_dir.exists():
            warnings.append(f"No label_2 directory in {self.split}")
            return warnings

        image_files = (
            list(images_dir.glob("*.png"))
            + list(images_dir.glob("*.jpg"))
            + list(images_dir.glob("*.jpeg"))
            + list(images_dir.glob("*.bmp"))
            + list(images_dir.glob("*.webp"))
        )
        label_files = list(labels_dir.glob("*.txt"))

        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}

        if len(image_stems) != len(label_stems):
            warnings.append(
                f"Image/label count mismatch in {self.split}: "
                f"{len(image_stems)} vs {len(label_stems)}"
            )

        missing_labels = image_stems - label_stems
        if missing_labels:
            warnings.append(f"{len(missing_labels)} images without labels in {self.split}")

        missing_images = label_stems - image_stems
        if missing_images:
            warnings.append(f"{len(missing_images)} labels without images in {self.split}")

        return warnings
