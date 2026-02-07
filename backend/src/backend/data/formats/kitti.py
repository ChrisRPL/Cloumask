"""KITTI format loader.

Supports KITTI 2D/3D detection format used in autonomous driving.
- One .txt label file per image
- 15 fields per object line
- Pixel coordinates (absolute, not normalized)

Implements spec: 06-data-pipeline/05-kitti-loader
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

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
