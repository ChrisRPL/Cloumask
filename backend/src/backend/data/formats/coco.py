"""COCO format loader and exporter.

Supports COCO detection and segmentation format.
- Single JSON annotation file
- Absolute pixel bounding boxes (x, y, w, h)
- Polygon or RLE segmentation masks

Implements spec: 06-data-pipeline/04-coco-loader
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from backend.data.formats.base import (
    FormatExporter,
    FormatLoader,
    FormatRegistry,
    ProgressCallback,
)
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)


def decode_rle(rle: dict, height: int, width: int) -> np.ndarray:
    """Decode COCO RLE segmentation to binary mask.

    Args:
        rle: RLE dict with 'counts' and 'size'
        height: Image height
        width: Image width

    Returns:
        Binary mask array (H, W)
    """
    counts = rle.get("counts", [])
    if isinstance(counts, str):
        # Compressed RLE - use pycocotools if available
        try:
            from pycocotools import mask as mask_utils

            return mask_utils.decode(rle).astype(np.uint8)
        except ImportError:
            logger.warning("pycocotools not installed, cannot decode compressed RLE")
            return np.zeros((height, width), dtype=np.uint8)

    # Uncompressed RLE
    mask = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    val = 0
    for count in counts:
        mask[pos : pos + count] = val
        pos += count
        val = 1 - val
    return mask.reshape((height, width), order="F")


def polygon_to_mask(polygon: list[list[float]], height: int, width: int) -> np.ndarray:
    """Convert COCO polygon to binary mask.

    Args:
        polygon: List of polygon coordinates [[x1,y1,x2,y2,...], ...]
        height: Image height
        width: Image width

    Returns:
        Binary mask array (H, W)
    """
    try:
        import cv2

        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygon:
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask
    except ImportError:
        logger.warning("cv2 not installed, cannot convert polygon to mask")
        return np.zeros((height, width), dtype=np.uint8)


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """Convert binary mask to COCO polygon contours.

    Args:
        mask: Binary mask array (H, W)

    Returns:
        COCO polygon list [[x1, y1, x2, y2, ...], ...]
    """
    if mask.ndim != 2:
        return []

    try:
        import cv2
    except ImportError:
        logger.warning("cv2 not installed, cannot convert mask to polygon")
        return []

    binary_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons: list[list[float]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue

        flattened = contour.reshape(-1, 2).astype(float).flatten().tolist()
        if len(flattened) >= 6:
            polygons.append(flattened)

    return polygons


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as uncompressed COCO RLE."""
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D for RLE export")

    binary_mask = (mask > 0).astype(np.uint8)
    pixels = binary_mask.reshape(-1, order="F")

    counts: list[int] = []
    prev = 0
    run_length = 0

    for pixel in pixels:
        pixel_value = int(pixel)
        if pixel_value != prev:
            counts.append(run_length)
            run_length = 1
            prev = pixel_value
        else:
            run_length += 1

    counts.append(run_length)

    return {
        "counts": counts,
        "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
    }


@FormatRegistry.register_loader
class CocoLoader(FormatLoader):
    """Load COCO format datasets.

    Expects:
    - annotations/ directory with JSON files
    - Image directories (train/, val/, etc.)

    Example:
        loader = CocoLoader(Path("/data/coco"))
        dataset = loader.load()
    """

    format_name = "coco"
    description = "COCO JSON format (detection/segmentation)"
    extensions = [".json"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        annotation_file: str | None = None,
        load_masks: bool = True,
    ) -> None:
        """Initialize COCO loader.

        Args:
            root_path: Dataset root directory
            class_names: Override class names from JSON
            progress_callback: Progress callback
            annotation_file: Specific annotation file to load
            load_masks: Whether to load segmentation masks
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.annotation_file = annotation_file
        self.load_masks = load_masks
        self._coco_data: dict | None = None
        self._categories: dict[int, dict] = {}
        self._image_info: dict[int, dict] = {}
        self._annotations: dict[int, list[dict]] = defaultdict(list)

    def _find_annotation_file(self) -> Path | None:
        """Find the annotation JSON file."""
        if self.annotation_file:
            path = self.root_path / self.annotation_file
            if path.exists():
                return path
            path = self.root_path / "annotations" / self.annotation_file
            if path.exists():
                return path
            return None

        # Look for common patterns
        ann_dir = self.root_path / "annotations"
        if ann_dir.exists():
            # Prefer instances_* files
            for pattern in ["instances_*.json", "*.json"]:
                files = list(ann_dir.glob(pattern))
                if files:
                    return sorted(files)[0]

        # Check root for JSON files
        for json_file in self.root_path.glob("*.json"):
            try:
                with json_file.open() as f:
                    data = json.load(f)
                if "images" in data and "annotations" in data:
                    return json_file
            except (json.JSONDecodeError, KeyError):
                continue

        return None

    def _load_json(self) -> None:
        """Load and parse the COCO JSON file."""
        if self._coco_data is not None:
            return

        ann_file = self._find_annotation_file()
        if ann_file is None:
            raise FileNotFoundError(f"No COCO annotation file found in {self.root_path}")

        logger.info(f"Loading COCO annotations from {ann_file}")
        with ann_file.open() as f:
            self._coco_data = json.load(f)

        # Index categories
        for cat in self._coco_data.get("categories", []):
            self._categories[cat["id"]] = cat

        # Index images
        for img in self._coco_data.get("images", []):
            self._image_info[img["id"]] = img

        # Group annotations by image
        for ann in self._coco_data.get("annotations", []):
            self._annotations[ann["image_id"]].append(ann)

    def _infer_class_names(self) -> list[str]:
        """Get class names from COCO categories."""
        self._load_json()
        # Sort by category ID to get ordered list
        sorted_cats = sorted(self._categories.values(), key=lambda x: x["id"])
        return [cat["name"] for cat in sorted_cats]

    def _find_image(self, image_info: dict) -> Path | None:
        """Find the actual image file.

        Args:
            image_info: COCO image dict with file_name

        Returns:
            Path to image or None
        """
        file_name = image_info["file_name"]

        # Try various locations
        candidates = [
            self.root_path / file_name,
            self.root_path / "images" / file_name,
            self.root_path / "train" / file_name,
            self.root_path / "train2017" / file_name,
            self.root_path / "val" / file_name,
            self.root_path / "val2017" / file_name,
        ]

        # Also try with path components from file_name
        if "/" in file_name:
            candidates.append(self.root_path / Path(file_name))

        for path in candidates:
            if path.exists():
                return path

        return None

    def _parse_annotation(
        self,
        ann: dict,
        img_width: int,
        img_height: int,
    ) -> Label | None:
        """Parse a single COCO annotation.

        Args:
            ann: COCO annotation dict
            img_width: Image width for normalization
            img_height: Image height for normalization

        Returns:
            Label object or None if invalid
        """
        cat_id = ann.get("category_id")
        if cat_id not in self._categories:
            logger.warning(f"Unknown category_id: {cat_id}")
            return None

        category = self._categories[cat_id]
        class_name = category["name"]

        # Get class_id as 0-based index
        sorted_cat_ids = sorted(self._categories.keys())
        class_id = sorted_cat_ids.index(cat_id)

        # Parse bbox (COCO: x, y, width, height in pixels)
        bbox_data = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox_data) != 4:
            return None

        x, y, w, h = bbox_data

        # Normalize bbox coordinates
        bbox = BBox.from_xywh(
            x / img_width,
            y / img_height,
            w / img_width,
            h / img_height,
        )

        # Parse segmentation mask
        mask = None
        if self.load_masks and "segmentation" in ann:
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                # RLE format
                mask = decode_rle(seg, img_height, img_width)
            elif isinstance(seg, list) and len(seg) > 0:
                # Polygon format
                mask = polygon_to_mask(seg, img_height, img_width)

        # Additional attributes
        attributes: dict[str, Any] = {}
        if "iscrowd" in ann:
            attributes["iscrowd"] = bool(ann["iscrowd"])
        if "area" in ann:
            attributes["area"] = ann["area"]

        return Label(
            class_name=class_name,
            class_id=class_id,
            bbox=bbox,
            mask=mask,
            attributes=attributes,
        )

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all samples in the dataset.

        Yields:
            Sample objects with labels
        """
        self._load_json()

        total = len(self._image_info)
        for idx, (image_id, img_info) in enumerate(self._image_info.items()):
            # Find image file
            image_path = self._find_image(img_info)
            if image_path is None:
                logger.warning(f"Image not found: {img_info['file_name']}")
                continue

            # Parse annotations for this image
            img_width = img_info.get("width", 0)
            img_height = img_info.get("height", 0)

            labels = []
            for ann in self._annotations.get(image_id, []):
                label = self._parse_annotation(ann, img_width, img_height)
                if label:
                    labels.append(label)

            yield Sample(
                image_path=image_path,
                labels=labels,
                image_width=img_width,
                image_height=img_height,
                metadata={
                    "coco_id": image_id,
                    "coco_info": {k: v for k, v in img_info.items() if k != "file_name"},
                },
            )

            self._report_progress(idx + 1, total, "Loading COCO")

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

        # Check for annotation file
        ann_file = self._find_annotation_file()
        if ann_file is None:
            warnings.append("No COCO annotation file found")
            return warnings

        # Load and validate JSON structure
        try:
            self._load_json()
        except json.JSONDecodeError as e:
            warnings.append(f"Invalid JSON: {e}")
            return warnings

        # Check required fields
        if not self._coco_data.get("images"):
            warnings.append("No images in annotation file")
        if not self._coco_data.get("annotations"):
            warnings.append("No annotations in annotation file")
        if not self._coco_data.get("categories"):
            warnings.append("No categories in annotation file")

        # Check for missing images
        missing = 0
        for img_info in self._image_info.values():
            if self._find_image(img_info) is None:
                missing += 1
        if missing > 0:
            warnings.append(f"{missing} images not found on disk")

        # Check for orphan annotations
        image_ids = set(self._image_info.keys())
        orphan_anns = sum(
            1
            for ann in self._coco_data.get("annotations", [])
            if ann["image_id"] not in image_ids
        )
        if orphan_anns > 0:
            warnings.append(f"{orphan_anns} annotations reference missing images")

        return warnings

    def summary(self) -> dict:
        """Get summary information."""
        self._load_json()
        base = super().summary()
        ann_file = self._find_annotation_file()
        base["annotation_file"] = str(ann_file) if ann_file else None
        base["num_images"] = len(self._image_info)
        base["num_annotations"] = sum(len(anns) for anns in self._annotations.values())
        base["num_categories"] = len(self._categories)
        return base


@FormatRegistry.register_exporter
class CocoExporter(FormatExporter):
    """Export datasets to COCO JSON format.

    Creates:
    - annotations/instances_<split>.json
    - images/ directory with exported images
    """

    format_name = "coco"
    description = "COCO JSON format"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: ProgressCallback | None = None,
        split: str = "train",
        export_masks: bool = True,
        mask_encoding: str = "polygon",
    ) -> None:
        """Initialize COCO exporter.

        Args:
            output_path: Output directory
            overwrite: Whether to overwrite existing files
            progress_callback: Progress callback
            split: Split name for annotations file (instances_<split>.json)
            export_masks: Whether to export segmentation
            mask_encoding: Segmentation encoding for label masks ("polygon" or "rle")
        """
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.split = split
        self.export_masks = export_masks
        self.mask_encoding = mask_encoding

        if self.mask_encoding not in {"polygon", "rle"}:
            raise ValueError("mask_encoding must be one of: {'polygon', 'rle'}")

    def _resolve_class_names(self, dataset: Dataset, samples: list[Sample]) -> list[str]:
        """Resolve class names from dataset metadata and label content."""
        names = list(dataset.class_names) if dataset.class_names else []

        max_class_id = -1
        for sample in samples:
            for label in sample.labels:
                if label.class_id > max_class_id:
                    max_class_id = label.class_id

                if label.class_id >= len(names):
                    continue

                if not names[label.class_id] and label.class_name:
                    names[label.class_id] = label.class_name

        if max_class_id >= len(names):
            names.extend(
                f"class_{class_id}" for class_id in range(len(names), max_class_id + 1)
            )

        for sample in samples:
            for label in sample.labels:
                if (
                    0 <= label.class_id < len(names)
                    and label.class_name
                    and names[label.class_id].startswith("class_")
                ):
                    names[label.class_id] = label.class_name

        return names

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
                "Unable to resolve image dimensions for %s. Falling back to 1x1.",
                sample.image_path,
            )
            return 1, 1

    def _convert_attribute_polygon(
        self,
        polygon_data: Any,
        img_width: int,
        img_height: int,
    ) -> list[list[float]]:
        """Convert polygon attribute into COCO polygon format."""
        if not isinstance(polygon_data, (list, tuple)):
            return []

        if polygon_data and isinstance(polygon_data[0], (list, tuple)):
            polygon_groups = polygon_data
        else:
            polygon_groups = [polygon_data]

        polygons: list[list[float]] = []
        for polygon in polygon_groups:
            if not isinstance(polygon, (list, tuple)):
                continue

            try:
                values = [float(value) for value in polygon]
            except (TypeError, ValueError):
                continue

            if len(values) < 6 or len(values) % 2 != 0:
                continue

            is_normalized = all(0.0 <= value <= 1.0 for value in values)
            if is_normalized:
                pixel_values = [
                    value * img_width if idx % 2 == 0 else value * img_height
                    for idx, value in enumerate(values)
                ]
            else:
                pixel_values = values

            polygons.append(pixel_values)

        return polygons

    def _annotation_segmentation(
        self,
        label: Label,
        img_width: int,
        img_height: int,
    ) -> list[list[float]] | dict[str, Any] | list:
        """Build segmentation payload for a COCO annotation."""
        if not self.export_masks:
            return []

        if label.mask is not None:
            if self.mask_encoding == "rle":
                return mask_to_rle(label.mask)

            polygons = mask_to_polygon(label.mask)
            if polygons:
                return polygons
            return []

        polygon = self._convert_attribute_polygon(
            label.attributes.get("polygon"),
            img_width=img_width,
            img_height=img_height,
        )
        if polygon:
            return polygon

        return []

    def _exported_image_name(
        self,
        sample: Sample,
        img_id: int,
        images_dir: Path,
        *,
        copy_images: bool,
        image_subdir: str,
    ) -> str:
        """Resolve exported image name and copy image when requested."""
        source = sample.image_path
        image_name = source.name
        destination = images_dir / image_name

        if destination.exists():
            image_name = f"{source.stem}_{img_id}{source.suffix}"
            destination = images_dir / image_name

        if copy_images:
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            else:
                logger.warning("Image not found during COCO export: %s", source)

            return str(Path(image_subdir) / image_name).replace("\\", "/")

        return source.as_posix()

    def export(
        self,
        dataset: Dataset,
        *,
        copy_images: bool = True,
        image_subdir: str = "images",
    ) -> Path:
        """Export dataset to COCO format.

        Args:
            dataset: Dataset to export
            copy_images: Whether to copy images
            image_subdir: Subdirectory for exported images

        Returns:
            Path to exported dataset root
        """
        self._ensure_output_dir()

        images_dir = self.output_path / image_subdir
        annotations_dir = self.output_path / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        samples = list(dataset)
        class_names = self._resolve_class_names(dataset, samples)

        coco_data: dict[str, Any] = {
            "info": {
                "description": f"Exported from {dataset.name}",
                "version": "1.0",
                "year": datetime.now(UTC).year,
                "date_created": datetime.now(UTC).isoformat(),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        for idx, class_name in enumerate(class_names, start=1):
            coco_data["categories"].append(
                {
                    "id": idx,
                    "name": class_name,
                    "supercategory": "object",
                }
            )

        ann_id = 1
        total = len(samples)

        for img_id, sample in enumerate(samples, start=1):
            img_width, img_height = self._resolve_image_dimensions(sample)
            file_name = self._exported_image_name(
                sample,
                img_id,
                images_dir,
                copy_images=copy_images,
                image_subdir=image_subdir,
            )

            coco_data["images"].append(
                {
                    "id": img_id,
                    "file_name": file_name,
                    "width": img_width,
                    "height": img_height,
                }
            )

            for label in sample.labels:
                if label.class_id < 0:
                    logger.warning(
                        "Skipping label with invalid class_id=%s for image %s",
                        label.class_id,
                        sample.image_path,
                    )
                    continue

                x, y, w, h = label.bbox.to_xywh()
                x_px = x * img_width
                y_px = y * img_height
                w_px = w * img_width
                h_px = h * img_height

                category_id = label.class_id + 1
                if category_id > len(class_names):
                    logger.warning(
                        "Skipping label with class_id=%s outside category range for image %s",
                        label.class_id,
                        sample.image_path,
                    )
                    continue

                annotation: dict[str, Any] = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x_px, y_px, w_px, h_px],
                    "area": float(w_px * h_px),
                    "iscrowd": int(bool(label.attributes.get("iscrowd", False))),
                    "segmentation": self._annotation_segmentation(
                        label,
                        img_width=img_width,
                        img_height=img_height,
                    ),
                }
                coco_data["annotations"].append(annotation)
                ann_id += 1

            self._report_progress(img_id, total, "Exporting COCO")

        ann_file = annotations_dir / f"instances_{self.split}.json"
        with ann_file.open("w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported COCO dataset and return warnings."""
        warnings: list[str] = []

        ann_file = self.output_path / "annotations" / f"instances_{self.split}.json"
        if not ann_file.exists():
            warnings.append(f"Annotation file not found: {ann_file.name}")
            return warnings

        try:
            with ann_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            return [f"Invalid COCO JSON: {exc}"]

        for key in ("images", "annotations", "categories"):
            if key not in data:
                warnings.append(f"Missing top-level key: {key}")
            elif not isinstance(data[key], list):
                warnings.append(f"Top-level key '{key}' must be a list")

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        if not images:
            warnings.append("No images in annotation file")
        if not categories:
            warnings.append("No categories in annotation file")

        image_ids: set[int] = set()
        for image in images:
            image_id = image.get("id")
            if image_id in image_ids:
                warnings.append(f"Duplicate image id: {image_id}")
                break
            image_ids.add(image_id)

            file_name = image.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                warnings.append("Image entry missing file_name")
                continue

            image_path = self.output_path / Path(file_name)
            if not image_path.exists():
                warnings.append(f"Missing image: {file_name}")
                break

        category_ids: set[int] = set()
        for category in categories:
            category_id = category.get("id")
            if category_id in category_ids:
                warnings.append(f"Duplicate category id: {category_id}")
                break
            category_ids.add(category_id)

        annotation_ids: set[int] = set()
        for annotation in annotations:
            annotation_id = annotation.get("id")
            if annotation_id in annotation_ids:
                warnings.append(f"Duplicate annotation id: {annotation_id}")
                break
            annotation_ids.add(annotation_id)

            if annotation.get("image_id") not in image_ids:
                warnings.append(f"Annotation {annotation_id} references unknown image_id")
                break

            if annotation.get("category_id") not in category_ids:
                warnings.append(f"Annotation {annotation_id} references unknown category_id")
                break

            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                warnings.append(f"Annotation {annotation_id} has invalid bbox")
                break

        return warnings
