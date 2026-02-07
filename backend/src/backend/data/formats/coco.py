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
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from backend.data.formats.base import FormatLoader, FormatRegistry, ProgressCallback
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
        class_names: Optional[list[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        annotation_file: Optional[str] = None,
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
        self._coco_data: Optional[dict] = None
        self._categories: dict[int, dict] = {}
        self._image_info: dict[int, dict] = {}
        self._annotations: dict[int, list[dict]] = defaultdict(list)

    def _find_annotation_file(self) -> Optional[Path]:
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
                with open(json_file) as f:
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
        with open(ann_file) as f:
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

    def _find_image(self, image_info: dict) -> Optional[Path]:
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
    ) -> Optional[Label]:
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
