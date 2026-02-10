"""Pascal VOC format loader.

Supports VOC 2007/2012 detection format with optional segmentation metadata.
- One XML annotation file per image
- Bounding boxes in absolute pixel coordinates
- Optional segmentation index masks (SegmentationClass/SegmentationObject)

Implements spec: 06-data-pipeline/06-voc-loader
"""

from __future__ import annotations

import logging
import shutil
import xml.etree.ElementTree as ET
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


# Standard VOC detection classes (VOC 2007/2012)
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_SEGMENTATION_IGNORE_VALUES = {0, 255}


def _parse_bool(value: str | None) -> bool:
    """Parse VOC bool-like values (0/1, true/false)."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def _to_voc_bool(value: object) -> str:
    """Convert a mixed-type attribute value to VOC 0/1 text."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return "1" if value != 0 else "0"
    if isinstance(value, str):
        return "1" if _parse_bool(value) else "0"
    return "0"


@FormatRegistry.register_loader
class VocLoader(FormatLoader):
    """Load Pascal VOC format datasets.

    Expects:
    - Annotations/ directory with XML files
    - JPEGImages/ directory with images
    - Optional SegmentationClass/ and SegmentationObject/ masks
    """

    format_name = "voc"
    description = "Pascal VOC XML format"
    extensions = [".xml"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        split: str | None = None,
        include_difficult: bool = True,
        load_segmentation_indices: bool = True,
        annotations_dir: str = "Annotations",
        images_dir: str = "JPEGImages",
        segmentation_class_dir: str = "SegmentationClass",
        segmentation_object_dir: str = "SegmentationObject",
    ) -> None:
        """Initialize VOC loader.

        Args:
            root_path: Dataset root directory
            class_names: Optional class name override
            progress_callback: Optional progress callback
            split: Optional split to load (train/val/trainval/test)
            include_difficult: Whether to include difficult objects
            load_segmentation_indices: Read unique segmentation IDs from mask PNGs
            annotations_dir: XML annotation subdirectory
            images_dir: Image subdirectory
            segmentation_class_dir: Segmentation-class mask subdirectory
            segmentation_object_dir: Segmentation-object mask subdirectory
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.split = split
        self.include_difficult = include_difficult
        self.load_segmentation_indices = load_segmentation_indices
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.segmentation_class_dir = segmentation_class_dir
        self.segmentation_object_dir = segmentation_object_dir

        self._discovered_classes: set[str] = set()
        self._extra_class_names: list[str] = []

    def _infer_class_names(self) -> list[str]:
        """Return VOC defaults with any extra discovered classes appended."""
        base = VOC_CLASSES.copy()
        for name in self._extra_class_names:
            if name not in base:
                base.append(name)
        return base

    def _find_existing_dir(self, preferred: str, alternatives: list[str]) -> Path | None:
        """Find an existing directory using preferred name and common fallbacks."""
        candidates = [preferred, *alternatives]
        for candidate in candidates:
            path = self.root_path / candidate
            if path.is_dir():
                return path
        return None

    def _get_annotation_dir(self) -> Path | None:
        return self._find_existing_dir(
            self.annotations_dir,
            ["annotations", "Annotation", "annotation", "labels"],
        )

    def _get_images_dir(self) -> Path | None:
        return self._find_existing_dir(
            self.images_dir,
            ["images", "Images", "JPEGs", "imgs"],
        )

    def _get_segmentation_class_dir(self) -> Path | None:
        return self._find_existing_dir(
            self.segmentation_class_dir,
            ["segmentation_class", "SegClass"],
        )

    def _get_segmentation_object_dir(self) -> Path | None:
        return self._find_existing_dir(
            self.segmentation_object_dir,
            ["segmentation_object", "SegObject"],
        )

    def _split_file_path(self) -> Path | None:
        """Get split file path if split is configured."""
        if not self.split:
            return None

        candidates = [
            self.root_path / "ImageSets" / "Main" / f"{self.split}.txt",
            self.root_path / "ImageSets" / f"{self.split}.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _get_split_stems(self) -> list[str] | None:
        """Read image stems from split file."""
        split_file = self._split_file_path()
        if split_file is None:
            return None
        if not split_file.exists():
            logger.warning("Split file not found: %s", split_file)
            return None

        stems: list[str] = []
        with split_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                stems.append(Path(parts[0]).stem)
        return stems

    def _resolve_xml_files(self, ann_dir: Path, split_stems: list[str] | None) -> list[Path]:
        """Resolve XML files to parse."""
        if not split_stems:
            return sorted(ann_dir.glob("*.xml"))

        xml_files: list[Path] = []
        for stem in split_stems:
            xml_path = ann_dir / f"{stem}.xml"
            if xml_path.exists():
                xml_files.append(xml_path)
            else:
                logger.warning("Split entry missing XML annotation: %s", stem)
        return xml_files

    def _find_image_path(self, filename: str) -> Path:
        """Find image file from VOC filename value."""
        file_path = Path(filename)
        image_roots = [self._get_images_dir(), self.root_path]

        for root in image_roots:
            if root is None:
                continue

            exact = root / file_path
            if exact.exists():
                return exact

            for ext in _IMAGE_EXTENSIONS:
                candidate = root / f"{file_path.stem}{ext}"
                if candidate.exists():
                    return candidate

        default_root = self._get_images_dir() or self.root_path
        return default_root / file_path

    def _read_segmentation_indices(self, mask_path: Path | None) -> list[int] | None:
        """Read unique segmentation indices from a VOC mask PNG."""
        if mask_path is None or not mask_path.exists() or not self.load_segmentation_indices:
            return None

        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            logger.debug("Pillow/Numpy not available, skipping segmentation index extraction")
            return None

        try:
            with Image.open(mask_path) as img:
                mask = np.asarray(img)
        except Exception as exc:
            logger.warning("Failed to read segmentation mask %s: %s", mask_path, exc)
            return None

        if mask.ndim == 3:
            # Palette-index masks should be 2D, but keep robust behavior.
            mask = mask[:, :, 0]

        unique_values = {int(value) for value in np.unique(mask).tolist()}
        return sorted(value for value in unique_values if value not in _SEGMENTATION_IGNORE_VALUES)

    def _get_class_id(self, class_name: str, base_class_names: list[str]) -> int:
        """Resolve class ID, extending class list with non-standard classes."""
        if class_name in base_class_names:
            return base_class_names.index(class_name)

        if class_name not in self._extra_class_names:
            self._extra_class_names.append(class_name)
            logger.warning("Discovered non-standard VOC class: %s", class_name)
        return len(base_class_names) + self._extra_class_names.index(class_name)

    def _get_image_dimensions(self, width: int, height: int, image_path: Path) -> tuple[int, int]:
        """Use XML size when present, otherwise attempt to read from image file."""
        if width > 0 and height > 0:
            return width, height

        if not image_path.exists():
            return width, height

        try:
            from PIL import Image
        except ImportError:
            return width, height

        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return width, height

    def _parse_object(
        self,
        object_node: ET.Element,
        base_class_names: list[str],
        img_width: int,
        img_height: int,
        xml_path: Path,
    ) -> Label | None:
        """Parse a single VOC object node into a Label."""
        name_node = object_node.find("name")
        class_name = name_node.text.strip() if name_node is not None and name_node.text else ""
        if not class_name:
            return None

        self._discovered_classes.add(class_name)

        is_difficult = _parse_bool(object_node.findtext("difficult"))
        if is_difficult and not self.include_difficult:
            return None

        bndbox = object_node.find("bndbox")
        if bndbox is None:
            logger.warning("%s: object '%s' missing bndbox", xml_path, class_name)
            return None

        try:
            xmin = float(bndbox.findtext("xmin", ""))
            ymin = float(bndbox.findtext("ymin", ""))
            xmax = float(bndbox.findtext("xmax", ""))
            ymax = float(bndbox.findtext("ymax", ""))
        except ValueError as exc:
            logger.warning("%s: invalid bbox for '%s': %s", xml_path, class_name, exc)
            return None

        if img_width <= 0 or img_height <= 0:
            logger.warning("%s: missing image dimensions; skipping bbox for '%s'", xml_path, class_name)
            return None

        # VOC coordinates are 1-based and inclusive.
        x1 = max(0.0, min(1.0, (xmin - 1.0) / img_width))
        y1 = max(0.0, min(1.0, (ymin - 1.0) / img_height))
        x2 = max(0.0, min(1.0, xmax / img_width))
        y2 = max(0.0, min(1.0, ymax / img_height))

        if x2 <= x1 or y2 <= y1:
            logger.warning("%s: invalid bbox coordinates for '%s'", xml_path, class_name)
            return None

        attributes: dict[str, bool | str] = {
            "difficult": is_difficult,
            "truncated": _parse_bool(object_node.findtext("truncated")),
            "occluded": _parse_bool(object_node.findtext("occluded")),
        }
        pose = object_node.findtext("pose")
        if pose:
            attributes["pose"] = pose.strip()

        return Label(
            class_name=class_name,
            class_id=self._get_class_id(class_name, base_class_names),
            bbox=BBox.from_xyxy(x1, y1, x2, y2),
            attributes=attributes,
        )

    def _parse_annotation(self, xml_path: Path, base_class_names: list[str]) -> Sample:
        """Parse a VOC XML annotation file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        if root.tag != "annotation":
            raise ValueError(f"Unexpected root tag '{root.tag}'")

        filename = root.findtext("filename")
        if not filename:
            filename = f"{xml_path.stem}.jpg"
            logger.warning("%s: missing filename, defaulting to %s", xml_path, filename)

        image_path = self._find_image_path(filename)

        size_node = root.find("size")
        width = int(size_node.findtext("width", "0")) if size_node is not None else 0
        height = int(size_node.findtext("height", "0")) if size_node is not None else 0
        width, height = self._get_image_dimensions(width, height, image_path)

        labels: list[Label] = []
        for object_node in root.findall("object"):
            label = self._parse_object(object_node, base_class_names, width, height, xml_path)
            if label is not None:
                labels.append(label)

        stem = Path(filename).stem
        seg_class_dir = self._get_segmentation_class_dir()
        seg_object_dir = self._get_segmentation_object_dir()
        seg_class_path = (seg_class_dir / f"{stem}.png") if seg_class_dir else None
        seg_object_path = (seg_object_dir / f"{stem}.png") if seg_object_dir else None

        metadata: dict[str, object] = {"annotation_file": str(xml_path)}
        if self.split:
            metadata["split"] = self.split

        if seg_class_path is not None and seg_class_path.exists():
            metadata["segmentation_class_mask"] = str(seg_class_path)
            class_indices = self._read_segmentation_indices(seg_class_path)
            if class_indices is not None:
                metadata["segmentation_class_indices"] = class_indices

        if seg_object_path is not None and seg_object_path.exists():
            metadata["segmentation_object_mask"] = str(seg_object_path)
            object_indices = self._read_segmentation_indices(seg_object_path)
            if object_indices is not None:
                metadata["segmentation_object_indices"] = object_indices

        return Sample(
            image_path=image_path,
            labels=labels,
            image_width=width if width > 0 else None,
            image_height=height if height > 0 else None,
            metadata=metadata,
        )

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all samples in the dataset."""
        ann_dir = self._get_annotation_dir()
        if ann_dir is None:
            logger.error("No annotations directory found in %s", self.root_path)
            return

        base_class_names = self._class_names or VOC_CLASSES
        split_stems = self._get_split_stems()
        xml_files = self._resolve_xml_files(ann_dir, split_stems)

        total = len(xml_files)
        for idx, xml_path in enumerate(xml_files, start=1):
            try:
                sample = self._parse_annotation(xml_path, base_class_names)
            except (ET.ParseError, ValueError) as exc:
                logger.warning("Skipping invalid XML %s: %s", xml_path, exc)
                continue

            yield sample
            self._report_progress(idx, total, "Loading VOC")

    def load(self) -> Dataset:
        """Load the full dataset."""
        samples = list(self.iter_samples())

        class_names = list(self._class_names) if self._class_names else VOC_CLASSES.copy()

        for name in self._extra_class_names:
            if name not in class_names:
                class_names.append(name)
        for name in sorted(self._discovered_classes):
            if name not in class_names:
                class_names.append(name)

        return Dataset(
            samples,
            name=self.root_path.name,
            class_names=class_names,
        )

    def validate(self) -> list[str]:
        """Validate VOC dataset structure and basic XML content."""
        warnings: list[str] = []

        ann_dir = self._get_annotation_dir()
        img_dir = self._get_images_dir()

        if ann_dir is None:
            warnings.append(f"Annotations directory not found ({self.annotations_dir})")
            return warnings

        xml_files = sorted(ann_dir.glob("*.xml"))
        if not xml_files:
            warnings.append("No XML annotation files found")
            return warnings

        if img_dir is None:
            warnings.append(f"Images directory not found ({self.images_dir})")

        split_file = self._split_file_path()
        if self.split and (split_file is None or not split_file.exists()):
            warnings.append(f"Split file not found: {self.split}.txt")
        elif self.split and split_file is not None:
            split_stems = self._get_split_stems() or []
            missing_xml = [stem for stem in split_stems if not (ann_dir / f"{stem}.xml").exists()]
            if missing_xml:
                warnings.append(
                    f"{len(missing_xml)} split entries reference missing XML annotations"
                )

        missing_images = 0
        invalid_xml = 0
        for xml_path in xml_files:
            try:
                tree = ET.parse(xml_path)
            except ET.ParseError:
                invalid_xml += 1
                continue

            root = tree.getroot()
            if root.tag != "annotation":
                invalid_xml += 1
                continue

            filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
            image_path = self._find_image_path(filename)
            if not image_path.exists():
                missing_images += 1

        if invalid_xml > 0:
            warnings.append(f"{invalid_xml} XML files have invalid structure")
        if missing_images > 0:
            warnings.append(f"{missing_images} annotation files reference missing images")

        return warnings

    def summary(self) -> dict:
        """Get summary information."""
        base = super().summary()
        ann_dir = self._get_annotation_dir()
        base["split"] = self.split
        base["include_difficult"] = self.include_difficult
        base["load_segmentation_indices"] = self.load_segmentation_indices
        base["num_xml_files"] = len(list(ann_dir.glob("*.xml"))) if ann_dir else 0
        return base


@FormatRegistry.register_exporter
class VocExporter(FormatExporter):
    """Export datasets to Pascal VOC format.

    Creates:
    - Annotations/ with XML files
    - JPEGImages/ with copied images
    - ImageSets/Main/ with split files
    """

    format_name = "voc"
    description = "Pascal VOC XML format"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: ProgressCallback | None = None,
        split: str = "trainval",
    ) -> None:
        """Initialize VOC exporter."""
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.split = split
        self._image_subdir = "JPEGImages"
        self._copied_images = True

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
                "Unable to resolve image dimensions for %s. Using VOC defaults 1920x1080.",
                sample.image_path,
            )
            return 1920, 1080

    def _bbox_to_voc_pixels(
        self,
        bbox: BBox,
        img_width: int,
        img_height: int,
    ) -> tuple[int, int, int, int]:
        """Convert normalized bbox to VOC 1-based inclusive pixel coordinates."""
        x1, y1, x2, y2 = bbox.to_xyxy()
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        xmin = int(round((x1 * img_width) + 1.0))
        ymin = int(round((y1 * img_height) + 1.0))
        xmax = int(round(x2 * img_width))
        ymax = int(round(y2 * img_height))

        xmin = min(max(1, xmin), img_width)
        ymin = min(max(1, ymin), img_height)
        xmax = min(max(1, xmax), img_width)
        ymax = min(max(1, ymax), img_height)

        if xmax < xmin:
            xmax = xmin
        if ymax < ymin:
            ymax = ymin

        return xmin, ymin, xmax, ymax

    def _create_annotation_xml(
        self,
        sample: Sample,
        filename: str,
        *,
        folder: str,
        img_width: int,
        img_height: int,
    ) -> str:
        """Create a VOC XML annotation payload for one sample."""
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = folder
        ET.SubElement(annotation, "filename").text = filename

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(img_width)
        ET.SubElement(size, "height").text = str(img_height)
        ET.SubElement(size, "depth").text = str(sample.metadata.get("depth", 3))

        for label in sample.labels:
            attrs = label.attributes if isinstance(label.attributes, dict) else {}
            xmin, ymin, xmax, ymax = self._bbox_to_voc_pixels(
                label.bbox,
                img_width=img_width,
                img_height=img_height,
            )

            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = label.class_name or f"class_{label.class_id}"

            pose = attrs.get("pose")
            pose_text = str(pose).strip() if pose is not None else ""
            ET.SubElement(obj, "pose").text = pose_text or "Unspecified"

            ET.SubElement(obj, "truncated").text = _to_voc_bool(attrs.get("truncated"))
            ET.SubElement(obj, "difficult").text = _to_voc_bool(attrs.get("difficult"))
            ET.SubElement(obj, "occluded").text = _to_voc_bool(attrs.get("occluded"))

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

        ET.indent(annotation, space="  ")
        xml_bytes = ET.tostring(annotation, encoding="utf-8", xml_declaration=True)
        return xml_bytes.decode("utf-8")

    def export(
        self,
        dataset: Dataset,
        *,
        copy_images: bool = True,
        image_subdir: str = "JPEGImages",
    ) -> Path:
        """Export dataset to VOC format."""
        self._ensure_output_dir()
        self._image_subdir = image_subdir
        self._copied_images = copy_images

        ann_dir = self.output_path / "Annotations"
        images_dir = self.output_path / image_subdir
        sets_dir = self.output_path / "ImageSets" / "Main"
        ann_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        sets_dir.mkdir(parents=True, exist_ok=True)

        samples = list(dataset)
        total = len(samples)
        split_members: dict[str, list[str]] = {}
        used_stems: set[str] = set()
        used_image_names: set[str] = set()

        for idx, sample in enumerate(samples):
            stem = sample.image_path.stem
            xml_stem = stem
            if xml_stem in used_stems:
                xml_stem = f"{stem}_{idx}"
            used_stems.add(xml_stem)

            image_name = sample.image_path.name
            if image_name in used_image_names:
                image_name = f"{xml_stem}{sample.image_path.suffix}"
            used_image_names.add(image_name)

            img_width, img_height = self._resolve_image_dimensions(sample)

            if copy_images:
                destination = images_dir / image_name
                if sample.image_path.exists():
                    if not destination.exists():
                        shutil.copy2(sample.image_path, destination)
                else:
                    logger.warning("Image not found during VOC export: %s", sample.image_path)

                filename_for_xml = image_name
            else:
                filename_for_xml = sample.image_path.as_posix()

            xml_content = self._create_annotation_xml(
                sample,
                filename_for_xml,
                folder=image_subdir,
                img_width=img_width,
                img_height=img_height,
            )
            (ann_dir / f"{xml_stem}.xml").write_text(xml_content, encoding="utf-8")

            split_name = str(sample.metadata.get("split", self.split)).strip() or self.split
            split_members.setdefault(split_name, []).append(xml_stem)

            self._report_progress(idx + 1, total, "Exporting VOC")

        for split_name, stems in split_members.items():
            split_file = sets_dir / f"{split_name}.txt"
            split_file.write_text(
                "\n".join(stems) + ("\n" if stems else ""),
                encoding="utf-8",
            )

        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported VOC dataset and return warnings."""
        warnings: list[str] = []

        ann_dir = self.output_path / "Annotations"
        images_dir = self.output_path / self._image_subdir
        sets_dir = self.output_path / "ImageSets" / "Main"

        if not ann_dir.exists():
            warnings.append("No Annotations directory")
            return warnings
        if not images_dir.exists():
            warnings.append(f"No {self._image_subdir} directory")
        if not sets_dir.exists():
            warnings.append("No ImageSets/Main directory")

        xml_files = sorted(ann_dir.glob("*.xml"))
        if not xml_files:
            warnings.append("No XML annotation files found")
            return warnings

        split_files = sorted(sets_dir.glob("*.txt")) if sets_dir.exists() else []
        if not split_files:
            warnings.append("No split files in ImageSets/Main")

        image_stems = {path.stem for path in images_dir.iterdir()} if images_dir.exists() else set()
        xml_stems = {path.stem for path in xml_files}
        if self._copied_images and image_stems and image_stems != xml_stems:
            missing_labels = image_stems - xml_stems
            missing_images = xml_stems - image_stems
            if missing_labels:
                warnings.append(f"{len(missing_labels)} images without XML annotations")
            if missing_images:
                warnings.append(f"{len(missing_images)} XML annotations without images")

        missing_images = 0
        invalid_xml = 0
        for xml_path in xml_files:
            try:
                tree = ET.parse(xml_path)
            except ET.ParseError:
                invalid_xml += 1
                continue

            root = tree.getroot()
            if root.tag != "annotation":
                invalid_xml += 1
                continue

            filename = root.findtext("filename")
            if not filename:
                missing_images += 1
                continue

            file_path = Path(filename)
            resolved_image = (
                file_path if file_path.is_absolute() else images_dir / file_path.name
            )

            if not resolved_image.exists():
                missing_images += 1

        if invalid_xml > 0:
            warnings.append(f"{invalid_xml} XML files have invalid structure")
        if missing_images > 0:
            warnings.append(f"{missing_images} XML files reference missing images")

        return warnings
