"""CVAT XML format loader.

Supports CVAT for images 1.1 export format.
- Single XML file with all annotations
- Box, polygon, polyline, points, and cuboid shapes
- Track annotations for video sequences

Implements spec: 06-data-pipeline/07-cvat-loader
"""

from __future__ import annotations

import json
import logging
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.data.formats.base import (
    FormatExporter,
    FormatLoader,
    FormatRegistry,
    ProgressCallback,
)
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_SHAPE_TYPES = ("box", "polygon", "polyline", "points", "cuboid")
_SHAPES_WITH_POINTS = {"polygon", "polyline", "points", "cuboid"}
_RESERVED_SHAPE_ATTRIBUTES = {
    "shape_type",
    "points",
    "occluded",
    "source",
    "z_order",
    "group_id",
    "rotation",
    "outside",
    "keyframe",
}


@dataclass
class _ImageRecord:
    """Internal representation of a CVAT image/frame and its annotations."""

    image_id: int
    name: str
    width: int
    height: int
    annotations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _TrackShapeRecord:
    """Track shape payload during CVAT export."""

    frame: int
    label: Label
    image_width: int
    image_height: int


@FormatRegistry.register_loader
class CvatLoader(FormatLoader):
    """Load CVAT XML export datasets."""

    format_name = "cvat"
    description = "CVAT XML format (single annotations file)"
    extensions = [".xml"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        xml_file: str | None = None,
        load_tracks: bool = True,
    ) -> None:
        """Initialize CVAT loader.

        Args:
            root_path: Dataset root directory
            class_names: Optional class-name override
            progress_callback: Optional progress callback
            xml_file: Explicit XML filename to use
            load_tracks: Expand track annotations to frame-level labels
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.xml_file = xml_file
        self.load_tracks = load_tracks

        self._parsed = False
        self._xml_path: Path | None = None
        self._version = ""
        self._images: dict[int, _ImageRecord] = {}
        self._class_names_discovered: list[str] = []
        self._has_tracks = False
        self._parse_warnings: list[str] = []
        self._warning_set: set[str] = set()

    def _add_warning(self, message: str) -> None:
        """Store warning message once."""
        if message in self._warning_set:
            return
        self._warning_set.add(message)
        self._parse_warnings.append(message)
        logger.warning(message)

    def _register_class_name(self, class_name: str) -> None:
        """Register class name while preserving insertion order."""
        if class_name and class_name not in self._class_names_discovered:
            self._class_names_discovered.append(class_name)

    @staticmethod
    def _parse_int(value: str | None, default: int = 0) -> int:
        """Parse integer value with fallback default."""
        try:
            return int(value) if value is not None else default
        except ValueError:
            return default

    @staticmethod
    def _parse_float(value: str | None, default: float = 0.0) -> float:
        """Parse float value with fallback default."""
        try:
            return float(value) if value is not None else default
        except ValueError:
            return default

    def _find_xml_file(self) -> Path | None:
        """Find CVAT XML file in dataset root."""
        candidates: list[Path] = []

        if self.xml_file:
            candidates.extend(
                [
                    self.root_path / self.xml_file,
                    self.root_path / "annotations" / self.xml_file,
                ]
            )
        else:
            candidates.extend(
                [
                    self.root_path / "annotations.xml",
                    self.root_path / "annotations" / "annotations.xml",
                ]
            )

        seen: set[Path] = set()
        for candidate in candidates:
            if candidate.exists():
                return candidate
            seen.add(candidate)

        extra_xmls = sorted(self.root_path.glob("*.xml"))
        ann_dir = self.root_path / "annotations"
        if ann_dir.is_dir():
            extra_xmls.extend(sorted(ann_dir.glob("*.xml")))

        for xml_path in extra_xmls:
            if xml_path in seen:
                continue
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                continue
            if root.tag == "annotations":
                return xml_path

        return None

    @staticmethod
    def _parse_points(points_raw: str) -> list[float]:
        """Parse CVAT point list `x,y;x,y;...` into flat float list."""
        points: list[float] = []
        for raw_point in points_raw.split(";"):
            raw_point = raw_point.strip()
            if not raw_point:
                continue
            coords = [value.strip() for value in raw_point.split(",")]
            if len(coords) != 2:
                continue
            try:
                x, y = float(coords[0]), float(coords[1])
            except ValueError:
                continue
            points.extend([x, y])
        return points

    def _parse_shape_attributes(self, elem: ET.Element) -> dict[str, str]:
        """Parse `<attribute name="...">value</attribute>` entries."""
        attributes: dict[str, str] = {}
        for attr_elem in elem.findall("attribute"):
            name = (attr_elem.get("name") or "").strip()
            value = (attr_elem.text or "").strip()
            if name:
                attributes[name] = value
        return attributes

    def _parse_shape(
        self,
        elem: ET.Element,
        shape_type: str,
        *,
        default_label: str = "",
    ) -> dict[str, Any] | None:
        """Parse a CVAT shape element."""
        label = (elem.get("label") or default_label).strip()
        if not label:
            self._add_warning(f"Skipping CVAT {shape_type} without a label")
            return None

        annotation: dict[str, Any] = {
            "type": shape_type,
            "label": label,
            "occluded": elem.get("occluded", "0") == "1",
            "attributes": self._parse_shape_attributes(elem),
        }

        for key in ("source", "z_order", "group_id", "rotation", "outside", "keyframe"):
            value = elem.get(key)
            if value is not None:
                annotation[key] = value

        if shape_type == "box":
            xtl = self._parse_float(elem.get("xtl"), float("nan"))
            ytl = self._parse_float(elem.get("ytl"), float("nan"))
            xbr = self._parse_float(elem.get("xbr"), float("nan"))
            ybr = self._parse_float(elem.get("ybr"), float("nan"))
            if any(coord != coord for coord in (xtl, ytl, xbr, ybr)):  # NaN check
                self._add_warning(f"Skipping CVAT box for '{label}' due to invalid coordinates")
                return None
            annotation.update({"xtl": xtl, "ytl": ytl, "xbr": xbr, "ybr": ybr})
            return annotation

        points = self._parse_points(elem.get("points", ""))
        min_points = 2 if shape_type == "points" else 4
        if len(points) < min_points:
            self._add_warning(f"Skipping CVAT {shape_type} for '{label}' due to invalid points")
            return None

        annotation["points"] = points
        return annotation

    def _parse_meta_labels(self, root: ET.Element) -> None:
        """Parse class names from CVAT `<meta>` section."""
        meta = root.find("meta")
        if meta is None:
            return

        for labels_elem in meta.findall(".//labels"):
            for label_elem in labels_elem.findall("label"):
                class_name = (label_elem.findtext("name") or "").strip()
                if class_name:
                    self._register_class_name(class_name)

    def _create_or_get_image_record(
        self,
        image_id: int,
        *,
        default_name: str = "",
        default_width: int = 0,
        default_height: int = 0,
    ) -> _ImageRecord:
        """Get existing image record or create one."""
        if image_id in self._images:
            record = self._images[image_id]
            if default_name and not record.name:
                record.name = default_name
            if default_width > 0 and record.width <= 0:
                record.width = default_width
            if default_height > 0 and record.height <= 0:
                record.height = default_height
            return record

        record = _ImageRecord(
            image_id=image_id,
            name=default_name,
            width=default_width,
            height=default_height,
        )
        self._images[image_id] = record
        return record

    def _parse_images(self, root: ET.Element) -> None:
        """Parse `<image>` blocks and contained annotations."""
        next_generated_id = 0

        for image_elem in root.findall("image"):
            raw_id = image_elem.get("id")
            image_id = self._parse_int(raw_id, default=-1) if raw_id is not None else -1

            if image_id < 0 or image_id in self._images:
                while next_generated_id in self._images:
                    next_generated_id += 1
                image_id = next_generated_id
                next_generated_id += 1

            name = (image_elem.get("name") or f"image_{image_id}.jpg").strip()
            width = self._parse_int(image_elem.get("width"), default=0)
            height = self._parse_int(image_elem.get("height"), default=0)

            record = self._create_or_get_image_record(
                image_id,
                default_name=name,
                default_width=width,
                default_height=height,
            )

            for shape_type in _SHAPE_TYPES:
                for shape_elem in image_elem.findall(shape_type):
                    ann = self._parse_shape(shape_elem, shape_type)
                    if ann is None:
                        continue
                    record.annotations.append(ann)
                    self._register_class_name(str(ann["label"]))

    def _parse_tracks(self, root: ET.Element) -> None:
        """Parse `<track>` blocks and map them to image frames."""
        for track_elem in root.findall("track"):
            track_id = self._parse_int(track_elem.get("id"), default=-1)
            track_label = (track_elem.get("label") or "").strip()

            for shape_type in _SHAPE_TYPES:
                for shape_elem in track_elem.findall(shape_type):
                    frame = self._parse_int(shape_elem.get("frame"), default=-1)
                    if frame < 0:
                        self._add_warning("Skipping track shape without valid frame index")
                        continue

                    if shape_elem.get("outside", "0") == "1":
                        # Object is not visible in this frame.
                        continue

                    ann = self._parse_shape(
                        shape_elem,
                        shape_type,
                        default_label=track_label,
                    )
                    if ann is None:
                        continue

                    if track_id >= 0:
                        ann["track_id"] = track_id
                    record = self._create_or_get_image_record(
                        frame,
                        default_name=f"frame_{frame:06d}.jpg",
                    )
                    record.annotations.append(ann)
                    self._register_class_name(str(ann["label"]))
                    self._has_tracks = True

    def _ensure_parsed(self) -> None:
        """Parse XML once and cache parsed structures."""
        if self._parsed:
            return

        xml_path = self._find_xml_file()
        if xml_path is None:
            raise FileNotFoundError(f"No CVAT XML file found in {self.root_path}")

        self._xml_path = xml_path
        root = ET.parse(xml_path).getroot()
        if root.tag != "annotations":
            raise ValueError(f"Unsupported XML root tag '{root.tag}', expected 'annotations'")

        self._version = (root.findtext("version") or "").strip()
        if self._version and self._version != "1.1":
            self._add_warning(
                f"CVAT XML version '{self._version}' detected; loader is validated for version 1.1"
            )

        self._parse_meta_labels(root)
        self._parse_images(root)
        if self.load_tracks:
            self._parse_tracks(root)

        self._parsed = True

    def _infer_class_names(self) -> list[str]:
        """Infer class names from XML meta and discovered annotations."""
        self._ensure_parsed()
        return self._class_names_discovered.copy()

    def _find_image(self, name: str) -> Path | None:
        """Resolve an image path from a CVAT image name."""
        if not name:
            return None

        image_name = Path(name)
        if image_name.is_absolute():
            return image_name if image_name.exists() else None

        candidates = [
            self.root_path / image_name,
            self.root_path / "images" / image_name,
            self.root_path / "data" / image_name,
            self.root_path / image_name.name,
            self.root_path / "images" / image_name.name,
            self.root_path / "data" / image_name.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        stem = image_name.stem
        for search_dir in (self.root_path, self.root_path / "images", self.root_path / "data"):
            if not search_dir.is_dir():
                continue
            for ext in _IMAGE_EXTENSIONS:
                candidate = search_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate

        return None

    @staticmethod
    def _read_image_dimensions(image_path: Path) -> tuple[int, int]:
        """Read image dimensions using Pillow when available."""
        if not image_path.exists():
            return (0, 0)

        try:
            from PIL import Image
        except ImportError:
            return (0, 0)

        try:
            with Image.open(image_path) as image:
                return image.size
        except Exception:
            return (0, 0)

    def _normalize_bbox(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float] | None:
        """Normalize absolute bbox coordinates to [0, 1]."""
        if image_width > 0 and image_height > 0:
            return (
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height,
            )

        # Fallback when XML stores already-normalized values.
        if all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
            return (x1, y1, x2, y2)

        return None

    def _normalize_points(
        self,
        points: list[float],
        *,
        image_width: int,
        image_height: int,
    ) -> list[float]:
        """Normalize point coordinates when image dimensions are known."""
        if image_width > 0 and image_height > 0:
            normalized: list[float] = []
            for index in range(0, len(points), 2):
                normalized.append(points[index] / image_width)
                normalized.append(points[index + 1] / image_height)
            return normalized
        return points.copy()

    def _annotation_to_label(
        self,
        annotation: dict[str, Any],
        *,
        image_width: int,
        image_height: int,
        class_ids: dict[str, int],
        class_names: list[str],
        image_name: str,
    ) -> Label | None:
        """Convert parsed CVAT annotation into internal Label."""
        class_name = str(annotation.get("label", "")).strip()
        if not class_name:
            return None

        if class_name not in class_ids:
            class_ids[class_name] = len(class_ids)
            class_names.append(class_name)

        shape_type = str(annotation.get("type", "box"))

        if shape_type == "box":
            x1 = float(annotation["xtl"])
            y1 = float(annotation["ytl"])
            x2 = float(annotation["xbr"])
            y2 = float(annotation["ybr"])
        elif shape_type in {"polygon", "polyline", "points", "cuboid"}:
            points = annotation.get("points", [])
            if not isinstance(points, list) or len(points) < 2:
                return None
            xs = [float(value) for value in points[0::2]]
            ys = [float(value) for value in points[1::2]]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            self._add_warning(f"Unsupported CVAT shape type '{shape_type}' in '{image_name}'")
            return None

        normalized = self._normalize_bbox(
            x1,
            y1,
            x2,
            y2,
            image_width=image_width,
            image_height=image_height,
        )
        if normalized is None:
            self._add_warning(
                f"Skipping annotation '{class_name}' in '{image_name}' due to missing dimensions"
            )
            return None

        attributes = {}
        raw_attributes = annotation.get("attributes")
        if isinstance(raw_attributes, dict):
            attributes = dict(raw_attributes)
        attributes["occluded"] = bool(annotation.get("occluded", False))
        attributes["shape_type"] = shape_type

        if "points" in annotation and isinstance(annotation["points"], list):
            attributes["points"] = self._normalize_points(
                annotation["points"],
                image_width=image_width,
                image_height=image_height,
            )

        for key in ("source", "z_order", "group_id", "outside", "keyframe", "rotation"):
            if key in annotation:
                attributes[key] = annotation[key]

        track_id: int | None = None
        if "track_id" in annotation:
            raw_track = annotation["track_id"]
            if isinstance(raw_track, int):
                track_id = raw_track
            elif isinstance(raw_track, str) and raw_track.isdigit():
                track_id = int(raw_track)

        return Label(
            class_name=class_name,
            class_id=class_ids[class_name],
            bbox=BBox.from_xyxy(*normalized),
            attributes=attributes,
            track_id=track_id,
        )

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all samples in the dataset."""
        self._ensure_parsed()

        class_names = self.get_class_names().copy()
        class_ids = {name: index for index, name in enumerate(class_names)}

        image_ids = sorted(self._images.keys())
        total = len(image_ids)

        for index, image_id in enumerate(image_ids):
            record = self._images[image_id]
            image_path = self._find_image(record.name)
            if image_path is None:
                image_path = self.root_path / record.name if record.name else self.root_path

            image_width = record.width
            image_height = record.height
            if image_width <= 0 or image_height <= 0:
                inferred_width, inferred_height = self._read_image_dimensions(image_path)
                if inferred_width > 0 and inferred_height > 0:
                    image_width = inferred_width
                    image_height = inferred_height

            labels: list[Label] = []
            for annotation in record.annotations:
                label = self._annotation_to_label(
                    annotation,
                    image_width=image_width,
                    image_height=image_height,
                    class_ids=class_ids,
                    class_names=class_names,
                    image_name=record.name or str(image_id),
                )
                if label is not None:
                    labels.append(label)

            metadata: dict[str, Any] = {"cvat_id": image_id}
            if record.name:
                metadata["cvat_name"] = record.name

            yield Sample(
                image_path=image_path,
                labels=labels,
                image_width=image_width if image_width > 0 else None,
                image_height=image_height if image_height > 0 else None,
                metadata=metadata,
            )

            self._report_progress(index + 1, total, "Loading CVAT")

    def load(self) -> Dataset:
        """Load full dataset into memory."""
        samples = list(self.iter_samples())
        return Dataset(
            samples=samples,
            name=self.root_path.name,
            class_names=self.get_class_names(),
        )

    def validate(self) -> list[str]:
        """Validate dataset structure and return warnings."""
        xml_path = self._find_xml_file()
        if xml_path is None:
            return ["No CVAT XML file found"]

        try:
            self._ensure_parsed()
        except ET.ParseError as exc:
            return [f"XML parse error: {exc}"]
        except Exception as exc:
            return [str(exc)]

        warnings = self._parse_warnings.copy()

        if not self._class_names_discovered:
            warnings.append("No labels defined in XML metadata or annotations")

        if not self._images:
            warnings.append("No images or frame annotations found")
            return warnings

        missing_images = sum(
            1 for image_record in self._images.values() if self._find_image(image_record.name) is None
        )
        if missing_images > 0:
            warnings.append(f"{missing_images} images referenced in XML were not found on disk")

        return warnings

    def summary(self) -> dict[str, Any]:
        """Get summary information for dataset."""
        base = super().summary()

        xml_path = self._find_xml_file()
        base["xml_file"] = str(xml_path) if xml_path else None

        if xml_path is None:
            return base

        try:
            self._ensure_parsed()
        except Exception as exc:
            base["parse_error"] = str(exc)
            return base

        base["version"] = self._version or None
        base["num_images"] = len(self._images)
        base["num_labels"] = len(self._class_names_discovered)
        base["has_tracks"] = self._has_tracks
        base["warnings"] = len(self._parse_warnings)
        return base


@FormatRegistry.register_exporter
class CvatExporter(FormatExporter):
    """Export datasets to CVAT XML format.

    Creates:
    - annotations.xml
    - images/ directory
    """

    format_name = "cvat"
    description = "CVAT XML format (single annotations file)"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: ProgressCallback | None = None,
        task_name: str = "exported",
    ) -> None:
        """Initialize CVAT exporter."""
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.task_name = task_name
        self._copied_images = True
        self._image_subdir = "images"

    @staticmethod
    def _as_bool(value: object, *, default: bool = False) -> bool:
        """Convert loose values to bool."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return default

    @staticmethod
    def _stringify_attribute(value: object) -> str:
        """Serialize attribute value to text."""
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (str, int, float)):
            return str(value)
        return json.dumps(value, sort_keys=True)

    @staticmethod
    def _resolve_class_names(dataset: Dataset, samples: list[Sample]) -> list[str]:
        """Resolve class names from dataset metadata and label content."""
        class_names = list(dataset.class_names) if dataset.class_names else []
        max_class_id = -1

        for sample in samples:
            for label in sample.labels:
                if label.class_id > max_class_id:
                    max_class_id = label.class_id

        if max_class_id >= len(class_names):
            class_names.extend(
                f"class_{class_id}" for class_id in range(len(class_names), max_class_id + 1)
            )

        for sample in samples:
            for label in sample.labels:
                if (
                    0 <= label.class_id < len(class_names)
                    and label.class_name
                    and (
                        not class_names[label.class_id]
                        or class_names[label.class_id].startswith("class_")
                    )
                ):
                    class_names[label.class_id] = label.class_name

        return class_names

    @staticmethod
    def _label_name(label: Label, class_names: list[str]) -> str:
        """Resolve the exported class name for a label."""
        if label.class_name:
            return label.class_name
        if 0 <= label.class_id < len(class_names):
            return class_names[label.class_id]
        return f"class_{label.class_id}"

    @staticmethod
    def _resolve_image_dimensions(sample: Sample) -> tuple[int, int]:
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
                "Unable to resolve image dimensions for %s. Using CVAT defaults 1920x1080.",
                sample.image_path,
            )
            return 1920, 1080

    @staticmethod
    def _bbox_to_pixels(bbox: BBox, img_width: int, img_height: int) -> tuple[float, float, float, float]:
        """Convert normalized bbox to absolute pixel coordinates."""
        x1, y1, x2, y2 = bbox.to_xyxy()
        xtl = max(0.0, min(float(img_width), x1 * img_width))
        ytl = max(0.0, min(float(img_height), y1 * img_height))
        xbr = max(0.0, min(float(img_width), x2 * img_width))
        ybr = max(0.0, min(float(img_height), y2 * img_height))

        if xbr < xtl:
            xtl, xbr = xbr, xtl
        if ybr < ytl:
            ytl, ybr = ybr, ytl

        return xtl, ytl, xbr, ybr

    @staticmethod
    def _points_to_cvat_string(
        points: object,
        *,
        img_width: int,
        img_height: int,
        min_pairs: int,
    ) -> str | None:
        """Convert point list to CVAT `x,y;x,y` format."""
        if not isinstance(points, (list, tuple)):
            return None

        try:
            raw_values = [float(value) for value in points]
        except (TypeError, ValueError):
            return None

        if len(raw_values) < (min_pairs * 2) or len(raw_values) % 2 != 0:
            return None

        is_normalized = all(0.0 <= value <= 1.0 for value in raw_values)
        encoded_points: list[str] = []
        for idx in range(0, len(raw_values), 2):
            x = raw_values[idx]
            y = raw_values[idx + 1]
            if is_normalized:
                x *= img_width
                y *= img_height
            encoded_points.append(f"{x:.2f},{y:.2f}")
        return ";".join(encoded_points)

    def _set_shape_attributes(
        self,
        shape_elem: ET.Element,
        attributes: dict[str, Any],
        *,
        include_track_fields: bool,
    ) -> None:
        """Set CVAT shape attributes and custom fields."""
        shape_elem.set("occluded", "1" if self._as_bool(attributes.get("occluded")) else "0")

        for key in ("source", "z_order", "group_id", "rotation"):
            if key in attributes and attributes[key] is not None:
                shape_elem.set(key, str(attributes[key]))

        if include_track_fields:
            shape_elem.set("outside", "1" if self._as_bool(attributes.get("outside")) else "0")
            if "keyframe" in attributes:
                shape_elem.set(
                    "keyframe",
                    "1" if self._as_bool(attributes.get("keyframe"), default=True) else "0",
                )

        for key, value in attributes.items():
            if key in _RESERVED_SHAPE_ATTRIBUTES:
                continue
            if value is None:
                continue
            attr_elem = ET.SubElement(shape_elem, "attribute")
            attr_elem.set("name", str(key))
            attr_elem.text = self._stringify_attribute(value)

    def _append_shape(
        self,
        parent: ET.Element,
        label: Label,
        *,
        class_name: str,
        img_width: int,
        img_height: int,
        frame: int | None = None,
        default_track_label: str | None = None,
    ) -> None:
        """Append a CVAT shape for one label."""
        attributes = label.attributes if isinstance(label.attributes, dict) else {}
        shape_type = str(attributes.get("shape_type", "box")).strip().lower() or "box"

        shape_attrs: dict[str, str] = {}
        if frame is None:
            shape_attrs["label"] = class_name
        else:
            shape_attrs["frame"] = str(frame)
            if class_name and class_name != (default_track_label or ""):
                shape_attrs["label"] = class_name

        min_pairs = 1 if shape_type == "points" else 2
        if shape_type in _SHAPES_WITH_POINTS:
            points_str = self._points_to_cvat_string(
                attributes.get("points"),
                img_width=img_width,
                img_height=img_height,
                min_pairs=min_pairs,
            )
            if points_str:
                shape_attrs["points"] = points_str
                shape_elem = ET.SubElement(parent, shape_type, shape_attrs)
                self._set_shape_attributes(
                    shape_elem,
                    attributes,
                    include_track_fields=frame is not None,
                )
                return

        xtl, ytl, xbr, ybr = self._bbox_to_pixels(label.bbox, img_width=img_width, img_height=img_height)
        shape_attrs.update(
            {
                "xtl": f"{xtl:.2f}",
                "ytl": f"{ytl:.2f}",
                "xbr": f"{xbr:.2f}",
                "ybr": f"{ybr:.2f}",
            }
        )
        shape_elem = ET.SubElement(parent, "box", shape_attrs)
        self._set_shape_attributes(
            shape_elem,
            attributes,
            include_track_fields=frame is not None,
        )

    def _copy_image_if_needed(
        self,
        sample: Sample,
        destination: Path,
        *,
        copy_images: bool,
    ) -> None:
        """Copy image into export tree when configured."""
        if not copy_images:
            return
        if not sample.image_path.exists():
            logger.warning("Image not found during CVAT export: %s", sample.image_path)
            return
        if destination.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, destination)

    def _build_meta(self, root: ET.Element, *, class_names: list[str], size: int) -> None:
        """Build CVAT metadata section."""
        meta_elem = ET.SubElement(root, "meta")
        task_elem = ET.SubElement(meta_elem, "task")
        ET.SubElement(task_elem, "name").text = self.task_name
        ET.SubElement(task_elem, "size").text = str(size)

        labels_elem = ET.SubElement(task_elem, "labels")
        for class_name in class_names:
            label_elem = ET.SubElement(labels_elem, "label")
            ET.SubElement(label_elem, "name").text = class_name

    def export(
        self,
        dataset: Dataset,
        *,
        copy_images: bool = True,
        image_subdir: str = "images",
    ) -> Path:
        """Export dataset to CVAT XML format."""
        self._ensure_output_dir()
        self._copied_images = copy_images
        self._image_subdir = image_subdir

        images_dir = self.output_path / image_subdir
        images_dir.mkdir(parents=True, exist_ok=True)

        samples = list(dataset)
        class_names = self._resolve_class_names(dataset, samples)

        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        self._build_meta(root, class_names=class_names, size=len(samples))

        used_image_names: set[str] = set()
        track_map: dict[int, list[_TrackShapeRecord]] = {}
        track_class_names: dict[int, str] = {}
        negative_track_id_map: dict[int, int] = {}
        fallback_track_id = 0
        total = len(samples)

        for idx, sample in enumerate(samples):
            img_width, img_height = self._resolve_image_dimensions(sample)

            if copy_images:
                image_name = sample.image_path.name
                if image_name in used_image_names:
                    image_name = f"{sample.image_path.stem}_{idx}{sample.image_path.suffix}"
                if not image_name:
                    image_name = f"image_{idx:06d}.jpg"
                used_image_names.add(image_name)
                self._copy_image_if_needed(sample, images_dir / image_name, copy_images=copy_images)
            else:
                image_name = sample.image_path.as_posix()

            image_elem = ET.SubElement(
                root,
                "image",
                {
                    "id": str(idx),
                    "name": image_name,
                    "width": str(img_width),
                    "height": str(img_height),
                },
            )

            for label in sample.labels:
                class_name = self._label_name(label, class_names)

                if label.track_id is None:
                    self._append_shape(
                        image_elem,
                        label,
                        class_name=class_name,
                        img_width=img_width,
                        img_height=img_height,
                    )
                    continue

                track_id = label.track_id
                if track_id < 0:
                    if track_id not in negative_track_id_map:
                        negative_track_id_map[track_id] = fallback_track_id
                        fallback_track_id += 1
                    track_id = negative_track_id_map[track_id]

                track_map.setdefault(track_id, []).append(
                    _TrackShapeRecord(
                        frame=idx,
                        label=label,
                        image_width=img_width,
                        image_height=img_height,
                    )
                )

                if track_id not in track_class_names:
                    track_class_names[track_id] = class_name
                elif track_class_names[track_id] != class_name:
                    logger.warning(
                        "Track %s has inconsistent labels (%s vs %s); keeping first label.",
                        track_id,
                        track_class_names[track_id],
                        class_name,
                    )

            self._report_progress(idx + 1, total, "Exporting CVAT")

        for track_id in sorted(track_map.keys()):
            track_label = track_class_names.get(track_id, "")
            track_elem = ET.SubElement(
                root,
                "track",
                {
                    "id": str(track_id),
                    "label": track_label,
                },
            )
            for record in sorted(track_map[track_id], key=lambda item: item.frame):
                self._append_shape(
                    track_elem,
                    record.label,
                    class_name=self._label_name(record.label, class_names),
                    img_width=record.image_width,
                    img_height=record.image_height,
                    frame=record.frame,
                    default_track_label=track_label,
                )

        ET.indent(root, space="  ")
        xml_path = self.output_path / "annotations.xml"
        xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        xml_path.write_bytes(xml_bytes)

        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported CVAT dataset."""
        warnings: list[str] = []
        xml_path = self.output_path / "annotations.xml"
        if not xml_path.exists():
            warnings.append("annotations.xml not found")
            return warnings

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            warnings.append(f"Invalid XML: {exc}")
            return warnings

        if root.tag != "annotations":
            warnings.append(f"Invalid root tag: {root.tag}")

        version = (root.findtext("version") or "").strip()
        if version != "1.1":
            warnings.append(f"Expected CVAT version 1.1, found '{version or 'missing'}'")

        image_entries = root.findall("image")
        track_entries = root.findall("track")
        if not image_entries and not track_entries:
            warnings.append("No image or track annotations found")

        if not self._copied_images:
            return warnings

        images_dir = self.output_path / self._image_subdir
        if not images_dir.exists():
            warnings.append(f"No {self._image_subdir} directory")
            return warnings

        missing_images = 0
        for image_elem in image_entries:
            name = (image_elem.get("name") or "").strip()
            if not name:
                missing_images += 1
                continue

            image_path = Path(name)
            if image_path.is_absolute():
                if not image_path.exists():
                    missing_images += 1
                continue

            direct_path = self.output_path / image_path
            subdir_path = images_dir / image_path.name
            if not direct_path.exists() and not subdir_path.exists():
                missing_images += 1

        if missing_images > 0:
            warnings.append(f"{missing_images} images referenced in XML were not found on disk")

        return warnings
