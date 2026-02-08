"""CVAT XML format loader.

Supports CVAT for images 1.1 export format.
- Single XML file with all annotations
- Box, polygon, polyline, points, and cuboid shapes
- Track annotations for video sequences

Implements spec: 06-data-pipeline/07-cvat-loader
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.data.formats.base import FormatLoader, FormatRegistry, ProgressCallback
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_SHAPE_TYPES = ("box", "polygon", "polyline", "points", "cuboid")


@dataclass
class _ImageRecord:
    """Internal representation of a CVAT image/frame and its annotations."""

    image_id: int
    name: str
    width: int
    height: int
    annotations: list[dict[str, Any]] = field(default_factory=list)


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
