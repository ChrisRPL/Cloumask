"""OpenLABEL format loader.

Supports OpenLABEL (ASAM) 1.x datasets:
- JSON annotations with `openlabel` root
- Frame-level 2D boxes and optional poly2d fallback
- Optional 3D cuboids and tracking IDs
- Actions/events/relations metadata
- Coordinate systems and frame transforms metadata

Implements spec: 06-data-pipeline/09-openlabel-loader
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from backend.data.formats.base import FormatLoader, FormatRegistry, ProgressCallback
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@FormatRegistry.register_loader
class OpenlabelLoader(FormatLoader):
    """Load OpenLABEL (ASAM) datasets."""

    format_name = "openlabel"
    description = "OpenLABEL (ASAM) format"
    extensions = [".json"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        json_file: str | None = None,
        stream_filter: list[str] | None = None,
    ) -> None:
        """Initialize OpenLABEL loader.

        Args:
            root_path: Dataset root directory
            class_names: Optional class-name override
            progress_callback: Optional progress callback
            json_file: Optional explicit JSON filename
            stream_filter: Optional stream whitelist (e.g. ["CAM_FRONT"])
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.json_file = json_file
        self.stream_filter = set(stream_filter or [])

        self._parsed = False
        self._json_path: Path | None = None
        self._data: dict[str, Any] = {}

        self._metadata: dict[str, Any] = {}
        self._coordinate_systems: dict[str, dict[str, Any]] = {}
        self._streams: dict[str, dict[str, Any]] = {}
        self._objects: dict[str, dict[str, Any]] = {}
        self._actions: dict[str, dict[str, Any]] = {}
        self._events: dict[str, dict[str, Any]] = {}
        self._relations: dict[str, dict[str, Any]] = {}
        self._frames: dict[str, dict[str, Any]] = {}

        self._track_ids: dict[str, int] = {}
        self._parse_warnings: list[str] = []
        self._warning_set: set[str] = set()

    def _add_warning(self, message: str) -> None:
        """Record warning once and log it."""
        if message in self._warning_set:
            return
        self._warning_set.add(message)
        self._parse_warnings.append(message)
        logger.warning(message)

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        """Return a dict-like value, otherwise an empty dict."""
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, dict[str, Any]]:
        """Return {str_key: dict_value} for OpenLABEL keyed sections."""
        mapping: dict[str, dict[str, Any]] = {}
        if not isinstance(value, dict):
            return mapping
        for key, item in value.items():
            if isinstance(item, dict):
                mapping[str(key)] = item
        return mapping

    @staticmethod
    def _looks_like_openlabel(payload: Any) -> bool:
        """Heuristic for OpenLABEL JSON payload."""
        if not isinstance(payload, dict):
            return False
        if isinstance(payload.get("openlabel"), dict):
            return True
        has_frames = isinstance(payload.get("frames"), dict)
        has_objects = isinstance(payload.get("objects"), dict)
        return has_frames and has_objects

    def _find_json_file(self) -> Path | None:
        """Find an OpenLABEL JSON file in dataset root."""
        candidates: list[Path] = []

        if self.json_file:
            candidates.extend(
                [
                    self.root_path / self.json_file,
                    self.root_path / "annotations" / self.json_file,
                ]
            )
        else:
            candidates.extend(
                [
                    self.root_path / "annotations.json",
                    self.root_path / "openlabel.json",
                ]
            )

        seen: set[Path] = set()
        for candidate in candidates:
            seen.add(candidate)
            if candidate.exists():
                try:
                    with candidate.open() as file:
                        payload = json.load(file)
                except json.JSONDecodeError:
                    continue
                if self._looks_like_openlabel(payload):
                    return candidate

        for json_path in sorted(self.root_path.glob("*.json")):
            if json_path in seen:
                continue
            try:
                with json_path.open() as file:
                    payload = json.load(file)
            except json.JSONDecodeError:
                continue
            if self._looks_like_openlabel(payload):
                return json_path

        return None

    def _ensure_parsed(self) -> None:
        """Parse OpenLABEL JSON once."""
        if self._parsed:
            return

        json_path = self._find_json_file()
        if json_path is None:
            raise FileNotFoundError(f"No OpenLABEL JSON file found in {self.root_path}")

        self._json_path = json_path
        with json_path.open() as file:
            payload = json.load(file)

        raw = payload.get("openlabel", payload)
        if not isinstance(raw, dict):
            raise ValueError("Invalid OpenLABEL payload: root must be an object")

        self._data = raw
        self._metadata = self._as_dict(raw.get("metadata"))
        self._coordinate_systems = self._as_mapping(raw.get("coordinate_systems"))
        self._streams = self._as_mapping(raw.get("streams"))
        self._objects = self._as_mapping(raw.get("objects"))
        self._actions = self._as_mapping(raw.get("actions"))
        self._events = self._as_mapping(raw.get("events"))
        self._relations = self._as_mapping(raw.get("relations"))
        self._frames = self._as_mapping(raw.get("frames"))

        for index, uid in enumerate(sorted(self._objects.keys()), start=1):
            self._track_ids[uid] = index

        self._parsed = True

    @staticmethod
    def _frame_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, int | str]:
        """Sort frame IDs numerically when possible."""
        frame_id = item[0]
        try:
            return (0, int(frame_id))
        except ValueError:
            return (1, frame_id)

    def _iter_frames(self) -> list[tuple[str, dict[str, Any]]]:
        """Get sorted frame entries."""
        return sorted(self._frames.items(), key=self._frame_sort_key)

    @staticmethod
    def _stream_from_data(data: dict[str, Any]) -> str | None:
        """Read stream name from object data."""
        stream = data.get("stream")
        if isinstance(stream, str) and stream.strip():
            return stream.strip()
        return None

    def _stream_allowed(self, stream_name: str | None) -> bool:
        """Check stream filter."""
        if not self.stream_filter:
            return True
        if stream_name is None:
            return True
        return stream_name in self.stream_filter

    @staticmethod
    def _extract_uri(data: dict[str, Any]) -> str | None:
        """Extract URI/path-like key from OpenLABEL stream metadata."""
        for key in ("uri", "path", "file", "filename"):
            raw = data.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()

        stream_props = data.get("stream_properties")
        if isinstance(stream_props, dict):
            for key in ("uri", "path", "file", "filename"):
                raw = stream_props.get(key)
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()

        return None

    def _resolve_path(self, raw_path: str) -> Path:
        """Resolve a file path from OpenLABEL data."""
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self.root_path / path

    def _frame_streams(self, frame_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Get per-frame stream entries."""
        frame_props = self._as_dict(frame_data.get("frame_properties"))
        return self._as_mapping(frame_props.get("streams"))

    def _find_frame_image(self, frame_id: str, frame_data: dict[str, Any]) -> Path | None:
        """Resolve frame image path from frame data and naming conventions."""
        frame_props = self._as_dict(frame_data.get("frame_properties"))

        for key in ("image_file", "image_path", "uri", "file", "filename"):
            raw = frame_props.get(key)
            if isinstance(raw, str) and raw.strip():
                candidate = self._resolve_path(raw.strip())
                if candidate.exists():
                    return candidate

        for stream_name, stream_data in self._frame_streams(frame_data).items():
            if not self._stream_allowed(stream_name):
                continue

            uri = self._extract_uri(stream_data)
            if uri:
                candidate = self._resolve_path(uri)
                if candidate.exists():
                    return candidate

            base_stream = self._streams.get(stream_name, {})
            base_uri = self._extract_uri(base_stream)
            if base_uri:
                candidate = self._resolve_path(base_uri)
                if candidate.exists():
                    return candidate

        frame_id_no_pad = frame_id
        frame_id_pad = frame_id
        with suppress(ValueError):
            frame_id_pad = f"{int(frame_id):06d}"

        search_dirs = [
            self.root_path,
            self.root_path / "images",
            self.root_path / "data",
            self.root_path / "samples",
        ]
        name_patterns = [
            f"frame_{frame_id_pad}",
            f"frame_{frame_id_no_pad}",
            f"image_{frame_id_pad}",
            f"image_{frame_id_no_pad}",
            frame_id_pad,
            frame_id_no_pad,
        ]

        for directory in search_dirs:
            if not directory.exists():
                continue
            for stem in name_patterns:
                for extension in _IMAGE_EXTENSIONS:
                    candidate = directory / f"{stem}{extension}"
                    if candidate.exists():
                        return candidate

        return None

    @staticmethod
    def _read_image_size(image_path: Path) -> tuple[int, int]:
        """Read image size when Pillow is available."""
        if not image_path.exists():
            return (0, 0)

        try:
            from PIL import Image
        except ImportError:
            return (0, 0)

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                return int(width), int(height)
        except Exception:
            return (0, 0)

    def _frame_dimensions(
        self,
        frame_data: dict[str, Any],
        image_path: Path | None,
    ) -> tuple[int, int]:
        """Resolve frame image dimensions."""
        frame_props = self._as_dict(frame_data.get("frame_properties"))

        width = frame_props.get("width", frame_data.get("width", 0))
        height = frame_props.get("height", frame_data.get("height", 0))

        try:
            image_width = int(width)
        except (TypeError, ValueError):
            image_width = 0
        try:
            image_height = int(height)
        except (TypeError, ValueError):
            image_height = 0

        if (image_width <= 0 or image_height <= 0) and image_path is not None:
            inferred_width, inferred_height = self._read_image_size(image_path)
            if inferred_width > 0 and inferred_height > 0:
                image_width, image_height = inferred_width, inferred_height

        if image_width <= 0:
            image_width = 1920
        if image_height <= 0:
            image_height = 1080

        return image_width, image_height

    @staticmethod
    def _as_entries(value: Any) -> list[dict[str, Any]]:
        """Normalize object_data payloads (list[dict] or dict)."""
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _to_float_list(value: Any, *, min_len: int) -> list[float] | None:
        """Convert iterable values to float list with minimum length."""
        if not isinstance(value, (list, tuple)):
            return None
        if len(value) < min_len:
            return None
        converted: list[float] = []
        for item in value:
            try:
                converted.append(float(item))
            except (TypeError, ValueError):
                return None
        return converted

    def _parse_bbox_2d(
        self,
        bbox_data: Any,
        image_width: int,
        image_height: int,
    ) -> BBox | None:
        """Parse OpenLABEL `bbox` object_data into normalized BBox."""
        for entry in self._as_entries(bbox_data):
            values = self._to_float_list(entry.get("val"), min_len=4)
            if values is None:
                continue
            x, y, w, h = values[:4]

            if max(abs(x), abs(y), abs(w), abs(h)) > 1.0:
                if image_width <= 0 or image_height <= 0:
                    continue
                return BBox.from_xywh(
                    x / image_width,
                    y / image_height,
                    w / image_width,
                    h / image_height,
                )

            return BBox.from_xywh(x, y, w, h)

        return None

    @staticmethod
    def _flatten_poly_points(value: Any) -> list[float]:
        """Flatten poly2d coordinate payload into [x1, y1, x2, y2, ...]."""
        if not isinstance(value, list):
            return []

        if value and all(isinstance(item, (int, float)) for item in value):
            return [float(item) for item in value]

        flattened: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    flattened.extend([float(item[0]), float(item[1])])
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, dict) and "x" in item and "y" in item:
                try:
                    flattened.extend([float(item["x"]), float(item["y"])])
                except (TypeError, ValueError):
                    continue
        return flattened

    def _parse_bbox_from_poly2d(
        self,
        poly_data: Any,
        image_width: int,
        image_height: int,
    ) -> BBox | None:
        """Build bbox from OpenLABEL poly2d data."""
        for entry in self._as_entries(poly_data):
            points = self._flatten_poly_points(entry.get("val"))
            if len(points) < 4:
                continue

            xs = points[0::2]
            ys = points[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            if max(abs(x_min), abs(y_min), abs(x_max), abs(y_max)) > 1.0:
                if image_width <= 0 or image_height <= 0:
                    continue
                return BBox.from_xyxy(
                    x_min / image_width,
                    y_min / image_height,
                    x_max / image_width,
                    y_max / image_height,
                )

            return BBox.from_xyxy(x_min, y_min, x_max, y_max)

        return None

    def _parse_cuboid_3d(self, cuboid_data: Any) -> dict[str, list[float]] | None:
        """Parse OpenLABEL cuboid data into position/rotation/size groups."""
        for entry in self._as_entries(cuboid_data):
            values = self._to_float_list(entry.get("val"), min_len=10)
            if values is None:
                continue
            return {
                "position": values[0:3],
                "rotation": values[3:7],
                "size": values[7:10],
            }
        return None

    @staticmethod
    def _merge_object_data(base_data: Any, frame_data: Any) -> dict[str, Any]:
        """Merge base object_data with frame object_data (frame overrides)."""
        merged: dict[str, Any] = {}
        if isinstance(base_data, dict):
            merged.update(base_data)
        if isinstance(frame_data, dict):
            merged.update(frame_data)
        return merged

    @staticmethod
    def _extract_named_attributes(object_data: dict[str, Any]) -> dict[str, Any]:
        """Extract text/num/boolean object_data into flat attributes."""
        attributes: dict[str, Any] = {}

        for bucket in ("text", "num", "boolean"):
            for entry in OpenlabelLoader._as_entries(object_data.get(bucket)):
                name = entry.get("name", bucket)
                value = entry.get("val")
                if isinstance(name, str) and name and value is not None:
                    attributes[name] = value

        return attributes

    @staticmethod
    def _parse_frame_number(frame_id: str) -> int | None:
        """Parse numeric frame ID if possible."""
        try:
            return int(frame_id)
        except ValueError:
            pass

        match = re.search(r"(\d+)$", frame_id)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _parse_interval_bound(value: Any) -> int | str | None:
        """Parse interval bound from OpenLABEL interval fields."""
        if isinstance(value, int):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return int(text)
            except ValueError:
                return text
        return None

    def _entity_active_in_frame(self, entity: dict[str, Any], frame_id: str) -> bool:
        """Check if entity with `frame_intervals` is active for current frame."""
        intervals = entity.get("frame_intervals")
        if not isinstance(intervals, list) or not intervals:
            return True

        frame_num = self._parse_frame_number(frame_id)
        for interval in intervals:
            if not isinstance(interval, dict):
                continue

            start_raw = interval.get("frame_start", interval.get("start"))
            end_raw = interval.get("frame_end", interval.get("end"))
            start = self._parse_interval_bound(start_raw)
            end = self._parse_interval_bound(end_raw)

            if frame_num is not None and isinstance(start, int) and isinstance(end, int):
                if start <= frame_num <= end:
                    return True
                continue

            start_text = str(start) if start is not None else frame_id
            end_text = str(end) if end is not None else frame_id
            if start_text <= frame_id <= end_text:
                return True

        return False

    def _merge_frame_entities(
        self,
        entity_name: str,
        frame_data: dict[str, Any],
        frame_id: str,
    ) -> dict[str, dict[str, Any]]:
        """Merge global and frame-scoped entities for one frame."""
        global_map: dict[str, dict[str, Any]]
        if entity_name == "actions":
            global_map = self._actions
        elif entity_name == "events":
            global_map = self._events
        else:
            global_map = self._relations

        frame_map = self._as_mapping(frame_data.get(entity_name))
        merged: dict[str, dict[str, Any]] = {}

        for uid, entry in global_map.items():
            if self._entity_active_in_frame(entry, frame_id):
                merged[uid] = dict(entry)

        for uid, entry in frame_map.items():
            current = dict(merged.get(uid, {}))
            current.update(entry)
            if self._entity_active_in_frame(current, frame_id):
                merged[uid] = current

        return merged

    @staticmethod
    def _normalize_entity_payload(uid: str, entity: dict[str, Any]) -> dict[str, Any]:
        """Build serializable frame metadata payload for action/event/relation."""
        payload: dict[str, Any] = {"uid": uid}

        for key in ("name", "type", "ontology_uid"):
            if key in entity:
                payload[key] = entity[key]

        for key in (
            "action_data",
            "event_data",
            "relation_data",
            "rdf_subjects",
            "rdf_objects",
            "subject",
            "target",
            "objects",
        ):
            if key in entity:
                payload[key] = entity[key]

        return payload

    @staticmethod
    def _extract_object_uids_from_entity(
        entity: Any,
        known_object_uids: set[str],
    ) -> set[str]:
        """Extract object UIDs referenced by action/event/relation payloads."""
        referenced: set[str] = set()

        def walk(value: Any) -> None:
            if isinstance(value, str):
                if value in known_object_uids:
                    referenced.add(value)
                return

            if isinstance(value, dict):
                value_type = value.get("type")
                value_uid = value.get("uid")
                if (
                    isinstance(value_type, str)
                    and value_type.lower() == "object"
                    and isinstance(value_uid, str)
                    and value_uid in known_object_uids
                ):
                    referenced.add(value_uid)

                for key in (
                    "object_uid",
                    "object",
                    "source",
                    "target",
                    "subject",
                    "from",
                    "to",
                ):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate in known_object_uids:
                        referenced.add(candidate)

                for key in (
                    "objects",
                    "rdf_subjects",
                    "rdf_objects",
                    "subjects",
                    "targets",
                    "participants",
                    "members",
                ):
                    child = value.get(key)
                    if child is not None:
                        walk(child)
                return

            if isinstance(value, list):
                for item in value:
                    walk(item)

        walk(entity)
        return referenced

    def _track_id_for_uid(self, object_uid: str) -> int:
        """Get stable numeric track ID for object UID."""
        existing = self._track_ids.get(object_uid)
        if existing is not None:
            return existing
        track_id = len(self._track_ids) + 1
        self._track_ids[object_uid] = track_id
        return track_id

    def _infer_class_names(self) -> list[str]:
        """Infer class names from object definitions and frame entries."""
        self._ensure_parsed()

        class_names: set[str] = set()

        for obj in self._objects.values():
            obj_type = obj.get("type")
            if isinstance(obj_type, str) and obj_type.strip():
                class_names.add(obj_type.strip())

        for frame in self._frames.values():
            for frame_object in self._as_mapping(frame.get("objects")).values():
                obj_type = frame_object.get("type")
                if isinstance(obj_type, str) and obj_type.strip():
                    class_names.add(obj_type.strip())

        return sorted(class_names)

    def _parse_frame_labels(
        self,
        frame_id: str,
        frame_data: dict[str, Any],
        image_width: int,
        image_height: int,
        class_names: list[str],
    ) -> tuple[list[Label], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Parse labels and frame entities for one frame."""
        frame_objects = self._as_mapping(frame_data.get("objects"))
        class_id_map = {name: index for index, name in enumerate(class_names)}
        known_object_uids = set(self._objects) | set(frame_objects)

        merged_actions = self._merge_frame_entities("actions", frame_data, frame_id)
        merged_events = self._merge_frame_entities("events", frame_data, frame_id)
        merged_relations = self._merge_frame_entities("relations", frame_data, frame_id)

        action_payloads: list[dict[str, Any]] = []
        event_payloads: list[dict[str, Any]] = []
        relation_payloads: list[dict[str, Any]] = []
        actions_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        events_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        relations_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for uid, action in merged_actions.items():
            payload = self._normalize_entity_payload(uid, action)
            action_payloads.append(payload)
            for object_uid in self._extract_object_uids_from_entity(action, known_object_uids):
                actions_by_object[object_uid].append(payload)

        for uid, event in merged_events.items():
            payload = self._normalize_entity_payload(uid, event)
            event_payloads.append(payload)
            for object_uid in self._extract_object_uids_from_entity(event, known_object_uids):
                events_by_object[object_uid].append(payload)

        for uid, relation in merged_relations.items():
            payload = self._normalize_entity_payload(uid, relation)
            relation_payloads.append(payload)
            for object_uid in self._extract_object_uids_from_entity(relation, known_object_uids):
                relations_by_object[object_uid].append(payload)

        labels: list[Label] = []
        for object_uid, frame_object in frame_objects.items():
            base_object = self._objects.get(object_uid, {})

            object_type_raw = frame_object.get("type", base_object.get("type", "unknown"))
            object_type = str(object_type_raw).strip() or "unknown"
            if object_type not in class_id_map:
                class_id_map[object_type] = len(class_id_map)
                class_names.append(object_type)
            class_id = class_id_map[object_type]

            object_name_raw = frame_object.get("name", base_object.get("name", object_uid))
            object_name = str(object_name_raw).strip() or object_uid

            object_data = self._merge_object_data(
                base_object.get("object_data"),
                frame_object.get("object_data"),
            )

            stream_name = self._stream_from_data(object_data)
            if stream_name is None:
                stream_name = self._stream_from_data(frame_object)
            if not self._stream_allowed(stream_name):
                continue

            bbox = self._parse_bbox_2d(object_data.get("bbox"), image_width, image_height)
            if bbox is None:
                bbox = self._parse_bbox_from_poly2d(
                    object_data.get("poly2d"),
                    image_width,
                    image_height,
                )
            if bbox is None:
                bbox = BBox(cx=0.5, cy=0.5, w=0.1, h=0.1)

            attributes: dict[str, Any] = {
                "object_uid": object_uid,
                "object_name": object_name,
            }

            if stream_name:
                attributes["stream"] = stream_name

            coordinate_system = object_data.get("coordinate_system")
            if coordinate_system is not None:
                attributes["coordinate_system"] = coordinate_system

            cuboid = self._parse_cuboid_3d(object_data.get("cuboid"))
            if cuboid is not None:
                attributes["cuboid_3d"] = cuboid

            attributes.update(self._extract_named_attributes(object_data))

            if actions_by_object.get(object_uid):
                attributes["actions"] = actions_by_object[object_uid]
            if events_by_object.get(object_uid):
                attributes["events"] = events_by_object[object_uid]
            if relations_by_object.get(object_uid):
                attributes["relations"] = relations_by_object[object_uid]

            labels.append(
                Label(
                    class_name=object_type,
                    class_id=class_id,
                    bbox=bbox,
                    attributes=attributes,
                    track_id=self._track_id_for_uid(object_uid),
                )
            )

        return labels, action_payloads, event_payloads, relation_payloads

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all OpenLABEL frames."""
        self._ensure_parsed()

        class_names = self.get_class_names().copy()
        frames = self._iter_frames()
        total = len(frames)

        for index, (frame_id, frame_data) in enumerate(frames, start=1):
            image_path = self._find_frame_image(frame_id, frame_data)
            if image_path is None:
                image_path = self.root_path / f"frame_{frame_id}"

            image_width, image_height = self._frame_dimensions(frame_data, image_path)
            labels, actions, events, relations = self._parse_frame_labels(
                frame_id,
                frame_data,
                image_width,
                image_height,
                class_names,
            )

            frame_props = self._as_dict(frame_data.get("frame_properties"))
            metadata: dict[str, Any] = {
                "frame_id": frame_id,
                "timestamp": frame_props.get("timestamp"),
                "schema_version": self._metadata.get("schema_version"),
                "actions": actions,
                "events": events,
                "relations": relations,
            }

            frame_streams = self._frame_streams(frame_data)
            if frame_streams:
                metadata["streams"] = sorted(frame_streams.keys())

            if self._coordinate_systems:
                metadata["coordinate_systems"] = self._coordinate_systems

            frame_coordinate_systems = frame_data.get("coordinate_systems")
            if frame_coordinate_systems is None:
                frame_coordinate_systems = frame_props.get("coordinate_systems")
            if frame_coordinate_systems is not None:
                metadata["frame_coordinate_systems"] = frame_coordinate_systems

            transforms = frame_data.get("transforms")
            if transforms is None:
                transforms = frame_props.get("transforms")
            if transforms is not None:
                metadata["transforms"] = transforms

            yield Sample(
                image_path=image_path,
                labels=labels,
                image_width=image_width,
                image_height=image_height,
                metadata=metadata,
            )

            self._report_progress(index, total, "Loading OpenLABEL")

    def load(self) -> Dataset:
        """Load full OpenLABEL dataset."""
        samples = list(self.iter_samples())
        return Dataset(
            samples,
            name=self.root_path.name,
            class_names=self.get_class_names(),
        )

    def validate(self) -> list[str]:
        """Validate OpenLABEL structure and references."""
        warnings: list[str] = []

        json_path = self._find_json_file()
        if json_path is None:
            warnings.append("No OpenLABEL JSON file found")
            return warnings

        try:
            self._ensure_parsed()
        except json.JSONDecodeError as exc:
            return [f"JSON parse error: {exc}"]
        except Exception as exc:
            return [str(exc)]

        if not self._frames:
            warnings.append("No frames found in OpenLABEL file")
        if not self._objects:
            warnings.append("No objects defined in OpenLABEL file")

        version = str(self._metadata.get("schema_version", "")).strip()
        if version and not version.startswith("1."):
            warnings.append(f"Untested schema version: {version}")

        if self.stream_filter:
            missing_streams = sorted(stream for stream in self.stream_filter if stream not in self._streams)
            if missing_streams:
                warnings.append(
                    "Requested stream_filter entries not found in dataset streams: "
                    + ", ".join(missing_streams)
                )

        missing_images = 0
        for frame_id, frame_data in self._iter_frames():
            if self._find_frame_image(frame_id, frame_data) is None:
                missing_images += 1
        if missing_images > 0:
            warnings.append(f"{missing_images} frames do not resolve to image files")

        warnings.extend(self._parse_warnings)
        return warnings

    def summary(self) -> dict[str, Any]:
        """Get OpenLABEL dataset summary."""
        self._ensure_parsed()
        base = super().summary()
        base.update(
            {
                "json_file": str(self._json_path) if self._json_path else None,
                "schema_version": self._metadata.get("schema_version", "unknown"),
                "num_frames": len(self._frames),
                "num_objects": len(self._objects),
                "num_streams": len(self._streams),
                "num_actions": len(self._actions),
                "num_events": len(self._events),
                "num_relations": len(self._relations),
            }
        )

        if self.stream_filter:
            base["stream_filter"] = sorted(self.stream_filter)

        return base
