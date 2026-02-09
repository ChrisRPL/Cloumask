"""nuScenes format loader.

Supports nuScenes autonomous driving dataset format:
- Table-based JSON metadata
- Multi-camera sample data
- 3D annotation attributes with optional 2D boxes

Implements spec: 06-data-pipeline/08-nuscenes-loader
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from backend.data.formats.base import FormatLoader, FormatRegistry, ProgressCallback
from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)


NUSCENES_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

_REQUIRED_TABLES = ("sample", "sample_data", "sample_annotation", "category")


@FormatRegistry.register_loader
class NuscenesLoader(FormatLoader):
    """Load nuScenes format datasets.

    Expects:
    - a `v1.0-*` directory with nuScenes table JSON files (or tables in root)
    - `samples/` directory with camera image files

    Notes:
    - nuScenes stores canonical 3D boxes in `sample_annotation`.
    - 2D boxes are not part of the core nuScenes schema, so this loader uses
      extension keys when present (`bbox`, `bbox_2d`, `box2d`) and otherwise
      falls back to a placeholder box while preserving 3D metadata.
    """

    format_name = "nuscenes"
    description = "nuScenes format (autonomous driving)"
    extensions = [".json"]

    def __init__(
        self,
        root_path: Path,
        *,
        class_names: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        version: str = "v1.0-trainval",
        cameras: list[str] | None = None,
        with_3d: bool = True,
    ) -> None:
        """Initialize nuScenes loader.

        Args:
            root_path: Dataset root directory
            class_names: Override class names
            progress_callback: Progress callback
            version: Preferred dataset version (`v1.0-mini`, `v1.0-trainval`, ...)
            cameras: Cameras to include (defaults to all nuScenes cameras)
            with_3d: Include 3D annotation data in label attributes
        """
        super().__init__(root_path, class_names=class_names, progress_callback=progress_callback)
        self.version = version
        self.cameras = cameras or NUSCENES_CAMERAS.copy()
        self.with_3d = with_3d

        self._version_dir: Path | None = None
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._indexes_built = False

        self._category_map: dict[str, dict[str, Any]] = {}
        self._attribute_map: dict[str, dict[str, Any]] = {}
        self._visibility_map: dict[str, dict[str, Any]] = {}
        self._instance_map: dict[str, dict[str, Any]] = {}
        self._sample_map: dict[str, dict[str, Any]] = {}
        self._sample_data_map: dict[str, dict[str, Any]] = {}
        self._sensor_map: dict[str, dict[str, Any]] = {}
        self._calibration_map: dict[str, dict[str, Any]] = {}
        self._ego_pose_map: dict[str, dict[str, Any]] = {}

        self._sample_data_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._annotations_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._track_id_map: dict[str, int] = {}

    def _get_version_dir(self) -> Path | None:
        """Resolve the directory containing nuScenes table files."""
        if self._version_dir is not None:
            return self._version_dir

        preferred = self.root_path / self.version
        if preferred.is_dir():
            self._version_dir = preferred
            return self._version_dir

        candidates = sorted(
            (
                subdir
                for subdir in self.root_path.iterdir()
                if subdir.is_dir() and subdir.name.startswith("v1.0")
            ),
            key=lambda item: item.name,
        )
        if candidates:
            self._version_dir = candidates[0]
            return self._version_dir

        if (self.root_path / "sample.json").exists():
            self._version_dir = self.root_path
            return self._version_dir

        return None

    def _load_table(self, name: str) -> list[dict[str, Any]]:
        """Load a nuScenes table JSON as a list of records."""
        if name in self._tables:
            return self._tables[name]

        version_dir = self._get_version_dir()
        if version_dir is None:
            self._tables[name] = []
            return self._tables[name]

        table_path = version_dir / f"{name}.json"
        if not table_path.exists():
            logger.debug("nuScenes table not found: %s", table_path)
            self._tables[name] = []
            return self._tables[name]

        try:
            with table_path.open() as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse nuScenes table %s: %s", table_path, exc)
            self._tables[name] = []
            return self._tables[name]

        if not isinstance(payload, list):
            logger.warning("Expected list table in %s, got %s", table_path, type(payload).__name__)
            self._tables[name] = []
            return self._tables[name]

        self._tables[name] = payload
        return self._tables[name]

    def _build_indexes(self) -> None:
        """Build token-based lookup indexes for table joins."""
        if self._indexes_built:
            return

        self._category_map = {
            str(row.get("token")): row for row in self._load_table("category") if row.get("token")
        }
        self._attribute_map = {
            str(row.get("token")): row for row in self._load_table("attribute") if row.get("token")
        }
        self._visibility_map = {
            str(row.get("token")): row for row in self._load_table("visibility") if row.get("token")
        }
        self._instance_map = {
            str(row.get("token")): row for row in self._load_table("instance") if row.get("token")
        }
        self._sample_map = {
            str(row.get("token")): row for row in self._load_table("sample") if row.get("token")
        }
        self._sample_data_map = {
            str(row.get("token")): row for row in self._load_table("sample_data") if row.get("token")
        }
        self._sensor_map = {
            str(row.get("token")): row for row in self._load_table("sensor") if row.get("token")
        }
        self._calibration_map = {
            str(row.get("token")): row
            for row in self._load_table("calibrated_sensor")
            if row.get("token")
        }
        self._ego_pose_map = {
            str(row.get("token")): row for row in self._load_table("ego_pose") if row.get("token")
        }

        for sample_data in self._sample_data_map.values():
            sample_token = str(sample_data.get("sample_token", ""))
            if sample_token:
                self._sample_data_by_sample[sample_token].append(sample_data)

        for annotation in self._load_table("sample_annotation"):
            sample_token = str(annotation.get("sample_token", ""))
            if sample_token:
                self._annotations_by_sample[sample_token].append(annotation)

        for index, token in enumerate(sorted(self._instance_map.keys()), start=1):
            self._track_id_map[token] = index

        self._indexes_built = True

    def _infer_class_names(self) -> list[str]:
        """Infer class names from `category` table."""
        categories = self._load_table("category")
        names = {
            str(category.get("name")).strip()
            for category in categories
            if category.get("name") is not None and str(category.get("name")).strip()
        }
        return sorted(names)

    def _get_sensor_channel(self, sample_data: dict[str, Any]) -> str | None:
        """Resolve camera channel for a `sample_data` row."""
        cal_token = str(sample_data.get("calibrated_sensor_token", ""))
        if not cal_token:
            return None

        calibration = self._calibration_map.get(cal_token)
        if calibration is None:
            return None

        sensor_token = str(calibration.get("sensor_token", ""))
        if not sensor_token:
            return None

        sensor = self._sensor_map.get(sensor_token)
        if sensor is None:
            return None

        channel = str(sensor.get("channel", "")).strip()
        if not channel:
            return None

        modality = str(sensor.get("modality", "")).strip().lower()
        if modality and modality != "camera":
            return None

        return channel

    def _get_camera_sample_data(self, sample_token: str) -> dict[str, dict[str, Any]]:
        """Get per-camera `sample_data` records for a sample token."""
        camera_records: dict[str, dict[str, Any]] = {}

        for sample_data in self._sample_data_by_sample.get(sample_token, []):
            channel = self._get_sensor_channel(sample_data)
            if channel is None or channel not in self.cameras:
                continue

            existing = camera_records.get(channel)
            if existing is None:
                camera_records[channel] = sample_data
                continue

            # Prefer keyframe records if multiple entries exist for a channel.
            existing_keyframe = bool(existing.get("is_key_frame", False))
            current_keyframe = bool(sample_data.get("is_key_frame", False))
            if current_keyframe and not existing_keyframe:
                camera_records[channel] = sample_data

        return camera_records

    def _resolve_image_path(self, filename: str, camera: str) -> Path:
        """Resolve an image path from a nuScenes `filename` field."""
        path = Path(filename)
        candidates = [
            self.root_path / path,
            self.root_path / "samples" / camera / path.name,
            self.root_path / "samples" / camera / filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _bbox_from_xywh(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        img_width: int,
        img_height: int,
    ) -> BBox | None:
        """Build BBox from xywh values, auto-detecting normalized input."""
        if w <= 0 or h <= 0:
            return None

        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
            return BBox.from_xywh(x, y, w, h)

        if img_width <= 0 or img_height <= 0:
            return None

        return BBox.from_xywh(x / img_width, y / img_height, w / img_width, h / img_height)

    def _bbox_from_xyxy(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        img_width: int,
        img_height: int,
    ) -> BBox | None:
        """Build BBox from xyxy values, auto-detecting normalized input."""
        if x2 <= x1 or y2 <= y1:
            return None

        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
            return BBox.from_xyxy(x1, y1, x2, y2)

        if img_width <= 0 or img_height <= 0:
            return None

        return BBox.from_xyxy(x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height)

    def _extract_bbox_2d(
        self,
        annotation: dict[str, Any],
        img_width: int,
        img_height: int,
    ) -> BBox | None:
        """Extract 2D bbox from extension fields when available."""
        for key in ("bbox_2d", "box2d", "box_2d", "bbox"):
            raw = annotation.get(key)
            if raw is None:
                continue

            if isinstance(raw, dict):
                lower = {str(k).lower(): v for k, v in raw.items()}
                if {"x", "y", "w", "h"} <= set(lower):
                    return self._bbox_from_xywh(
                        float(lower["x"]),
                        float(lower["y"]),
                        float(lower["w"]),
                        float(lower["h"]),
                        img_width,
                        img_height,
                    )
                if {"xmin", "ymin", "xmax", "ymax"} <= set(lower):
                    return self._bbox_from_xyxy(
                        float(lower["xmin"]),
                        float(lower["ymin"]),
                        float(lower["xmax"]),
                        float(lower["ymax"]),
                        img_width,
                        img_height,
                    )
                if {"x1", "y1", "x2", "y2"} <= set(lower):
                    return self._bbox_from_xyxy(
                        float(lower["x1"]),
                        float(lower["y1"]),
                        float(lower["x2"]),
                        float(lower["y2"]),
                        img_width,
                        img_height,
                    )
                continue

            if isinstance(raw, (list, tuple)) and len(raw) >= 4:
                values = [float(value) for value in raw[:4]]
                return self._bbox_from_xywh(
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    img_width,
                    img_height,
                )

        return None

    def _annotation_class_name(self, annotation: dict[str, Any]) -> str:
        """Resolve class name from annotation and instance/category joins."""
        category_token = str(annotation.get("category_token", ""))

        if not category_token:
            instance_token = str(annotation.get("instance_token", ""))
            if instance_token:
                instance = self._instance_map.get(instance_token, {})
                category_token = str(instance.get("category_token", ""))

        category = self._category_map.get(category_token, {})
        class_name = str(category.get("name", "")).strip()
        return class_name or "unknown"

    def _annotation_track_id(self, annotation: dict[str, Any]) -> int | None:
        """Resolve deterministic numeric track ID from instance token."""
        instance_token = str(annotation.get("instance_token", ""))
        if not instance_token:
            return None

        track_id = self._track_id_map.get(instance_token)
        if track_id is not None:
            return track_id

        track_id = len(self._track_id_map) + 1
        self._track_id_map[instance_token] = track_id
        return track_id

    def _annotation_attributes(self, annotation: dict[str, Any]) -> dict[str, Any]:
        """Build label attributes dictionary from nuScenes fields."""
        attributes: dict[str, Any] = {}

        for attr_token in annotation.get("attribute_tokens", []):
            attr = self._attribute_map.get(str(attr_token), {})
            attr_name = str(attr.get("name", "")).strip()
            if attr_name:
                attributes[attr_name] = True

        visibility_token = str(annotation.get("visibility_token", ""))
        if visibility_token:
            visibility = self._visibility_map.get(visibility_token, {})
            if visibility:
                level = visibility.get("level", visibility.get("description"))
                if level is not None:
                    attributes["visibility"] = level

        if "num_lidar_pts" in annotation:
            attributes["num_lidar_pts"] = annotation.get("num_lidar_pts")
        if "num_radar_pts" in annotation:
            attributes["num_radar_pts"] = annotation.get("num_radar_pts")

        if self.with_3d:
            attributes["translation_3d"] = annotation.get("translation", [0.0, 0.0, 0.0])
            attributes["size_3d"] = annotation.get("size", [0.0, 0.0, 0.0])
            attributes["rotation_3d"] = annotation.get("rotation", [1.0, 0.0, 0.0, 0.0])
            if "velocity" in annotation:
                attributes["velocity_3d"] = annotation.get("velocity")

        return attributes

    def _get_annotations_for_sample(
        self,
        sample_token: str,
        img_width: int,
        img_height: int,
    ) -> list[Label]:
        """Get labels for a sample token."""
        class_names = self.get_class_names()
        labels: list[Label] = []

        for annotation in self._annotations_by_sample.get(sample_token, []):
            class_name = self._annotation_class_name(annotation)
            class_id = class_names.index(class_name) if class_name in class_names else len(class_names)

            bbox = self._extract_bbox_2d(annotation, img_width, img_height)
            attributes = self._annotation_attributes(annotation)
            if bbox is None:
                bbox = BBox(cx=0.5, cy=0.5, w=0.1, h=0.1)
                attributes["bbox_source"] = "placeholder"
            else:
                attributes["bbox_source"] = "provided_2d"

            labels.append(
                Label(
                    class_name=class_name,
                    class_id=class_id,
                    bbox=bbox,
                    attributes=attributes,
                    track_id=self._annotation_track_id(annotation),
                )
            )

        return labels

    def iter_samples(self) -> Iterator[Sample]:
        """Iterate over all camera samples in the dataset."""
        self._build_indexes()
        samples = sorted(
            self._sample_map.values(),
            key=lambda sample: (int(sample.get("timestamp", 0)), str(sample.get("token", ""))),
        )

        total = sum(
            len(self._get_camera_sample_data(str(sample.get("token", "")))) for sample in samples
        )
        processed = 0

        for sample in samples:
            sample_token = str(sample.get("token", ""))
            if not sample_token:
                continue

            camera_data = self._get_camera_sample_data(sample_token)
            for camera in self.cameras:
                sample_data = camera_data.get(camera)
                if sample_data is None:
                    continue

                filename = str(sample_data.get("filename", "")).strip()
                if not filename:
                    logger.warning("Missing filename for sample_data token %s", sample_data.get("token"))
                    continue

                image_path = self._resolve_image_path(filename, camera)
                img_width = int(sample_data.get("width", 1600) or 1600)
                img_height = int(sample_data.get("height", 900) or 900)
                labels = self._get_annotations_for_sample(sample_token, img_width, img_height)

                metadata: dict[str, Any] = {
                    "sample_token": sample_token,
                    "sample_data_token": str(sample_data.get("token", "")),
                    "camera": camera,
                    "timestamp": sample.get("timestamp", sample_data.get("timestamp", 0)),
                    "is_key_frame": bool(sample_data.get("is_key_frame", True)),
                }

                if sample.get("scene_token"):
                    metadata["scene_token"] = sample.get("scene_token")
                if sample_data.get("prev"):
                    metadata["sample_data_prev"] = sample_data.get("prev")
                if sample_data.get("next"):
                    metadata["sample_data_next"] = sample_data.get("next")

                cal_token = str(sample_data.get("calibrated_sensor_token", ""))
                if cal_token:
                    metadata["calibrated_sensor_token"] = cal_token

                ego_pose_token = str(sample_data.get("ego_pose_token", ""))
                if ego_pose_token:
                    metadata["ego_pose_token"] = ego_pose_token
                    ego_pose = self._ego_pose_map.get(ego_pose_token, {})
                    if ego_pose:
                        metadata["ego_pose"] = {
                            "translation": ego_pose.get("translation"),
                            "rotation": ego_pose.get("rotation"),
                        }

                yield Sample(
                    image_path=image_path,
                    labels=labels,
                    image_width=img_width,
                    image_height=img_height,
                    metadata=metadata,
                )

                processed += 1
                self._report_progress(processed, total, f"Loading {camera}")

    def load(self) -> Dataset:
        """Load the full dataset into memory."""
        samples = list(self.iter_samples())
        return Dataset(samples, name=self.root_path.name, class_names=self.get_class_names())

    def validate(self) -> list[str]:
        """Validate dataset structure and return warnings."""
        warnings: list[str] = []
        version_dir = self._get_version_dir()
        if version_dir is None:
            warnings.append(f"No version directory found (expected {self.version} or any v1.0-*)")
            return warnings

        for table in _REQUIRED_TABLES:
            if not (version_dir / f"{table}.json").exists():
                warnings.append(f"Missing table: {table}.json")

        samples_dir = self.root_path / "samples"
        if not samples_dir.exists():
            warnings.append("No samples/ directory found")
        else:
            for camera in self.cameras:
                if not (samples_dir / camera).exists():
                    warnings.append(f"Camera directory not found: {camera}")

        self._build_indexes()
        if self._sample_map and not self._sample_data_map:
            warnings.append("No sample_data records found")

        has_camera_data = False
        for sample_token in self._sample_map:
            if self._get_camera_sample_data(sample_token):
                has_camera_data = True
                break
        if self._sample_map and not has_camera_data:
            warnings.append("No camera sample_data records found for selected cameras")

        return warnings

    def summary(self) -> dict[str, Any]:
        """Get dataset summary with nuScenes-specific metadata."""
        self._build_indexes()
        base = super().summary()
        base.update(
            {
                "version": self._get_version_dir().name if self._get_version_dir() else None,
                "requested_version": self.version,
                "cameras": self.cameras,
                "with_3d": self.with_3d,
                "num_samples": len(self._sample_map),
                "num_sample_data": len(self._sample_data_map),
                "num_annotations": len(self._load_table("sample_annotation")),
                "num_instances": len(self._instance_map),
                "num_categories": len(self._category_map),
            }
        )
        return base
