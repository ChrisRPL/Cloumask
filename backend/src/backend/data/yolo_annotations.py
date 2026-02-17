"""Helpers for reading YOLO annotation files for review workflows."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_stem(stem: str) -> str:
    """
    Normalize generated YOLO export stems to source image stems.

    Detect export names may include an integer prefix such as:
    - 00000_frame_001
    which should map back to:
    - frame_001
    """
    prefix, sep, remainder = stem.partition("_")
    if sep and prefix.isdigit() and remainder:
        return remainder
    return stem


def _iter_candidate_roots(path: Path, max_levels: int = 3) -> list[Path]:
    roots: list[Path] = []
    current = path.parent if path.is_file() else path

    for _ in range(max_levels + 1):
        if current not in roots:
            roots.append(current)
        if current.parent == current:
            break
        current = current.parent

    return roots


def load_yolo_class_names(source_path: Path) -> list[str]:
    """Load class names from the nearest YOLO YAML config."""
    for root in _iter_candidate_roots(source_path):
        yaml_candidates = [root / "data.yaml", root / "data.yml"]
        yaml_candidates.extend(sorted(root.glob("*.yaml")))
        yaml_candidates.extend(sorted(root.glob("*.yml")))

        for yaml_path in yaml_candidates:
            if not yaml_path.exists() or not yaml_path.is_file():
                continue
            try:
                with yaml_path.open(encoding="utf-8") as file:
                    payload = yaml.safe_load(file) or {}
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping YAML %s: %s", yaml_path, exc)
                continue

            names = payload.get("names")
            if isinstance(names, list):
                return [str(name) for name in names]

            if isinstance(names, dict):
                parsed: dict[int, str] = {}
                for raw_key, raw_value in names.items():
                    try:
                        class_id = int(raw_key)
                    except (TypeError, ValueError):
                        continue
                    parsed[class_id] = str(raw_value)

                if not parsed:
                    continue

                max_id = max(parsed.keys())
                class_names = [f"class_{i}" for i in range(max_id + 1)]
                for class_id, class_name in parsed.items():
                    class_names[class_id] = class_name
                return class_names

    return []


def find_yolo_labels_dir(source_path: Path) -> Path | None:
    """Resolve a directory that contains YOLO `.txt` label files."""
    if source_path.is_file() and source_path.suffix == ".txt":
        source_dir = source_path.parent
    elif source_path.is_file():
        source_dir = source_path.parent
    else:
        source_dir = source_path

    if not source_dir.exists():
        return None

    direct_candidates = [
        source_dir,
        source_dir / "labels",
        source_dir / "train" / "labels",
        source_dir / "val" / "labels",
        source_dir / "test" / "labels",
    ]
    for candidate in direct_candidates:
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate

    # Fallback for nested exports.
    for pattern in ("*/labels", "*/*/labels"):
        for candidate in sorted(source_dir.glob(pattern)):
            if candidate.is_dir() and any(candidate.glob("*.txt")):
                return candidate

    return None


def build_annotation_index(
    labels_dir: Path,
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Build exact + normalized annotation indices for robust file matching."""
    exact: dict[str, Path] = {}
    normalized: dict[str, list[Path]] = defaultdict(list)

    for annotation_file in sorted(labels_dir.glob("*.txt")):
        stem = annotation_file.stem
        exact.setdefault(stem, annotation_file)
        normalized[_normalize_stem(stem)].append(annotation_file)

    return exact, dict(normalized)


def find_annotation_for_image(
    image_path: Path,
    exact_index: dict[str, Path],
    normalized_index: dict[str, list[Path]],
) -> Path | None:
    """Find best matching annotation file for an image path."""
    image_stem = image_path.stem

    exact = exact_index.get(image_stem)
    if exact:
        return exact

    normalized_candidates = normalized_index.get(image_stem, [])
    if normalized_candidates:
        return sorted(normalized_candidates)[0]

    return None


def parse_yolo_annotation_file(annotation_path: Path, class_names: list[str]) -> list[dict[str, Any]]:
    """
    Parse a YOLO `.txt` label file into review-friendly annotation payloads.

    Supports line formats:
    - `<class_id> <cx> <cy> <w> <h>`
    - `<class_id> <cx> <cy> <w> <h> <confidence>`
    """
    annotations: list[dict[str, Any]] = []

    try:
        with annotation_path.open(encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                parts = stripped.split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    confidence = float(parts[5]) if len(parts) > 5 else 1.0
                except ValueError:
                    continue

                class_name = (
                    class_names[class_id]
                    if 0 <= class_id < len(class_names) and class_names[class_id]
                    else f"class_{class_id}"
                )

                x = _clamp01(x_center - width / 2.0)
                y = _clamp01(y_center - height / 2.0)
                width = _clamp01(width)
                height = _clamp01(height)
                width = min(width, 1.0 - x)
                height = min(height, 1.0 - y)
                if width <= 0.0 or height <= 0.0:
                    continue

                annotations.append({
                    "label": class_name,
                    "confidence": _clamp01(confidence),
                    "bbox": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                    },
                })
    except Exception as exc:  # noqa: BLE001
        logger.error("Error parsing annotation file %s: %s", annotation_path, exc)

    return annotations
