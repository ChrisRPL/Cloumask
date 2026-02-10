"""Core data models for label representation.

All format loaders convert to these types. All exporters read from them.
Coordinates are normalized (0-1 range) relative to image dimensions.

Implements spec: 06-data-pipeline/01-data-models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import numpy as np


class BBoxFormat(str, Enum):
    """Bounding box coordinate formats."""

    XYXY = "xyxy"  # x1, y1, x2, y2 (top-left, bottom-right)
    XYWH = "xywh"  # x, y, width, height (top-left corner)
    CXCYWH = "cxcywh"  # center_x, center_y, width, height


@dataclass
class BBox:
    """Normalized bounding box (0-1 range).

    Internal format is CXCYWH (center-x, center-y, width, height).
    Use class methods to create from other formats.

    Attributes:
        cx: Center x coordinate (0-1)
        cy: Center y coordinate (0-1)
        w: Width (0-1)
        h: Height (0-1)
    """

    cx: float
    cy: float
    w: float
    h: float

    def __post_init__(self) -> None:
        """Validate coordinates are in 0-1 range."""
        for name, val in [("cx", self.cx), ("cy", self.cy), ("w", self.w), ("h", self.h)]:
            if not 0 <= val <= 1:
                # Clamp with warning instead of raising
                clamped = max(0.0, min(1.0, val))
                setattr(self, name, clamped)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        """Create from top-left/bottom-right corners."""
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return cls(cx=cx, cy=cy, w=w, h=h)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        """Create from top-left corner + dimensions."""
        cx = x + w / 2
        cy = y + h / 2
        return cls(cx=cx, cy=cy, w=w, h=h)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BBox:
        """Create from center + dimensions (native format)."""
        return cls(cx=cx, cy=cy, w=w, h=h)

    @classmethod
    def from_absolute(
        cls,
        coords: tuple[float, float, float, float],
        img_width: int,
        img_height: int,
        fmt: BBoxFormat = BBoxFormat.XYXY,
    ) -> BBox:
        """Create from absolute pixel coordinates."""
        if fmt == BBoxFormat.XYXY:
            x1, y1, x2, y2 = coords
            return cls.from_xyxy(
                x1 / img_width,
                y1 / img_height,
                x2 / img_width,
                y2 / img_height,
            )
        elif fmt == BBoxFormat.XYWH:
            x, y, w, h = coords
            return cls.from_xywh(
                x / img_width,
                y / img_height,
                w / img_width,
                h / img_height,
            )
        else:  # CXCYWH
            cx, cy, w, h = coords
            return cls.from_cxcywh(
                cx / img_width,
                cy / img_height,
                w / img_width,
                h / img_height,
            )

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        x1 = self.cx - self.w / 2
        y1 = self.cy - self.h / 2
        x2 = self.cx + self.w / 2
        y2 = self.cy + self.h / 2
        return (x1, y1, x2, y2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to (x, y, width, height) format."""
        x = self.cx - self.w / 2
        y = self.cy - self.h / 2
        return (x, y, self.w, self.h)

    def to_cxcywh(self) -> tuple[float, float, float, float]:
        """Convert to (center_x, center_y, width, height) format."""
        return (self.cx, self.cy, self.w, self.h)

    def to_absolute(
        self,
        img_width: int,
        img_height: int,
        fmt: BBoxFormat = BBoxFormat.XYXY,
    ) -> tuple[float, float, float, float]:
        """Convert to absolute pixel coordinates."""
        if fmt == BBoxFormat.XYXY:
            x1, y1, x2, y2 = self.to_xyxy()
            return (x1 * img_width, y1 * img_height, x2 * img_width, y2 * img_height)
        elif fmt == BBoxFormat.XYWH:
            x, y, w, h = self.to_xywh()
            return (x * img_width, y * img_height, w * img_width, h * img_height)
        else:  # CXCYWH
            return (
                self.cx * img_width,
                self.cy * img_height,
                self.w * img_width,
                self.h * img_height,
            )

    def area(self) -> float:
        """Compute normalized area."""
        return self.w * self.h

    def iou(self, other: BBox) -> float:
        """Compute Intersection over Union with another box."""
        x1_a, y1_a, x2_a, y2_a = self.to_xyxy()
        x1_b, y1_b, x2_b, y2_b = other.to_xyxy()

        # Intersection
        x1_i = max(x1_a, x1_b)
        y1_i = max(y1_a, y1_b)
        x2_i = min(x2_a, x2_b)
        y2_i = min(y2_a, y2_b)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = self.area() + other.area() - intersection

        return intersection / union if union > 0 else 0.0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"cx": self.cx, "cy": self.cy, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, data: dict) -> BBox:
        """Deserialize from dictionary."""
        return cls(cx=data["cx"], cy=data["cy"], w=data["w"], h=data["h"])


@dataclass
class Label:
    """A single annotation label.

    Attributes:
        class_name: Human-readable class name (e.g., "car", "person")
        class_id: Numeric class index (0-based)
        bbox: Normalized bounding box
        mask: Optional segmentation mask (H, W) binary array
        confidence: Prediction confidence (1.0 for ground truth)
        attributes: Additional metadata (occluded, truncated, etc.)
        track_id: Optional tracking ID for video sequences
    """

    class_name: str
    class_id: int
    bbox: BBox
    mask: Optional["np.ndarray"] = None
    confidence: float = 1.0
    attributes: dict = field(default_factory=dict)
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate label data."""
        if self.class_id < 0:
            raise ValueError(f"class_id must be >= 0, got {self.class_id}")
        if not 0 <= self.confidence <= 1:
            self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict:
        """Serialize to dictionary (mask excluded)."""
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "attributes": self.attributes,
            "track_id": self.track_id,
            "has_mask": self.mask is not None,
        }

    @classmethod
    def from_dict(cls, data: dict, mask: Optional["np.ndarray"] = None) -> Label:
        """Deserialize from dictionary."""
        return cls(
            class_name=data["class_name"],
            class_id=data["class_id"],
            bbox=BBox.from_dict(data["bbox"]),
            mask=mask,
            confidence=data.get("confidence", 1.0),
            attributes=data.get("attributes", {}),
            track_id=data.get("track_id"),
        )


@dataclass
class Sample:
    """A single image with its annotations.

    Attributes:
        image_path: Absolute path to image file
        labels: List of Label annotations
        image_width: Image width in pixels (optional, loaded lazily)
        image_height: Image height in pixels (optional, loaded lazily)
        metadata: Additional sample metadata (frame_id, timestamp, etc.)
    """

    image_path: Path
    labels: list[Label] = field(default_factory=list)
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure path is Path object."""
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)

    @property
    def num_labels(self) -> int:
        """Number of annotations."""
        return len(self.labels)

    @property
    def class_names(self) -> set[str]:
        """Unique class names in this sample."""
        return {lbl.class_name for lbl in self.labels}

    @property
    def has_labels(self) -> bool:
        """Whether sample has any annotations."""
        return len(self.labels) > 0

    def filter_by_class(self, class_names: Sequence[str]) -> Sample:
        """Return new Sample with only specified classes."""
        filtered = [lbl for lbl in self.labels if lbl.class_name in class_names]
        return Sample(
            image_path=self.image_path,
            labels=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
            metadata=self.metadata.copy(),
        )

    def filter_by_confidence(self, min_confidence: float) -> Sample:
        """Return new Sample with labels above confidence threshold."""
        filtered = [lbl for lbl in self.labels if lbl.confidence >= min_confidence]
        return Sample(
            image_path=self.image_path,
            labels=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "image_path": str(self.image_path),
            "labels": [lbl.to_dict() for lbl in self.labels],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Sample:
        """Deserialize from dictionary."""
        return cls(
            image_path=Path(data["image_path"]),
            labels=[Label.from_dict(lbl) for lbl in data.get("labels", [])],
            image_width=data.get("image_width"),
            image_height=data.get("image_height"),
            metadata=data.get("metadata", {}),
        )


class Dataset:
    """Collection of samples with metadata.

    Provides iteration, filtering, statistics, and split functionality.
    Uses lazy loading - samples are yielded, not all loaded at once.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        name: str = "dataset",
        class_names: Optional[list[str]] = None,
    ) -> None:
        """Initialize dataset.

        Args:
            samples: Sequence of Sample objects
            name: Dataset identifier
            class_names: Ordered list of class names (index = class_id)
        """
        self._samples = list(samples)
        self.name = name
        self._class_names = class_names

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]

    @property
    def samples(self) -> list[Sample]:
        """Access underlying sample list."""
        return self._samples

    @property
    def class_names(self) -> list[str]:
        """Get ordered class names. Infers from data if not set."""
        if self._class_names is not None:
            return self._class_names
        # Infer from samples
        names: dict[int, str] = {}
        for sample in self._samples:
            for label in sample.labels:
                if label.class_id not in names:
                    names[label.class_id] = label.class_name
        return [names[i] for i in sorted(names.keys())]

    @class_names.setter
    def class_names(self, names: list[str]) -> None:
        self._class_names = names

    @property
    def num_classes(self) -> int:
        """Number of unique classes."""
        return len(self.class_names)

    def class_distribution(self) -> dict[str, int]:
        """Count labels per class."""
        counts: dict[str, int] = {}
        for sample in self._samples:
            for label in sample.labels:
                counts[label.class_name] = counts.get(label.class_name, 0) + 1
        return counts

    def samples_per_class(self) -> dict[str, int]:
        """Count samples containing each class."""
        counts: dict[str, int] = {}
        for sample in self._samples:
            seen: set[str] = set()
            for label in sample.labels:
                if label.class_name not in seen:
                    counts[label.class_name] = counts.get(label.class_name, 0) + 1
                    seen.add(label.class_name)
        return counts

    def total_labels(self) -> int:
        """Total number of labels across all samples."""
        return sum(sample.num_labels for sample in self._samples)

    def unlabeled_samples(self) -> list[Sample]:
        """Samples with no annotations."""
        return [s for s in self._samples if not s.has_labels]

    def filter_by_class(self, class_names: Sequence[str]) -> Dataset:
        """Return new Dataset with only specified classes."""
        filtered = [s.filter_by_class(class_names) for s in self._samples]
        # Keep only samples that still have labels
        filtered = [s for s in filtered if s.has_labels]
        return Dataset(filtered, name=self.name, class_names=list(class_names))

    def subset(self, indices: Sequence[int]) -> Dataset:
        """Return new Dataset with only specified sample indices."""
        samples = [self._samples[i] for i in indices]
        return Dataset(samples, name=self.name, class_names=self._class_names)

    def merge(self, other: Dataset) -> Dataset:
        """Merge with another dataset."""
        combined = self._samples + other._samples

        # Merge class names preserving order. Use resolved class names so datasets
        # without explicit metadata still contribute inferred labels.
        names: list[str] = []
        for class_name in self.class_names:
            if class_name not in names:
                names.append(class_name)
        for class_name in other.class_names:
            if class_name not in names:
                names.append(class_name)

        return Dataset(
            combined,
            name=f"{self.name}+{other.name}",
            class_names=names or None,
        )

    def stats(self) -> dict:
        """Compute dataset statistics."""
        dist = self.class_distribution()
        return {
            "name": self.name,
            "num_samples": len(self._samples),
            "num_labels": self.total_labels(),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "class_distribution": dist,
            "samples_per_class": self.samples_per_class(),
            "unlabeled_count": len(self.unlabeled_samples()),
            "avg_labels_per_sample": self.total_labels() / max(len(self._samples), 1),
        }

    def to_dict(self) -> dict:
        """Serialize dataset to dictionary."""
        return {
            "name": self.name,
            "class_names": self.class_names,
            "samples": [s.to_dict() for s in self._samples],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Dataset:
        """Deserialize dataset from dictionary."""
        samples = [Sample.from_dict(s) for s in data.get("samples", [])]
        return cls(
            samples=samples,
            name=data.get("name", "dataset"),
            class_names=data.get("class_names"),
        )


# Pydantic models for API serialization


class BBoxSchema(BaseModel):
    """Pydantic schema for BBox serialization."""

    model_config = ConfigDict(from_attributes=True)

    cx: float
    cy: float
    w: float
    h: float


class LabelSchema(BaseModel):
    """Pydantic schema for Label serialization (without mask)."""

    model_config = ConfigDict(from_attributes=True)

    class_name: str
    class_id: int
    bbox: BBoxSchema
    confidence: float = 1.0
    attributes: dict = {}
    track_id: Optional[int] = None


class SampleSchema(BaseModel):
    """Pydantic schema for Sample serialization."""

    model_config = ConfigDict(from_attributes=True)

    image_path: str
    labels: list[LabelSchema] = []
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    metadata: dict = {}


class DatasetStatsSchema(BaseModel):
    """Pydantic schema for dataset statistics."""

    name: str
    num_samples: int
    num_labels: int
    num_classes: int
    class_names: list[str]
    class_distribution: dict[str, int]
    samples_per_class: dict[str, int]
    unlabeled_count: int
    avg_labels_per_sample: float
