"""Dataset augmentation utilities backed by Albumentations.

Implements spec: 06-data-pipeline/20-augmentation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.data.models import BBox, Dataset, Label, Sample

logger = logging.getLogger(__name__)

try:
    import albumentations as A  # type: ignore[import-not-found]

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without optional dependency
    A = None
    ALBUMENTATIONS_AVAILABLE = False

_PRESET_NAMES: tuple[str, ...] = ("light", "medium", "heavy", "geometric", "photometric")


def _require_albumentations() -> None:
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "Albumentations is not installed. Install with `pip install albumentations` "
            "or include cv dependencies."
        )


def _bbox_params() -> Any:
    _require_albumentations()
    return A.BboxParams(
        format="yolo",
        label_fields=["bbox_indices"],
        min_visibility=0.01,
    )


@dataclass
class AugmentedSample:
    """Result of augmenting a sample."""

    image: np.ndarray
    labels: list[Label]
    original: Sample
    transform_name: str


def available_presets() -> tuple[str, ...]:
    """Return supported preset names."""
    return _PRESET_NAMES


def get_preset_pipeline(preset: str) -> Any:
    """Get a preset Albumentations pipeline."""
    _require_albumentations()

    normalized = preset.strip().lower()
    if normalized not in _PRESET_NAMES:
        raise ValueError(f"Unknown preset: {preset}. Available presets: {', '.join(_PRESET_NAMES)}")

    if normalized == "light":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        ]
    elif normalized == "medium":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12, rotate_limit=12, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.2,
            ),
            A.CLAHE(clip_limit=2.0, p=0.2),
        ]
    elif normalized == "heavy":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=25, p=0.7),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=0.35,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08, p=0.3),
        ]
    elif normalized == "geometric":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.18, scale_limit=0.2, rotate_limit=20, p=0.7),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.Affine(shear=(-8, 8), p=0.25),
        ]
    else:  # photometric
        transforms = [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.1),
        ]

    return A.Compose(transforms, bbox_params=_bbox_params())


def _clone_label_with_bbox(source: Label, bbox: BBox) -> Label:
    """Create a label copy with updated bbox."""
    return Label(
        class_name=source.class_name,
        class_id=source.class_id,
        bbox=bbox,
        mask=None,
        confidence=source.confidence,
        attributes=source.attributes.copy(),
        track_id=source.track_id,
    )


class AugmentationPipeline:
    """Apply Albumentations transforms to samples."""

    def __init__(
        self,
        preset: str | None = None,
        transform: Any | None = None,
    ) -> None:
        _require_albumentations()
        if transform is not None:
            self.transform = transform
            self.name = "custom"
        else:
            selected = preset or "medium"
            self.transform = get_preset_pipeline(selected)
            self.name = selected

    def _load_image(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as img:
            return np.array(img.convert("RGB"))

    def _labels_to_albumentations(
        self,
        labels: list[Label],
    ) -> tuple[list[tuple[float, float, float, float]], list[int]]:
        bboxes: list[tuple[float, float, float, float]] = []
        bbox_indices: list[int] = []

        for idx, label in enumerate(labels):
            bboxes.append(label.bbox.to_cxcywh())
            bbox_indices.append(idx)

        return bboxes, bbox_indices

    def _albumentations_to_labels(
        self,
        bboxes: list[tuple[float, float, float, float]],
        bbox_indices: list[int],
        original_labels: list[Label],
    ) -> list[Label]:
        new_labels: list[Label] = []

        for bbox_values, index in zip(bboxes, bbox_indices, strict=False):
            original = original_labels[int(index)]
            cx, cy, width, height = bbox_values
            new_bbox = BBox.from_cxcywh(cx, cy, width, height)
            new_labels.append(_clone_label_with_bbox(original, new_bbox))

        return new_labels

    def augment(self, sample: Sample) -> AugmentedSample:
        """Apply augmentation to a sample."""
        image = self._load_image(sample.image_path)
        bboxes, bbox_indices = self._labels_to_albumentations(sample.labels)

        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            bbox_indices=bbox_indices,
        )
        transformed_bboxes = transformed.get("bboxes", [])
        transformed_indices = transformed.get("bbox_indices", [])

        labels = self._albumentations_to_labels(
            transformed_bboxes,
            transformed_indices,
            sample.labels,
        )

        return AugmentedSample(
            image=transformed["image"],
            labels=labels,
            original=sample,
            transform_name=self.name,
        )

    def preview(self, sample: Sample, n: int = 5) -> list[AugmentedSample]:
        """Generate multiple augmented variations of a single sample."""
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        return [self.augment(sample) for _ in range(n)]


def augment_sample(sample: Sample, preset: str = "medium") -> AugmentedSample:
    """Augment a single sample using a preset pipeline."""
    pipeline = AugmentationPipeline(preset=preset)
    return pipeline.augment(sample)


def preview_augmentations(sample: Sample, preset: str = "medium", n: int = 5) -> list[AugmentedSample]:
    """Generate `n` preview augmentations for a sample."""
    pipeline = AugmentationPipeline(preset=preset)
    return pipeline.preview(sample, n=n)


def _augmented_image_name(sample: Sample, preset: str, sample_index: int, copy_index: int) -> str:
    suffix = sample.image_path.suffix if sample.image_path.suffix else ".jpg"
    stem = sample.image_path.stem if sample.image_path.stem else "image"
    return f"{sample_index:06d}_{stem}__aug_{preset}_{copy_index:02d}{suffix}"


def _write_augmented_image(image: np.ndarray, output_path: Path) -> None:
    image_uint8 = image.astype(np.uint8, copy=False)
    Image.fromarray(image_uint8).save(output_path)


def augment_dataset(
    dataset: Dataset,
    *,
    output_dir: Path,
    preset: str = "medium",
    copies_per_sample: int = 1,
    include_original: bool = True,
) -> Dataset:
    """Create an augmented dataset and write transformed images to ``output_dir``."""
    if copies_per_sample < 1:
        raise ValueError(f"copies_per_sample must be >= 1, got {copies_per_sample}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = AugmentationPipeline(preset=preset)
    class_names = list(dataset.class_names)
    generated_samples: list[Sample] = []

    for sample_index, sample in enumerate(dataset):
        if include_original:
            generated_samples.append(sample)

        for copy_index in range(1, copies_per_sample + 1):
            augmented = pipeline.augment(sample)
            augmented_path = output_dir / _augmented_image_name(
                sample,
                preset=preset,
                sample_index=sample_index,
                copy_index=copy_index,
            )
            _write_augmented_image(augmented.image, augmented_path)

            image_height, image_width = augmented.image.shape[:2]
            metadata = sample.metadata.copy()
            metadata["augmentation"] = {
                "preset": preset,
                "copy_index": copy_index,
                "source_image": str(sample.image_path),
            }

            generated_samples.append(
                Sample(
                    image_path=augmented_path,
                    labels=augmented.labels,
                    image_width=image_width,
                    image_height=image_height,
                    metadata=metadata,
                )
            )

    return Dataset(
        generated_samples,
        name=f"{dataset.name}_augmented_{preset}",
        class_names=class_names,
    )


__all__ = [
    "ALBUMENTATIONS_AVAILABLE",
    "AugmentedSample",
    "AugmentationPipeline",
    "augment_dataset",
    "augment_sample",
    "available_presets",
    "get_preset_pipeline",
    "preview_augmentations",
]
