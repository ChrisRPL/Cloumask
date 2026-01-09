# Augmentation

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models
> **Blocks:** None (optional feature)

## Objective

Implement data augmentation using Albumentations library with preset pipelines and preview functionality.

## Acceptance Criteria

- [ ] Integrate Albumentations transforms
- [ ] Preset augmentation pipelines (light, medium, heavy)
- [ ] Bbox-safe transforms (adjust coordinates)
- [ ] Preview augmented samples
- [ ] Apply augmentation during export
- [ ] Unit tests for transform application

## Implementation Steps

### 1. Create augmentation.py

Create `backend/data/augmentation.py`:

```python
"""Data augmentation using Albumentations.

Provides preset pipelines and bbox-safe transforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from backend.data.models import BBox, Label, Sample

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    from albumentations.core.composition import BboxParams
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.warning("Albumentations not installed. Install with: pip install albumentations")


@dataclass
class AugmentedSample:
    """Result of augmenting a sample.

    Attributes:
        image: Augmented image as numpy array
        labels: Updated labels with transformed bboxes
        original: Reference to original sample
        transform_name: Name of applied transform
    """
    image: np.ndarray
    labels: list[Label]
    original: Sample
    transform_name: str


def get_preset_pipeline(preset: str) -> "A.Compose":
    """Get a preset augmentation pipeline.

    Args:
        preset: One of "light", "medium", "heavy", "geometric", "photometric"

    Returns:
        Albumentations Compose pipeline
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Albumentations not installed")

    if preset == "light":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(var_limit=(5, 25), p=0.3),
        ], bbox_params=BboxParams(format="yolo", label_fields=["class_labels"]))

    elif preset == "medium":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),
        ], bbox_params=BboxParams(format="yolo", label_fields=["class_labels"]))

    elif preset == "heavy":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.MedianBlur(blur_limit=5, p=1),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(20, 80), p=1),
                A.ISONoise(color_shift=(0.01, 0.05), p=1),
            ], p=0.3),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        ], bbox_params=BboxParams(format="yolo", label_fields=["class_labels"]))

    elif preset == "geometric":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.7),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.Affine(shear=(-10, 10), p=0.3),
        ], bbox_params=BboxParams(format="yolo", label_fields=["class_labels"]))

    elif preset == "photometric":
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.ToGray(p=0.1),
            A.ChannelShuffle(p=0.1),
        ], bbox_params=BboxParams(format="yolo", label_fields=["class_labels"]))

    else:
        raise ValueError(f"Unknown preset: {preset}. Use: light, medium, heavy, geometric, photometric")


class AugmentationPipeline:
    """Apply augmentations to samples.

    Example:
        pipeline = AugmentationPipeline(preset="medium")
        augmented = pipeline.augment(sample)
        # augmented.image is the transformed image
        # augmented.labels has updated bbox coordinates
    """

    def __init__(
        self,
        preset: Optional[str] = None,
        transform: Optional["A.Compose"] = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            preset: Use preset pipeline (light, medium, heavy)
            transform: Custom Albumentations Compose (overrides preset)
        """
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("Albumentations not installed")

        if transform is not None:
            self.transform = transform
            self.name = "custom"
        elif preset is not None:
            self.transform = get_preset_pipeline(preset)
            self.name = preset
        else:
            self.transform = get_preset_pipeline("medium")
            self.name = "medium"

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as numpy array."""
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def _labels_to_albumentations(
        self,
        labels: list[Label],
    ) -> tuple[list[list[float]], list[str]]:
        """Convert labels to Albumentations format.

        Returns:
            (bboxes in YOLO format, class labels)
        """
        bboxes = []
        class_labels = []

        for label in labels:
            cx, cy, w, h = label.bbox.to_cxcywh()
            bboxes.append([cx, cy, w, h])
            class_labels.append(label.class_name)

        return bboxes, class_labels

    def _albumentations_to_labels(
        self,
        bboxes: list[list[float]],
        class_labels: list[str],
        original_labels: list[Label],
    ) -> list[Label]:
        """Convert Albumentations output back to Labels."""
        new_labels = []

        for i, (bbox, class_name) in enumerate(zip(bboxes, class_labels)):
            cx, cy, w, h = bbox
            new_bbox = BBox.from_cxcywh(cx, cy, w, h)

            # Preserve original attributes
            if i < len(original_labels):
                orig = original_labels[i]
                new_labels.append(Label(
                    class_name=class_name,
                    class_id=orig.class_id,
                    bbox=new_bbox,
                    mask=None,  # Masks need separate handling
                    confidence=orig.confidence,
                    attributes=orig.attributes.copy(),
                    track_id=orig.track_id,
                ))
            else:
                # Shouldn't happen, but handle gracefully
                new_labels.append(Label(
                    class_name=class_name,
                    class_id=0,
                    bbox=new_bbox,
                ))

        return new_labels

    def augment(self, sample: Sample) -> AugmentedSample:
        """Apply augmentation to a sample.

        Args:
            sample: Sample to augment

        Returns:
            AugmentedSample with transformed image and labels
        """
        # Load image
        image = self._load_image(sample.image_path)

        # Convert labels
        bboxes, class_labels = self._labels_to_albumentations(sample.labels)

        # Apply transform
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels,
        )

        # Convert back
        new_labels = self._albumentations_to_labels(
            transformed["bboxes"],
            transformed["class_labels"],
            sample.labels,
        )

        return AugmentedSample(
            image=transformed["image"],
            labels=new_labels,
            original=sample,
            transform_name=self.name,
        )

    def preview(
        self,
        sample: Sample,
        n: int = 5,
    ) -> list[AugmentedSample]:
        """Generate multiple augmentation previews.

        Args:
            sample: Sample to preview
            n: Number of variations to generate

        Returns:
            List of augmented samples
        """
        return [self.augment(sample) for _ in range(n)]


def augment_sample(
    sample: Sample,
    preset: str = "medium",
) -> AugmentedSample:
    """Convenience function to augment a single sample.

    Args:
        sample: Sample to augment
        preset: Augmentation preset

    Returns:
        AugmentedSample
    """
    pipeline = AugmentationPipeline(preset=preset)
    return pipeline.augment(sample)


def preview_augmentations(
    sample: Sample,
    preset: str = "medium",
    n: int = 5,
) -> list[AugmentedSample]:
    """Generate augmentation previews.

    Args:
        sample: Sample to preview
        preset: Augmentation preset
        n: Number of previews

    Returns:
        List of augmented samples
    """
    pipeline = AugmentationPipeline(preset=preset)
    return pipeline.preview(sample, n)
```

### 2. Create unit tests

Create `backend/tests/data/test_augmentation.py`:

```python
"""Tests for augmentation."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.data.models import BBox, Label, Sample


@pytest.fixture
def sample_with_image(tmp_path):
    """Create a sample with actual image."""
    img = Image.new("RGB", (640, 480), color="red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    return Sample(
        image_path=img_path,
        labels=[
            Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2)),
            Label(class_name="person", class_id=1, bbox=BBox(0.3, 0.4, 0.1, 0.3)),
        ],
        image_width=640,
        image_height=480,
    )


class TestAugmentation:
    """Tests for augmentation pipeline."""

    @pytest.mark.skipif(not True, reason="Albumentations required")
    def test_import(self):
        """Test module imports."""
        try:
            from backend.data.augmentation import AugmentationPipeline
            assert True
        except ImportError:
            pytest.skip("Albumentations not installed")

    @pytest.mark.skipif(not True, reason="Albumentations required")
    def test_augment_sample(self, sample_with_image):
        """Test augmenting a sample."""
        try:
            from backend.data.augmentation import AugmentationPipeline

            pipeline = AugmentationPipeline(preset="light")
            result = pipeline.augment(sample_with_image)

            assert result.image is not None
            assert isinstance(result.image, np.ndarray)
            assert len(result.labels) <= len(sample_with_image.labels)
        except ImportError:
            pytest.skip("Albumentations not installed")

    @pytest.mark.skipif(not True, reason="Albumentations required")
    def test_preview(self, sample_with_image):
        """Test generating previews."""
        try:
            from backend.data.augmentation import preview_augmentations

            previews = preview_augmentations(sample_with_image, preset="light", n=3)
            assert len(previews) == 3
        except ImportError:
            pytest.skip("Albumentations not installed")

    @pytest.mark.skipif(not True, reason="Albumentations required")
    def test_presets(self, sample_with_image):
        """Test different presets."""
        try:
            from backend.data.augmentation import get_preset_pipeline

            for preset in ["light", "medium", "heavy", "geometric", "photometric"]:
                pipeline = get_preset_pipeline(preset)
                assert pipeline is not None
        except ImportError:
            pytest.skip("Albumentations not installed")
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/augmentation.py` | Create | Augmentation implementation |
| `backend/data/__init__.py` | Modify | Export augmentation module |
| `backend/tests/data/test_augmentation.py` | Create | Unit tests |

## Verification

```bash
pip install albumentations
cd backend
pytest tests/data/test_augmentation.py -v
```

## Notes

- Albumentations handles bbox coordinate updates automatically
- YOLO format used for bbox params (cx, cy, w, h normalized)
- Masks require separate handling (not implemented here)
- Heavy preset may make images unrecognizable
- Geometric transforms useful for multi-angle robustness
- Photometric transforms for lighting invariance
