"""Tests for Albumentations-backed dataset augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.data.augmentation import (
    ALBUMENTATIONS_AVAILABLE,
    AugmentationPipeline,
    augment_dataset,
    available_presets,
    get_preset_pipeline,
    preview_augmentations,
)
from backend.data.models import BBox, Dataset, Label, Sample

albumentations_required = pytest.mark.skipif(
    not ALBUMENTATIONS_AVAILABLE,
    reason="Albumentations is required for augmentation tests",
)


@pytest.fixture
def sample_with_image(tmp_path: Path) -> Sample:
    """Create a sample with an on-disk image and two labels."""
    image_path = tmp_path / "frame_001.jpg"
    image = np.full((480, 640, 3), fill_value=120, dtype=np.uint8)
    image[120:220, 200:340] = 200
    Image.fromarray(image).save(image_path)

    return Sample(
        image_path=image_path,
        labels=[
            Label(
                class_name="car",
                class_id=0,
                bbox=BBox(0.5, 0.5, 0.3, 0.25),
                confidence=0.95,
                attributes={"occluded": False},
                track_id=11,
            ),
            Label(
                class_name="person",
                class_id=1,
                bbox=BBox(0.25, 0.45, 0.1, 0.2),
                confidence=0.8,
                attributes={"pose": "standing"},
            ),
        ],
        image_width=640,
        image_height=480,
    )


class TestPresets:
    """Tests for preset handling."""

    def test_available_presets(self) -> None:
        assert {"light", "medium", "heavy"} <= set(available_presets())

    @albumentations_required
    def test_get_preset_pipeline(self) -> None:
        for preset in available_presets():
            pipeline = get_preset_pipeline(preset)
            assert pipeline is not None

    @albumentations_required
    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_pipeline("unknown")


@albumentations_required
class TestAugmentationPipeline:
    """Tests for augmentation pipeline behavior."""

    def test_augment_sample(self, sample_with_image: Sample) -> None:
        pipeline = AugmentationPipeline(preset="light")
        result = pipeline.augment(sample_with_image)

        assert isinstance(result.image, np.ndarray)
        assert result.image.shape[:2] == (480, 640)
        assert result.transform_name == "light"
        assert len(result.labels) <= len(sample_with_image.labels)
        for label in result.labels:
            assert 0.0 <= label.bbox.cx <= 1.0
            assert 0.0 <= label.bbox.cy <= 1.0
            assert 0.0 <= label.bbox.w <= 1.0
            assert 0.0 <= label.bbox.h <= 1.0

    def test_custom_transform_preserves_metadata(self, sample_with_image: Sample) -> None:
        import albumentations as A

        transform = A.Compose(
            [A.HorizontalFlip(p=1.0)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["bbox_indices"]),
        )
        pipeline = AugmentationPipeline(transform=transform)
        result = pipeline.augment(sample_with_image)

        assert len(result.labels) == len(sample_with_image.labels)
        for transformed, original in zip(result.labels, sample_with_image.labels, strict=True):
            assert transformed.class_id == original.class_id
            assert transformed.class_name == original.class_name
            assert transformed.attributes == original.attributes
            assert transformed.track_id == original.track_id
            assert transformed.mask is None

        assert result.labels[0].bbox.cx == pytest.approx(1 - sample_with_image.labels[0].bbox.cx, rel=1e-3)

    def test_preview(self, sample_with_image: Sample) -> None:
        previews = preview_augmentations(sample_with_image, preset="light", n=3)
        assert len(previews) == 3
        assert all(item.transform_name == "light" for item in previews)

    def test_augment_dataset(self, sample_with_image: Sample, tmp_path: Path) -> None:
        dataset = Dataset([sample_with_image], name="demo", class_names=["car", "person"])
        output_dir = tmp_path / "augmented-images"

        augmented = augment_dataset(
            dataset,
            output_dir=output_dir,
            preset="light",
            copies_per_sample=2,
            include_original=True,
        )

        assert len(augmented) == 3
        assert augmented.class_names == ["car", "person"]
        generated = [sample for sample in augmented if "augmentation" in sample.metadata]
        assert len(generated) == 2
        for sample in generated:
            assert sample.image_path.exists()
            assert sample.image_path.parent == output_dir
            assert sample.image_width == 640
            assert sample.image_height == 480
            assert sample.metadata["augmentation"]["preset"] == "light"
