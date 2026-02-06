"""Tests for data models.

Tests BBox coordinate conversions, Label validation, Sample filtering,
and Dataset statistics/operations.
"""

from pathlib import Path

import numpy as np
import pytest

from backend.data.models import BBox, BBoxFormat, Dataset, Label, Sample


class TestBBox:
    """Tests for BBox dataclass."""

    def test_from_xyxy(self) -> None:
        """Test creation from corner coordinates."""
        bbox = BBox.from_xyxy(0.1, 0.2, 0.5, 0.6)
        assert bbox.cx == pytest.approx(0.3)
        assert bbox.cy == pytest.approx(0.4)
        assert bbox.w == pytest.approx(0.4)
        assert bbox.h == pytest.approx(0.4)

    def test_from_xywh(self) -> None:
        """Test creation from top-left + dimensions."""
        bbox = BBox.from_xywh(0.1, 0.2, 0.4, 0.4)
        assert bbox.cx == pytest.approx(0.3)
        assert bbox.cy == pytest.approx(0.4)

    def test_to_xyxy(self) -> None:
        """Test conversion to corner format."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.4, h=0.4)
        x1, y1, x2, y2 = bbox.to_xyxy()
        assert x1 == pytest.approx(0.3)
        assert y1 == pytest.approx(0.3)
        assert x2 == pytest.approx(0.7)
        assert y2 == pytest.approx(0.7)

    def test_roundtrip_xyxy(self) -> None:
        """Test xyxy -> BBox -> xyxy preserves values."""
        original = (0.1, 0.2, 0.6, 0.8)
        bbox = BBox.from_xyxy(*original)
        result = bbox.to_xyxy()
        for a, b in zip(original, result):
            assert a == pytest.approx(b)

    def test_from_absolute(self) -> None:
        """Test creation from pixel coordinates."""
        bbox = BBox.from_absolute((100, 200, 300, 400), 1000, 1000, BBoxFormat.XYXY)
        assert bbox.cx == pytest.approx(0.2)
        assert bbox.cy == pytest.approx(0.3)
        assert bbox.w == pytest.approx(0.2)
        assert bbox.h == pytest.approx(0.2)

    def test_to_absolute(self) -> None:
        """Test conversion to pixel coordinates."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        x1, y1, x2, y2 = bbox.to_absolute(1000, 1000, BBoxFormat.XYXY)
        assert x1 == pytest.approx(400)
        assert y1 == pytest.approx(400)
        assert x2 == pytest.approx(600)
        assert y2 == pytest.approx(600)

    def test_area(self) -> None:
        """Test area computation."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.5, h=0.5)
        assert bbox.area() == pytest.approx(0.25)

    def test_iou_no_overlap(self) -> None:
        """Test IoU with non-overlapping boxes."""
        box1 = BBox(cx=0.25, cy=0.5, w=0.2, h=0.2)
        box2 = BBox(cx=0.75, cy=0.5, w=0.2, h=0.2)
        assert box1.iou(box2) == pytest.approx(0.0)

    def test_iou_full_overlap(self) -> None:
        """Test IoU with identical boxes."""
        box1 = BBox(cx=0.5, cy=0.5, w=0.4, h=0.4)
        box2 = BBox(cx=0.5, cy=0.5, w=0.4, h=0.4)
        assert box1.iou(box2) == pytest.approx(1.0)

    def test_iou_partial_overlap(self) -> None:
        """Test IoU with partial overlap."""
        box1 = BBox.from_xyxy(0.0, 0.0, 0.5, 0.5)
        box2 = BBox.from_xyxy(0.25, 0.25, 0.75, 0.75)
        # Intersection: 0.25 * 0.25 = 0.0625
        # Union: 0.25 + 0.25 - 0.0625 = 0.4375
        expected = 0.0625 / 0.4375
        assert box1.iou(box2) == pytest.approx(expected, rel=0.01)

    def test_clamps_out_of_bounds(self) -> None:
        """Test that out-of-bounds values are clamped."""
        bbox = BBox(cx=1.5, cy=-0.1, w=0.5, h=0.5)
        assert bbox.cx == 1.0
        assert bbox.cy == 0.0

    def test_serialization(self) -> None:
        """Test to_dict/from_dict roundtrip."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.3, h=0.4)
        data = bbox.to_dict()
        restored = BBox.from_dict(data)
        assert restored.cx == bbox.cx
        assert restored.w == bbox.w


class TestLabel:
    """Tests for Label dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic label creation."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        label = Label(class_name="car", class_id=0, bbox=bbox)
        assert label.class_name == "car"
        assert label.confidence == 1.0
        assert label.mask is None

    def test_with_mask(self) -> None:
        """Test label with segmentation mask."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        mask = np.zeros((100, 100), dtype=np.uint8)
        label = Label(class_name="car", class_id=0, bbox=bbox, mask=mask)
        assert label.mask is not None
        assert label.mask.shape == (100, 100)

    def test_with_attributes(self) -> None:
        """Test label with custom attributes."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        label = Label(
            class_name="car",
            class_id=0,
            bbox=bbox,
            attributes={"occluded": True, "truncated": False},
        )
        assert label.attributes["occluded"] is True

    def test_clamps_confidence(self) -> None:
        """Test that invalid confidence is clamped."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        label = Label(class_name="car", class_id=0, bbox=bbox, confidence=1.5)
        assert label.confidence == 1.0

    def test_rejects_negative_class_id(self) -> None:
        """Test that negative class_id raises error."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        with pytest.raises(ValueError):
            Label(class_name="car", class_id=-1, bbox=bbox)

    def test_serialization(self) -> None:
        """Test to_dict/from_dict roundtrip."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        label = Label(
            class_name="car",
            class_id=0,
            bbox=bbox,
            confidence=0.95,
            attributes={"color": "red"},
        )
        data = label.to_dict()
        restored = Label.from_dict(data)
        assert restored.class_name == label.class_name
        assert restored.confidence == label.confidence
        assert restored.attributes == label.attributes


class TestSample:
    """Tests for Sample dataclass."""

    def test_basic_creation(self) -> None:
        """Test sample with no labels."""
        sample = Sample(image_path=Path("/data/img001.jpg"))
        assert sample.num_labels == 0
        assert not sample.has_labels

    def test_with_labels(self) -> None:
        """Test sample with labels."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        labels = [
            Label(class_name="car", class_id=0, bbox=bbox),
            Label(class_name="person", class_id=1, bbox=bbox),
        ]
        sample = Sample(image_path=Path("/data/img001.jpg"), labels=labels)
        assert sample.num_labels == 2
        assert sample.has_labels
        assert sample.class_names == {"car", "person"}

    def test_filter_by_class(self) -> None:
        """Test filtering labels by class name."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        labels = [
            Label(class_name="car", class_id=0, bbox=bbox),
            Label(class_name="person", class_id=1, bbox=bbox),
            Label(class_name="car", class_id=0, bbox=bbox),
        ]
        sample = Sample(image_path=Path("/data/img.jpg"), labels=labels)
        filtered = sample.filter_by_class(["car"])
        assert filtered.num_labels == 2
        assert all(lbl.class_name == "car" for lbl in filtered.labels)

    def test_filter_by_confidence(self) -> None:
        """Test filtering labels by confidence."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        labels = [
            Label(class_name="car", class_id=0, bbox=bbox, confidence=0.9),
            Label(class_name="car", class_id=0, bbox=bbox, confidence=0.5),
            Label(class_name="car", class_id=0, bbox=bbox, confidence=0.3),
        ]
        sample = Sample(image_path=Path("/data/img.jpg"), labels=labels)
        filtered = sample.filter_by_confidence(0.5)
        assert filtered.num_labels == 2

    def test_string_path_conversion(self) -> None:
        """Test that string paths are converted to Path objects."""
        sample = Sample(image_path="/data/img.jpg")  # type: ignore
        assert isinstance(sample.image_path, Path)

    def test_serialization(self) -> None:
        """Test to_dict/from_dict roundtrip."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        label = Label(class_name="car", class_id=0, bbox=bbox)
        sample = Sample(
            image_path=Path("/data/img.jpg"),
            labels=[label],
            image_width=1920,
            image_height=1080,
            metadata={"frame_id": 42},
        )
        data = sample.to_dict()
        restored = Sample.from_dict(data)
        assert restored.num_labels == 1
        assert restored.image_width == 1920
        assert restored.metadata["frame_id"] == 42


class TestDataset:
    """Tests for Dataset class."""

    def test_basic_creation(self) -> None:
        """Test dataset creation."""
        samples = [Sample(image_path=Path(f"/data/img{i:03d}.jpg")) for i in range(10)]
        ds = Dataset(samples, name="test")
        assert len(ds) == 10
        assert ds.name == "test"

    def test_iteration(self) -> None:
        """Test dataset iteration."""
        samples = [Sample(image_path=Path(f"/data/img{i:03d}.jpg")) for i in range(5)]
        ds = Dataset(samples)
        count = sum(1 for _ in ds)
        assert count == 5

    def test_indexing(self) -> None:
        """Test dataset indexing."""
        samples = [Sample(image_path=Path(f"/data/img{i:03d}.jpg")) for i in range(5)]
        ds = Dataset(samples)
        assert ds[0].image_path == Path("/data/img000.jpg")
        assert ds[4].image_path == Path("/data/img004.jpg")

    def test_class_distribution(self) -> None:
        """Test class distribution counting."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        samples = [
            Sample(
                image_path=Path("/data/img1.jpg"),
                labels=[
                    Label(class_name="car", class_id=0, bbox=bbox),
                    Label(class_name="car", class_id=0, bbox=bbox),
                ],
            ),
            Sample(
                image_path=Path("/data/img2.jpg"),
                labels=[
                    Label(class_name="person", class_id=1, bbox=bbox),
                ],
            ),
        ]
        ds = Dataset(samples, class_names=["car", "person"])
        dist = ds.class_distribution()
        assert dist["car"] == 2
        assert dist["person"] == 1

    def test_samples_per_class(self) -> None:
        """Test samples per class counting."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        samples = [
            Sample(
                image_path=Path("/data/img1.jpg"),
                labels=[
                    Label(class_name="car", class_id=0, bbox=bbox),
                    Label(class_name="car", class_id=0, bbox=bbox),  # Same sample
                ],
            ),
            Sample(
                image_path=Path("/data/img2.jpg"),
                labels=[
                    Label(class_name="car", class_id=0, bbox=bbox),
                ],
            ),
        ]
        ds = Dataset(samples)
        spc = ds.samples_per_class()
        assert spc["car"] == 2  # 2 samples contain "car"

    def test_subset(self) -> None:
        """Test creating subset by indices."""
        samples = [Sample(image_path=Path(f"/data/img{i:03d}.jpg")) for i in range(10)]
        ds = Dataset(samples)
        subset = ds.subset([0, 2, 4])
        assert len(subset) == 3

    def test_merge(self) -> None:
        """Test merging two datasets."""
        ds1 = Dataset(
            [Sample(image_path=Path("/data/a/img1.jpg"))],
            name="ds1",
            class_names=["car"],
        )
        ds2 = Dataset(
            [Sample(image_path=Path("/data/b/img2.jpg"))],
            name="ds2",
            class_names=["person"],
        )
        merged = ds1.merge(ds2)
        assert len(merged) == 2
        assert merged.class_names == ["car", "person"]

    def test_filter_by_class(self) -> None:
        """Test filtering dataset by class."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        samples = [
            Sample(
                image_path=Path("/data/img1.jpg"),
                labels=[Label(class_name="car", class_id=0, bbox=bbox)],
            ),
            Sample(
                image_path=Path("/data/img2.jpg"),
                labels=[Label(class_name="person", class_id=1, bbox=bbox)],
            ),
        ]
        ds = Dataset(samples)
        filtered = ds.filter_by_class(["car"])
        assert len(filtered) == 1
        assert filtered[0].labels[0].class_name == "car"

    def test_unlabeled_samples(self) -> None:
        """Test finding unlabeled samples."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        samples = [
            Sample(
                image_path=Path("/data/img1.jpg"),
                labels=[Label(class_name="car", class_id=0, bbox=bbox)],
            ),
            Sample(image_path=Path("/data/img2.jpg"), labels=[]),
        ]
        ds = Dataset(samples)
        unlabeled = ds.unlabeled_samples()
        assert len(unlabeled) == 1
        assert unlabeled[0].image_path == Path("/data/img2.jpg")

    def test_stats(self) -> None:
        """Test statistics computation."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        samples = [
            Sample(
                image_path=Path("/data/img1.jpg"),
                labels=[Label(class_name="car", class_id=0, bbox=bbox)],
            ),
            Sample(image_path=Path("/data/img2.jpg"), labels=[]),
        ]
        ds = Dataset(samples, name="test", class_names=["car"])
        stats = ds.stats()
        assert stats["num_samples"] == 2
        assert stats["num_labels"] == 1
        assert stats["unlabeled_count"] == 1

    def test_serialization(self) -> None:
        """Test to_dict/from_dict roundtrip."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        samples = [
            Sample(
                image_path=Path("/data/img1.jpg"),
                labels=[Label(class_name="car", class_id=0, bbox=bbox)],
            ),
        ]
        ds = Dataset(samples, name="test", class_names=["car"])
        data = ds.to_dict()
        restored = Dataset.from_dict(data)
        assert len(restored) == 1
        assert restored.name == "test"
        assert restored.class_names == ["car"]

    def test_infer_class_names(self) -> None:
        """Test automatic class name inference from data."""
        bbox = BBox(cx=0.5, cy=0.5, w=0.2, h=0.2)
        samples = [
            Sample(
                image_path=Path("/data/img1.jpg"),
                labels=[
                    Label(class_name="car", class_id=0, bbox=bbox),
                    Label(class_name="person", class_id=1, bbox=bbox),
                ],
            ),
        ]
        ds = Dataset(samples)  # No class_names provided
        assert ds.class_names == ["car", "person"]
