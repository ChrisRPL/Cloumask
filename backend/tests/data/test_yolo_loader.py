"""Tests for YOLO format loader."""

from pathlib import Path

import pytest

from backend.data.formats.yolo import YoloLoader
from backend.data.models import Dataset


@pytest.fixture
def yolo_dataset(tmp_path: Path) -> Path:
    """Create a sample YOLO dataset."""
    # Create structure
    train_images = tmp_path / "train" / "images"
    train_labels = tmp_path / "train" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)

    val_images = tmp_path / "val" / "images"
    val_labels = tmp_path / "val" / "labels"
    val_images.mkdir(parents=True)
    val_labels.mkdir(parents=True)

    # Create data.yaml
    (tmp_path / "data.yaml").write_text(
        """
train: train/images
val: val/images
names:
  0: person
  1: car
  2: bicycle
"""
    )

    # Create sample images (empty files for testing)
    (train_images / "img001.jpg").touch()
    (train_images / "img002.jpg").touch()
    (val_images / "img003.jpg").touch()

    # Create labels
    (train_labels / "img001.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.25 0.75 0.1 0.15\n")
    (train_labels / "img002.txt").write_text("0 0.3 0.4 0.15 0.2\n")
    (val_labels / "img003.txt").write_text("2 0.6 0.6 0.1 0.1\n")

    return tmp_path


class TestYoloLoader:
    """Tests for YoloLoader."""

    def test_load_dataset(self, yolo_dataset: Path) -> None:
        """Test loading a YOLO dataset."""
        loader = YoloLoader(yolo_dataset)
        ds = loader.load()

        assert isinstance(ds, Dataset)
        assert len(ds) == 3
        assert ds.class_names == ["person", "car", "bicycle"]

    def test_load_specific_splits(self, yolo_dataset: Path) -> None:
        """Test loading specific splits only."""
        loader = YoloLoader(yolo_dataset, splits=["train"])
        ds = loader.load()

        assert len(ds) == 2
        assert all(s.metadata.get("split") == "train" for s in ds)

    def test_parse_labels(self, yolo_dataset: Path) -> None:
        """Test that labels are parsed correctly."""
        loader = YoloLoader(yolo_dataset)
        ds = loader.load()

        # Find img001
        img001 = next(s for s in ds if s.image_path.stem == "img001")
        assert len(img001.labels) == 2

        # Check first label
        lbl = img001.labels[0]
        assert lbl.class_name == "person"
        assert lbl.class_id == 0
        assert lbl.bbox.cx == pytest.approx(0.5)
        assert lbl.bbox.cy == pytest.approx(0.5)

    def test_class_names_from_yaml(self, yolo_dataset: Path) -> None:
        """Test class names are read from data.yaml."""
        loader = YoloLoader(yolo_dataset)
        assert loader.get_class_names() == ["person", "car", "bicycle"]

    def test_class_names_override(self, yolo_dataset: Path) -> None:
        """Test class names can be overridden."""
        loader = YoloLoader(yolo_dataset, class_names=["a", "b", "c"])
        assert loader.get_class_names() == ["a", "b", "c"]

    def test_iter_samples(self, yolo_dataset: Path) -> None:
        """Test lazy iteration."""
        loader = YoloLoader(yolo_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 3

    def test_validate(self, yolo_dataset: Path) -> None:
        """Test validation returns no warnings for valid dataset."""
        loader = YoloLoader(yolo_dataset)
        warnings = loader.validate()
        assert len(warnings) == 0

    def test_validate_missing_yaml(self, tmp_path: Path) -> None:
        """Test validation warns on missing data.yaml."""
        (tmp_path / "train" / "images").mkdir(parents=True)
        loader = YoloLoader(tmp_path)
        warnings = loader.validate()
        assert any("data.yaml" in w for w in warnings)

    def test_summary(self, yolo_dataset: Path) -> None:
        """Test summary method."""
        loader = YoloLoader(yolo_dataset)
        summary = loader.summary()
        assert summary["format"] == "yolo"
        assert "train" in summary["splits"]
        assert "val" in summary["splits"]

    def test_segmentation_polygon(self, tmp_path: Path) -> None:
        """Test parsing segmentation polygons."""
        images = tmp_path / "train" / "images"
        labels = tmp_path / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)

        (tmp_path / "data.yaml").write_text("train: train/images\nnames: [obj]")
        (images / "seg.jpg").touch()
        # Polygon format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        (labels / "seg.txt").write_text("0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")

        loader = YoloLoader(tmp_path)
        ds = loader.load()
        sample = ds[0]

        assert len(sample.labels) == 1
        assert "polygon" in sample.labels[0].attributes
        assert len(sample.labels[0].attributes["polygon"]) == 8

    def test_progress_callback(self, yolo_dataset: Path) -> None:
        """Test progress callback is called."""
        progress_calls: list[tuple[int, int, str]] = []

        def callback(current: int, total: int, msg: str) -> None:
            progress_calls.append((current, total, msg))

        loader = YoloLoader(yolo_dataset, progress_callback=callback)
        loader.load()

        assert len(progress_calls) == 3  # 3 images
        assert progress_calls[-1][0] == progress_calls[-1][1]  # Last call: current == total

    def test_list_format_class_names(self, tmp_path: Path) -> None:
        """Test parsing class names as list format."""
        images = tmp_path / "train" / "images"
        labels = tmp_path / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)

        # Use list format for names instead of dict
        (tmp_path / "data.yaml").write_text("train: train/images\nnames: [cat, dog, bird]")
        (images / "test.jpg").touch()
        (labels / "test.txt").write_text("1 0.5 0.5 0.2 0.3\n")

        loader = YoloLoader(tmp_path)
        assert loader.get_class_names() == ["cat", "dog", "bird"]

        ds = loader.load()
        assert ds[0].labels[0].class_name == "dog"

    def test_empty_label_file(self, tmp_path: Path) -> None:
        """Test handling of empty label files (no objects)."""
        images = tmp_path / "train" / "images"
        labels = tmp_path / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)

        (tmp_path / "data.yaml").write_text("train: train/images\nnames: [obj]")
        (images / "empty.jpg").touch()
        (labels / "empty.txt").write_text("")  # Empty file

        loader = YoloLoader(tmp_path)
        ds = loader.load()

        assert len(ds) == 1
        assert len(ds[0].labels) == 0

    def test_missing_label_file(self, tmp_path: Path) -> None:
        """Test handling of missing label files (unlabeled image)."""
        images = tmp_path / "train" / "images"
        images.mkdir(parents=True)

        (tmp_path / "data.yaml").write_text("train: train/images\nnames: [obj]")
        (images / "unlabeled.jpg").touch()
        # No corresponding label file

        loader = YoloLoader(tmp_path)
        ds = loader.load()

        assert len(ds) == 1
        assert len(ds[0].labels) == 0
