"""Tests for COCO format loader."""

import json
from pathlib import Path

import numpy as np
import pytest

from backend.data.formats.coco import CocoExporter, CocoLoader
from backend.data.models import Dataset


@pytest.fixture
def coco_dataset(tmp_path: Path) -> Path:
    """Create a sample COCO dataset."""
    # Create structure
    ann_dir = tmp_path / "annotations"
    train_dir = tmp_path / "train"
    ann_dir.mkdir()
    train_dir.mkdir()

    # Create annotation file
    coco_data = {
        "info": {"description": "Test dataset"},
        "images": [
            {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img002.jpg", "width": 800, "height": 600},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "area": 30000,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [300, 200, 100, 80],
                "area": 8000,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [50, 50, 300, 250],
                "area": 75000,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
        ],
    }

    (ann_dir / "instances_train.json").write_text(json.dumps(coco_data))

    # Create image files
    (train_dir / "img001.jpg").touch()
    (train_dir / "img002.jpg").touch()

    return tmp_path


class TestCocoLoader:
    """Tests for CocoLoader."""

    def test_load_dataset(self, coco_dataset: Path) -> None:
        """Test loading a COCO dataset."""
        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        assert isinstance(ds, Dataset)
        assert len(ds) == 2

    def test_class_names(self, coco_dataset: Path) -> None:
        """Test class names from categories."""
        loader = CocoLoader(coco_dataset)
        names = loader.get_class_names()
        assert names == ["person", "car"]

    def test_parse_bbox(self, coco_dataset: Path) -> None:
        """Test bounding box parsing."""
        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        # Find img001
        sample = next(s for s in ds if s.image_path.stem == "img001")
        assert len(sample.labels) == 2

        # Check bbox normalization
        lbl = sample.labels[0]
        assert lbl.class_name == "person"
        # Original: x=100, y=100, w=200, h=150 on 640x480 image
        # cx = (100 + 100) / 640 = 200/640 = 0.3125
        # cy = (100 + 75) / 480 = 175/480 ≈ 0.3646
        assert lbl.bbox.cx == pytest.approx(200 / 640, rel=0.01)
        assert lbl.bbox.cy == pytest.approx(175 / 480, rel=0.01)

    def test_image_dimensions(self, coco_dataset: Path) -> None:
        """Test image dimensions are stored."""
        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        sample = ds[0]
        assert sample.image_width == 640
        assert sample.image_height == 480

    def test_metadata(self, coco_dataset: Path) -> None:
        """Test COCO metadata is preserved."""
        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        sample = ds[0]
        assert "coco_id" in sample.metadata
        assert sample.metadata["coco_id"] == 1

    def test_iter_samples(self, coco_dataset: Path) -> None:
        """Test lazy iteration."""
        loader = CocoLoader(coco_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 2

    def test_validate_valid_dataset(self, coco_dataset: Path) -> None:
        """Test validation on valid dataset."""
        loader = CocoLoader(coco_dataset)
        warnings = loader.validate()
        assert len(warnings) == 0

    def test_validate_missing_images(self, coco_dataset: Path) -> None:
        """Test validation detects missing images."""
        # Remove an image
        (coco_dataset / "train" / "img002.jpg").unlink()

        loader = CocoLoader(coco_dataset)
        warnings = loader.validate()
        assert any("not found" in w for w in warnings)

    def test_summary(self, coco_dataset: Path) -> None:
        """Test summary method."""
        loader = CocoLoader(coco_dataset)
        summary = loader.summary()

        assert summary["format"] == "coco"
        assert summary["num_images"] == 2
        assert summary["num_annotations"] == 3
        assert summary["num_categories"] == 2

    def test_polygon_segmentation(self, tmp_path: Path) -> None:
        """Test loading polygon segmentation."""
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()

        coco_data = {
            "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 30, 30],
                    "segmentation": [[10, 10, 40, 10, 40, 40, 10, 40]],
                    "area": 900,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "obj"}],
        }
        (ann_dir / "instances.json").write_text(json.dumps(coco_data))
        (tmp_path / "img.jpg").touch()

        loader = CocoLoader(tmp_path, load_masks=True)
        ds = loader.load()
        # Mask loading depends on cv2 being available
        # Just verify label is parsed
        assert len(ds[0].labels) == 1

    def test_specific_annotation_file(self, coco_dataset: Path) -> None:
        """Test loading specific annotation file."""
        loader = CocoLoader(coco_dataset, annotation_file="instances_train.json")
        ds = loader.load()
        assert len(ds) == 2

    def test_label_attributes(self, coco_dataset: Path) -> None:
        """Test label attributes are preserved."""
        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        sample = ds[0]
        label = sample.labels[0]
        assert "iscrowd" in label.attributes
        assert label.attributes["iscrowd"] is False
        assert "area" in label.attributes
        assert label.attributes["area"] == 30000

    def test_progress_callback(self, coco_dataset: Path) -> None:
        """Test progress callback is called."""
        progress_calls: list[tuple[int, int, str]] = []

        def callback(current: int, total: int, msg: str) -> None:
            progress_calls.append((current, total, msg))

        loader = CocoLoader(coco_dataset, progress_callback=callback)
        loader.load()

        assert len(progress_calls) == 2  # 2 images
        assert progress_calls[-1][0] == progress_calls[-1][1]  # Last: current == total

    def test_class_names_override(self, coco_dataset: Path) -> None:
        """Test class names can be overridden."""
        loader = CocoLoader(coco_dataset, class_names=["a", "b"])
        assert loader.get_class_names() == ["a", "b"]

    def test_non_contiguous_category_ids(self, tmp_path: Path) -> None:
        """Test handling of non-contiguous category IDs."""
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()

        coco_data = {
            "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 5,  # Non-contiguous
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 5, "name": "dog"},  # Non-contiguous
            ],
        }
        (ann_dir / "instances.json").write_text(json.dumps(coco_data))
        (tmp_path / "img.jpg").touch()

        loader = CocoLoader(tmp_path)
        ds = loader.load()

        # class_id should be 0-based index based on sorted category IDs
        assert ds[0].labels[0].class_name == "dog"
        assert ds[0].labels[0].class_id == 1  # Second in sorted order

    def test_missing_annotation_file(self, tmp_path: Path) -> None:
        """Test error when no annotation file found."""
        loader = CocoLoader(tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load()


class TestCocoExporter:
    """Tests for CocoExporter."""

    def test_export_basic(self, coco_dataset: Path, tmp_path: Path) -> None:
        """Test basic COCO export writes expected structure."""
        dataset = CocoLoader(coco_dataset).load()

        output = tmp_path / "export"
        exporter = CocoExporter(output)
        exported = exporter.export(dataset)

        assert exported == output
        assert (output / "annotations" / "instances_train.json").exists()
        assert (output / "images" / "img001.jpg").exists()
        assert (output / "images" / "img002.jpg").exists()

    def test_export_roundtrip(self, coco_dataset: Path, tmp_path: Path) -> None:
        """Test load -> export -> load roundtrip."""
        original = CocoLoader(coco_dataset).load()

        output = tmp_path / "export"
        CocoExporter(output).export(original)

        exported = CocoLoader(output).load()

        assert len(exported) == len(original)
        assert exported.class_names == original.class_names

        original_counts = {sample.image_path.stem: len(sample.labels) for sample in original}
        exported_counts = {sample.image_path.stem: len(sample.labels) for sample in exported}
        assert exported_counts == original_counts

    def test_export_categories(self, coco_dataset: Path, tmp_path: Path) -> None:
        """Test categories are exported with 1-indexed IDs."""
        dataset = CocoLoader(coco_dataset).load()

        output = tmp_path / "export"
        CocoExporter(output).export(dataset)

        with (output / "annotations" / "instances_train.json").open() as f:
            data = json.load(f)

        assert len(data["categories"]) == len(dataset.class_names)
        assert [category["id"] for category in data["categories"]] == list(
            range(1, len(dataset.class_names) + 1)
        )

    def test_export_mask_rle(self, coco_dataset: Path, tmp_path: Path) -> None:
        """Test segmentation mask export using RLE encoding."""
        dataset = CocoLoader(coco_dataset).load()
        dataset[0].labels[0].mask = np.array(
            [[0, 1, 1], [0, 1, 0], [0, 0, 0]],
            dtype=np.uint8,
        )

        output = tmp_path / "export"
        exporter = CocoExporter(output, mask_encoding="rle")
        exporter.export(dataset)

        with (output / "annotations" / "instances_train.json").open() as f:
            data = json.load(f)

        segmentation = data["annotations"][0]["segmentation"]
        assert isinstance(segmentation, dict)
        assert "counts" in segmentation
        assert "size" in segmentation

    def test_validate_export(self, coco_dataset: Path, tmp_path: Path) -> None:
        """Test export validation."""
        dataset = CocoLoader(coco_dataset).load()

        output = tmp_path / "export"
        exporter = CocoExporter(output)
        exporter.export(dataset)

        warnings = exporter.validate_export()
        assert warnings == []
