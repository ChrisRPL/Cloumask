"""Tests for KITTI format loader."""

from pathlib import Path

import pytest

from backend.data.formats.kitti import KITTI_CLASSES, KittiLoader
from backend.data.models import Dataset


@pytest.fixture
def kitti_dataset(tmp_path: Path) -> Path:
    """Create a sample KITTI dataset."""
    # Create structure
    train_img = tmp_path / "training" / "image_2"
    train_lbl = tmp_path / "training" / "label_2"
    train_img.mkdir(parents=True)
    train_lbl.mkdir(parents=True)

    # Create dummy images (we need real images for dimension detection)
    # Using PIL to create minimal test images
    try:
        from PIL import Image

        img = Image.new("RGB", (1242, 375), color="black")
        img.save(train_img / "000000.png")
        img.save(train_img / "000001.png")
    except ImportError:
        # Fallback: touch files (will use default dimensions)
        (train_img / "000000.png").touch()
        (train_img / "000001.png").touch()

    # Create labels with full 15-field KITTI format
    (train_lbl / "000000.txt").write_text(
        "Car 0.00 0 -1.82 517.0 174.0 636.0 224.0 1.47 1.60 3.69 1.04 1.82 9.64 -1.57\n"
        "Pedestrian 0.00 1 0.21 397.0 181.0 434.0 268.0 1.72 0.50 0.80 -5.52 1.77 10.85 0.17\n"
    )
    (train_lbl / "000001.txt").write_text(
        "Cyclist 0.50 2 1.23 100.0 150.0 200.0 300.0 1.80 0.60 1.90 2.50 1.60 15.20 0.85\n"
    )

    return tmp_path


class TestKittiLoader:
    """Tests for KittiLoader."""

    def test_load_dataset(self, kitti_dataset: Path) -> None:
        """Test loading a KITTI dataset."""
        loader = KittiLoader(kitti_dataset)
        ds = loader.load()

        assert isinstance(ds, Dataset)
        assert len(ds) == 2

    def test_class_names(self, kitti_dataset: Path) -> None:
        """Test class names."""
        loader = KittiLoader(kitti_dataset)
        names = loader.get_class_names()
        # Should not include DontCare by default
        assert "DontCare" not in names
        assert "Car" in names

    def test_include_dontcare(self, kitti_dataset: Path) -> None:
        """Test including DontCare class."""
        loader = KittiLoader(kitti_dataset, include_dontcare=True)
        names = loader.get_class_names()
        assert "DontCare" in names

    def test_parse_labels(self, kitti_dataset: Path) -> None:
        """Test label parsing."""
        loader = KittiLoader(kitti_dataset)
        ds = loader.load()

        sample = next(s for s in ds if s.image_path.stem == "000000")
        assert len(sample.labels) == 2

        car_label = next(lbl for lbl in sample.labels if lbl.class_name == "Car")
        assert car_label.class_name == "Car"
        # Check 3D attributes preserved
        assert "dimensions_3d" in car_label.attributes
        assert "location_3d" in car_label.attributes
        assert car_label.attributes["truncated"] == 0.0
        assert car_label.attributes["occluded"] == 0

    def test_bbox_normalization(self, kitti_dataset: Path) -> None:
        """Test bbox is normalized."""
        loader = KittiLoader(kitti_dataset)
        ds = loader.load()

        sample = ds[0]
        for label in sample.labels:
            assert 0 <= label.bbox.cx <= 1
            assert 0 <= label.bbox.cy <= 1
            assert 0 <= label.bbox.w <= 1
            assert 0 <= label.bbox.h <= 1

    def test_3d_attributes(self, kitti_dataset: Path) -> None:
        """Test 3D attributes are preserved."""
        loader = KittiLoader(kitti_dataset)
        ds = loader.load()

        sample = ds[0]
        label = sample.labels[0]

        assert "dimensions_3d" in label.attributes
        dims = label.attributes["dimensions_3d"]
        assert "height" in dims
        assert "width" in dims
        assert "length" in dims

        assert "location_3d" in label.attributes
        loc = label.attributes["location_3d"]
        assert "x" in loc
        assert "y" in loc
        assert "z" in loc

        assert "rotation_y" in label.attributes
        assert "alpha" in label.attributes

    def test_iter_samples(self, kitti_dataset: Path) -> None:
        """Test lazy iteration."""
        loader = KittiLoader(kitti_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 2

    def test_validate(self, kitti_dataset: Path) -> None:
        """Test validation."""
        loader = KittiLoader(kitti_dataset)
        warnings = loader.validate()
        assert len(warnings) == 0

    def test_validate_missing_labels(self, kitti_dataset: Path) -> None:
        """Test validation detects missing labels."""
        # Remove a label file
        (kitti_dataset / "training" / "label_2" / "000001.txt").unlink()

        loader = KittiLoader(kitti_dataset)
        warnings = loader.validate()
        assert any("without labels" in w for w in warnings)

    def test_summary(self, kitti_dataset: Path) -> None:
        """Test summary method."""
        loader = KittiLoader(kitti_dataset)
        summary = loader.summary()

        assert summary["format"] == "kitti"
        assert "training" in summary["splits"]

    def test_kitti_classes_constant(self) -> None:
        """Test KITTI_CLASSES constant."""
        assert "Car" in KITTI_CLASSES
        assert "Pedestrian" in KITTI_CLASSES
        assert "DontCare" in KITTI_CLASSES

    def test_direct_structure(self, tmp_path: Path) -> None:
        """Test loading without split subdirectory."""
        img_dir = tmp_path / "image_2"
        lbl_dir = tmp_path / "label_2"
        img_dir.mkdir()
        lbl_dir.mkdir()

        try:
            from PIL import Image

            img = Image.new("RGB", (1242, 375), color="black")
            img.save(img_dir / "000000.png")
        except ImportError:
            (img_dir / "000000.png").touch()

        (lbl_dir / "000000.txt").write_text(
            "Car 0.00 0 0.00 100.0 100.0 200.0 200.0 1.5 1.6 3.5 1.0 1.0 10.0 0.0\n"
        )

        loader = KittiLoader(tmp_path)
        ds = loader.load()
        assert len(ds) == 1

    def test_empty_label_file(self, tmp_path: Path) -> None:
        """Test handling of empty label files."""
        train_img = tmp_path / "training" / "image_2"
        train_lbl = tmp_path / "training" / "label_2"
        train_img.mkdir(parents=True)
        train_lbl.mkdir(parents=True)

        try:
            from PIL import Image

            img = Image.new("RGB", (1242, 375), color="black")
            img.save(train_img / "empty.png")
        except ImportError:
            (train_img / "empty.png").touch()

        (train_lbl / "empty.txt").write_text("")

        loader = KittiLoader(tmp_path)
        ds = loader.load()
        assert len(ds) == 1
        assert len(ds[0].labels) == 0

    def test_invalid_line_skipped(self, tmp_path: Path) -> None:
        """Test that invalid lines are skipped."""
        train_img = tmp_path / "training" / "image_2"
        train_lbl = tmp_path / "training" / "label_2"
        train_img.mkdir(parents=True)
        train_lbl.mkdir(parents=True)

        try:
            from PIL import Image

            img = Image.new("RGB", (1242, 375), color="black")
            img.save(train_img / "test.png")
        except ImportError:
            (train_img / "test.png").touch()

        # Line with only 10 fields (invalid)
        (train_lbl / "test.txt").write_text("Car 0.00 0 0.00 100.0 100.0 200.0 200.0 1.5 1.6\n")

        loader = KittiLoader(tmp_path)
        ds = loader.load()
        assert len(ds) == 1
        assert len(ds[0].labels) == 0  # Invalid line skipped

    def test_progress_callback(self, kitti_dataset: Path) -> None:
        """Test progress callback is called."""
        progress_calls: list[tuple[int, int, str]] = []

        def callback(current: int, total: int, msg: str) -> None:
            progress_calls.append((current, total, msg))

        loader = KittiLoader(kitti_dataset, progress_callback=callback)
        loader.load()

        assert len(progress_calls) == 2  # 2 images
        assert progress_calls[-1][0] == progress_calls[-1][1]  # Last: current == total

    def test_class_names_override(self, kitti_dataset: Path) -> None:
        """Test class names can be overridden."""
        loader = KittiLoader(kitti_dataset, class_names=["A", "B", "C"])
        assert loader.get_class_names() == ["A", "B", "C"]
