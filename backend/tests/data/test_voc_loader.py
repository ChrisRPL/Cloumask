"""Tests for Pascal VOC format loader."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from backend.data.formats.voc import VOC_CLASSES, VocExporter, VocLoader
from backend.data.models import Dataset


@pytest.fixture
def voc_dataset(tmp_path: Path) -> Path:
    """Create a sample VOC dataset."""
    ann_dir = tmp_path / "Annotations"
    img_dir = tmp_path / "JPEGImages"
    seg_class_dir = tmp_path / "SegmentationClass"
    seg_object_dir = tmp_path / "SegmentationObject"
    imagesets_dir = tmp_path / "ImageSets" / "Main"

    ann_dir.mkdir(parents=True)
    img_dir.mkdir()
    seg_class_dir.mkdir()
    seg_object_dir.mkdir()
    imagesets_dir.mkdir(parents=True)

    xml_img001 = """<?xml version="1.0"?>
<annotation>
  <folder>VOC2012</folder>
  <filename>img001.jpg</filename>
  <size>
    <width>640</width>
    <height>480</height>
    <depth>3</depth>
  </size>
  <object>
    <name>person</name>
    <pose>Frontal</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <occluded>1</occluded>
    <bndbox>
      <xmin>100</xmin>
      <ymin>100</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
  <object>
    <name>car</name>
    <truncated>1</truncated>
    <difficult>1</difficult>
    <bndbox>
      <xmin>350</xmin>
      <ymin>180</ymin>
      <xmax>600</xmax>
      <ymax>380</ymax>
    </bndbox>
  </object>
</annotation>
"""
    (ann_dir / "img001.xml").write_text(xml_img001)

    xml_img002 = """<?xml version="1.0"?>
<annotation>
  <filename>img002.jpg</filename>
  <size>
    <width>800</width>
    <height>600</height>
    <depth>3</depth>
  </size>
  <object>
    <name>forklift</name>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>50</xmin>
      <ymin>75</ymin>
      <xmax>200</xmax>
      <ymax>320</ymax>
    </bndbox>
  </object>
</annotation>
"""
    (ann_dir / "img002.xml").write_text(xml_img002)

    (img_dir / "img001.jpg").touch()
    (img_dir / "img002.jpg").touch()

    (imagesets_dir / "train.txt").write_text("img001 1\n")
    (imagesets_dir / "trainval.txt").write_text("img001\nimg002\n")

    try:
        import numpy as np
        from PIL import Image

        class_mask = np.array(
            [
                [0, 7, 7],
                [0, 15, 15],
                [0, 0, 255],
            ],
            dtype=np.uint8,
        )
        object_mask = np.array(
            [
                [0, 1, 1],
                [0, 2, 2],
                [0, 0, 255],
            ],
            dtype=np.uint8,
        )
        Image.fromarray(class_mask).save(seg_class_dir / "img001.png")
        Image.fromarray(object_mask).save(seg_object_dir / "img001.png")
    except ImportError:
        # Segmentation metadata tests are skipped when dependencies are unavailable.
        pass

    return tmp_path


class TestVocLoader:
    """Tests for VocLoader."""

    def test_load_dataset(self, voc_dataset: Path) -> None:
        """Test loading a VOC dataset."""
        loader = VocLoader(voc_dataset)
        ds = loader.load()

        assert isinstance(ds, Dataset)
        assert len(ds) == 2

    def test_parse_labels_and_attributes(self, voc_dataset: Path) -> None:
        """Test object parsing, bbox normalization, and VOC attributes."""
        loader = VocLoader(voc_dataset)
        ds = loader.load()

        sample = next(s for s in ds if s.image_path.stem == "img001")
        assert len(sample.labels) == 2

        person = next(label for label in sample.labels if label.class_name == "person")
        assert person.bbox.cx == pytest.approx(0.3125, rel=0.01)
        assert person.attributes["difficult"] is False
        assert person.attributes["truncated"] is False
        assert person.attributes["occluded"] is True
        assert person.attributes["pose"] == "Frontal"

        car = next(label for label in sample.labels if label.class_name == "car")
        assert car.attributes["difficult"] is True
        assert car.attributes["truncated"] is True

    def test_exclude_difficult_objects(self, voc_dataset: Path) -> None:
        """Test filtering difficult objects."""
        loader = VocLoader(voc_dataset, include_difficult=False)
        ds = loader.load()

        sample = next(s for s in ds if s.image_path.stem == "img001")
        assert len(sample.labels) == 1
        assert sample.labels[0].class_name == "person"

    def test_split_loading(self, voc_dataset: Path) -> None:
        """Test loading only files listed in split file."""
        loader = VocLoader(voc_dataset, split="train")
        ds = loader.load()

        assert len(ds) == 1
        assert ds[0].image_path.stem == "img001"
        assert ds[0].metadata["split"] == "train"

    def test_non_standard_class_discovery(self, voc_dataset: Path) -> None:
        """Test that non-standard VOC classes are appended to class_names."""
        loader = VocLoader(voc_dataset)
        ds = loader.load()

        assert "forklift" in ds.class_names

        sample = next(s for s in ds if s.image_path.stem == "img002")
        label = sample.labels[0]
        assert label.class_name == "forklift"
        assert label.class_id >= len(VOC_CLASSES)

    def test_segmentation_indices_metadata(self, voc_dataset: Path) -> None:
        """Test extraction of segmentation class/object indices from masks."""
        if not (voc_dataset / "SegmentationClass" / "img001.png").exists():
            pytest.skip("Pillow/Numpy not installed")

        loader = VocLoader(voc_dataset)
        ds = loader.load()
        sample = next(s for s in ds if s.image_path.stem == "img001")

        assert sample.metadata["segmentation_class_indices"] == [7, 15]
        assert sample.metadata["segmentation_object_indices"] == [1, 2]
        assert "segmentation_class_mask" in sample.metadata
        assert "segmentation_object_mask" in sample.metadata

    def test_iter_samples(self, voc_dataset: Path) -> None:
        """Test lazy iteration."""
        loader = VocLoader(voc_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 2

    def test_validate_valid_dataset(self, voc_dataset: Path) -> None:
        """Test validation reports no warnings for valid VOC dataset."""
        loader = VocLoader(voc_dataset)
        warnings = loader.validate()
        assert len(warnings) == 0

    def test_validate_missing_dirs(self, tmp_path: Path) -> None:
        """Test validation warns when VOC directories are missing."""
        loader = VocLoader(tmp_path)
        warnings = loader.validate()

        assert any("Annotations directory not found" in warning for warning in warnings)

    def test_summary(self, voc_dataset: Path) -> None:
        """Test summary includes VOC-specific fields."""
        loader = VocLoader(voc_dataset, split="train", include_difficult=False)
        summary = loader.summary()

        assert summary["format"] == "voc"
        assert summary["split"] == "train"
        assert summary["include_difficult"] is False
        assert summary["num_xml_files"] == 2

    def test_progress_callback(self, voc_dataset: Path) -> None:
        """Test progress callback is called for each sample."""
        progress_calls: list[tuple[int, int, str]] = []

        def callback(current: int, total: int, message: str) -> None:
            progress_calls.append((current, total, message))

        loader = VocLoader(voc_dataset, progress_callback=callback)
        loader.load()

        assert len(progress_calls) == 2
        assert progress_calls[-1][0] == progress_calls[-1][1]


class TestVocExporter:
    """Tests for VocExporter."""

    def test_export_basic(self, voc_dataset: Path, tmp_path: Path) -> None:
        """Test basic VOC export writes expected structure."""
        dataset = VocLoader(voc_dataset).load()

        output = tmp_path / "export"
        exporter = VocExporter(output)
        exported = exporter.export(dataset)

        assert exported == output
        assert (output / "Annotations").exists()
        assert (output / "JPEGImages").exists()
        assert (output / "ImageSets" / "Main").exists()
        assert (output / "Annotations" / "img001.xml").exists()
        assert (output / "Annotations" / "img002.xml").exists()
        assert (output / "JPEGImages" / "img001.jpg").exists()
        assert (output / "JPEGImages" / "img002.jpg").exists()

        split_file = output / "ImageSets" / "Main" / "trainval.txt"
        assert split_file.exists()
        assert split_file.read_text(encoding="utf-8").splitlines() == ["img001", "img002"]

    def test_export_roundtrip(self, voc_dataset: Path, tmp_path: Path) -> None:
        """Test load -> export -> load roundtrip."""
        original = VocLoader(voc_dataset).load()

        output = tmp_path / "export"
        VocExporter(output).export(original)

        reloaded = VocLoader(output).load()

        assert len(reloaded) == len(original)
        assert reloaded.total_labels() == original.total_labels()
        assert "forklift" in reloaded.class_names

        original_counts = {sample.image_path.stem: len(sample.labels) for sample in original}
        reloaded_counts = {sample.image_path.stem: len(sample.labels) for sample in reloaded}
        assert reloaded_counts == original_counts

        original_person = next(
            label for label in next(s for s in original if s.image_path.stem == "img001").labels
            if label.class_name == "person"
        )
        reloaded_person = next(
            label for label in next(s for s in reloaded if s.image_path.stem == "img001").labels
            if label.class_name == "person"
        )
        assert reloaded_person.attributes["difficult"] == original_person.attributes["difficult"]
        assert reloaded_person.attributes["truncated"] == original_person.attributes["truncated"]
        assert reloaded_person.attributes["occluded"] == original_person.attributes["occluded"]
        assert reloaded_person.bbox.cx == pytest.approx(original_person.bbox.cx, abs=0.02)
        assert reloaded_person.bbox.cy == pytest.approx(original_person.bbox.cy, abs=0.02)

    def test_export_includes_voc_attributes(self, voc_dataset: Path, tmp_path: Path) -> None:
        """Test exporter writes difficult/truncated/occluded tags."""
        dataset = VocLoader(voc_dataset).load()

        output = tmp_path / "export"
        VocExporter(output).export(dataset)

        root = ET.parse(output / "Annotations" / "img001.xml").getroot()
        objects = {
            obj.findtext("name", default=""): obj
            for obj in root.findall("object")
        }

        person = objects["person"]
        assert person.findtext("truncated") == "0"
        assert person.findtext("difficult") == "0"
        assert person.findtext("occluded") == "1"

        car = objects["car"]
        assert car.findtext("truncated") == "1"
        assert car.findtext("difficult") == "1"
        assert car.findtext("occluded") == "0"

    def test_export_uses_sample_split_metadata(self, voc_dataset: Path, tmp_path: Path) -> None:
        """Test split file generation uses sample metadata split when available."""
        dataset = VocLoader(voc_dataset, split="train").load()

        output = tmp_path / "export"
        VocExporter(output, split="trainval").export(dataset)

        train_file = output / "ImageSets" / "Main" / "train.txt"
        assert train_file.exists()
        assert train_file.read_text(encoding="utf-8").splitlines() == ["img001"]

    def test_validate_export(self, voc_dataset: Path, tmp_path: Path) -> None:
        """Test exported VOC structure validation."""
        dataset = VocLoader(voc_dataset).load()

        output = tmp_path / "export"
        exporter = VocExporter(output)
        exporter.export(dataset)

        warnings = exporter.validate_export()
        assert warnings == []
