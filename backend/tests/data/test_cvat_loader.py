"""Tests for CVAT XML format loader."""

from pathlib import Path

import pytest

from backend.data.formats.cvat import CvatLoader
from backend.data.models import Dataset


@pytest.fixture
def cvat_dataset(tmp_path: Path) -> Path:
    """Create a sample CVAT image dataset."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label>
          <name>car</name>
          <attributes>
            <attribute>
              <name>color</name>
              <values>red,blue</values>
            </attribute>
          </attributes>
        </label>
        <label><name>person</name></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="frame_000000.jpg" width="1920" height="1080">
    <box label="car" xtl="100" ytl="200" xbr="400" ybr="500" occluded="0">
      <attribute name="color">red</attribute>
    </box>
    <polygon label="person" points="500,300;600,300;600,500;500,500" occluded="1" />
  </image>
  <image id="1" name="frame_000001.jpg" width="1920" height="1080">
    <box label="person" xtl="200" ytl="100" xbr="350" ybr="400" occluded="0" />
  </image>
</annotations>
"""
    (tmp_path / "annotations.xml").write_text(xml_content)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "frame_000000.jpg").touch()
    (images_dir / "frame_000001.jpg").touch()
    return tmp_path


@pytest.fixture
def cvat_tracks_dataset(tmp_path: Path) -> Path:
    """Create a sample CVAT dataset with track annotations."""
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label><name>car</name></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="frame_000000.jpg" width="1280" height="720" />
  <image id="1" name="frame_000001.jpg" width="1280" height="720" />
  <track id="7" label="car">
    <box frame="0" xtl="100" ytl="120" xbr="220" ybr="260" outside="0" occluded="0" />
    <box frame="1" xtl="130" ytl="120" xbr="240" ybr="260" outside="0" occluded="1" />
    <box frame="2" xtl="160" ytl="120" xbr="270" ybr="260" outside="1" occluded="0" />
  </track>
</annotations>
"""
    (tmp_path / "annotations.xml").write_text(xml_content)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "frame_000000.jpg").touch()
    (images_dir / "frame_000001.jpg").touch()
    return tmp_path


class TestCvatLoader:
    """Tests for CvatLoader."""

    def test_load_dataset(self, cvat_dataset: Path) -> None:
        """Test loading a CVAT dataset."""
        loader = CvatLoader(cvat_dataset)
        dataset = loader.load()

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 2
        assert dataset.class_names == ["car", "person"]

    def test_parse_box_and_attributes(self, cvat_dataset: Path) -> None:
        """Test CVAT box parsing with attributes."""
        loader = CvatLoader(cvat_dataset)
        dataset = loader.load()

        sample = next(item for item in dataset if item.image_path.stem == "frame_000000")
        car = next(label for label in sample.labels if label.class_name == "car")

        assert car.bbox.cx == pytest.approx(250 / 1920, rel=0.01)
        assert car.bbox.cy == pytest.approx(350 / 1080, rel=0.01)
        assert car.attributes["shape_type"] == "box"
        assert car.attributes["color"] == "red"
        assert car.attributes["occluded"] is False

    def test_parse_polygon(self, cvat_dataset: Path) -> None:
        """Test polygon conversion to bbox while preserving points."""
        loader = CvatLoader(cvat_dataset)
        dataset = loader.load()

        sample = next(item for item in dataset if item.image_path.stem == "frame_000000")
        person = next(label for label in sample.labels if label.class_name == "person")

        assert person.attributes["shape_type"] == "polygon"
        assert "points" in person.attributes
        assert len(person.attributes["points"]) == 8
        assert person.attributes["occluded"] is True

    def test_parse_polyline_points_and_cuboid(self, tmp_path: Path) -> None:
        """Test support for polyline, points, and cuboid annotations."""
        xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label><name>lane</name></label>
        <label><name>keypoint</name></label>
        <label><name>box3d</name></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="scene.jpg" width="100" height="100">
    <polyline label="lane" points="10,10;40,20;70,25" />
    <points label="keypoint" points="50,50" />
    <cuboid label="box3d" points="20,20;40,20;40,40;20,40" />
  </image>
</annotations>
"""
        (tmp_path / "annotations.xml").write_text(xml_content)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "scene.jpg").touch()

        loader = CvatLoader(tmp_path)
        dataset = loader.load()
        sample = dataset[0]

        shape_types = {label.attributes.get("shape_type") for label in sample.labels}
        assert shape_types == {"polyline", "points", "cuboid"}

    def test_track_annotations(self, cvat_tracks_dataset: Path) -> None:
        """Test loading track annotations and skipping outside=1 frames."""
        loader = CvatLoader(cvat_tracks_dataset, load_tracks=True)
        dataset = loader.load()

        assert len(dataset) == 2  # frame=2 is outside=1 and not included

        for sample in dataset:
            assert len(sample.labels) == 1
            assert sample.labels[0].track_id == 7
            assert sample.labels[0].class_name == "car"

        assert dataset[1].labels[0].attributes["occluded"] is True

    def test_disable_track_loading(self, cvat_tracks_dataset: Path) -> None:
        """Test that track parsing can be disabled."""
        loader = CvatLoader(cvat_tracks_dataset, load_tracks=False)
        dataset = loader.load()

        assert len(dataset) == 2
        assert all(len(sample.labels) == 0 for sample in dataset)

    def test_iter_samples(self, cvat_dataset: Path) -> None:
        """Test lazy sample iteration."""
        loader = CvatLoader(cvat_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 2

    def test_validate_valid_dataset(self, cvat_dataset: Path) -> None:
        """Test validation on valid dataset."""
        loader = CvatLoader(cvat_dataset)
        warnings = loader.validate()
        assert warnings == []

    def test_validate_missing_xml(self, tmp_path: Path) -> None:
        """Test validation detects missing XML file."""
        loader = CvatLoader(tmp_path)
        warnings = loader.validate()
        assert any("XML" in warning for warning in warnings)

    def test_validate_missing_images(self, cvat_dataset: Path) -> None:
        """Test validation detects missing image files referenced in XML."""
        (cvat_dataset / "images" / "frame_000001.jpg").unlink()
        loader = CvatLoader(cvat_dataset)
        warnings = loader.validate()

        assert any("not found on disk" in warning for warning in warnings)

    def test_summary(self, cvat_tracks_dataset: Path) -> None:
        """Test summary includes CVAT-specific fields."""
        loader = CvatLoader(cvat_tracks_dataset)
        summary = loader.summary()

        assert summary["format"] == "cvat"
        assert summary["num_images"] == 2
        assert summary["num_labels"] == 1
        assert summary["has_tracks"] is True
        assert summary["xml_file"] is not None
