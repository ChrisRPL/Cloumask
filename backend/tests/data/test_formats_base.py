"""Tests for format base classes and registry."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.data.formats.base import (
    FormatExporter,
    FormatLoader,
    FormatRegistry,
    convert,
    detect_format,
    get_exporter,
    get_loader,
)
from backend.data.models import Dataset, Sample


class MockLoader(FormatLoader):
    """Test loader implementation."""

    format_name = "mock"
    description = "Mock format for testing"
    extensions = [".mock"]

    def load(self) -> Dataset:
        return Dataset(list(self.iter_samples()), name="mock")

    def iter_samples(self) -> Iterator[Sample]:
        for i in range(3):
            yield Sample(image_path=self.root_path / f"img{i}.jpg")

    def validate(self) -> list[str]:
        return []


class MockExporter(FormatExporter):
    """Test exporter implementation."""

    format_name = "mock"
    description = "Mock format for testing"

    def export(
        self, dataset: Dataset, *, copy_images: bool = True, image_subdir: str = "images"
    ) -> Path:
        self._ensure_output_dir()
        return self.output_path

    def validate_export(self) -> list[str]:
        return []


class TestFormatLoader:
    """Tests for FormatLoader base class."""

    def test_requires_existing_path(self, tmp_path: Path) -> None:
        """Test that non-existent path raises error."""
        with pytest.raises(FileNotFoundError):
            MockLoader(tmp_path / "nonexistent")

    def test_load_returns_dataset(self, tmp_path: Path) -> None:
        """Test that load returns a Dataset."""
        loader = MockLoader(tmp_path)
        ds = loader.load()
        assert isinstance(ds, Dataset)
        assert len(ds) == 3

    def test_iter_samples_yields(self, tmp_path: Path) -> None:
        """Test that iter_samples yields Sample objects."""
        loader = MockLoader(tmp_path)
        samples = list(loader.iter_samples())
        assert len(samples) == 3
        assert all(isinstance(s, Sample) for s in samples)

    def test_progress_callback(self, tmp_path: Path) -> None:
        """Test that progress callback is called."""
        callback = MagicMock()
        loader = MockLoader(tmp_path, progress_callback=callback)
        loader._report_progress(1, 10, "test")
        callback.assert_called_once_with(1, 10, "test")

    def test_summary_returns_dict(self, tmp_path: Path) -> None:
        """Test that summary returns format info."""
        loader = MockLoader(tmp_path)
        summary = loader.summary()
        assert summary["format"] == "mock"
        assert summary["root_path"] == str(tmp_path)

    def test_class_names_from_constructor(self, tmp_path: Path) -> None:
        """Test class names can be set via constructor."""
        loader = MockLoader(tmp_path, class_names=["car", "person"])
        assert loader.get_class_names() == ["car", "person"]


class TestFormatExporter:
    """Tests for FormatExporter base class."""

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Test that export creates output directory."""
        output = tmp_path / "output"
        exporter = MockExporter(output)
        ds = Dataset([])
        exporter.export(ds)
        assert output.exists()

    def test_fails_on_non_empty_without_overwrite(self, tmp_path: Path) -> None:
        """Test that non-empty dir raises without overwrite."""
        output = tmp_path / "output"
        output.mkdir()
        (output / "existing.txt").touch()
        exporter = MockExporter(output, overwrite=False)
        with pytest.raises(FileExistsError):
            exporter.export(Dataset([]))

    def test_allows_overwrite(self, tmp_path: Path) -> None:
        """Test that overwrite=True allows non-empty dir."""
        output = tmp_path / "output"
        output.mkdir()
        (output / "existing.txt").touch()
        exporter = MockExporter(output, overwrite=True)
        exporter.export(Dataset([]))  # Should not raise

    def test_progress_callback(self, tmp_path: Path) -> None:
        """Test that progress callback is called."""
        callback = MagicMock()
        exporter = MockExporter(tmp_path, progress_callback=callback)
        exporter._report_progress(5, 10, "exporting")
        callback.assert_called_once_with(5, 10, "exporting")


class TestFormatRegistry:
    """Tests for FormatRegistry."""

    def setup_method(self) -> None:
        """Register mock format for tests."""
        FormatRegistry.register_loader(MockLoader)
        FormatRegistry.register_exporter(MockExporter)

    def test_register_loader(self) -> None:
        """Test loader registration."""
        assert "mock" in FormatRegistry._loaders

    def test_register_exporter(self) -> None:
        """Test exporter registration."""
        assert "mock" in FormatRegistry._exporters

    def test_get_loader_explicit_format(self, tmp_path: Path) -> None:
        """Test getting loader with explicit format."""
        loader = get_loader(tmp_path, format_name="mock")
        assert isinstance(loader, MockLoader)

    def test_get_exporter(self, tmp_path: Path) -> None:
        """Test getting exporter."""
        exporter = get_exporter(tmp_path, "mock")
        assert isinstance(exporter, MockExporter)

    def test_get_loader_unknown_format(self, tmp_path: Path) -> None:
        """Test that unknown format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown format"):
            get_loader(tmp_path, format_name="unknown_format_xyz")

    def test_get_exporter_unknown_format(self, tmp_path: Path) -> None:
        """Test that unknown export format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown export format"):
            get_exporter(tmp_path, "unknown_format_xyz")

    def test_list_formats(self) -> None:
        """Test listing available formats."""
        formats = FormatRegistry.list_formats()
        assert "mock" in formats
        assert formats["mock"]["loader"] is True
        assert formats["mock"]["exporter"] is True

    def test_singleton_pattern(self) -> None:
        """Test that FormatRegistry is a singleton."""
        r1 = FormatRegistry()
        r2 = FormatRegistry()
        assert r1 is r2


class TestDetectFormat:
    """Tests for format auto-detection."""

    def test_detect_yolo_by_data_yaml(self, tmp_path: Path) -> None:
        """Test YOLO detection by data.yaml."""
        (tmp_path / "data.yaml").touch()
        assert detect_format(tmp_path) == "yolo"

    def test_detect_coco_by_annotations_dir(self, tmp_path: Path) -> None:
        """Test COCO detection by annotations directory."""
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        (ann_dir / "instances_train.json").write_text('{"images": [], "annotations": []}')
        assert detect_format(tmp_path) == "coco"

    def test_detect_coco_by_json_structure(self, tmp_path: Path) -> None:
        """Test COCO detection by JSON structure."""
        import json

        (tmp_path / "instances.json").write_text(
            json.dumps({"images": [], "annotations": [], "categories": []})
        )
        assert detect_format(tmp_path) == "coco"

    def test_detect_voc_by_annotations_xml(self, tmp_path: Path) -> None:
        """Test VOC detection by Annotations directory."""
        ann_dir = tmp_path / "Annotations"
        ann_dir.mkdir()
        (ann_dir / "img001.xml").touch()
        assert detect_format(tmp_path) == "voc"

    def test_detect_kitti_by_label_2(self, tmp_path: Path) -> None:
        """Test KITTI detection by label_2 directory."""
        (tmp_path / "label_2").mkdir()
        assert detect_format(tmp_path) == "kitti"

    def test_detect_kitti_by_split_label_2(self, tmp_path: Path) -> None:
        """Test KITTI detection by split/label_2 directory."""
        (tmp_path / "training" / "label_2").mkdir(parents=True)
        assert detect_format(tmp_path) == "kitti"

    def test_detect_cvat_by_xml(self, tmp_path: Path) -> None:
        """Test CVAT detection by XML with annotations root."""
        xml_content = '<?xml version="1.0"?><annotations></annotations>'
        (tmp_path / "annotations.xml").write_text(xml_content)
        assert detect_format(tmp_path) == "cvat"

    def test_detect_nuscenes_by_version_dir(self, tmp_path: Path) -> None:
        """Test nuScenes detection by v1.0-* directory."""
        ns_dir = tmp_path / "v1.0-trainval"
        ns_dir.mkdir()
        (ns_dir / "sample_annotation.json").touch()
        assert detect_format(tmp_path) == "nuscenes"

    def test_detect_openlabel_by_json(self, tmp_path: Path) -> None:
        """Test OpenLABEL detection by JSON with openlabel key."""
        import json

        (tmp_path / "labels.json").write_text(json.dumps({"openlabel": {"metadata": {}}}))
        assert detect_format(tmp_path) == "openlabel"

    def test_returns_none_for_unknown(self, tmp_path: Path) -> None:
        """Test that unknown format returns None."""
        (tmp_path / "random_file.txt").touch()
        assert detect_format(tmp_path) is None

    def test_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        """Test that empty directory returns None."""
        assert detect_format(tmp_path) is None


class TestConvert:
    """Tests for format conversion helper."""

    def test_augmentation_requires_copy_images(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Loader:
            def load(self) -> Dataset:
                return Dataset([])

        monkeypatch.setattr(
            "backend.data.formats.base.get_loader",
            lambda *args, **kwargs: _Loader(),
        )

        with pytest.raises(ValueError, match="copy_images=True"):
            convert(
                Path("/tmp/input"),
                Path("/tmp/output"),
                "mock",
                augment=True,
                copy_images=False,
            )

    def test_augmentation_path_uses_augmented_dataset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source_dataset = Dataset([Sample(image_path=tmp_path / "frame.jpg")], name="source")
        augmented_dataset = Dataset([Sample(image_path=tmp_path / "aug.jpg")], name="augmented")
        calls: dict[str, object] = {}

        class _Loader:
            def load(self) -> Dataset:
                return source_dataset

        class _Exporter:
            def export(
                self,
                dataset: Dataset,
                *,
                copy_images: bool = True,
                image_subdir: str = "images",
            ) -> Path:
                calls["export_dataset"] = dataset
                calls["copy_images"] = copy_images
                calls["image_subdir"] = image_subdir
                return tmp_path / "converted"

        def _fake_augment_dataset(
            dataset: Dataset,
            *,
            output_dir: Path,
            preset: str,
            copies_per_sample: int,
            include_original: bool,
        ) -> Dataset:
            calls["augment_dataset"] = dataset
            calls["augment_output_dir"] = output_dir
            calls["augment_preset"] = preset
            calls["augment_copies"] = copies_per_sample
            calls["augment_include_original"] = include_original
            return augmented_dataset

        monkeypatch.setattr(
            "backend.data.formats.base.get_loader",
            lambda *args, **kwargs: _Loader(),
        )
        monkeypatch.setattr(
            "backend.data.formats.base.get_exporter",
            lambda *args, **kwargs: _Exporter(),
        )
        monkeypatch.setattr(
            "backend.data.augmentation.augment_dataset",
            _fake_augment_dataset,
        )

        output = convert(
            input_path=tmp_path / "input",
            output_path=tmp_path / "output",
            target_format="mock",
            augment=True,
            augmentation_preset="heavy",
            augmentation_copies=3,
            include_original=False,
            copy_images=True,
        )

        assert output == tmp_path / "converted"
        assert calls["augment_dataset"] is source_dataset
        assert isinstance(calls["augment_output_dir"], Path)
        assert calls["augment_preset"] == "heavy"
        assert calls["augment_copies"] == 3
        assert calls["augment_include_original"] is False
        assert calls["export_dataset"] is augmented_dataset
        assert calls["copy_images"] is True
