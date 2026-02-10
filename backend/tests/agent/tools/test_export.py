"""Tests for export agent tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.agent.tools.export import ExportTool
from backend.data.models import BBox, Dataset, Label, Sample


def _make_dataset(dataset_path: Path) -> Dataset:
    """Create a deterministic dataset fixture for export tests."""
    samples: list[Sample] = []

    for idx in range(10):
        image_path = dataset_path / f"frame_{idx:03d}.jpg"
        image_path.write_bytes(b"image-data")

        class_name = "car" if idx < 6 else "person"
        class_id = 0 if class_name == "car" else 1
        confidence = 0.95 if idx % 2 == 0 else 0.45

        samples.append(
            Sample(
                image_path=image_path,
                labels=[
                    Label(
                        class_name=class_name,
                        class_id=class_id,
                        bbox=BBox.from_cxcywh(0.5, 0.5, 0.25, 0.2),
                        confidence=confidence,
                    )
                ],
            )
        )

    return Dataset(samples=samples, name="export-fixture", class_names=["car", "person"])


class _FakeLoader:
    def __init__(
        self,
        dataset: Dataset,
        *,
        warnings: list[str] | None = None,
        raise_on_load: Exception | None = None,
    ) -> None:
        self._dataset = dataset
        self._warnings = warnings or []
        self._raise_on_load = raise_on_load

    def validate(self) -> list[str]:
        return list(self._warnings)

    def load(self) -> Dataset:
        if self._raise_on_load:
            raise self._raise_on_load
        return self._dataset


class _FakeExporter:
    def __init__(self, output_path: Path, *, warnings: list[str] | None = None) -> None:
        self.output_path = output_path
        self._warnings = warnings or []
        self.export_calls: list[dict[str, Any]] = []

    def export(
        self,
        dataset: Dataset,
        *,
        copy_images: bool = True,
        image_subdir: str = "images",
    ) -> Path:
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.export_calls.append(
            {
                "dataset": dataset,
                "copy_images": copy_images,
                "image_subdir": image_subdir,
            }
        )
        return self.output_path

    def validate_export(self) -> list[str]:
        return list(self._warnings)


def _format_map() -> dict[str, dict[str, bool]]:
    return {
        "yolo": {"loader": True, "exporter": True},
        "coco": {"loader": True, "exporter": True},
        "voc": {"loader": True, "exporter": True},
    }


class TestExportTool:
    @pytest.mark.asyncio
    async def test_export_success_with_filters_and_split(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        dataset = _make_dataset(source)
        loader = _FakeLoader(dataset, warnings=["source warning"])
        calls: dict[str, Any] = {"exporters": []}

        monkeypatch.setattr("backend.agent.tools.export.list_formats", _format_map)
        monkeypatch.setattr("backend.agent.tools.export.detect_format", lambda _p: "yolo")

        def _fake_get_loader(path: Path, *, format_name: str | None = None, **kwargs: Any) -> _FakeLoader:
            calls["loader_path"] = path
            calls["loader_format"] = format_name
            calls["loader_progress"] = kwargs.get("progress_callback")
            return loader

        def _fake_get_exporter(path: Path, format_name: str, **kwargs: Any) -> _FakeExporter:
            exporter = _FakeExporter(path, warnings=[f"{path.name} warning"])
            calls["exporters"].append(
                {
                    "path": path,
                    "format": format_name,
                    "overwrite": kwargs.get("overwrite"),
                    "progress_callback": kwargs.get("progress_callback"),
                    "exporter": exporter,
                }
            )
            return exporter

        monkeypatch.setattr("backend.agent.tools.export.get_loader", _fake_get_loader)
        monkeypatch.setattr("backend.agent.tools.export.get_exporter", _fake_get_exporter)

        tool = ExportTool()
        result = await tool.run(
            source_path=str(source),
            output_path=str(output),
            output_format="coco",
            classes=["car", "bus"],
            min_confidence=0.5,
            split=True,
            train_ratio=6.0,
            val_ratio=2.0,
            test_ratio=2.0,
            stratify=True,
            seed=17,
            copy_images=False,
            overwrite=True,
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["source_path"] == str(source)
        assert result.data["output_path"] == str(output)
        assert result.data["source_format"] == "yolo"
        assert result.data["output_format"] == "coco"
        assert result.data["format"] == "coco"
        assert result.data["split_requested"] is True
        assert result.data["filtered_classes"] == ["car", "bus"]
        assert result.data["confidence_threshold"] == 0.5
        assert result.data["images_copied"] is False
        assert result.data["requested_split_ratios"] == {
            "train": 0.6,
            "val": 0.2,
            "test": 0.2,
        }
        assert result.data["dataset_warnings"] == ["source warning"]
        assert len(result.data["filter_warnings"]) == 1
        assert "not found" in result.data["filter_warnings"][0].lower()
        assert len(result.data["export_warnings"]) == 3
        assert result.data["num_samples"] == 6
        assert result.data["num_labels"] == 3
        assert result.data["num_classes"] == 2

        splits = result.data["splits"]
        assert splits is not None
        assert set(splits.keys()) == {"train", "val", "test"}
        assert sum(split["num_samples"] for split in splits.values()) == 6

        split_output_paths = result.data["split_output_paths"]
        assert split_output_paths is not None
        assert set(split_output_paths.keys()) == {"train", "val", "test"}

        assert calls["loader_path"] == source
        assert calls["loader_format"] == "yolo"
        assert callable(calls["loader_progress"])
        assert len(calls["exporters"]) == 3
        assert {entry["path"].name for entry in calls["exporters"]} == {"train", "val", "test"}
        assert {entry["format"] for entry in calls["exporters"]} == {"coco"}
        assert all(entry["overwrite"] is True for entry in calls["exporters"])
        assert all(callable(entry["progress_callback"]) for entry in calls["exporters"])
        assert all(
            entry["exporter"].export_calls[0]["copy_images"] is False
            for entry in calls["exporters"]
        )

    @pytest.mark.asyncio
    async def test_export_success_without_split(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        dataset = _make_dataset(source)
        loader = _FakeLoader(dataset)
        exporter = _FakeExporter(output, warnings=["export warning"])
        calls: dict[str, Any] = {}

        monkeypatch.setattr("backend.agent.tools.export.list_formats", _format_map)

        def _unexpected_detect(_: Path) -> str:
            raise AssertionError("detect_format should not be called with source_format")

        monkeypatch.setattr("backend.agent.tools.export.detect_format", _unexpected_detect)

        def _fake_get_loader(path: Path, *, format_name: str | None = None, **kwargs: Any) -> _FakeLoader:
            calls["loader_path"] = path
            calls["loader_format"] = format_name
            return loader

        def _fake_get_exporter(path: Path, format_name: str, **kwargs: Any) -> _FakeExporter:
            calls["exporter_path"] = path
            calls["exporter_format"] = format_name
            calls["overwrite"] = kwargs.get("overwrite")
            return exporter

        monkeypatch.setattr("backend.agent.tools.export.get_loader", _fake_get_loader)
        monkeypatch.setattr("backend.agent.tools.export.get_exporter", _fake_get_exporter)

        tool = ExportTool()
        result = await tool.run(
            source_path=str(source),
            output_path=str(output),
            output_format="voc",
            source_format="yolo",
            split=False,
            copy_images=True,
            overwrite=False,
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["source_format"] == "yolo"
        assert result.data["output_format"] == "voc"
        assert result.data["split_requested"] is False
        assert result.data["splits"] is None
        assert result.data["split_output_paths"] is None
        assert result.data["output_path"] == str(output)
        assert result.data["export_warnings"] == ["export warning"]
        assert calls["loader_path"] == source
        assert calls["loader_format"] == "yolo"
        assert calls["exporter_path"] == output
        assert calls["exporter_format"] == "voc"
        assert calls["overwrite"] is False
        assert len(exporter.export_calls) == 1
        assert exporter.export_calls[0]["copy_images"] is True

    @pytest.mark.asyncio
    async def test_export_fails_for_missing_source(self, tmp_path: Path) -> None:
        tool = ExportTool()
        result = await tool.run(
            source_path=str(tmp_path / "missing"),
            output_path=str(tmp_path / "output"),
            output_format="yolo",
        )

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_export_fails_for_invalid_confidence(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()

        tool = ExportTool()
        result = await tool.run(
            source_path=str(source),
            output_path=str(tmp_path / "output"),
            output_format="yolo",
            min_confidence=1.2,
        )

        assert result.success is False
        assert result.error is not None
        assert "min_confidence" in result.error.lower()

    @pytest.mark.asyncio
    async def test_export_fails_when_source_format_cannot_be_detected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()

        monkeypatch.setattr("backend.agent.tools.export.list_formats", _format_map)
        monkeypatch.setattr("backend.agent.tools.export.detect_format", lambda _p: None)

        tool = ExportTool()
        result = await tool.run(
            source_path=str(source),
            output_path=str(tmp_path / "output"),
            output_format="yolo",
        )

        assert result.success is False
        assert result.error is not None
        assert "could not detect source format" in result.error.lower()

    @pytest.mark.asyncio
    async def test_export_fails_for_invalid_split_ratios(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()

        monkeypatch.setattr("backend.agent.tools.export.list_formats", _format_map)
        monkeypatch.setattr("backend.agent.tools.export.detect_format", lambda _p: "yolo")

        tool = ExportTool()
        result = await tool.run(
            source_path=str(source),
            output_path=str(tmp_path / "output"),
            output_format="yolo",
            split=True,
            train_ratio=0.0,
            val_ratio=0.0,
            test_ratio=0.0,
        )

        assert result.success is False
        assert result.error is not None
        assert "at least one split ratio" in result.error.lower()

    @pytest.mark.asyncio
    async def test_export_returns_error_when_loader_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()

        dataset = _make_dataset(source)
        loader = _FakeLoader(dataset, raise_on_load=RuntimeError("boom"))

        monkeypatch.setattr("backend.agent.tools.export.list_formats", _format_map)
        monkeypatch.setattr("backend.agent.tools.export.detect_format", lambda _p: "yolo")
        monkeypatch.setattr(
            "backend.agent.tools.export.get_loader",
            lambda _path, *, format_name=None, **kwargs: loader,
        )

        tool = ExportTool()
        result = await tool.run(
            source_path=str(source),
            output_path=str(tmp_path / "output"),
            output_format="coco",
        )

        assert result.success is False
        assert result.error is not None
        assert "dataset export failed" in result.error.lower()
        assert "boom" in result.error.lower()


class TestExportToolRegistration:
    def test_export_tool_registered(self) -> None:
        """export should be available in the global tool registry."""
        from backend.agent.tools import get_tool_registry, initialize_tools

        initialize_tools()
        registry = get_tool_registry()

        assert registry.has("export")
        tool = registry.get("export")
        assert tool is not None
        schema = tool.get_schema()
        properties = schema["function"]["parameters"]["properties"]
        assert schema["function"]["name"] == "export"
        assert "source_path" in properties
        assert "output_path" in properties
        assert "output_format" in properties
