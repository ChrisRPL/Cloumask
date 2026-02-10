"""Tests for split_dataset agent tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.agent.tools.split import SplitDatasetTool
from backend.data.models import BBox, Dataset, Label, Sample


def _make_dataset(dataset_path: Path) -> Dataset:
    """Create a small deterministic dataset fixture."""
    samples: list[Sample] = []

    for idx in range(10):
        image_path = dataset_path / f"frame_{idx:03d}.jpg"
        image_path.write_bytes(b"image-data")

        class_name = "car" if idx < 7 else "person"
        class_id = 0 if class_name == "car" else 1
        samples.append(
            Sample(
                image_path=image_path,
                labels=[
                    Label(
                        class_name=class_name,
                        class_id=class_id,
                        bbox=BBox.from_cxcywh(0.5, 0.5, 0.25, 0.2),
                    )
                ],
            )
        )

    return Dataset(samples=samples, name="split-fixture", class_names=["car", "person"])


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
        self.export_calls.append({
            "dataset": dataset,
            "copy_images": copy_images,
            "image_subdir": image_subdir,
        })
        return self.output_path

    def validate_export(self) -> list[str]:
        return list(self._warnings)


def _format_map() -> dict[str, dict[str, bool]]:
    return {
        "yolo": {"loader": True, "exporter": True},
        "coco": {"loader": True, "exporter": True},
        "voc": {"loader": True, "exporter": True},
    }


class TestSplitDatasetTool:
    @pytest.mark.asyncio
    async def test_split_success_with_auto_detect(
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

        monkeypatch.setattr("backend.agent.tools.split.list_formats", _format_map)
        monkeypatch.setattr("backend.agent.tools.split.detect_format", lambda _p: "yolo")

        def _fake_get_loader(path: Path, *, format_name: str | None = None, **kwargs: Any) -> _FakeLoader:
            calls["loader_path"] = path
            calls["loader_format"] = format_name
            calls["loader_progress"] = kwargs.get("progress_callback")
            return loader

        def _fake_get_exporter(path: Path, format_name: str, **kwargs: Any) -> _FakeExporter:
            exporter = _FakeExporter(path, warnings=[f"{path.name} warning"])
            calls["exporters"].append({
                "path": path,
                "format": format_name,
                "overwrite": kwargs.get("overwrite"),
                "progress_callback": kwargs.get("progress_callback"),
                "exporter": exporter,
            })
            return exporter

        monkeypatch.setattr("backend.agent.tools.split.get_loader", _fake_get_loader)
        monkeypatch.setattr("backend.agent.tools.split.get_exporter", _fake_get_exporter)

        tool = SplitDatasetTool()
        result = await tool.run(
            path=str(source),
            output_path=str(output),
            train_ratio=8.0,
            val_ratio=1.0,
            test_ratio=1.0,
            stratify=True,
            seed=9,
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["source_path"] == str(source)
        assert result.data["output_path"] == str(output)
        assert result.data["source_format"] == "yolo"
        assert result.data["output_format"] == "yolo"
        assert result.data["stratified"] is True
        assert result.data["seed"] == 9
        assert result.data["requested_ratios"] == {"train": 0.8, "val": 0.1, "test": 0.1}
        assert result.data["total_samples"] == 10
        assert result.data["total_labels"] == 10
        assert result.data["dataset_warnings"] == ["source warning"]
        assert len(result.data["splits"]) == 3
        assert [split["name"] for split in result.data["splits"]] == ["train", "val", "test"]
        assert sum(split["num_samples"] for split in result.data["splits"]) == 10
        assert len(result.data["export_warnings"]) == 3

        assert calls["loader_path"] == source
        assert calls["loader_format"] == "yolo"
        assert callable(calls["loader_progress"])
        assert len(calls["exporters"]) == 3
        assert {entry["path"].name for entry in calls["exporters"]} == {"train", "val", "test"}
        assert {entry["format"] for entry in calls["exporters"]} == {"yolo"}
        assert all(entry["overwrite"] is True for entry in calls["exporters"])
        assert all(callable(entry["progress_callback"]) for entry in calls["exporters"])
        assert all(
            entry["exporter"].export_calls[0]["copy_images"] is True for entry in calls["exporters"]
        )

    @pytest.mark.asyncio
    async def test_split_uses_explicit_formats(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        dataset = _make_dataset(source)
        loader = _FakeLoader(dataset)
        calls: dict[str, Any] = {"export_formats": [], "copy_images": []}

        monkeypatch.setattr("backend.agent.tools.split.list_formats", _format_map)

        def _unexpected_detect(_: Path) -> str:
            raise AssertionError("detect_format should not be called when format is provided")

        monkeypatch.setattr("backend.agent.tools.split.detect_format", _unexpected_detect)
        monkeypatch.setattr(
            "backend.agent.tools.split.get_loader",
            lambda _path, *, format_name=None, **kwargs: loader,
        )

        def _fake_get_exporter(path: Path, format_name: str, **kwargs: Any) -> _FakeExporter:
            exporter = _FakeExporter(path)
            calls["export_formats"].append(format_name)
            calls["overwrite"] = kwargs.get("overwrite")
            calls["exporter"] = exporter
            return exporter

        monkeypatch.setattr("backend.agent.tools.split.get_exporter", _fake_get_exporter)

        tool = SplitDatasetTool()
        result = await tool.run(
            path=str(source),
            output_path=str(output),
            format="coco",
            output_format="voc",
            stratify=False,
            copy_images=False,
            overwrite=False,
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["source_format"] == "coco"
        assert result.data["output_format"] == "voc"
        assert result.data["stratified"] is False
        assert calls["overwrite"] is False
        assert set(calls["export_formats"]) == {"voc"}
        assert calls["exporter"].export_calls[0]["copy_images"] is False

    @pytest.mark.asyncio
    async def test_split_fails_for_missing_source(self, tmp_path: Path) -> None:
        tool = SplitDatasetTool()
        result = await tool.run(
            path=str(tmp_path / "missing"),
            output_path=str(tmp_path / "output"),
        )

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_split_fails_when_format_cannot_be_detected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()

        monkeypatch.setattr("backend.agent.tools.split.list_formats", _format_map)
        monkeypatch.setattr("backend.agent.tools.split.detect_format", lambda _p: None)

        tool = SplitDatasetTool()
        result = await tool.run(
            path=str(source),
            output_path=str(tmp_path / "output"),
        )

        assert result.success is False
        assert result.error is not None
        assert "could not detect dataset format" in result.error.lower()

    @pytest.mark.asyncio
    async def test_split_fails_for_unsupported_output_format(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()

        monkeypatch.setattr(
            "backend.agent.tools.split.list_formats",
            lambda: {"yolo": {"loader": True, "exporter": True}},
        )

        tool = SplitDatasetTool()
        result = await tool.run(
            path=str(source),
            output_path=str(tmp_path / "output"),
            format="yolo",
            output_format="voc",
        )

        assert result.success is False
        assert result.error is not None
        assert "unsupported output_format" in result.error.lower()

    @pytest.mark.asyncio
    async def test_split_fails_for_invalid_ratios(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()

        monkeypatch.setattr("backend.agent.tools.split.list_formats", _format_map)

        tool = SplitDatasetTool()
        result = await tool.run(
            path=str(source),
            output_path=str(tmp_path / "output"),
            format="yolo",
            train_ratio=0.0,
            val_ratio=0.0,
            test_ratio=0.0,
        )

        assert result.success is False
        assert result.error is not None
        assert "at least one split ratio" in result.error.lower()

    @pytest.mark.asyncio
    async def test_split_returns_error_when_loader_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()

        dataset = _make_dataset(source)
        loader = _FakeLoader(dataset, raise_on_load=RuntimeError("boom"))

        monkeypatch.setattr("backend.agent.tools.split.list_formats", _format_map)
        monkeypatch.setattr("backend.agent.tools.split.detect_format", lambda _p: "yolo")
        monkeypatch.setattr(
            "backend.agent.tools.split.get_loader",
            lambda _path, *, format_name=None, **kwargs: loader,
        )

        tool = SplitDatasetTool()
        result = await tool.run(
            path=str(source),
            output_path=str(tmp_path / "output"),
        )

        assert result.success is False
        assert result.error is not None
        assert "dataset split failed" in result.error.lower()
        assert "boom" in result.error.lower()


class TestSplitDatasetToolRegistration:
    def test_split_dataset_tool_registered(self) -> None:
        """split_dataset should be available in the global tool registry."""
        from backend.agent.tools import get_tool_registry, initialize_tools

        initialize_tools()
        registry = get_tool_registry()

        assert registry.has("split_dataset")
        tool = registry.get("split_dataset")
        assert tool is not None
        schema = tool.get_schema()
        properties = schema["function"]["parameters"]["properties"]
        assert schema["function"]["name"] == "split_dataset"
        assert "path" in properties
        assert "output_path" in properties
        assert "train_ratio" in properties
        assert "val_ratio" in properties
        assert "test_ratio" in properties
        assert "stratify" in properties
        assert "output_format" in properties
