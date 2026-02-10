"""Tests for convert_format agent tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.agent.tools.convert import ConvertFormatTool
from backend.data.models import BBox, Dataset, Label, Sample


def _make_dataset(tmp_path: Path) -> Dataset:
    """Create a tiny labeled dataset for conversion tests."""
    image_path = tmp_path / "frame_0001.jpg"
    image_path.write_bytes(b"fake-image")

    sample = Sample(
        image_path=image_path,
        labels=[
            Label(
                class_name="car",
                class_id=0,
                bbox=BBox.from_cxcywh(0.5, 0.5, 0.4, 0.3),
            )
        ],
    )
    return Dataset([sample], name="source", class_names=["car"])


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
        "kitti": {"loader": True, "exporter": True},
    }


class TestConvertFormatTool:
    @pytest.mark.asyncio
    async def test_convert_success_with_auto_detect(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        dataset = _make_dataset(tmp_path)
        loader = _FakeLoader(dataset, warnings=["source warning"])
        exporter = _FakeExporter(output, warnings=["export warning"])
        calls: dict[str, Any] = {}

        monkeypatch.setattr("backend.agent.tools.convert.list_formats", _format_map)
        monkeypatch.setattr("backend.agent.tools.convert.detect_format", lambda _p: "yolo")

        def _fake_get_loader(path: Path, *, format_name: str | None = None, **kwargs: Any) -> _FakeLoader:
            calls["loader_path"] = path
            calls["loader_format"] = format_name
            calls["loader_progress"] = kwargs.get("progress_callback")
            return loader

        def _fake_get_exporter(path: Path, format_name: str, **kwargs: Any) -> _FakeExporter:
            calls["exporter_path"] = path
            calls["exporter_format"] = format_name
            calls["overwrite"] = kwargs.get("overwrite")
            calls["exporter_progress"] = kwargs.get("progress_callback")
            return exporter

        monkeypatch.setattr("backend.agent.tools.convert.get_loader", _fake_get_loader)
        monkeypatch.setattr("backend.agent.tools.convert.get_exporter", _fake_get_exporter)

        tool = ConvertFormatTool()
        result = await tool.run(
            source_path=str(source),
            output_path=str(output),
            target_format="coco",
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["source_format"] == "yolo"
        assert result.data["target_format"] == "coco"
        assert result.data["num_samples"] == 1
        assert result.data["num_labels"] == 1
        assert result.data["warnings"] == ["source warning", "export warning"]
        assert result.data["output_path"] == str(output)
        assert callable(calls["loader_progress"])
        assert callable(calls["exporter_progress"])
        assert calls["loader_path"] == source
        assert calls["loader_format"] == "yolo"
        assert calls["exporter_path"] == output
        assert calls["exporter_format"] == "coco"
        assert calls["overwrite"] is True
        assert len(exporter.export_calls) == 1
        assert exporter.export_calls[0]["copy_images"] is True

    @pytest.mark.asyncio
    async def test_convert_uses_explicit_source_format(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        dataset = _make_dataset(tmp_path)
        loader = _FakeLoader(dataset)
        exporter = _FakeExporter(output)
        calls: dict[str, Any] = {}

        monkeypatch.setattr("backend.agent.tools.convert.list_formats", _format_map)

        def _unexpected_detect(_: Path) -> str:
            raise AssertionError("detect_format should not be called when source_format is provided")

        monkeypatch.setattr("backend.agent.tools.convert.detect_format", _unexpected_detect)

        def _fake_get_loader(path: Path, *, format_name: str | None = None, **kwargs: Any) -> _FakeLoader:
            calls["loader_format"] = format_name
            return loader

        def _fake_get_exporter(path: Path, format_name: str, **kwargs: Any) -> _FakeExporter:
            calls["overwrite"] = kwargs.get("overwrite")
            return exporter

        monkeypatch.setattr("backend.agent.tools.convert.get_loader", _fake_get_loader)
        monkeypatch.setattr("backend.agent.tools.convert.get_exporter", _fake_get_exporter)

        tool = ConvertFormatTool()
        result = await tool.run(
            source_path=str(source),
            output_path=str(output),
            target_format="yolo",
            source_format="coco",
            copy_images=False,
            overwrite=False,
        )

        assert result.success is True
        assert calls["loader_format"] == "coco"
        assert calls["overwrite"] is False
        assert exporter.export_calls[0]["copy_images"] is False

    @pytest.mark.asyncio
    async def test_convert_fails_for_missing_source(self, tmp_path: Path) -> None:
        tool = ConvertFormatTool()
        result = await tool.run(
            source_path=str(tmp_path / "missing"),
            output_path=str(tmp_path / "output"),
            target_format="yolo",
        )

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_convert_fails_for_unsupported_target_format(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()

        monkeypatch.setattr(
            "backend.agent.tools.convert.list_formats",
            lambda: {"yolo": {"loader": True, "exporter": True}},
        )

        tool = ConvertFormatTool()
        result = await tool.run(
            source_path=str(source),
            output_path=str(tmp_path / "output"),
            target_format="coco",
        )

        assert result.success is False
        assert result.error is not None
        assert "unsupported target_format" in result.error.lower()

    @pytest.mark.asyncio
    async def test_convert_fails_when_source_format_cannot_be_detected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()

        monkeypatch.setattr("backend.agent.tools.convert.list_formats", _format_map)
        monkeypatch.setattr("backend.agent.tools.convert.detect_format", lambda _p: None)

        tool = ConvertFormatTool()
        result = await tool.run(
            source_path=str(source),
            output_path=str(tmp_path / "output"),
            target_format="yolo",
        )

        assert result.success is False
        assert result.error is not None
        assert "could not detect source format" in result.error.lower()

    @pytest.mark.asyncio
    async def test_convert_returns_error_when_loader_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source"
        source.mkdir()

        dataset = _make_dataset(tmp_path)
        loader = _FakeLoader(dataset, raise_on_load=RuntimeError("boom"))

        monkeypatch.setattr("backend.agent.tools.convert.list_formats", _format_map)
        monkeypatch.setattr("backend.agent.tools.convert.detect_format", lambda _p: "yolo")
        monkeypatch.setattr(
            "backend.agent.tools.convert.get_loader",
            lambda _path, *, format_name=None, **kwargs: loader,
        )

        tool = ConvertFormatTool()
        result = await tool.run(
            source_path=str(source),
            output_path=str(tmp_path / "output"),
            target_format="coco",
        )

        assert result.success is False
        assert result.error is not None
        assert "conversion failed" in result.error.lower()
        assert "boom" in result.error.lower()


class TestConvertFormatToolRegistration:
    def test_convert_format_tool_registered(self) -> None:
        """convert_format should be available in the global tool registry."""
        from backend.agent.tools import get_tool_registry, initialize_tools

        initialize_tools()
        registry = get_tool_registry()

        assert registry.has("convert_format")
        tool = registry.get("convert_format")
        assert tool is not None
        schema = tool.get_schema()
        properties = schema["function"]["parameters"]["properties"]
        assert schema["function"]["name"] == "convert_format"
        assert "source_path" in properties
        assert "output_path" in properties
        assert "target_format" in properties
