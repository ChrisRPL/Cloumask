"""Tests for label_qa agent tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.agent.tools.qa import LabelQATool
from backend.data.models import BBox, Dataset, Label, Sample
from backend.data.qa import IssueType, QAIssue, QAResult


def _make_dataset(dataset_path: Path) -> Dataset:
    image_a = dataset_path / "frame_a.jpg"
    image_b = dataset_path / "frame_b.jpg"
    image_a.write_bytes(b"image-a")
    image_b.write_bytes(b"image-b")

    return Dataset(
        samples=[
            Sample(
                image_path=image_a,
                labels=[
                    Label(
                        class_name="car",
                        class_id=0,
                        bbox=BBox.from_cxcywh(0.5, 0.5, 0.3, 0.2),
                    )
                ],
            ),
            Sample(
                image_path=image_b,
                labels=[],
            ),
        ],
        name="qa-fixture",
        class_names=["car"],
    )


def _make_qa_result(dataset: Dataset, dataset_path: Path) -> QAResult:
    return QAResult(
        issues=[
            QAIssue(
                issue_type=IssueType.MISSING_LABELS,
                severity="warning",
                sample_path=dataset_path / "frame_b.jpg",
                label_index=None,
                description="missing labels",
            ),
            QAIssue(
                issue_type=IssueType.MISSING_IMAGE,
                severity="error",
                sample_path=dataset_path / "missing.jpg",
                label_index=None,
                description="missing image",
            ),
            QAIssue(
                issue_type=IssueType.OVERLAPPING_BOXES,
                severity="warning",
                sample_path=dataset_path / "frame_a.jpg",
                label_index=0,
                description="overlap",
            ),
        ],
        stats=dataset.stats(),
        checks_run=["missing_labels", "missing_images", "overlapping_boxes"],
    )


class _FakeLoader:
    def __init__(self, dataset: Dataset, *, warnings: list[str] | None = None) -> None:
        self._dataset = dataset
        self._warnings = warnings or []

    def validate(self) -> list[str]:
        return list(self._warnings)

    def load(self) -> Dataset:
        return self._dataset


class TestLabelQATool:
    @pytest.mark.asyncio
    async def test_label_qa_success_with_auto_detect_and_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        dataset = _make_dataset(dataset_path)
        qa_result = _make_qa_result(dataset, dataset_path)
        loader = _FakeLoader(dataset, warnings=["dataset warning"])
        calls: dict[str, Any] = {}

        monkeypatch.setattr(
            "backend.agent.tools.qa.list_formats",
            lambda: {"yolo": {"loader": True}, "coco": {"loader": True}},
        )
        monkeypatch.setattr("backend.agent.tools.qa.detect_format", lambda _p: "yolo")

        def _fake_get_loader(path: Path, *, format_name: str | None = None, **kwargs: Any) -> _FakeLoader:
            calls["loader_path"] = path
            calls["loader_format"] = format_name
            calls["loader_progress"] = kwargs.get("progress_callback")
            return loader

        def _fake_run_qa(
            dataset_input: Dataset,
            *,
            checks: list[str] | None = None,
            iou_threshold: float = 0.8,
            progress_callback: Any = None,
        ) -> QAResult:
            calls["qa_dataset"] = dataset_input
            calls["qa_checks"] = checks
            calls["qa_iou"] = iou_threshold
            calls["qa_progress"] = progress_callback
            return qa_result

        def _fake_generate_report(
            dataset_input: Dataset,
            qa_input: QAResult,
            output_path: Path,
        ) -> Path:
            calls["report_dataset"] = dataset_input
            calls["report_result"] = qa_input
            calls["report_output"] = output_path
            output_path.write_text("<html></html>", encoding="utf-8")
            return output_path

        monkeypatch.setattr("backend.agent.tools.qa.get_loader", _fake_get_loader)
        monkeypatch.setattr("backend.agent.tools.qa.run_qa", _fake_run_qa)
        monkeypatch.setattr("backend.agent.tools.qa.generate_qa_report", _fake_generate_report)

        tool = LabelQATool()
        result = await tool.run(
            path=str(dataset_path),
            generate_report=True,
            checks=["missing_labels"],
            iou_threshold=0.7,
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["dataset_path"] == str(dataset_path)
        assert result.data["format"] == "yolo"
        assert result.data["num_samples"] == 2
        assert result.data["num_labels"] == 1
        assert result.data["total_issues"] == 3
        assert result.data["errors"] == 1
        assert result.data["warnings"] == 2
        assert result.data["info"] == 0
        assert result.data["checks_run"] == [
            "missing_labels",
            "missing_images",
            "overlapping_boxes",
        ]
        assert result.data["dataset_warnings"] == ["dataset warning"]
        assert result.data["report_path"] == str(dataset_path / "qa_report.html")
        assert callable(calls["loader_progress"])
        assert callable(calls["qa_progress"])
        assert calls["loader_path"] == dataset_path
        assert calls["loader_format"] == "yolo"
        assert calls["qa_dataset"] is dataset
        assert calls["qa_checks"] == ["missing_labels"]
        assert calls["qa_iou"] == 0.7
        assert calls["report_dataset"] is dataset
        assert calls["report_result"] is qa_result
        assert calls["report_output"] == dataset_path / "qa_report.html"

        issue_types = {summary["issue_type"] for summary in result.data["issues_by_type"]}
        assert issue_types == {"missing_image", "missing_labels", "overlapping_boxes"}

        recommendations = result.data["recommendations"]
        assert any("error-level issues first" in item for item in recommendations)
        assert any("unlabeled images" in item for item in recommendations)
        assert any("missing image files" in item for item in recommendations)
        assert any("high-overlap boxes" in item for item in recommendations)

    @pytest.mark.asyncio
    async def test_label_qa_without_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        dataset = _make_dataset(dataset_path)
        qa_result = _make_qa_result(dataset, dataset_path)

        monkeypatch.setattr(
            "backend.agent.tools.qa.list_formats",
            lambda: {"yolo": {"loader": True}},
        )
        monkeypatch.setattr("backend.agent.tools.qa.get_loader", lambda _p, **_k: _FakeLoader(dataset))
        monkeypatch.setattr("backend.agent.tools.qa.run_qa", lambda *_a, **_k: qa_result)

        def _unexpected_report(*_args: Any, **_kwargs: Any) -> Path:
            raise AssertionError("generate_qa_report should not be called when generate_report=False")

        monkeypatch.setattr("backend.agent.tools.qa.generate_qa_report", _unexpected_report)

        tool = LabelQATool()
        result = await tool.run(
            path=str(dataset_path),
            format="yolo",
            generate_report=False,
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["report_path"] is None

    @pytest.mark.asyncio
    async def test_label_qa_invalid_iou_threshold(self, tmp_path: Path) -> None:
        tool = LabelQATool()
        result = await tool.run(path=str(tmp_path), iou_threshold=1.5)

        assert result.success is False
        assert result.error is not None
        assert "iou_threshold" in result.error

    @pytest.mark.asyncio
    async def test_label_qa_invalid_checks(self, tmp_path: Path) -> None:
        tool = LabelQATool()
        result = await tool.run(path=str(tmp_path), checks=["not-a-check"])

        assert result.success is False
        assert result.error is not None
        assert "unsupported checks" in result.error.lower()

    @pytest.mark.asyncio
    async def test_label_qa_detect_format_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        monkeypatch.setattr(
            "backend.agent.tools.qa.list_formats",
            lambda: {"yolo": {"loader": True}},
        )
        monkeypatch.setattr("backend.agent.tools.qa.detect_format", lambda _p: None)

        tool = LabelQATool()
        result = await tool.run(path=str(dataset_path))

        assert result.success is False
        assert result.error is not None
        assert "could not detect dataset format" in result.error.lower()

    @pytest.mark.asyncio
    async def test_label_qa_missing_path(self, tmp_path: Path) -> None:
        tool = LabelQATool()
        result = await tool.run(path=str(tmp_path / "missing"))

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()


class TestLabelQAToolRegistration:
    def test_label_qa_tool_registered(self) -> None:
        """label_qa should be available in the global tool registry."""
        from backend.agent.tools import get_tool_registry, initialize_tools

        initialize_tools()
        registry = get_tool_registry()

        assert registry.has("label_qa")
        tool = registry.get("label_qa")
        assert tool is not None
        schema = tool.get_schema()
        properties = schema["function"]["parameters"]["properties"]
        assert schema["function"]["name"] == "label_qa"
        assert "path" in properties
        assert "format" in properties
        assert "generate_report" in properties
        assert "checks" in properties
        assert "iou_threshold" in properties
