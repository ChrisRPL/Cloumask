"""Run label quality assurance checks for datasets.

Implements spec: 06-data-pipeline/24-label-qa-tool
Integration points: backend/data/qa.py and backend/data/report.py
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.registry import register_tool
from backend.data.formats import detect_format, get_loader, list_formats
from backend.data.qa import QAIssue, QAResult, run_qa
from backend.data.report import generate_qa_report

logger = logging.getLogger(__name__)

SUPPORTED_QA_CHECKS = [
    "missing_labels",
    "missing_images",
    "overlapping_boxes",
    "out_of_bounds",
    "box_sizes",
    "class_imbalance",
    "duplicate_labels",
]

SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}

ISSUE_RECOMMENDATIONS = {
    "missing_labels": "Review unlabeled images and annotate or remove invalid samples.",
    "missing_image": "Restore missing image files or remove orphaned annotations.",
    "overlapping_boxes": "Inspect high-overlap boxes to remove accidental duplicate annotations.",
    "out_of_bounds": "Fix boxes that extend outside image bounds.",
    "tiny_box": "Review tiny boxes; they are often annotation noise.",
    "huge_box": "Check huge boxes for coarse or incorrect annotations.",
    "class_imbalance": "Collect or augment underrepresented classes to balance training data.",
    "duplicate_label": "Remove duplicate labels for the same object instance.",
}


def _default_report_path(dataset_path: Path) -> Path:
    if dataset_path.is_dir():
        return dataset_path / "qa_report.html"
    return dataset_path.parent / f"{dataset_path.stem}_qa_report.html"


def _highest_severity(issues: list[QAIssue]) -> str:
    if not issues:
        return "info"

    return min(
        (issue.severity for issue in issues),
        key=lambda severity: SEVERITY_RANK.get(severity, 99),
    )


def _sample_paths(issues: list[QAIssue], limit: int = 5) -> list[str]:
    samples: list[str] = []
    seen: set[str] = set()

    for issue in issues:
        if issue.sample_path is None:
            continue
        sample = str(issue.sample_path)
        if sample in seen:
            continue
        seen.add(sample)
        samples.append(sample)
        if len(samples) >= limit:
            break

    return samples


def _summarize_issues(qa_result: QAResult) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    grouped = qa_result.issues_by_type()

    for issue_type in sorted(grouped.keys(), key=lambda value: value.value):
        issues = grouped[issue_type]
        summaries.append(
            {
                "issue_type": issue_type.value,
                "count": len(issues),
                "severity": _highest_severity(issues),
                "sample_paths": _sample_paths(issues),
            }
        )

    return summaries


def _build_recommendations(qa_result: QAResult) -> list[str]:
    recommendations: list[str] = []

    if qa_result.num_errors > 0:
        recommendations.append("Fix error-level issues first; they indicate broken dataset assets.")

    issue_types = {issue.issue_type.value for issue in qa_result.issues}
    for issue_type in sorted(issue_types):
        recommendation = ISSUE_RECOMMENDATIONS.get(issue_type)
        if recommendation:
            recommendations.append(recommendation)

    if not recommendations and qa_result.total_issues == 0:
        recommendations.append("No issues found. Dataset is ready for training.")

    return recommendations


@register_tool
class LabelQATool(BaseTool):
    """Run quality assurance checks against dataset labels."""

    name = "label_qa"
    description = """Run label quality checks on an annotated dataset and optionally generate an HTML report.
Detects missing labels/images, overlaps, out-of-bounds boxes, anomalous box sizes,
class imbalance, and duplicate labels."""
    category = ToolCategory.UTILITY

    parameters = [
        ToolParameter(
            name="path",
            type=str,
            description="Path to dataset root (or annotation root for supported formats)",
            required=True,
        ),
        ToolParameter(
            name="format",
            type=str,
            description="Optional format override (auto-detected if omitted)",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="generate_report",
            type=bool,
            description="Generate an HTML QA report in the dataset directory",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="checks",
            type=list,
            description=(
                "Optional subset of checks: missing_labels, missing_images, "
                "overlapping_boxes, out_of_bounds, box_sizes, class_imbalance, "
                "duplicate_labels"
            ),
            required=False,
            default=None,
        ),
        ToolParameter(
            name="iou_threshold",
            type=float,
            description="IoU threshold used for overlapping box detection (0-1)",
            required=False,
            default=0.8,
        ),
    ]

    async def execute(
        self,
        path: str,
        format: str | None = None,
        generate_report: bool = True,
        checks: list[str] | None = None,
        iou_threshold: float = 0.8,
    ) -> ToolResult:
        """Execute QA analysis for a dataset."""
        dataset_path = Path(path)
        normalized_format = format.lower() if format else None

        if not dataset_path.exists():
            return error_result(f"Input path not found: {path}")

        if not 0.0 <= iou_threshold <= 1.0:
            return error_result(
                f"Invalid iou_threshold: {iou_threshold}. Must be between 0 and 1."
            )

        if checks is not None:
            unknown_checks = sorted(set(checks) - set(SUPPORTED_QA_CHECKS))
            if unknown_checks:
                return error_result(
                    f"Unsupported checks: {unknown_checks}. "
                    f"Supported checks: {SUPPORTED_QA_CHECKS}"
                )

        available_formats = list_formats()
        loader_formats = sorted(
            name for name, details in available_formats.items() if details.get("loader")
        )

        if normalized_format and normalized_format not in loader_formats:
            return error_result(
                f"Unsupported format '{format}'. Available source formats: {loader_formats}"
            )

        if normalized_format is None:
            normalized_format = detect_format(dataset_path)
            if normalized_format is None:
                return error_result(
                    "Could not detect dataset format. "
                    f"Please provide format explicitly ({loader_formats})."
                )

        try:
            result_data = await asyncio.to_thread(
                self._run_qa,
                dataset_path,
                normalized_format,
                generate_report,
                checks,
                iou_threshold,
            )
            return success_result(result_data)
        except Exception as exc:
            logger.exception("Label QA failed")
            return error_result(
                f"Label QA failed: {exc}",
                path=str(dataset_path),
                format=normalized_format,
            )

    def _run_qa(
        self,
        dataset_path: Path,
        format_name: str,
        generate_report: bool,
        checks: list[str] | None,
        iou_threshold: float,
    ) -> dict[str, Any]:
        """Run synchronous QA logic in a worker thread."""
        self.report_progress(0, 3, "Initializing label QA")

        loader = get_loader(
            dataset_path,
            format_name=format_name,
            progress_callback=self.report_progress,
        )
        dataset_warnings = loader.validate()
        dataset = loader.load()
        self.report_progress(1, 3, f"Loaded {len(dataset)} samples from {format_name}")

        qa_result = run_qa(
            dataset,
            checks=checks,
            iou_threshold=iou_threshold,
            progress_callback=self.report_progress,
        )
        self.report_progress(2, 3, "QA checks complete")

        report_path: str | None = None
        if generate_report:
            report_file = _default_report_path(dataset_path)
            generate_qa_report(dataset, qa_result, report_file)
            report_path = str(report_file)
            self.report_progress(3, 3, f"Report generated: {report_file.name}")
        else:
            self.report_progress(3, 3, "Label QA complete")

        return {
            "dataset_path": str(dataset_path),
            "format": format_name,
            "num_samples": len(dataset),
            "num_labels": dataset.total_labels(),
            "total_issues": qa_result.total_issues,
            "errors": qa_result.num_errors,
            "warnings": qa_result.num_warnings,
            "info": qa_result.num_info,
            "checks_run": qa_result.checks_run,
            "issues_by_type": _summarize_issues(qa_result),
            "recommendations": _build_recommendations(qa_result),
            "report_path": report_path,
            "dataset_warnings": dataset_warnings,
        }
