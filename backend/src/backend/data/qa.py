"""Label quality assurance checks.

Identifies common annotation issues in datasets.

Implements spec: 06-data-pipeline/18-label-qa
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from backend.data.models import Dataset

logger = logging.getLogger(__name__)

Severity = Literal["error", "warning", "info"]
ProgressCallback = Callable[[int, int, str], None]


class IssueType(str, Enum):
    """Types of QA issues."""

    MISSING_LABELS = "missing_labels"
    OVERLAPPING_BOXES = "overlapping_boxes"
    OUT_OF_BOUNDS = "out_of_bounds"
    TINY_BOX = "tiny_box"
    HUGE_BOX = "huge_box"
    CLASS_IMBALANCE = "class_imbalance"
    MISSING_IMAGE = "missing_image"
    DUPLICATE_LABEL = "duplicate_label"


@dataclass
class QAIssue:
    """A single QA issue."""

    issue_type: IssueType
    severity: Severity
    sample_path: Path | None
    label_index: int | None
    description: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class QAResult:
    """Result of QA analysis."""

    issues: list[QAIssue]
    stats: dict[str, object]
    checks_run: list[str]

    @property
    def total_issues(self) -> int:
        """Return total number of issues."""

        return len(self.issues)

    @property
    def num_errors(self) -> int:
        """Return count of issues with error severity."""

        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def num_warnings(self) -> int:
        """Return count of issues with warning severity."""

        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def num_info(self) -> int:
        """Return count of issues with info severity."""

        return sum(1 for issue in self.issues if issue.severity == "info")

    def issues_by_type(self) -> dict[IssueType, list[QAIssue]]:
        """Group issues by issue type."""

        grouped: dict[IssueType, list[QAIssue]] = {}
        for issue in self.issues:
            grouped.setdefault(issue.issue_type, []).append(issue)
        return grouped

    def to_dict(self) -> dict[str, object]:
        """Serialize result to dictionary."""

        return {
            "issues": [
                {
                    "type": issue.issue_type.value,
                    "severity": issue.severity,
                    "sample_path": str(issue.sample_path) if issue.sample_path else None,
                    "label_index": issue.label_index,
                    "description": issue.description,
                    "details": issue.details,
                }
                for issue in self.issues
            ],
            "stats": self.stats,
            "checks_run": self.checks_run,
            "summary": {
                "total_issues": self.total_issues,
                "errors": self.num_errors,
                "warnings": self.num_warnings,
                "info": self.num_info,
            },
        }


class LabelQA:
    """Label quality assurance checker."""

    def __init__(
        self,
        iou_threshold: float = 0.8,
        tiny_box_threshold: float = 0.001,
        huge_box_threshold: float = 0.9,
        imbalance_ratio: float = 10.0,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Initialize QA checker with configurable thresholds."""

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be between 0 and 1, got {iou_threshold}")
        if not 0.0 <= tiny_box_threshold <= 1.0:
            raise ValueError(
                f"tiny_box_threshold must be between 0 and 1, got {tiny_box_threshold}"
            )
        if not 0.0 <= huge_box_threshold <= 1.0:
            raise ValueError(
                f"huge_box_threshold must be between 0 and 1, got {huge_box_threshold}"
            )
        if imbalance_ratio <= 0.0:
            raise ValueError(f"imbalance_ratio must be > 0, got {imbalance_ratio}")

        self.iou_threshold = iou_threshold
        self.tiny_box_threshold = tiny_box_threshold
        self.huge_box_threshold = huge_box_threshold
        self.imbalance_ratio = imbalance_ratio
        self.progress_callback = progress_callback

    def _report_progress(self, current: int, total: int, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def check_missing_labels(self, dataset: Dataset) -> list[QAIssue]:
        """Find images with no annotations."""

        issues: list[QAIssue] = []
        for sample in dataset:
            if sample.has_labels:
                continue
            issues.append(
                QAIssue(
                    issue_type=IssueType.MISSING_LABELS,
                    severity="warning",
                    sample_path=sample.image_path,
                    label_index=None,
                    description=f"Image has no annotations: {sample.image_path.name}",
                )
            )
        return issues

    def check_missing_images(self, dataset: Dataset) -> list[QAIssue]:
        """Find annotations with missing image files."""

        issues: list[QAIssue] = []
        for sample in dataset:
            if sample.image_path.exists():
                continue
            issues.append(
                QAIssue(
                    issue_type=IssueType.MISSING_IMAGE,
                    severity="error",
                    sample_path=sample.image_path,
                    label_index=None,
                    description=f"Image file not found: {sample.image_path}",
                )
            )
        return issues

    def check_overlapping_boxes(self, dataset: Dataset) -> list[QAIssue]:
        """Find highly overlapping bounding boxes."""

        issues: list[QAIssue] = []
        for sample in dataset:
            labels = sample.labels
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    iou = labels[i].bbox.iou(labels[j].bbox)
                    if iou <= self.iou_threshold:
                        continue
                    issues.append(
                        QAIssue(
                            issue_type=IssueType.OVERLAPPING_BOXES,
                            severity="warning",
                            sample_path=sample.image_path,
                            label_index=i,
                            description=f"Boxes {i} and {j} overlap with IoU={iou:.2f}",
                            details={"box_i": i, "box_j": j, "iou": iou},
                        )
                    )
        return issues

    def check_out_of_bounds(self, dataset: Dataset) -> list[QAIssue]:
        """Find boxes extending outside normalized image bounds."""

        issues: list[QAIssue] = []
        for sample in dataset:
            for idx, label in enumerate(sample.labels):
                x1, y1, x2, y2 = label.bbox.to_xyxy()
                if x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                    continue
                issues.append(
                    QAIssue(
                        issue_type=IssueType.OUT_OF_BOUNDS,
                        severity="warning",
                        sample_path=sample.image_path,
                        label_index=idx,
                        description=f"Box {idx} extends outside image bounds",
                        details={"bbox": label.bbox.to_dict()},
                    )
                )
        return issues

    def check_box_sizes(self, dataset: Dataset) -> list[QAIssue]:
        """Find boxes with anomalous area."""

        issues: list[QAIssue] = []
        for sample in dataset:
            for idx, label in enumerate(sample.labels):
                area = label.bbox.area()

                if area < self.tiny_box_threshold:
                    issues.append(
                        QAIssue(
                            issue_type=IssueType.TINY_BOX,
                            severity="warning",
                            sample_path=sample.image_path,
                            label_index=idx,
                            description=f"Box {idx} is very small (area={area:.4f})",
                            details={"area": area, "class": label.class_name},
                        )
                    )

                if area > self.huge_box_threshold:
                    issues.append(
                        QAIssue(
                            issue_type=IssueType.HUGE_BOX,
                            severity="info",
                            sample_path=sample.image_path,
                            label_index=idx,
                            description=f"Box {idx} is very large (area={area:.4f})",
                            details={"area": area, "class": label.class_name},
                        )
                    )

        return issues

    def check_class_imbalance(self, dataset: Dataset) -> list[QAIssue]:
        """Check for class distribution imbalance."""

        distribution = Counter(dataset.class_distribution())
        for class_name in dataset.class_names:
            distribution.setdefault(class_name, 0)

        if not distribution:
            return []

        max_count = max(distribution.values())
        if max_count == 0:
            return []

        issues: list[QAIssue] = []
        for class_name, count in distribution.items():
            ratio = float("inf") if count == 0 else max_count / count
            if ratio <= self.imbalance_ratio:
                continue
            issues.append(
                QAIssue(
                    issue_type=IssueType.CLASS_IMBALANCE,
                    severity="warning",
                    sample_path=None,
                    label_index=None,
                    description=f"Class '{class_name}' is underrepresented ({count} labels)",
                    details={
                        "class": class_name,
                        "count": count,
                        "max_count": max_count,
                        "ratio_to_max": ratio,
                    },
                )
            )
        return issues

    def check_duplicate_labels(self, dataset: Dataset) -> list[QAIssue]:
        """Find near-identical duplicate labels for the same class."""

        issues: list[QAIssue] = []
        for sample in dataset:
            labels = sample.labels
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if labels[i].class_id != labels[j].class_id:
                        continue
                    iou = labels[i].bbox.iou(labels[j].bbox)
                    if iou <= 0.95:
                        continue
                    issues.append(
                        QAIssue(
                            issue_type=IssueType.DUPLICATE_LABEL,
                            severity="warning",
                            sample_path=sample.image_path,
                            label_index=i,
                            description=(
                                f"Possible duplicate: boxes {i} and {j} same class "
                                f"with IoU={iou:.2f}"
                            ),
                            details={"box_i": i, "box_j": j, "class": labels[i].class_name},
                        )
                    )
        return issues

    def run(self, dataset: Dataset, checks: list[str] | None = None) -> QAResult:
        """Run selected QA checks on dataset."""

        all_checks = {
            "missing_labels": self.check_missing_labels,
            "missing_images": self.check_missing_images,
            "overlapping_boxes": self.check_overlapping_boxes,
            "out_of_bounds": self.check_out_of_bounds,
            "box_sizes": self.check_box_sizes,
            "class_imbalance": self.check_class_imbalance,
            "duplicate_labels": self.check_duplicate_labels,
        }

        selected_checks = checks or list(all_checks.keys())
        issues: list[QAIssue] = []
        checks_run: list[str] = []
        total = len(selected_checks)

        for idx, check_name in enumerate(selected_checks, start=1):
            check = all_checks.get(check_name)
            if check is None:
                logger.warning("Skipping unknown QA check '%s'", check_name)
                self._report_progress(idx, total, f"Skipping {check_name} (unknown)")
                continue

            issues.extend(check(dataset))
            checks_run.append(check_name)
            self._report_progress(idx, total, f"Running {check_name}")

        return QAResult(issues=issues, stats=dataset.stats(), checks_run=checks_run)


def run_qa(dataset: Dataset, checks: list[str] | None = None, **kwargs: object) -> QAResult:
    """Convenience entry point for running label QA."""

    qa = LabelQA(**kwargs)
    return qa.run(dataset, checks=checks)

