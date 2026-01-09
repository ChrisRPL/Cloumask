# Label Quality Assurance

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models
> **Blocks:** 21-html-reports, 24-label-qa-tool

## Objective

Implement label quality assurance checks to identify annotation issues such as missing labels, overlapping boxes, out-of-bounds boxes, class imbalance, and anomalous box sizes.

## Acceptance Criteria

- [ ] Detect missing labels (images with no annotations)
- [ ] Find overlapping boxes (IoU above threshold)
- [ ] Identify out-of-bounds boxes
- [ ] Report class distribution imbalance
- [ ] Flag tiny/huge boxes (outliers)
- [ ] Generate structured QA report
- [ ] Unit tests for each check type

## Implementation Steps

### 1. Create qa.py

Create `backend/data/qa.py`:

```python
"""Label quality assurance checks.

Identifies common annotation issues in datasets.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from backend.data.models import Dataset, Label, Sample

logger = logging.getLogger(__name__)


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
    """A single QA issue.

    Attributes:
        issue_type: Type of issue
        severity: Severity level (error, warning, info)
        sample_path: Path to affected image
        label_index: Index of affected label (if applicable)
        description: Human-readable description
        details: Additional details dict
    """
    issue_type: IssueType
    severity: str
    sample_path: Optional[Path]
    label_index: Optional[int]
    description: str
    details: dict = field(default_factory=dict)


@dataclass
class QAResult:
    """Result of QA analysis.

    Attributes:
        issues: List of issues found
        stats: Dataset statistics
        checks_run: List of check names run
    """
    issues: list[QAIssue]
    stats: dict
    checks_run: list[str]

    @property
    def num_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def num_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def num_info(self) -> int:
        return sum(1 for i in self.issues if i.severity == "info")

    def issues_by_type(self) -> dict[IssueType, list[QAIssue]]:
        """Group issues by type."""
        grouped: dict[IssueType, list[QAIssue]] = {}
        for issue in self.issues:
            if issue.issue_type not in grouped:
                grouped[issue.issue_type] = []
            grouped[issue.issue_type].append(issue)
        return grouped

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "issues": [
                {
                    "type": i.issue_type.value,
                    "severity": i.severity,
                    "sample_path": str(i.sample_path) if i.sample_path else None,
                    "label_index": i.label_index,
                    "description": i.description,
                    "details": i.details,
                }
                for i in self.issues
            ],
            "stats": self.stats,
            "checks_run": self.checks_run,
            "summary": {
                "total_issues": len(self.issues),
                "errors": self.num_errors,
                "warnings": self.num_warnings,
                "info": self.num_info,
            },
        }


class LabelQA:
    """Label quality assurance checker.

    Runs configurable checks on datasets to find annotation issues.

    Example:
        qa = LabelQA()
        result = qa.run(dataset)
        print(f"Found {result.num_warnings} warnings")
    """

    def __init__(
        self,
        iou_threshold: float = 0.8,
        tiny_box_threshold: float = 0.001,
        huge_box_threshold: float = 0.9,
        imbalance_ratio: float = 10.0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """Initialize QA checker.

        Args:
            iou_threshold: IoU threshold for overlap detection
            tiny_box_threshold: Area threshold for tiny boxes (normalized)
            huge_box_threshold: Area threshold for huge boxes (normalized)
            imbalance_ratio: Max ratio for class imbalance warning
            progress_callback: Progress callback
        """
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
        issues = []
        for sample in dataset:
            if not sample.has_labels:
                issues.append(QAIssue(
                    issue_type=IssueType.MISSING_LABELS,
                    severity="warning",
                    sample_path=sample.image_path,
                    label_index=None,
                    description=f"Image has no annotations: {sample.image_path.name}",
                ))
        return issues

    def check_missing_images(self, dataset: Dataset) -> list[QAIssue]:
        """Find annotations with missing image files."""
        issues = []
        for sample in dataset:
            if not sample.image_path.exists():
                issues.append(QAIssue(
                    issue_type=IssueType.MISSING_IMAGE,
                    severity="error",
                    sample_path=sample.image_path,
                    label_index=None,
                    description=f"Image file not found: {sample.image_path}",
                ))
        return issues

    def check_overlapping_boxes(self, dataset: Dataset) -> list[QAIssue]:
        """Find highly overlapping bounding boxes."""
        issues = []
        for sample in dataset:
            labels = sample.labels
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    iou = labels[i].bbox.iou(labels[j].bbox)
                    if iou > self.iou_threshold:
                        issues.append(QAIssue(
                            issue_type=IssueType.OVERLAPPING_BOXES,
                            severity="warning",
                            sample_path=sample.image_path,
                            label_index=i,
                            description=f"Boxes {i} and {j} overlap with IoU={iou:.2f}",
                            details={"box_i": i, "box_j": j, "iou": iou},
                        ))
        return issues

    def check_out_of_bounds(self, dataset: Dataset) -> list[QAIssue]:
        """Find boxes extending outside image bounds."""
        issues = []
        for sample in dataset:
            for i, label in enumerate(sample.labels):
                x1, y1, x2, y2 = label.bbox.to_xyxy()
                if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                    issues.append(QAIssue(
                        issue_type=IssueType.OUT_OF_BOUNDS,
                        severity="warning",
                        sample_path=sample.image_path,
                        label_index=i,
                        description=f"Box {i} extends outside image bounds",
                        details={"bbox": label.bbox.to_dict()},
                    ))
        return issues

    def check_box_sizes(self, dataset: Dataset) -> list[QAIssue]:
        """Find anomalously sized boxes."""
        issues = []
        for sample in dataset:
            for i, label in enumerate(sample.labels):
                area = label.bbox.area()

                if area < self.tiny_box_threshold:
                    issues.append(QAIssue(
                        issue_type=IssueType.TINY_BOX,
                        severity="warning",
                        sample_path=sample.image_path,
                        label_index=i,
                        description=f"Box {i} is very small (area={area:.4f})",
                        details={"area": area, "class": label.class_name},
                    ))

                if area > self.huge_box_threshold:
                    issues.append(QAIssue(
                        issue_type=IssueType.HUGE_BOX,
                        severity="info",
                        sample_path=sample.image_path,
                        label_index=i,
                        description=f"Box {i} is very large (area={area:.4f})",
                        details={"area": area, "class": label.class_name},
                    ))
        return issues

    def check_class_imbalance(self, dataset: Dataset) -> list[QAIssue]:
        """Check for class distribution imbalance."""
        issues = []
        distribution = dataset.class_distribution()

        if not distribution:
            return issues

        counts = list(distribution.values())
        max_count = max(counts)
        min_count = min(counts)

        if min_count > 0 and max_count / min_count > self.imbalance_ratio:
            # Find the imbalanced classes
            avg_count = sum(counts) / len(counts)
            for class_name, count in distribution.items():
                if count < avg_count / self.imbalance_ratio:
                    issues.append(QAIssue(
                        issue_type=IssueType.CLASS_IMBALANCE,
                        severity="warning",
                        sample_path=None,
                        label_index=None,
                        description=f"Class '{class_name}' is underrepresented ({count} labels)",
                        details={"class": class_name, "count": count, "avg": avg_count},
                    ))

        return issues

    def check_duplicate_labels(self, dataset: Dataset) -> list[QAIssue]:
        """Find duplicate labels (same class, nearly identical box)."""
        issues = []
        for sample in dataset:
            labels = sample.labels
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if labels[i].class_id == labels[j].class_id:
                        iou = labels[i].bbox.iou(labels[j].bbox)
                        if iou > 0.95:  # Nearly identical
                            issues.append(QAIssue(
                                issue_type=IssueType.DUPLICATE_LABEL,
                                severity="warning",
                                sample_path=sample.image_path,
                                label_index=i,
                                description=f"Possible duplicate: boxes {i} and {j} same class with IoU={iou:.2f}",
                                details={"box_i": i, "box_j": j, "class": labels[i].class_name},
                            ))
        return issues

    def run(
        self,
        dataset: Dataset,
        checks: Optional[list[str]] = None,
    ) -> QAResult:
        """Run QA checks on dataset.

        Args:
            dataset: Dataset to check
            checks: Specific checks to run (default: all)

        Returns:
            QAResult with issues and stats
        """
        all_checks = {
            "missing_labels": self.check_missing_labels,
            "missing_images": self.check_missing_images,
            "overlapping_boxes": self.check_overlapping_boxes,
            "out_of_bounds": self.check_out_of_bounds,
            "box_sizes": self.check_box_sizes,
            "class_imbalance": self.check_class_imbalance,
            "duplicate_labels": self.check_duplicate_labels,
        }

        if checks is None:
            checks = list(all_checks.keys())

        issues: list[QAIssue] = []
        checks_run = []
        total = len(checks)

        for idx, check_name in enumerate(checks):
            if check_name in all_checks:
                check_func = all_checks[check_name]
                issues.extend(check_func(dataset))
                checks_run.append(check_name)
            self._report_progress(idx + 1, total, f"Running {check_name}")

        return QAResult(
            issues=issues,
            stats=dataset.stats(),
            checks_run=checks_run,
        )


# Convenience function
def run_qa(
    dataset: Dataset,
    checks: Optional[list[str]] = None,
    **kwargs,
) -> QAResult:
    """Run QA checks on dataset.

    Args:
        dataset: Dataset to check
        checks: Specific checks to run
        **kwargs: LabelQA constructor arguments

    Returns:
        QAResult
    """
    qa = LabelQA(**kwargs)
    return qa.run(dataset, checks)
```

### 2. Create unit tests

Create `backend/tests/data/test_qa.py`:

```python
"""Tests for label QA checks."""

from pathlib import Path

import pytest

from backend.data.models import BBox, Dataset, Label, Sample
from backend.data.qa import IssueType, LabelQA, run_qa


@pytest.fixture
def sample_dataset():
    """Create a sample dataset with various issues."""
    samples = [
        # Normal sample
        Sample(
            image_path=Path("/data/img1.jpg"),
            labels=[
                Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2)),
            ],
        ),
        # Missing labels
        Sample(
            image_path=Path("/data/img2.jpg"),
            labels=[],
        ),
        # Overlapping boxes
        Sample(
            image_path=Path("/data/img3.jpg"),
            labels=[
                Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2)),
                Label(class_name="car", class_id=0, bbox=BBox(0.52, 0.52, 0.2, 0.2)),
            ],
        ),
        # Out of bounds
        Sample(
            image_path=Path("/data/img4.jpg"),
            labels=[
                Label(class_name="car", class_id=0, bbox=BBox(0.95, 0.5, 0.2, 0.2)),
            ],
        ),
        # Tiny box
        Sample(
            image_path=Path("/data/img5.jpg"),
            labels=[
                Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.01, 0.01)),
            ],
        ),
    ]
    return Dataset(samples, class_names=["car"])


class TestLabelQA:
    """Tests for LabelQA."""

    def test_check_missing_labels(self, sample_dataset):
        """Test missing labels detection."""
        qa = LabelQA()
        issues = qa.check_missing_labels(sample_dataset)

        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.MISSING_LABELS
        assert "img2" in str(issues[0].sample_path)

    def test_check_overlapping_boxes(self, sample_dataset):
        """Test overlapping boxes detection."""
        qa = LabelQA(iou_threshold=0.5)
        issues = qa.check_overlapping_boxes(sample_dataset)

        assert len(issues) >= 1
        assert any(i.issue_type == IssueType.OVERLAPPING_BOXES for i in issues)

    def test_check_out_of_bounds(self, sample_dataset):
        """Test out of bounds detection."""
        qa = LabelQA()
        issues = qa.check_out_of_bounds(sample_dataset)

        assert len(issues) >= 1
        assert any(i.issue_type == IssueType.OUT_OF_BOUNDS for i in issues)

    def test_check_box_sizes(self, sample_dataset):
        """Test tiny/huge box detection."""
        qa = LabelQA(tiny_box_threshold=0.001)
        issues = qa.check_box_sizes(sample_dataset)

        assert len(issues) >= 1
        assert any(i.issue_type == IssueType.TINY_BOX for i in issues)

    def test_run_all_checks(self, sample_dataset):
        """Test running all checks."""
        result = run_qa(sample_dataset)

        assert len(result.issues) > 0
        assert len(result.checks_run) == 7
        assert result.stats is not None

    def test_run_specific_checks(self, sample_dataset):
        """Test running specific checks."""
        result = run_qa(sample_dataset, checks=["missing_labels"])

        assert len(result.checks_run) == 1
        assert result.checks_run[0] == "missing_labels"

    def test_result_to_dict(self, sample_dataset):
        """Test result serialization."""
        result = run_qa(sample_dataset)
        data = result.to_dict()

        assert "issues" in data
        assert "stats" in data
        assert "summary" in data

    def test_class_imbalance(self):
        """Test class imbalance detection."""
        samples = [
            Sample(
                image_path=Path(f"/data/img{i}.jpg"),
                labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
            )
            for i in range(100)
        ] + [
            Sample(
                image_path=Path("/data/rare.jpg"),
                labels=[Label(class_name="rare", class_id=1, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
            )
        ]
        ds = Dataset(samples, class_names=["car", "rare"])

        qa = LabelQA(imbalance_ratio=10.0)
        issues = qa.check_class_imbalance(ds)

        assert len(issues) >= 1
        assert any(i.details.get("class") == "rare" for i in issues)
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/qa.py` | Create | QA checks implementation |
| `backend/data/__init__.py` | Modify | Export QA module |
| `backend/tests/data/test_qa.py` | Create | Unit tests |

## Verification

```bash
cd backend
pytest tests/data/test_qa.py -v
```

## Notes

- IoU threshold 0.8+ for suspicious overlap
- Tiny box threshold based on normalized area
- Class imbalance uses ratio of max/min counts
- Results can be serialized to JSON for reporting
- Severity levels: error (must fix), warning (should fix), info (FYI)
