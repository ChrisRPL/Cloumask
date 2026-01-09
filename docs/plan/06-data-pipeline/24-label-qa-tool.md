# Label QA Agent Tool

> **Parent:** 06-data-pipeline
> **Depends on:** 18-label-qa, 21-html-reports
> **Blocks:** None

## Objective

Implement the `label_qa` LangGraph agent tool for running quality assurance checks on dataset labels.

## Acceptance Criteria

- [ ] Tool callable from LangGraph agent
- [ ] Run configurable QA checks
- [ ] Generate HTML report
- [ ] Return issue summary
- [ ] Suggest fixes for common issues
- [ ] Return structured result

## Implementation Steps

### 1. Create qa.py

Create `backend/agent/tools/qa.py`:

```python
"""Label QA agent tool.

Runs quality assurance checks on dataset labels.
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class IssueSummary(BaseModel):
    """Summary of a QA issue type."""
    issue_type: str
    count: int
    severity: str
    sample_paths: list[str]


class LabelQAResult(BaseModel):
    """Result of label QA analysis."""
    success: bool
    dataset_path: str
    num_samples: int
    num_labels: int
    total_issues: int
    errors: int
    warnings: int
    info: int
    issues_by_type: list[IssueSummary]
    report_path: Optional[str] = None
    recommendations: list[str] = []
    error: Optional[str] = None


@tool
def label_qa(
    path: str = Field(description="Path to dataset"),
    format: Optional[str] = Field(default=None, description="Dataset format (auto-detect if not provided)"),
    generate_report: bool = Field(default=True, description="Generate HTML report"),
    checks: Optional[list[str]] = Field(default=None, description="Specific checks to run (default: all)"),
    iou_threshold: float = Field(default=0.8, description="IoU threshold for overlap detection"),
) -> LabelQAResult:
    """Run quality assurance checks on dataset labels.

    Checks for:
    - Missing labels (images with no annotations)
    - Missing images (annotations without image files)
    - Overlapping boxes (high IoU between boxes)
    - Out-of-bounds boxes (extending outside image)
    - Tiny/huge boxes (anomalous sizes)
    - Class imbalance (underrepresented classes)
    - Duplicate labels (same class, same location)

    Example:
        label_qa("/data/yolo_dataset", generate_report=True)
    """
    from backend.data.formats import detect_format, get_loader
    from backend.data.qa import run_qa
    from backend.data.report import generate_qa_report

    dataset_path = Path(path)

    try:
        # Detect and load dataset
        if format is None:
            format = detect_format(dataset_path)
            if format is None:
                return LabelQAResult(
                    success=False,
                    dataset_path=str(dataset_path),
                    num_samples=0,
                    num_labels=0,
                    total_issues=0,
                    errors=0,
                    warnings=0,
                    info=0,
                    issues_by_type=[],
                    error="Could not detect dataset format",
                )

        loader = get_loader(dataset_path, format_name=format)
        dataset = loader.load()

        # Run QA
        qa_result = run_qa(dataset, checks=checks, iou_threshold=iou_threshold)

        # Build issue summaries
        issues_by_type = []
        for issue_type, issues in qa_result.issues_by_type().items():
            issues_by_type.append(IssueSummary(
                issue_type=issue_type.value,
                count=len(issues),
                severity=issues[0].severity if issues else "info",
                sample_paths=[str(i.sample_path) for i in issues[:5] if i.sample_path],
            ))

        # Generate report
        report_path = None
        if generate_report:
            report_file = dataset_path / "qa_report.html"
            generate_qa_report(dataset, qa_result, report_file)
            report_path = str(report_file)

        # Generate recommendations
        recommendations = []
        if qa_result.num_errors > 0:
            recommendations.append("Fix errors first - these indicate broken data")
        if any(i.issue_type.value == "missing_labels" for i in qa_result.issues):
            recommendations.append("Review images without labels - may need annotation")
        if any(i.issue_type.value == "overlapping_boxes" for i in qa_result.issues):
            recommendations.append("Check overlapping boxes - may be duplicate annotations")
        if any(i.issue_type.value == "class_imbalance" for i in qa_result.issues):
            recommendations.append("Consider augmenting underrepresented classes")
        if any(i.issue_type.value == "tiny_box" for i in qa_result.issues):
            recommendations.append("Review tiny boxes - may be annotation errors")

        return LabelQAResult(
            success=True,
            dataset_path=str(dataset_path),
            num_samples=len(dataset),
            num_labels=dataset.total_labels(),
            total_issues=len(qa_result.issues),
            errors=qa_result.num_errors,
            warnings=qa_result.num_warnings,
            info=qa_result.num_info,
            issues_by_type=issues_by_type,
            report_path=report_path,
            recommendations=recommendations,
        )

    except Exception as e:
        return LabelQAResult(
            success=False,
            dataset_path=str(dataset_path),
            num_samples=0,
            num_labels=0,
            total_issues=0,
            errors=0,
            warnings=0,
            info=0,
            issues_by_type=[],
            error=str(e),
        )
```

### 2. Register tool

Add to `backend/agent/tools/__init__.py`:

```python
from backend.agent.tools.qa import label_qa

DATA_TOOLS = [
    convert_format,
    find_duplicates,
    label_qa,
]
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/agent/tools/qa.py` | Create | Label QA tool |
| `backend/agent/tools/__init__.py` | Modify | Register tool |

## Verification

```bash
python -c "
from backend.agent.tools.qa import label_qa
result = label_qa.invoke({'path': '/data/yolo_dataset'})
print(f'Found {result.total_issues} issues')
for rec in result.recommendations:
    print(f'  - {rec}')
"
```

## Notes

- Generates HTML report by default
- Returns actionable recommendations
- Sample paths limited to 5 per issue type
- iou_threshold controls overlap sensitivity
