# HTML Report Generation

> **Parent:** 06-data-pipeline
> **Depends on:** 18-label-qa
> **Blocks:** 24-label-qa-tool

## Objective

Generate interactive HTML reports for dataset statistics and QA results with visualizations and issue summaries.

## Acceptance Criteria

- [ ] Generate standalone HTML report
- [ ] Include dataset statistics with charts
- [ ] Display QA issues grouped by type
- [ ] Interactive class distribution chart
- [ ] Sample gallery with issues highlighted
- [ ] Export-ready format (single HTML file)
- [ ] Unit tests for report generation

## Implementation Steps

### 1. Create report.py

Create `backend/data/report.py`:

```python
"""HTML report generation for datasets and QA results.

Generates standalone HTML reports with embedded charts.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from backend.data.models import Dataset
from backend.data.qa import QAResult

logger = logging.getLogger(__name__)


HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 0; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #4CAF50; }}
        .stat-label {{ color: #666; font-size: 0.9em; }}
        .chart {{ height: 400px; }}
        .issue {{ padding: 10px; border-left: 4px solid #ff9800; margin: 10px 0; background: #fff8e1; }}
        .issue.error {{ border-color: #f44336; background: #ffebee; }}
        .issue.warning {{ border-color: #ff9800; background: #fff8e1; }}
        .issue.info {{ border-color: #2196f3; background: #e3f2fd; }}
        .issue-type {{ font-weight: bold; margin-bottom: 5px; }}
        .issue-path {{ font-family: monospace; color: #666; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; }}
        .badge-error {{ background: #f44336; color: white; }}
        .badge-warning {{ background: #ff9800; color: white; }}
        .badge-info {{ background: #2196f3; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
    <script>
        {scripts}
    </script>
</body>
</html>'''


class ReportGenerator:
    """Generate HTML reports for datasets and QA results.

    Example:
        generator = ReportGenerator()
        generator.generate_qa_report(dataset, qa_result, Path("report.html"))
    """

    def __init__(self) -> None:
        pass

    def _render_stats_section(self, stats: dict) -> str:
        """Render dataset statistics section."""
        return f'''
        <div class="card">
            <h2>Dataset Statistics</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{stats.get("num_samples", 0)}</div>
                    <div class="stat-label">Images</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats.get("num_labels", 0)}</div>
                    <div class="stat-label">Labels</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats.get("num_classes", 0)}</div>
                    <div class="stat-label">Classes</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats.get("unlabeled_count", 0)}</div>
                    <div class="stat-label">Unlabeled</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats.get("avg_labels_per_sample", 0):.1f}</div>
                    <div class="stat-label">Avg Labels/Image</div>
                </div>
            </div>
        </div>
        '''

    def _render_class_distribution_chart(self, stats: dict) -> tuple[str, str]:
        """Render class distribution chart."""
        dist = stats.get("class_distribution", {})
        if not dist:
            return '<div class="card"><h2>Class Distribution</h2><p>No data</p></div>', ""

        labels = list(dist.keys())
        values = list(dist.values())

        chart_data = {
            "labels": labels,
            "values": values,
        }

        html = '''
        <div class="card">
            <h2>Class Distribution</h2>
            <div id="classDistChart" class="chart"></div>
        </div>
        '''

        script = f'''
        var distData = {json.dumps(chart_data)};
        Plotly.newPlot('classDistChart', [{{
            x: distData.labels,
            y: distData.values,
            type: 'bar',
            marker: {{ color: '#4CAF50' }}
        }}], {{
            margin: {{ t: 20, b: 80 }},
            xaxis: {{ tickangle: -45 }},
            yaxis: {{ title: 'Count' }}
        }});
        '''

        return html, script

    def _render_issues_section(self, qa_result: QAResult) -> str:
        """Render QA issues section."""
        if not qa_result.issues:
            return '''
            <div class="card">
                <h2>Quality Issues</h2>
                <p style="color: #4CAF50; font-weight: bold;">No issues found!</p>
            </div>
            '''

        # Group by type
        by_type = qa_result.issues_by_type()

        html = '''
        <div class="card">
            <h2>Quality Issues</h2>
            <p>
                <span class="badge badge-error">{errors} errors</span>
                <span class="badge badge-warning">{warnings} warnings</span>
                <span class="badge badge-info">{info} info</span>
            </p>
        '''.format(
            errors=qa_result.num_errors,
            warnings=qa_result.num_warnings,
            info=qa_result.num_info,
        )

        for issue_type, issues in by_type.items():
            html += f'<h3>{issue_type.value.replace("_", " ").title()} ({len(issues)})</h3>'
            for issue in issues[:10]:  # Limit to 10 per type
                html += f'''
                <div class="issue {issue.severity}">
                    <div class="issue-type">{issue.description}</div>
                    {f'<div class="issue-path">{issue.sample_path}</div>' if issue.sample_path else ''}
                </div>
                '''
            if len(issues) > 10:
                html += f'<p>... and {len(issues) - 10} more</p>'

        html += '</div>'
        return html

    def _render_class_table(self, stats: dict) -> str:
        """Render class statistics table."""
        dist = stats.get("class_distribution", {})
        samples_per_class = stats.get("samples_per_class", {})

        if not dist:
            return ""

        rows = ""
        for class_name in sorted(dist.keys()):
            count = dist.get(class_name, 0)
            samples = samples_per_class.get(class_name, 0)
            rows += f'''
            <tr>
                <td>{class_name}</td>
                <td>{count}</td>
                <td>{samples}</td>
                <td>{count / samples if samples > 0 else 0:.1f}</td>
            </tr>
            '''

        return f'''
        <div class="card">
            <h2>Class Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Total Labels</th>
                        <th>Images with Class</th>
                        <th>Avg per Image</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        '''

    def generate_qa_report(
        self,
        dataset: Dataset,
        qa_result: QAResult,
        output_path: Path,
    ) -> Path:
        """Generate QA report HTML.

        Args:
            dataset: Dataset analyzed
            qa_result: QA analysis result
            output_path: Output HTML file path

        Returns:
            Path to generated report
        """
        stats = dataset.stats()

        # Build content sections
        content_parts = [
            f'<h1>Dataset QA Report: {dataset.name}</h1>',
            f'<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
            self._render_stats_section(stats),
        ]

        scripts = []

        # Class distribution chart
        chart_html, chart_script = self._render_class_distribution_chart(stats)
        content_parts.append(chart_html)
        if chart_script:
            scripts.append(chart_script)

        # Issues section
        content_parts.append(self._render_issues_section(qa_result))

        # Class table
        content_parts.append(self._render_class_table(stats))

        # Assemble HTML
        html = HTML_TEMPLATE.format(
            title=f"QA Report - {dataset.name}",
            content="\n".join(content_parts),
            scripts="\n".join(scripts),
        )

        output_path.write_text(html)
        return output_path

    def generate_dataset_report(
        self,
        dataset: Dataset,
        output_path: Path,
    ) -> Path:
        """Generate dataset statistics report.

        Args:
            dataset: Dataset to report on
            output_path: Output HTML file path

        Returns:
            Path to generated report
        """
        stats = dataset.stats()

        content_parts = [
            f'<h1>Dataset Report: {dataset.name}</h1>',
            f'<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
            self._render_stats_section(stats),
        ]

        scripts = []

        chart_html, chart_script = self._render_class_distribution_chart(stats)
        content_parts.append(chart_html)
        if chart_script:
            scripts.append(chart_script)

        content_parts.append(self._render_class_table(stats))

        html = HTML_TEMPLATE.format(
            title=f"Dataset Report - {dataset.name}",
            content="\n".join(content_parts),
            scripts="\n".join(scripts),
        )

        output_path.write_text(html)
        return output_path


# Convenience functions
def generate_qa_report(
    dataset: Dataset,
    qa_result: QAResult,
    output_path: Path,
) -> Path:
    """Generate QA report."""
    generator = ReportGenerator()
    return generator.generate_qa_report(dataset, qa_result, output_path)


def generate_dataset_report(
    dataset: Dataset,
    output_path: Path,
) -> Path:
    """Generate dataset statistics report."""
    generator = ReportGenerator()
    return generator.generate_dataset_report(dataset, output_path)
```

### 2. Create unit tests

Create `backend/tests/data/test_report.py`:

```python
"""Tests for report generation."""

from pathlib import Path

import pytest

from backend.data.models import BBox, Dataset, Label, Sample
from backend.data.qa import run_qa
from backend.data.report import generate_dataset_report, generate_qa_report


@pytest.fixture
def sample_dataset():
    """Create sample dataset."""
    samples = [
        Sample(
            image_path=Path(f"/data/img{i}.jpg"),
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        )
        for i in range(10)
    ]
    return Dataset(samples, class_names=["car"])


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_dataset_report(self, sample_dataset, tmp_path):
        """Test dataset report generation."""
        output = tmp_path / "report.html"
        result = generate_dataset_report(sample_dataset, output)

        assert result.exists()
        content = result.read_text()
        assert "Dataset Report" in content
        assert "10" in content  # num_samples

    def test_generate_qa_report(self, sample_dataset, tmp_path):
        """Test QA report generation."""
        qa_result = run_qa(sample_dataset)
        output = tmp_path / "qa_report.html"
        result = generate_qa_report(sample_dataset, qa_result, output)

        assert result.exists()
        content = result.read_text()
        assert "QA Report" in content

    def test_report_contains_chart(self, sample_dataset, tmp_path):
        """Test report contains chart."""
        output = tmp_path / "report.html"
        result = generate_dataset_report(sample_dataset, output)

        content = result.read_text()
        assert "plotly" in content.lower()
        assert "classDistChart" in content
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/report.py` | Create | Report generation |
| `backend/data/__init__.py` | Modify | Export report module |
| `backend/tests/data/test_report.py` | Create | Unit tests |

## Verification

```bash
cd backend
pytest tests/data/test_report.py -v
```

## Notes

- Uses Plotly.js CDN for charts (requires internet)
- Single HTML file with embedded styles
- Badge styling for issue severity
- Limits issues shown per type to prevent huge reports
- Can be extended with sample gallery images
