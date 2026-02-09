"""HTML report generation for datasets and QA results.

Produces standalone single-file reports with inline styles/scripts and optional
embedded image previews for QA issue samples.

Implements spec: 06-data-pipeline/21-html-reports
"""

from __future__ import annotations

import base64
import html
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backend.data.models import Dataset
from backend.data.qa import QAIssue, QAResult

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
SEVERITY_TEXT = {"error": "Error", "warning": "Warning", "info": "Info"}

SUPPORTED_IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

MAX_ISSUES_PER_TYPE_DEFAULT = 25
MAX_GALLERY_SAMPLES_DEFAULT = 12
MAX_GALLERY_ISSUES_PER_SAMPLE_DEFAULT = 5
MAX_EMBEDDED_IMAGE_BYTES = 8_000_000

REPORT_CSS = """
* {
    box-sizing: border-box;
}

:root {
    --forest: #166534;
    --forest-strong: #14532d;
    --forest-soft: #dcfce7;
    --cream: #faf7f0;
    --cream-strong: #f1ede4;
    --ink: #1f2937;
    --muted: #4b5563;
    --border: #d9d4c8;
    --error: #b91c1c;
    --warning: #b45309;
    --info: #0f766e;
}

body {
    margin: 0;
    min-height: 100vh;
    color: var(--ink);
    background:
        radial-gradient(circle at 0% 0%, rgba(22, 101, 52, 0.08), transparent 45%),
        radial-gradient(circle at 100% 100%, rgba(22, 101, 52, 0.06), transparent 40%),
        var(--cream);
    font-family:
        "JetBrains Mono",
        "IBM Plex Mono",
        "Fira Code",
        "Cascadia Code",
        "SFMono-Regular",
        Menlo,
        Consolas,
        "Liberation Mono",
        monospace;
    line-height: 1.5;
}

.container {
    width: min(1180px, 94%);
    margin: 28px auto 42px;
}

.card {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 14px;
    backdrop-filter: blur(3px);
    box-shadow: 0 8px 24px rgba(20, 83, 45, 0.06);
}

.section-title {
    margin: 0 0 12px;
    font-size: 1.05rem;
    color: var(--forest-strong);
}

.muted {
    color: var(--muted);
}

.report-header {
    border-left: 4px solid var(--forest);
}

.header-row {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: center;
    flex-wrap: wrap;
}

.brand {
    display: flex;
    gap: 12px;
    align-items: center;
}

.brand-logo {
    width: auto;
    max-width: min(46vw, 244px);
    height: 50px;
    object-fit: contain;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--cream-strong);
    padding: 4px 8px;
}

@media (max-width: 720px) {
    .brand-logo {
        max-width: 188px;
        height: 42px;
    }
}

.brand-name {
    margin: 0;
    font-size: 1.25rem;
    color: var(--forest-strong);
}

.brand-subtitle {
    margin: 0;
    font-size: 0.85rem;
    color: var(--muted);
}

.meta-block {
    text-align: right;
}

.meta-block p {
    margin: 0;
    color: var(--muted);
    font-size: 0.84rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
}

.stat-box {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    min-height: 84px;
}

.stat-label {
    color: var(--muted);
    font-size: 0.8rem;
    margin-bottom: 4px;
}

.stat-value {
    color: var(--forest-strong);
    font-size: 1.7rem;
    line-height: 1.1;
    font-weight: 700;
}

.checks-run {
    margin: 0;
    padding-left: 18px;
}

.checks-run li {
    color: var(--muted);
}

.chart-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}

.chart-btn {
    border: 1px solid var(--border);
    background: var(--cream);
    color: var(--ink);
    border-radius: 8px;
    padding: 5px 10px;
    cursor: pointer;
    font: inherit;
    font-size: 0.83rem;
}

.chart-btn.active {
    border-color: var(--forest);
    color: var(--forest-strong);
    background: var(--forest-soft);
}

.chart-limit {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    font-size: 0.83rem;
}

.chart-limit input {
    width: 72px;
    border: 1px solid var(--border);
    background: #fff;
    border-radius: 6px;
    padding: 4px 6px;
    font: inherit;
}

.bar-chart {
    display: grid;
    gap: 8px;
}

.bar-row {
    display: grid;
    grid-template-columns: minmax(120px, 220px) 1fr auto;
    align-items: center;
    gap: 10px;
}

.bar-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--ink);
    font-size: 0.84rem;
}

.bar-track {
    position: relative;
    height: 20px;
    background: #e6e2d8;
    border-radius: 999px;
    overflow: hidden;
}

.bar-fill {
    position: absolute;
    inset: 0 auto 0 0;
    min-width: 2px;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--forest), #1f7a44);
}

.bar-count {
    color: var(--forest-strong);
    min-width: 38px;
    text-align: right;
    font-size: 0.82rem;
}

.issue-summary {
    margin: 0 0 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.badge {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    border: 1px solid transparent;
    padding: 3px 9px;
    font-size: 0.8rem;
}

.badge-error {
    color: var(--error);
    border-color: rgba(185, 28, 28, 0.25);
    background: rgba(185, 28, 28, 0.08);
}

.badge-warning {
    color: var(--warning);
    border-color: rgba(180, 83, 9, 0.3);
    background: rgba(180, 83, 9, 0.09);
}

.badge-info {
    color: var(--info);
    border-color: rgba(15, 118, 110, 0.25);
    background: rgba(15, 118, 110, 0.08);
}

.issue-group {
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 10px;
    background: #fff;
}

.issue-group summary {
    cursor: pointer;
    padding: 9px 12px;
    color: var(--forest-strong);
    font-weight: 600;
}

.issue-list {
    margin: 0;
    padding: 0 12px 12px;
    list-style: none;
    display: grid;
    gap: 8px;
}

.issue-item {
    border-left: 4px solid transparent;
    border-radius: 8px;
    padding: 8px 10px;
    background: var(--cream);
}

.issue-item.error {
    border-left-color: var(--error);
}

.issue-item.warning {
    border-left-color: var(--warning);
}

.issue-item.info {
    border-left-color: var(--info);
}

.issue-description {
    margin: 0;
}

.issue-meta {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 0.78rem;
}

.gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 10px;
}

.gallery-item {
    border: 1px solid var(--border);
    border-left: 4px solid var(--forest);
    border-radius: 10px;
    background: #fff;
    overflow: hidden;
}

.gallery-item.error {
    border-left-color: var(--error);
}

.gallery-item.warning {
    border-left-color: var(--warning);
}

.gallery-item.info {
    border-left-color: var(--info);
}

.gallery-image {
    display: block;
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
    background: var(--cream-strong);
}

.gallery-placeholder {
    width: 100%;
    aspect-ratio: 4 / 3;
    display: grid;
    place-items: center;
    background: var(--cream-strong);
    color: var(--muted);
    font-size: 0.82rem;
    text-align: center;
    padding: 8px;
}

.gallery-content {
    padding: 10px 12px 12px;
}

.gallery-path {
    margin: 0 0 8px;
    font-size: 0.78rem;
    color: var(--muted);
    overflow-wrap: anywhere;
}

.gallery-issues {
    margin: 0;
    padding-left: 18px;
    font-size: 0.8rem;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}

th,
td {
    border-bottom: 1px solid var(--border);
    padding: 8px 10px;
    text-align: left;
}

th {
    color: var(--forest-strong);
}

@media (max-width: 780px) {
    .meta-block {
        text-align: left;
    }

    .chart-limit {
        margin-left: 0;
    }

    .bar-row {
        grid-template-columns: 1fr;
        gap: 4px;
    }

    .bar-count {
        text-align: left;
    }
}
"""


@dataclass
class GalleryEntry:
    """Prepared data for a sample gallery card."""

    sample_path: Path
    severity: str
    issues: list[QAIssue]
    image_data_uri: str | None


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _severity_rank(severity: str) -> int:
    return SEVERITY_ORDER.get(severity, 99)


def _display_issue_type(issue_type_value: str) -> str:
    return issue_type_value.replace("_", " ").title()


def _format_timestamp(now: datetime | None = None) -> str:
    local_now = (now or datetime.now()).astimezone()
    return local_now.strftime("%Y-%m-%d %H:%M:%S %Z")


def _find_default_logo_path() -> Path | None:
    module_path = Path(__file__).resolve()
    logo_candidates = ("icon.png", "icon_large.png")

    for parent in module_path.parents:
        assets_dir = parent / "assets"
        if not assets_dir.exists():
            continue
        for logo_name in logo_candidates:
            candidate = assets_dir / logo_name
            if candidate.exists():
                return candidate

    return None


def _encode_file_as_data_uri(path: Path) -> str | None:
    mime = SUPPORTED_IMAGE_MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        return None

    try:
        payload = path.read_bytes()
    except OSError:
        logger.warning("Unable to read image for report preview: %s", path)
        return None

    if len(payload) > MAX_EMBEDDED_IMAGE_BYTES:
        logger.info("Skipping preview for %s because file is too large", path)
        return None

    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class ReportGenerator:
    """Generate standalone HTML reports for datasets and QA results."""

    def __init__(
        self,
        max_issues_per_type: int = MAX_ISSUES_PER_TYPE_DEFAULT,
        max_gallery_samples: int = MAX_GALLERY_SAMPLES_DEFAULT,
        max_gallery_issues_per_sample: int = MAX_GALLERY_ISSUES_PER_SAMPLE_DEFAULT,
        logo_path: Path | None = None,
    ) -> None:
        self.max_issues_per_type = max(1, max_issues_per_type)
        self.max_gallery_samples = max(1, max_gallery_samples)
        self.max_gallery_issues_per_sample = max(1, max_gallery_issues_per_sample)
        self.logo_path = logo_path

    def _render_header(
        self,
        *,
        title: str,
        dataset_name: str,
        generated_at: str,
        logo_data_uri: str | None,
    ) -> str:
        logo_html = ""
        if logo_data_uri:
            logo_html = (
                f'<img class="brand-logo" src="{html.escape(logo_data_uri, quote=True)}" '
                'alt="Cloumask logo">'
            )

        return (
            '<section class="card report-header">'
            '<div class="header-row">'
            '<div class="brand">'
            f"{logo_html}"
            "<div>"
            f'<h1 class="brand-name">{html.escape(title)}</h1>'
            f'<p class="brand-subtitle">Dataset: {html.escape(dataset_name)}</p>'
            "</div>"
            "</div>"
            '<div class="meta-block">'
            f"<p>Generated: {html.escape(generated_at)}</p>"
            "<p>Format: Single-file HTML</p>"
            "</div>"
            "</div>"
            "</section>"
        )

    def _render_stats_section(self, stats: dict[str, object]) -> str:
        stat_cards = [
            ("Samples", _safe_int(stats.get("num_samples"))),
            ("Labels", _safe_int(stats.get("num_labels"))),
            ("Classes", _safe_int(stats.get("num_classes"))),
            ("Unlabeled", _safe_int(stats.get("unlabeled_count"))),
            ("Avg labels/sample", float(stats.get("avg_labels_per_sample", 0.0))),
        ]

        boxes: list[str] = []
        for label, value in stat_cards:
            display = f"{value:.2f}" if isinstance(value, float) else f"{value:,}"
            boxes.append(
                '<div class="stat-box">'
                f'<div class="stat-label">{html.escape(label)}</div>'
                f'<div class="stat-value">{display}</div>'
                "</div>"
            )

        return (
            '<section class="card">'
            '<h2 class="section-title">Dataset Statistics</h2>'
            '<div class="stats-grid">'
            + "".join(boxes)
            + "</div>"
            "</section>"
        )

    def _render_checks_section(self, qa_result: QAResult) -> str:
        if not qa_result.checks_run:
            return ""

        checks = "".join(f"<li>{html.escape(check)}</li>" for check in qa_result.checks_run)
        return (
            '<section class="card">'
            '<h2 class="section-title">Checks Run</h2>'
            f'<ul class="checks-run">{checks}</ul>'
            "</section>"
        )

    def _render_class_distribution_section(self, stats: dict[str, object]) -> tuple[str, str]:
        raw_distribution = stats.get("class_distribution", {})
        if not isinstance(raw_distribution, dict):
            raw_distribution = {}

        chart_data: list[dict[str, object]] = []
        for class_name, count in raw_distribution.items():
            chart_data.append({"class_name": str(class_name), "count": _safe_int(count)})

        if not chart_data:
            html_section = (
                '<section class="card">'
                '<h2 class="section-title">Class Distribution</h2>'
                '<p class="muted">No class labels were found.</p>'
                "</section>"
            )
            return html_section, ""

        max_limit = len(chart_data)
        default_limit = min(max_limit, 20)
        html_section = (
            '<section class="card">'
            '<h2 class="section-title">Class Distribution</h2>'
            '<div class="chart-toolbar">'
            '<button type="button" class="chart-btn active" data-class-sort="count">Sort by Count</button>'
            '<button type="button" class="chart-btn" data-class-sort="name">Sort by Name</button>'
            '<label class="chart-limit" for="classDistLimit">'
            'Top classes'
            f'<input id="classDistLimit" type="number" min="1" max="{max_limit}" value="{default_limit}">'
            "</label>"
            "</div>"
            '<div id="classDistChart" class="bar-chart"></div>'
            '<p id="classDistMeta" class="muted"></p>'
            "</section>"
        )

        script = self._build_class_distribution_script(chart_data, default_limit)
        return html_section, script

    def _build_class_distribution_script(
        self,
        chart_data: list[dict[str, object]],
        default_limit: int,
    ) -> str:
        chart_json = json.dumps(chart_data, ensure_ascii=False)
        lines = [
            "const classDistributionData = " + chart_json + ";",
            "(function renderClassDistributionChart() {",
            "  const chartRoot = document.getElementById('classDistChart');",
            "  const limitInput = document.getElementById('classDistLimit');",
            "  const meta = document.getElementById('classDistMeta');",
            "  const buttons = Array.from(document.querySelectorAll('[data-class-sort]'));",
            "  if (!chartRoot) { return; }",
            "  let sortMode = 'count';",
            f"  let limit = {default_limit};",
            "  const escapeHtml = (value) => String(value).replace(/[&<>\"']/g, (char) => {",
            "    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', \"'\": '&#39;' };",
            "    return map[char] || char;",
            "  });",
            "  const clampLimit = (value) => {",
            "    const max = Math.max(classDistributionData.length, 1);",
            "    return Math.max(1, Math.min(max, value));",
            "  };",
            "  const getSorted = () => {",
            "    const items = classDistributionData.slice();",
            "    if (sortMode === 'name') {",
            "      items.sort((a, b) => String(a.class_name).localeCompare(String(b.class_name)));",
            "    } else {",
            "      items.sort((a, b) => Number(b.count) - Number(a.count) || "
            "String(a.class_name).localeCompare(String(b.class_name)));",
            "    }",
            "    return items.slice(0, limit);",
            "  };",
            "  const render = () => {",
            "    const rows = getSorted();",
            "    if (rows.length === 0) {",
            "      chartRoot.innerHTML = '<p class=\"muted\">No class data available.</p>';",
            "      if (meta) { meta.textContent = ''; }",
            "      return;",
            "    }",
            "    const maxCount = Math.max(...rows.map((item) => Number(item.count)), 1);",
            "    chartRoot.innerHTML = rows.map((item) => {",
            "      const width = Math.max((Number(item.count) / maxCount) * 100, 2);",
            "      return `",
            "        <div class=\"bar-row\">",
            "          <span class=\"bar-name\" title=\"${escapeHtml(item.class_name)}\">${escapeHtml(item.class_name)}</span>",
            "          <div class=\"bar-track\">",
            "            <div class=\"bar-fill\" style=\"width:${width.toFixed(2)}%\"></div>",
            "          </div>",
            "          <span class=\"bar-count\">${Number(item.count)}</span>",
            "        </div>`;",
            "    }).join('');",
            "    if (meta) {",
            "      meta.textContent = `Showing ${rows.length} of ${classDistributionData.length} classes`;",
            "    }",
            "  };",
            "  buttons.forEach((button) => {",
            "    button.addEventListener('click', () => {",
            "      sortMode = button.dataset.classSort === 'name' ? 'name' : 'count';",
            "      buttons.forEach((btn) => btn.classList.remove('active'));",
            "      button.classList.add('active');",
            "      render();",
            "    });",
            "  });",
            "  if (limitInput) {",
            "    limitInput.addEventListener('input', () => {",
            "      const parsed = Number(limitInput.value);",
            "      limit = clampLimit(Number.isFinite(parsed) ? parsed : classDistributionData.length);",
            "      limitInput.value = String(limit);",
            "      render();",
            "    });",
            "  }",
            "  render();",
            "})();",
        ]
        return "\n".join(lines)

    def _render_issues_section(self, qa_result: QAResult) -> str:
        if not qa_result.issues:
            return (
                '<section class="card">'
                '<h2 class="section-title">Quality Issues</h2>'
                '<p class="muted">No issues found.</p>'
                "</section>"
            )

        grouped = qa_result.issues_by_type()
        groups_html: list[str] = []

        for issue_type in sorted(grouped.keys(), key=lambda key: key.value):
            issues = grouped[issue_type]
            group_title = _display_issue_type(issue_type.value)
            issue_items: list[str] = []
            shown = issues[: self.max_issues_per_type]

            for issue in shown:
                sample_path = (
                    html.escape(str(issue.sample_path))
                    if issue.sample_path is not None
                    else "Dataset level"
                )
                label_info = (
                    f"label #{issue.label_index}" if issue.label_index is not None else "sample-level"
                )
                issue_items.append(
                    f'<li class="issue-item {issue.severity}">'
                    f'<p class="issue-description">{html.escape(issue.description)}</p>'
                    f'<p class="issue-meta">{sample_path} · {html.escape(label_info)}</p>'
                    "</li>"
                )

            hidden_count = len(issues) - len(shown)
            if hidden_count > 0:
                issue_items.append(
                    f'<li class="issue-item info"><p class="issue-description muted">'
                    f"... and {hidden_count} more issues in this category.</p></li>"
                )

            groups_html.append(
                '<details class="issue-group" open>'
                f"<summary>{html.escape(group_title)} ({len(issues)})</summary>"
                f'<ul class="issue-list">{"".join(issue_items)}</ul>'
                "</details>"
            )

        return (
            '<section class="card">'
            '<h2 class="section-title">Quality Issues</h2>'
            '<p class="issue-summary">'
            f'<span class="badge badge-error">{qa_result.num_errors} errors</span>'
            f'<span class="badge badge-warning">{qa_result.num_warnings} warnings</span>'
            f'<span class="badge badge-info">{qa_result.num_info} info</span>'
            "</p>"
            + "".join(groups_html)
            + "</section>"
        )

    def _build_gallery_entries(self, qa_result: QAResult) -> list[GalleryEntry]:
        grouped: dict[Path, list[QAIssue]] = defaultdict(list)

        for issue in qa_result.issues:
            if issue.sample_path is None:
                continue
            grouped[issue.sample_path].append(issue)

        entries: list[GalleryEntry] = []
        for sample_path, issues in grouped.items():
            severity = min((issue.severity for issue in issues), key=_severity_rank, default="info")
            preview_data_uri = _encode_file_as_data_uri(sample_path) if sample_path.exists() else None
            entries.append(
                GalleryEntry(
                    sample_path=sample_path,
                    severity=severity,
                    issues=issues,
                    image_data_uri=preview_data_uri,
                )
            )

        entries.sort(
            key=lambda item: (
                _severity_rank(item.severity),
                -len(item.issues),
                str(item.sample_path),
            )
        )
        return entries[: self.max_gallery_samples]

    def _render_sample_gallery(self, qa_result: QAResult) -> str:
        entries = self._build_gallery_entries(qa_result)
        if not entries:
            return (
                '<section class="card">'
                '<h2 class="section-title">Issue Sample Gallery</h2>'
                '<p class="muted">No sample-linked issues to preview.</p>'
                "</section>"
            )

        cards: list[str] = []
        for entry in entries:
            severity_text = SEVERITY_TEXT.get(entry.severity, "Issue")
            preview_html: str
            if entry.image_data_uri:
                preview_html = (
                    f'<img class="gallery-image" src="{html.escape(entry.image_data_uri, quote=True)}" '
                    f'alt="Preview of {html.escape(entry.sample_path.name)}">'
                )
            else:
                preview_html = '<div class="gallery-placeholder">Preview unavailable</div>'

            issue_items = "".join(
                f"<li>{html.escape(issue.description)}</li>"
                for issue in entry.issues[: self.max_gallery_issues_per_sample]
            )
            hidden_count = len(entry.issues) - self.max_gallery_issues_per_sample
            if hidden_count > 0:
                issue_items += f"<li>... and {hidden_count} more.</li>"

            cards.append(
                f'<article class="gallery-item {entry.severity}">'
                f"{preview_html}"
                '<div class="gallery-content">'
                '<p class="issue-summary">'
                f'<span class="badge badge-{entry.severity}">{severity_text}</span>'
                f'<span class="badge badge-info">{len(entry.issues)} issue(s)</span>'
                "</p>"
                f'<p class="gallery-path">{html.escape(str(entry.sample_path))}</p>'
                f'<ul class="gallery-issues">{issue_items}</ul>'
                "</div>"
                "</article>"
            )

        return (
            '<section class="card">'
            '<h2 class="section-title">Issue Sample Gallery</h2>'
            '<div class="gallery-grid">'
            + "".join(cards)
            + "</div>"
            "</section>"
        )

    def _render_class_table(self, stats: dict[str, object]) -> str:
        distribution = stats.get("class_distribution", {})
        samples_per_class = stats.get("samples_per_class", {})
        if not isinstance(distribution, dict) or not distribution:
            return ""
        if not isinstance(samples_per_class, dict):
            samples_per_class = {}

        rows: list[str] = []
        for class_name in sorted(distribution.keys(), key=str):
            label_count = _safe_int(distribution.get(class_name))
            sample_count = _safe_int(samples_per_class.get(class_name))
            avg = label_count / sample_count if sample_count else 0.0
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(class_name))}</td>"
                f"<td>{label_count:,}</td>"
                f"<td>{sample_count:,}</td>"
                f"<td>{avg:.2f}</td>"
                "</tr>"
            )

        return (
            '<section class="card">'
            '<h2 class="section-title">Class Details</h2>'
            "<table>"
            "<thead><tr><th>Class</th><th>Total Labels</th><th>Samples</th><th>Avg / Sample</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
            "</section>"
        )

    def _render_document(self, title: str, content_sections: list[str], scripts: list[str]) -> str:
        content = "\n".join(section for section in content_sections if section)
        script_block = "\n".join(script for script in scripts if script).strip()
        if not script_block:
            script_block = "// No client-side scripts needed for this report."

        lines = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(title)}</title>",
            "<style>",
            REPORT_CSS.strip(),
            "</style>",
            "</head>",
            "<body>",
            '<main class="container">',
            content,
            "</main>",
            "<script>",
            script_block,
            "</script>",
            "</body>",
            "</html>",
        ]
        return "\n".join(lines)

    def _resolve_logo_data_uri(self) -> str | None:
        logo_file = self.logo_path or _find_default_logo_path()
        if logo_file is None:
            return None
        return _encode_file_as_data_uri(logo_file)

    def generate_qa_report(
        self,
        dataset: Dataset,
        qa_result: QAResult,
        output_path: Path,
    ) -> Path:
        """Generate a QA report for dataset + QA analysis results."""

        stats = dataset.stats()
        generated_at = _format_timestamp()

        class_dist_html, class_dist_script = self._render_class_distribution_section(stats)

        sections = [
            self._render_header(
                title="Cloumask Dataset QA Report",
                dataset_name=dataset.name,
                generated_at=generated_at,
                logo_data_uri=self._resolve_logo_data_uri(),
            ),
            self._render_stats_section(stats),
            self._render_checks_section(qa_result),
            class_dist_html,
            self._render_issues_section(qa_result),
            self._render_sample_gallery(qa_result),
            self._render_class_table(stats),
        ]

        html_doc = self._render_document(
            title=f"QA Report - {dataset.name}",
            content_sections=sections,
            scripts=[class_dist_script],
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_doc, encoding="utf-8")
        return output_path

    def generate_dataset_report(
        self,
        dataset: Dataset,
        output_path: Path,
    ) -> Path:
        """Generate a dataset-only report with descriptive statistics."""

        stats = dataset.stats()
        generated_at = _format_timestamp()
        class_dist_html, class_dist_script = self._render_class_distribution_section(stats)

        sections = [
            self._render_header(
                title="Cloumask Dataset Report",
                dataset_name=dataset.name,
                generated_at=generated_at,
                logo_data_uri=self._resolve_logo_data_uri(),
            ),
            self._render_stats_section(stats),
            class_dist_html,
            self._render_class_table(stats),
        ]

        html_doc = self._render_document(
            title=f"Dataset Report - {dataset.name}",
            content_sections=sections,
            scripts=[class_dist_script],
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_doc, encoding="utf-8")
        return output_path


def generate_qa_report(
    dataset: Dataset,
    qa_result: QAResult,
    output_path: Path,
) -> Path:
    """Convenience function to generate QA report."""
    return ReportGenerator().generate_qa_report(dataset, qa_result, output_path)


def generate_dataset_report(
    dataset: Dataset,
    output_path: Path,
) -> Path:
    """Convenience function to generate dataset report."""
    return ReportGenerator().generate_dataset_report(dataset, output_path)
