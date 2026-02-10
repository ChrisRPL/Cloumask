"""Tests for standalone HTML report generation."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from backend.data.models import BBox, Dataset, Label, Sample
from backend.data.qa import run_qa
from backend.data.report import ReportGenerator, generate_dataset_report, generate_qa_report

ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO0NfVcAAAAASUVORK5CYII="
)


def _write_png(path: Path) -> None:
    path.write_bytes(ONE_PIXEL_PNG)


@pytest.fixture
def report_dataset(tmp_path: Path) -> Dataset:
    """Create a dataset that triggers multiple QA issue categories."""

    img_ok = tmp_path / "img_ok.png"
    img_unlabeled = tmp_path / "img_unlabeled.png"
    img_overlap = tmp_path / "img_overlap.png"
    img_tiny = tmp_path / "img_tiny.png"
    missing = tmp_path / "missing.png"

    for image_path in (img_ok, img_unlabeled, img_overlap, img_tiny):
        _write_png(image_path)

    samples = [
        Sample(
            image_path=img_ok,
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.3, 0.3))],
        ),
        Sample(image_path=img_unlabeled, labels=[]),
        Sample(
            image_path=img_overlap,
            labels=[
                Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2)),
                Label(class_name="person", class_id=1, bbox=BBox(0.51, 0.51, 0.2, 0.2)),
                Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2)),
            ],
        ),
        Sample(
            image_path=img_tiny,
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.01, 0.01))],
        ),
        Sample(
            image_path=missing,
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.95, 0.5, 0.2, 0.2))],
        ),
    ]
    return Dataset(samples, name="qa_dataset", class_names=["car", "person"])


class TestReportGeneration:
    """Tests for HTML report generation."""

    def test_generate_dataset_report(self, report_dataset: Dataset, tmp_path: Path) -> None:
        output = tmp_path / "dataset_report.html"

        result = generate_dataset_report(report_dataset, output)

        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "Cloumask Dataset Report" in content
        assert "#166534" in content
        assert "#faf7f0" in content
        assert "classDistributionData" in content
        assert "data-class-sort=\"count\"" in content

    def test_generate_qa_report_contains_grouped_issues(
        self,
        report_dataset: Dataset,
        tmp_path: Path,
    ) -> None:
        qa_result = run_qa(report_dataset)
        output = tmp_path / "qa_report.html"

        result = generate_qa_report(report_dataset, qa_result, output)

        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "Cloumask Dataset QA Report" in content
        assert "Quality Issues" in content
        assert "Missing Labels" in content
        assert "Overlapping Boxes" in content
        assert "Issue Sample Gallery" in content
        assert "missing_labels" in content

    def test_gallery_embeds_images_and_falls_back_for_missing_files(
        self,
        report_dataset: Dataset,
        tmp_path: Path,
    ) -> None:
        qa_result = run_qa(report_dataset)
        output = tmp_path / "gallery_report.html"

        generator = ReportGenerator(max_gallery_samples=10, max_gallery_issues_per_sample=3)
        generator.generate_qa_report(report_dataset, qa_result, output)

        content = output.read_text(encoding="utf-8")
        assert "data:image/png;base64," in content
        assert "Preview unavailable" in content
        assert "badge-error" in content
