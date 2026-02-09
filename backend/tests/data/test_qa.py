"""Tests for label quality assurance checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.data.models import BBox, Dataset, Label, Sample
from backend.data.qa import IssueType, LabelQA, run_qa


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Dataset:
    """Create sample dataset with deterministic QA issues."""

    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img3 = tmp_path / "img3.jpg"
    img4 = tmp_path / "img4.jpg"
    img5 = tmp_path / "img5.jpg"
    img6 = tmp_path / "img6.jpg"
    missing = tmp_path / "missing.jpg"

    for image_path in (img1, img2, img3, img4, img5, img6):
        image_path.touch()

    samples = [
        # Normal sample
        Sample(
            image_path=img1,
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ),
        # Missing labels
        Sample(image_path=img2, labels=[]),
        # Overlapping + duplicate label pair
        Sample(
            image_path=img3,
            labels=[
                Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2)),
                Label(class_name="person", class_id=1, bbox=BBox(0.52, 0.52, 0.2, 0.2)),
                Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2)),
            ],
        ),
        # Out-of-bounds box
        Sample(
            image_path=img4,
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.95, 0.5, 0.2, 0.2))],
        ),
        # Tiny box
        Sample(
            image_path=img5,
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.01, 0.01))],
        ),
        # Huge box
        Sample(
            image_path=img6,
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.96, 0.96))],
        ),
        # Missing image file
        Sample(
            image_path=missing,
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ),
    ]
    return Dataset(samples, class_names=["car", "person"])


class TestLabelQA:
    """Tests for LabelQA checks."""

    def test_check_missing_labels(self, sample_dataset: Dataset) -> None:
        qa = LabelQA()
        issues = qa.check_missing_labels(sample_dataset)

        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.MISSING_LABELS
        assert issues[0].sample_path is not None
        assert issues[0].sample_path.name == "img2.jpg"

    def test_check_missing_images(self, sample_dataset: Dataset) -> None:
        qa = LabelQA()
        issues = qa.check_missing_images(sample_dataset)

        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.MISSING_IMAGE
        assert issues[0].severity == "error"
        assert issues[0].sample_path is not None
        assert issues[0].sample_path.name == "missing.jpg"

    def test_check_overlapping_boxes(self, sample_dataset: Dataset) -> None:
        qa = LabelQA(iou_threshold=0.8)
        issues = qa.check_overlapping_boxes(sample_dataset)

        assert len(issues) >= 1
        assert any(issue.issue_type == IssueType.OVERLAPPING_BOXES for issue in issues)

    def test_check_out_of_bounds(self, sample_dataset: Dataset) -> None:
        qa = LabelQA()
        issues = qa.check_out_of_bounds(sample_dataset)

        assert len(issues) >= 1
        assert any(issue.issue_type == IssueType.OUT_OF_BOUNDS for issue in issues)

    def test_check_box_sizes(self, sample_dataset: Dataset) -> None:
        qa = LabelQA(tiny_box_threshold=0.001, huge_box_threshold=0.9)
        issues = qa.check_box_sizes(sample_dataset)

        issue_types = {issue.issue_type for issue in issues}
        assert IssueType.TINY_BOX in issue_types
        assert IssueType.HUGE_BOX in issue_types

    def test_check_class_imbalance(self, tmp_path: Path) -> None:
        samples = []
        for idx in range(40):
            image_path = tmp_path / f"car_{idx}.jpg"
            image_path.touch()
            samples.append(
                Sample(
                    image_path=image_path,
                    labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
                )
            )

        rare_path = tmp_path / "rare.jpg"
        rare_path.touch()
        samples.append(
            Sample(
                image_path=rare_path,
                labels=[Label(class_name="rare", class_id=1, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
            )
        )

        dataset = Dataset(samples, class_names=["car", "rare"])
        qa = LabelQA(imbalance_ratio=10.0)
        issues = qa.check_class_imbalance(dataset)

        assert len(issues) >= 1
        assert any(issue.details.get("class") == "rare" for issue in issues)

    def test_check_duplicate_labels(self, sample_dataset: Dataset) -> None:
        qa = LabelQA()
        issues = qa.check_duplicate_labels(sample_dataset)

        assert len(issues) >= 1
        assert any(issue.issue_type == IssueType.DUPLICATE_LABEL for issue in issues)

    def test_run_all_checks(self, sample_dataset: Dataset) -> None:
        result = run_qa(sample_dataset)

        assert result.total_issues > 0
        assert set(result.checks_run) == {
            "missing_labels",
            "missing_images",
            "overlapping_boxes",
            "out_of_bounds",
            "box_sizes",
            "class_imbalance",
            "duplicate_labels",
        }
        assert result.stats["num_samples"] == len(sample_dataset)

    def test_run_specific_checks(self, sample_dataset: Dataset) -> None:
        result = run_qa(sample_dataset, checks=["missing_labels", "unknown_check"])

        assert result.checks_run == ["missing_labels"]
        assert all(issue.issue_type == IssueType.MISSING_LABELS for issue in result.issues)

    def test_result_to_dict(self, sample_dataset: Dataset) -> None:
        result = run_qa(sample_dataset)
        data = result.to_dict()

        assert "issues" in data
        assert "stats" in data
        assert "summary" in data
        assert data["summary"]["total_issues"] == result.total_issues

    def test_progress_callback(self, sample_dataset: Dataset) -> None:
        callbacks: list[tuple[int, int, str]] = []

        def callback(current: int, total: int, message: str) -> None:
            callbacks.append((current, total, message))

        qa = LabelQA(progress_callback=callback)
        qa.run(sample_dataset, checks=["missing_labels", "missing_images"])

        assert callbacks
        assert callbacks[-1][0] == 2
        assert callbacks[-1][1] == 2
