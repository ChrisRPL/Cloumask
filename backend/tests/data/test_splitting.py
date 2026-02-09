"""Tests for dataset splitting utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.data.models import BBox, Dataset, Label, Sample
from backend.data.splitting import (
    DatasetSplitter,
    create_folds,
    cross_validation_indices,
    split_dataset,
    split_indices,
    stratified_split_indices,
)


@pytest.fixture
def sample_dataset() -> Dataset:
    """Create deterministic dataset with 70/30 class split."""
    samples: list[Sample] = []

    for idx in range(100):
        class_name = "car" if idx < 70 else "person"
        class_id = 0 if class_name == "car" else 1
        samples.append(
            Sample(
                image_path=Path(f"/data/img{idx:03d}.jpg"),
                labels=[
                    Label(
                        class_name=class_name,
                        class_id=class_id,
                        bbox=BBox(0.5, 0.5, 0.2, 0.2),
                    )
                ],
            )
        )

    return Dataset(samples, name="sample", class_names=["car", "person"])


@pytest.fixture
def tiny_dataset() -> Dataset:
    """Create small dataset with rare and unlabeled samples."""
    samples = [
        Sample(
            image_path=Path("/data/car_1.jpg"),
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ),
        Sample(
            image_path=Path("/data/car_2.jpg"),
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ),
        Sample(
            image_path=Path("/data/car_3.jpg"),
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ),
        Sample(
            image_path=Path("/data/car_4.jpg"),
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ),
        Sample(
            image_path=Path("/data/car_5.jpg"),
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ),
        Sample(
            image_path=Path("/data/car_6.jpg"),
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ),
        Sample(
            image_path=Path("/data/car_7.jpg"),
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ),
        Sample(
            image_path=Path("/data/car_8.jpg"),
            labels=[Label(class_name="car", class_id=0, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ),
        Sample(
            image_path=Path("/data/person.jpg"),
            labels=[Label(class_name="person", class_id=1, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ),
        Sample(image_path=Path("/data/unlabeled.jpg"), labels=[]),
    ]
    return Dataset(samples, name="tiny", class_names=["car", "person"])


class TestSplitIndices:
    """Tests for non-stratified index splitting."""

    def test_basic_split(self) -> None:
        result = split_indices(100, {"train": 0.8, "val": 0.2}, seed=42)

        assert len(result["train"]) == 80
        assert len(result["val"]) == 20
        assert not (set(result["train"]) & set(result["val"]))
        assert sorted(result["train"] + result["val"]) == list(range(100))

    def test_ratio_normalization(self) -> None:
        result = split_indices(10, {"train": 8.0, "val": 2.0, "test": 0.0}, seed=42)

        assert len(result["train"]) == 8
        assert len(result["val"]) == 2
        assert len(result["test"]) == 0

    def test_reproducibility(self) -> None:
        first = split_indices(50, {"train": 0.7, "val": 0.3}, seed=7)
        second = split_indices(50, {"train": 0.7, "val": 0.3}, seed=7)
        assert first == second

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 0"):
            split_indices(-1, {"train": 1.0})

        with pytest.raises(ValueError, match="ratios must not be empty"):
            split_indices(10, {})

        with pytest.raises(ValueError, match="must be >= 0"):
            split_indices(10, {"train": -1.0, "val": 2.0})

        with pytest.raises(ValueError, match="at least one positive"):
            split_indices(10, {"train": 0.0, "val": 0.0})


class TestStratifiedSplitIndices:
    """Tests for stratified splitting."""

    def test_maintains_class_proportions(self, sample_dataset: Dataset) -> None:
        samples = list(sample_dataset)
        result = stratified_split_indices(
            samples,
            {"train": 0.8, "val": 0.1, "test": 0.1},
            seed=42,
        )

        full_ratio = 70 / 100
        for split_indices_list in result.values():
            car_count = sum(1 for idx in split_indices_list if samples[idx].labels[0].class_name == "car")
            split_ratio = car_count / max(len(split_indices_list), 1)
            assert abs(split_ratio - full_ratio) <= 0.12

        all_indices = result["train"] + result["val"] + result["test"]
        assert sorted(all_indices) == list(range(len(samples)))

    def test_handles_rare_and_unlabeled_samples(self, tiny_dataset: Dataset) -> None:
        samples = list(tiny_dataset)
        result = stratified_split_indices(
            samples,
            {"train": 0.6, "val": 0.2, "test": 0.2},
            seed=123,
        )

        all_indices = result["train"] + result["val"] + result["test"]
        assert sorted(all_indices) == list(range(len(samples)))

        unlabeled_idx = next(idx for idx, sample in enumerate(samples) if not sample.labels)
        containing_splits = sum(unlabeled_idx in split for split in result.values())
        assert containing_splits == 1

    def test_reproducibility(self, sample_dataset: Dataset) -> None:
        samples = list(sample_dataset)
        first = stratified_split_indices(samples, {"train": 0.8, "val": 0.2}, seed=11)
        second = stratified_split_indices(samples, {"train": 0.8, "val": 0.2}, seed=11)
        assert first == second


class TestCrossValidationIndices:
    """Tests for k-fold index generation."""

    def test_fold_coverage_and_disjoint_validation(self) -> None:
        folds = cross_validation_indices(23, k=5, seed=99)
        assert len(folds) == 5

        all_val_indices: list[int] = []
        for train_indices, val_indices in folds:
            assert len(train_indices) + len(val_indices) == 23
            assert set(train_indices).isdisjoint(val_indices)
            all_val_indices.extend(val_indices)

        assert sorted(all_val_indices) == list(range(23))

    def test_k_larger_than_dataset_size(self) -> None:
        folds = cross_validation_indices(3, k=5, seed=5)
        assert len(folds) == 5
        assert sum(len(val_indices) for _, val_indices in folds) == 3
        assert sum(1 for _, val_indices in folds if len(val_indices) == 0) == 2

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 2"):
            cross_validation_indices(10, k=1)

    def test_reproducibility(self) -> None:
        first = cross_validation_indices(20, k=4, seed=21)
        second = cross_validation_indices(20, k=4, seed=21)
        assert first == second


class TestDatasetSplitter:
    """Tests for high-level dataset splitting API."""

    def test_split_dataset_defaults(self, sample_dataset: Dataset) -> None:
        result = split_dataset(sample_dataset, seed=42)

        assert {"train", "val", "test"} == set(result.splits.keys())
        assert len(result.train) + len(result.val) + len(result.test) == len(sample_dataset)
        assert result.train.class_names == sample_dataset.class_names
        assert result.val.class_names == sample_dataset.class_names
        assert result.test.class_names == sample_dataset.class_names

    def test_non_stratified_split(self, sample_dataset: Dataset) -> None:
        splitter = DatasetSplitter(
            ratios={"train": 0.75, "val": 0.25},
            stratify=False,
            seed=9,
        )
        result = splitter.split(sample_dataset)

        assert {"train", "val"} == set(result.splits.keys())
        assert len(result.train) + len(result.val) == len(sample_dataset)
        assert pytest.approx(result.ratios["train"], rel=0.05) == 0.75

    def test_cross_validate_returns_folds(self, sample_dataset: Dataset) -> None:
        folds = create_folds(sample_dataset, k=5, seed=42)

        assert len(folds) == 5
        for train_ds, val_ds in folds:
            assert len(train_ds) + len(val_ds) == len(sample_dataset)
            assert train_ds.class_names == sample_dataset.class_names
            assert val_ds.class_names == sample_dataset.class_names

    def test_empty_dataset(self) -> None:
        empty = Dataset([], name="empty", class_names=["car"])
        result = split_dataset(empty, seed=1)

        assert len(result.train) == 0
        assert len(result.val) == 0
        assert len(result.test) == 0
        assert all(ratio == 0.0 for ratio in result.ratios.values())
