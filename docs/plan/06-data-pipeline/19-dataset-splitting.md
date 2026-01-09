# Dataset Splitting

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models
> **Blocks:** 25-split-dataset-tool

## Objective

Implement dataset splitting functionality for train/val/test splits with stratification to maintain class distribution across splits.

## Acceptance Criteria

- [ ] Split dataset by ratio (e.g., 80/10/10)
- [ ] Stratified splitting maintains class proportions
- [ ] Support cross-validation fold generation
- [ ] Handle edge cases (small datasets, rare classes)
- [ ] Reproducible splits with random seed
- [ ] Unit tests for all split modes

## Implementation Steps

### 1. Create splitting.py

Create `backend/data/splitting.py`:

```python
"""Dataset splitting utilities.

Supports train/val/test splits with stratification.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from backend.data.models import Dataset, Sample


@dataclass
class SplitResult:
    """Result of dataset splitting.

    Attributes:
        splits: Dict of split name to Dataset
        ratios: Actual ratios achieved
        seed: Random seed used
    """
    splits: dict[str, Dataset]
    ratios: dict[str, float]
    seed: Optional[int]

    def __getitem__(self, key: str) -> Dataset:
        return self.splits[key]

    @property
    def train(self) -> Dataset:
        return self.splits.get("train", Dataset([]))

    @property
    def val(self) -> Dataset:
        return self.splits.get("val", Dataset([]))

    @property
    def test(self) -> Dataset:
        return self.splits.get("test", Dataset([]))


def split_indices(
    n: int,
    ratios: dict[str, float],
    seed: Optional[int] = None,
) -> dict[str, list[int]]:
    """Split indices into groups by ratio.

    Args:
        n: Total number of items
        ratios: Dict of split name to ratio (should sum to ~1.0)
        seed: Random seed for reproducibility

    Returns:
        Dict of split name to list of indices
    """
    if seed is not None:
        random.seed(seed)

    # Normalize ratios
    total_ratio = sum(ratios.values())
    normalized = {k: v / total_ratio for k, v in ratios.items()}

    # Shuffle indices
    indices = list(range(n))
    random.shuffle(indices)

    # Split
    result: dict[str, list[int]] = {}
    start = 0
    remaining_names = list(normalized.keys())

    for name in remaining_names[:-1]:
        count = int(n * normalized[name])
        result[name] = indices[start:start + count]
        start += count

    # Last split gets remaining
    result[remaining_names[-1]] = indices[start:]

    return result


def stratified_split_indices(
    samples: Sequence[Sample],
    ratios: dict[str, float],
    seed: Optional[int] = None,
) -> dict[str, list[int]]:
    """Split indices with stratification by class.

    Maintains class proportions across splits.

    Args:
        samples: List of samples
        ratios: Dict of split name to ratio
        seed: Random seed

    Returns:
        Dict of split name to list of indices
    """
    if seed is not None:
        random.seed(seed)

    # Group samples by primary class (first label)
    class_indices: dict[str, list[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        if sample.labels:
            # Use first label's class as primary
            primary_class = sample.labels[0].class_name
        else:
            primary_class = "__unlabeled__"
        class_indices[primary_class].append(i)

    # Normalize ratios
    total_ratio = sum(ratios.values())
    normalized = {k: v / total_ratio for k, v in ratios.items()}

    # Split each class group
    result: dict[str, list[int]] = {name: [] for name in ratios.keys()}

    for class_name, indices in class_indices.items():
        random.shuffle(indices)
        n = len(indices)
        start = 0
        split_names = list(normalized.keys())

        for name in split_names[:-1]:
            count = max(1, int(n * normalized[name])) if n > len(split_names) else 0
            count = min(count, n - start - (len(split_names) - split_names.index(name) - 1))
            result[name].extend(indices[start:start + count])
            start += count

        # Remaining to last split
        result[split_names[-1]].extend(indices[start:])

    return result


def cross_validation_indices(
    n: int,
    k: int = 5,
    seed: Optional[int] = None,
) -> list[tuple[list[int], list[int]]]:
    """Generate k-fold cross-validation splits.

    Args:
        n: Total number of items
        k: Number of folds
        seed: Random seed

    Returns:
        List of (train_indices, val_indices) tuples
    """
    if seed is not None:
        random.seed(seed)

    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]
        folds.append((train_indices, val_indices))

    return folds


class DatasetSplitter:
    """Split datasets into train/val/test with various strategies.

    Example:
        splitter = DatasetSplitter(ratios={"train": 0.8, "val": 0.1, "test": 0.1})
        result = splitter.split(dataset)
        train_ds = result.train
    """

    def __init__(
        self,
        ratios: Optional[dict[str, float]] = None,
        stratify: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize splitter.

        Args:
            ratios: Split ratios (default: 80/10/10)
            stratify: Use stratified splitting
            seed: Random seed for reproducibility
        """
        self.ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
        self.stratify = stratify
        self.seed = seed

    def split(self, dataset: Dataset) -> SplitResult:
        """Split dataset according to configuration.

        Args:
            dataset: Dataset to split

        Returns:
            SplitResult with split datasets
        """
        samples = list(dataset)

        if self.stratify:
            indices = stratified_split_indices(samples, self.ratios, self.seed)
        else:
            indices = split_indices(len(samples), self.ratios, self.seed)

        # Create split datasets
        splits = {}
        actual_ratios = {}

        for name, idxs in indices.items():
            split_samples = [samples[i] for i in idxs]
            splits[name] = Dataset(
                split_samples,
                name=f"{dataset.name}_{name}",
                class_names=dataset.class_names,
            )
            actual_ratios[name] = len(idxs) / len(samples) if samples else 0

        return SplitResult(
            splits=splits,
            ratios=actual_ratios,
            seed=self.seed,
        )

    def cross_validate(
        self,
        dataset: Dataset,
        k: int = 5,
    ) -> list[tuple[Dataset, Dataset]]:
        """Generate k-fold cross-validation splits.

        Args:
            dataset: Dataset to split
            k: Number of folds

        Returns:
            List of (train_dataset, val_dataset) tuples
        """
        samples = list(dataset)
        folds_indices = cross_validation_indices(len(samples), k, self.seed)

        folds = []
        for train_idx, val_idx in folds_indices:
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            train_ds = Dataset(train_samples, name=f"{dataset.name}_train", class_names=dataset.class_names)
            val_ds = Dataset(val_samples, name=f"{dataset.name}_val", class_names=dataset.class_names)

            folds.append((train_ds, val_ds))

        return folds


# Convenience functions
def split_dataset(
    dataset: Dataset,
    ratios: Optional[dict[str, float]] = None,
    stratify: bool = True,
    seed: Optional[int] = None,
) -> SplitResult:
    """Split dataset into train/val/test.

    Args:
        dataset: Dataset to split
        ratios: Split ratios
        stratify: Use stratified splitting
        seed: Random seed

    Returns:
        SplitResult
    """
    splitter = DatasetSplitter(ratios=ratios, stratify=stratify, seed=seed)
    return splitter.split(dataset)


def create_folds(
    dataset: Dataset,
    k: int = 5,
    seed: Optional[int] = None,
) -> list[tuple[Dataset, Dataset]]:
    """Create k-fold cross-validation splits.

    Args:
        dataset: Dataset to split
        k: Number of folds
        seed: Random seed

    Returns:
        List of (train, val) dataset tuples
    """
    splitter = DatasetSplitter(seed=seed)
    return splitter.cross_validate(dataset, k)
```

### 2. Create unit tests

Create `backend/tests/data/test_splitting.py`:

```python
"""Tests for dataset splitting."""

from pathlib import Path

import pytest

from backend.data.models import BBox, Dataset, Label, Sample
from backend.data.splitting import (
    DatasetSplitter,
    create_folds,
    split_dataset,
    split_indices,
    stratified_split_indices,
)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset."""
    samples = []
    for i in range(100):
        class_name = "car" if i < 70 else "person"
        class_id = 0 if class_name == "car" else 1
        samples.append(Sample(
            image_path=Path(f"/data/img{i:03d}.jpg"),
            labels=[Label(class_name=class_name, class_id=class_id, bbox=BBox(0.5, 0.5, 0.2, 0.2))],
        ))
    return Dataset(samples, class_names=["car", "person"])


class TestSplitIndices:
    """Tests for split_indices."""

    def test_basic_split(self):
        """Test basic index splitting."""
        result = split_indices(100, {"train": 0.8, "val": 0.2}, seed=42)

        assert len(result["train"]) == 80
        assert len(result["val"]) == 20
        assert set(result["train"]) & set(result["val"]) == set()

    def test_three_way_split(self):
        """Test three-way split."""
        result = split_indices(100, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)

        total = sum(len(v) for v in result.values())
        assert total == 100

    def test_reproducibility(self):
        """Test seed reproducibility."""
        r1 = split_indices(100, {"train": 0.8, "val": 0.2}, seed=42)
        r2 = split_indices(100, {"train": 0.8, "val": 0.2}, seed=42)

        assert r1["train"] == r2["train"]


class TestStratifiedSplit:
    """Tests for stratified splitting."""

    def test_maintains_proportions(self, sample_dataset):
        """Test class proportions are maintained."""
        result = stratified_split_indices(
            list(sample_dataset),
            {"train": 0.8, "val": 0.2},
            seed=42,
        )

        # Count classes in each split
        samples = list(sample_dataset)
        train_cars = sum(1 for i in result["train"] if samples[i].labels[0].class_name == "car")
        val_cars = sum(1 for i in result["val"] if samples[i].labels[0].class_name == "car")

        # Proportions should be similar (allow some variance)
        train_ratio = train_cars / len(result["train"])
        val_ratio = val_cars / len(result["val"])

        assert abs(train_ratio - val_ratio) < 0.1


class TestDatasetSplitter:
    """Tests for DatasetSplitter."""

    def test_split_basic(self, sample_dataset):
        """Test basic dataset split."""
        result = split_dataset(sample_dataset, seed=42)

        assert "train" in result.splits
        assert "val" in result.splits
        assert "test" in result.splits
        assert len(result.train) + len(result.val) + len(result.test) == len(sample_dataset)

    def test_split_ratios(self, sample_dataset):
        """Test split ratios are approximately correct."""
        result = split_dataset(
            sample_dataset,
            ratios={"train": 0.8, "val": 0.1, "test": 0.1},
            seed=42,
        )

        assert 0.75 < result.ratios["train"] < 0.85
        assert len(result.train) > len(result.val)

    def test_cross_validation(self, sample_dataset):
        """Test k-fold cross-validation."""
        folds = create_folds(sample_dataset, k=5, seed=42)

        assert len(folds) == 5
        for train_ds, val_ds in folds:
            assert len(train_ds) > len(val_ds)
            assert len(train_ds) + len(val_ds) == len(sample_dataset)

    def test_class_names_preserved(self, sample_dataset):
        """Test class names are preserved in splits."""
        result = split_dataset(sample_dataset, seed=42)

        assert result.train.class_names == sample_dataset.class_names
        assert result.val.class_names == sample_dataset.class_names
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/splitting.py` | Create | Splitting implementation |
| `backend/data/__init__.py` | Modify | Export splitting module |
| `backend/tests/data/test_splitting.py` | Create | Unit tests |

## Verification

```bash
cd backend
pytest tests/data/test_splitting.py -v
```

## Notes

- Stratification uses first label's class as primary
- Unlabeled samples are treated as separate class
- Cross-validation useful for hyperparameter tuning
- Seed ensures reproducible splits
- Small classes may have imperfect stratification
