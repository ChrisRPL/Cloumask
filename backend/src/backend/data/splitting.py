"""Dataset splitting utilities.

Supports train/val/test splitting with optional stratification and
cross-validation fold generation.

Implements spec: 06-data-pipeline/19-dataset-splitting
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from backend.data.models import Dataset, Sample

DEFAULT_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
UNLABELED_CLASS = "__unlabeled__"


def _normalize_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    """Validate and normalize split ratios."""
    if not ratios:
        raise ValueError("ratios must not be empty")

    cleaned: dict[str, float] = {}
    for split_name, ratio in ratios.items():
        name = str(split_name).strip()
        if not name:
            raise ValueError("split names must be non-empty")

        try:
            value = float(ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ratio for split '{name}': {ratio}") from exc

        if value < 0:
            raise ValueError(f"Ratio for split '{name}' must be >= 0, got {value}")

        cleaned[name] = value

    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("ratios must contain at least one positive value")

    return {name: value / total for name, value in cleaned.items()}


def _allocate_counts(total_items: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Allocate integer counts per split while preserving total size."""
    split_names = list(ratios.keys())
    if total_items == 0:
        return {name: 0 for name in split_names}

    expected = {name: total_items * ratios[name] for name in split_names}
    counts = {name: int(expected[name]) for name in split_names}
    remaining = total_items - sum(counts.values())

    if remaining <= 0:
        return counts

    # Largest remainder method with stable tie-breaking by split order.
    ranked = sorted(
        enumerate(split_names),
        key=lambda item: (-(expected[item[1]] - counts[item[1]]), item[0]),
    )
    for idx in range(remaining):
        counts[ranked[idx % len(ranked)][1]] += 1

    return counts


def _primary_class(sample: Sample) -> str:
    """Get a deterministic primary class key for stratification."""
    if not sample.labels:
        return UNLABELED_CLASS

    first_label = sample.labels[0]
    class_name = first_label.class_name.strip()
    if class_name:
        return class_name
    return f"__class_{first_label.class_id}__"


@dataclass
class SplitResult:
    """Result of dataset splitting.

    Attributes:
        splits: Mapping from split name to split dataset
        ratios: Actual split ratios (based on resulting sample counts)
        seed: Random seed used for reproducibility
    """

    splits: dict[str, Dataset]
    ratios: dict[str, float]
    seed: int | None

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
    seed: int | None = None,
) -> dict[str, list[int]]:
    """Split indices into groups based on target ratios."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    normalized = _normalize_ratios(ratios)
    rng = random.Random(seed)

    indices = list(range(n))
    rng.shuffle(indices)

    counts = _allocate_counts(n, normalized)
    split_map: dict[str, list[int]] = {}
    start = 0

    for split_name, count in counts.items():
        end = start + count
        split_map[split_name] = indices[start:end]
        start = end

    return split_map


def stratified_split_indices(
    samples: Sequence[Sample],
    ratios: dict[str, float],
    seed: int | None = None,
) -> dict[str, list[int]]:
    """Split sample indices with class-aware stratification."""
    normalized = _normalize_ratios(ratios)
    rng = random.Random(seed)

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        grouped_indices[_primary_class(sample)].append(idx)

    split_names = list(normalized.keys())
    split_map: dict[str, list[int]] = {name: [] for name in split_names}

    for group in grouped_indices.values():
        rng.shuffle(group)
        counts = _allocate_counts(len(group), normalized)
        start = 0

        for split_name in split_names:
            end = start + counts[split_name]
            split_map[split_name].extend(group[start:end])
            start = end

    for split_name in split_names:
        rng.shuffle(split_map[split_name])

    return split_map


def cross_validation_indices(
    n: int,
    k: int = 5,
    seed: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Generate k-fold cross-validation index splits.

    Returns a list of ``(train_indices, val_indices)`` tuples.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    fold_sizes = [n // k] * k
    for idx in range(n % k):
        fold_sizes[idx] += 1

    folds: list[tuple[list[int], list[int]]] = []
    start = 0

    for fold_size in fold_sizes:
        end = start + fold_size
        val_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]
        folds.append((train_indices, val_indices))
        start = end

    return folds


class DatasetSplitter:
    """Split datasets into train/val/test subsets."""

    def __init__(
        self,
        ratios: dict[str, float] | None = None,
        stratify: bool = True,
        seed: int | None = None,
    ) -> None:
        self.ratios = _normalize_ratios(ratios or DEFAULT_SPLIT_RATIOS)
        self.stratify = stratify
        self.seed = seed

    def split(self, dataset: Dataset) -> SplitResult:
        """Split a dataset according to configured ratios."""
        samples = list(dataset)
        if self.stratify:
            split_map = stratified_split_indices(samples, self.ratios, self.seed)
        else:
            split_map = split_indices(len(samples), self.ratios, self.seed)

        class_names = list(dataset.class_names)
        total = len(samples)
        splits: dict[str, Dataset] = {}
        actual_ratios: dict[str, float] = {}

        for split_name, indices in split_map.items():
            split_samples = [samples[idx] for idx in indices]
            splits[split_name] = Dataset(
                split_samples,
                name=f"{dataset.name}_{split_name}",
                class_names=list(class_names),
            )
            actual_ratios[split_name] = (len(indices) / total) if total else 0.0

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
        """Create train/validation dataset pairs for k folds."""
        samples = list(dataset)
        class_names = list(dataset.class_names)
        folds = cross_validation_indices(len(samples), k=k, seed=self.seed)

        datasets: list[tuple[Dataset, Dataset]] = []
        for fold_idx, (train_indices, val_indices) in enumerate(folds, start=1):
            train_samples = [samples[idx] for idx in train_indices]
            val_samples = [samples[idx] for idx in val_indices]

            train_ds = Dataset(
                train_samples,
                name=f"{dataset.name}_fold{fold_idx}_train",
                class_names=list(class_names),
            )
            val_ds = Dataset(
                val_samples,
                name=f"{dataset.name}_fold{fold_idx}_val",
                class_names=list(class_names),
            )
            datasets.append((train_ds, val_ds))

        return datasets


def split_dataset(
    dataset: Dataset,
    ratios: dict[str, float] | None = None,
    stratify: bool = True,
    seed: int | None = None,
) -> SplitResult:
    """Split dataset into train/val/test sets."""
    splitter = DatasetSplitter(ratios=ratios, stratify=stratify, seed=seed)
    return splitter.split(dataset)


def create_folds(
    dataset: Dataset,
    k: int = 5,
    seed: int | None = None,
) -> list[tuple[Dataset, Dataset]]:
    """Create k-fold train/validation dataset pairs."""
    splitter = DatasetSplitter(seed=seed)
    return splitter.cross_validate(dataset, k=k)


__all__ = [
    "SplitResult",
    "DatasetSplitter",
    "split_indices",
    "stratified_split_indices",
    "cross_validation_indices",
    "split_dataset",
    "create_folds",
]
