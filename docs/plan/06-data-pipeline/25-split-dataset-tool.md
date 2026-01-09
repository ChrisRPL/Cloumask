# Split Dataset Agent Tool

> **Parent:** 06-data-pipeline
> **Depends on:** 19-dataset-splitting
> **Blocks:** None

## Objective

Implement the `split_dataset` LangGraph agent tool for splitting datasets into train/val/test sets.

## Acceptance Criteria

- [ ] Tool callable from LangGraph agent
- [ ] Support custom split ratios
- [ ] Stratified splitting option
- [ ] Export splits to directories
- [ ] Return split statistics
- [ ] Return structured result

## Implementation Steps

### 1. Create split.py

Create `backend/agent/tools/split.py`:

```python
"""Split dataset agent tool.

Splits datasets into train/val/test sets.
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SplitInfo(BaseModel):
    """Info about a split."""
    name: str
    num_samples: int
    num_labels: int
    ratio: float
    output_path: str


class SplitDatasetResult(BaseModel):
    """Result of dataset splitting."""
    success: bool
    source_path: str
    format: str
    stratified: bool
    seed: Optional[int]
    splits: list[SplitInfo]
    total_samples: int
    error: Optional[str] = None


@tool
def split_dataset(
    path: str = Field(description="Path to source dataset"),
    output_path: str = Field(description="Path for split output"),
    format: Optional[str] = Field(default=None, description="Dataset format (auto-detect if not provided)"),
    train_ratio: float = Field(default=0.8, description="Training set ratio"),
    val_ratio: float = Field(default=0.1, description="Validation set ratio"),
    test_ratio: float = Field(default=0.1, description="Test set ratio"),
    stratify: bool = Field(default=True, description="Use stratified splitting to maintain class ratios"),
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility"),
    output_format: Optional[str] = Field(default=None, description="Output format (default: same as source)"),
) -> SplitDatasetResult:
    """Split a dataset into train/val/test sets.

    Supports stratified splitting to maintain class proportions across splits.
    Exports each split to a separate directory in the specified format.

    Example:
        split_dataset("/data/coco", "/data/coco_split", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    """
    from backend.data.formats import detect_format, get_exporter, get_loader
    from backend.data.splitting import split_dataset as do_split

    source = Path(path)
    output = Path(output_path)

    try:
        # Detect format
        if format is None:
            format = detect_format(source)
            if format is None:
                return SplitDatasetResult(
                    success=False,
                    source_path=str(source),
                    format="unknown",
                    stratified=stratify,
                    seed=seed,
                    splits=[],
                    total_samples=0,
                    error="Could not detect dataset format",
                )

        # Load dataset
        loader = get_loader(source, format_name=format)
        dataset = loader.load()

        # Split
        ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
        # Normalize ratios
        total = sum(ratios.values())
        ratios = {k: v / total for k, v in ratios.items()}

        result = do_split(dataset, ratios=ratios, stratify=stratify, seed=seed)

        # Export each split
        target_format = output_format or format
        splits_info = []

        for split_name, split_ds in result.splits.items():
            split_output = output / split_name
            exporter = get_exporter(split_output, target_format, overwrite=True)
            exporter.export(split_ds)

            splits_info.append(SplitInfo(
                name=split_name,
                num_samples=len(split_ds),
                num_labels=split_ds.total_labels(),
                ratio=result.ratios[split_name],
                output_path=str(split_output),
            ))

        return SplitDatasetResult(
            success=True,
            source_path=str(source),
            format=format,
            stratified=stratify,
            seed=seed,
            splits=splits_info,
            total_samples=len(dataset),
        )

    except Exception as e:
        return SplitDatasetResult(
            success=False,
            source_path=str(source),
            format=format or "unknown",
            stratified=stratify,
            seed=seed,
            splits=[],
            total_samples=0,
            error=str(e),
        )
```

### 2. Register tool

Add to `backend/agent/tools/__init__.py`:

```python
from backend.agent.tools.split import split_dataset

DATA_TOOLS = [
    convert_format,
    find_duplicates,
    label_qa,
    split_dataset,
]
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/agent/tools/split.py` | Create | Split dataset tool |
| `backend/agent/tools/__init__.py` | Modify | Register tool |

## Verification

```bash
python -c "
from backend.agent.tools.split import split_dataset
result = split_dataset.invoke({
    'path': '/data/yolo',
    'output_path': '/data/yolo_split',
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
})
print(f'Split {result.total_samples} samples:')
for split in result.splits:
    print(f'  {split.name}: {split.num_samples} ({split.ratio:.0%})')
"
```

## Notes

- Ratios are normalized to sum to 1.0
- Stratified split uses first label's class
- Each split exported to separate directory
- output_format allows format conversion during split
