# Export Agent Tool

> **Parent:** 06-data-pipeline
> **Depends on:** All exporters (10-16)
> **Blocks:** None

## Objective

Implement a comprehensive `export` LangGraph agent tool that combines dataset operations (filter, split, convert) and exports to any format.

## Acceptance Criteria

- [ ] Tool callable from LangGraph agent
- [ ] Support all export formats
- [ ] Optional filtering by class
- [ ] Optional confidence threshold filtering
- [ ] Optional train/val/test splitting
- [ ] Copy or link images
- [ ] Return comprehensive statistics
- [ ] Return structured result

## Implementation Steps

### 1. Create export.py

Create `backend/agent/tools/export.py`:

```python
"""Export agent tool.

Comprehensive dataset export with filtering and splitting.
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ExportStats(BaseModel):
    """Statistics for exported data."""
    num_samples: int
    num_labels: int
    num_classes: int
    class_distribution: dict[str, int]


class ExportResult(BaseModel):
    """Result of export operation."""
    success: bool
    source_path: str
    source_format: str
    output_path: str
    output_format: str
    stats: ExportStats
    splits: Optional[dict[str, ExportStats]] = None
    filtered_classes: Optional[list[str]] = None
    confidence_threshold: Optional[float] = None
    images_copied: bool
    error: Optional[str] = None


@tool
def export(
    source_path: str = Field(description="Path to source dataset"),
    output_path: str = Field(description="Path for exported output"),
    output_format: str = Field(description="Output format (yolo, coco, kitti, voc, cvat, nuscenes, openlabel)"),
    source_format: Optional[str] = Field(default=None, description="Source format (auto-detect if not provided)"),
    classes: Optional[list[str]] = Field(default=None, description="Only export these classes (filter)"),
    min_confidence: Optional[float] = Field(default=None, description="Minimum confidence threshold"),
    split: bool = Field(default=False, description="Split into train/val/test"),
    train_ratio: float = Field(default=0.8, description="Training ratio if splitting"),
    val_ratio: float = Field(default=0.1, description="Validation ratio if splitting"),
    test_ratio: float = Field(default=0.1, description="Test ratio if splitting"),
    copy_images: bool = Field(default=True, description="Copy images to output"),
) -> ExportResult:
    """Export a dataset with optional filtering, splitting, and format conversion.

    This is a comprehensive export tool that can:
    1. Load a dataset in any supported format
    2. Filter by class names
    3. Filter by confidence threshold
    4. Split into train/val/test sets
    5. Export to any supported format

    Example:
        # Full pipeline: filter cars, split, export to YOLO
        export(
            "/data/coco",
            "/output/yolo_cars",
            "yolo",
            classes=["car", "truck"],
            split=True,
            train_ratio=0.8,
        )
    """
    from backend.data.formats import detect_format, get_exporter, get_loader
    from backend.data.splitting import split_dataset

    source = Path(source_path)
    output = Path(output_path)

    try:
        # Detect and load
        if source_format is None:
            source_format = detect_format(source)
            if source_format is None:
                return ExportResult(
                    success=False,
                    source_path=str(source),
                    source_format="unknown",
                    output_path=str(output),
                    output_format=output_format,
                    stats=ExportStats(num_samples=0, num_labels=0, num_classes=0, class_distribution={}),
                    images_copied=copy_images,
                    error="Could not detect source format",
                )

        loader = get_loader(source, format_name=source_format)
        dataset = loader.load()

        # Filter by classes
        if classes:
            dataset = dataset.filter_by_class(classes)

        # Filter by confidence
        if min_confidence is not None:
            from backend.data.models import Dataset
            filtered_samples = [
                sample.filter_by_confidence(min_confidence)
                for sample in dataset
            ]
            dataset = Dataset(filtered_samples, name=dataset.name, class_names=dataset.class_names)

        # Split if requested
        splits_stats = None
        if split:
            ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
            result = split_dataset(dataset, ratios=ratios, stratify=True)

            splits_stats = {}
            for split_name, split_ds in result.splits.items():
                split_output = output / split_name
                exporter = get_exporter(split_output, output_format, overwrite=True)
                exporter.export(split_ds, copy_images=copy_images)

                splits_stats[split_name] = ExportStats(
                    num_samples=len(split_ds),
                    num_labels=split_ds.total_labels(),
                    num_classes=split_ds.num_classes,
                    class_distribution=split_ds.class_distribution(),
                )
        else:
            # Export directly
            exporter = get_exporter(output, output_format, overwrite=True)
            exporter.export(dataset, copy_images=copy_images)

        # Compute stats
        stats = ExportStats(
            num_samples=len(dataset),
            num_labels=dataset.total_labels(),
            num_classes=dataset.num_classes,
            class_distribution=dataset.class_distribution(),
        )

        return ExportResult(
            success=True,
            source_path=str(source),
            source_format=source_format,
            output_path=str(output),
            output_format=output_format,
            stats=stats,
            splits=splits_stats,
            filtered_classes=classes,
            confidence_threshold=min_confidence,
            images_copied=copy_images,
        )

    except Exception as e:
        return ExportResult(
            success=False,
            source_path=str(source),
            source_format=source_format or "unknown",
            output_path=str(output),
            output_format=output_format,
            stats=ExportStats(num_samples=0, num_labels=0, num_classes=0, class_distribution={}),
            images_copied=copy_images,
            error=str(e),
        )
```

### 2. Register tool

Add to `backend/agent/tools/__init__.py`:

```python
from backend.agent.tools.export import export

DATA_TOOLS = [
    convert_format,
    find_duplicates,
    label_qa,
    split_dataset,
    export,
]
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/agent/tools/export.py` | Create | Export tool |
| `backend/agent/tools/__init__.py` | Modify | Register tool |

## Verification

```bash
python -c "
from backend.agent.tools.export import export
result = export.invoke({
    'source_path': '/data/coco',
    'output_path': '/output/yolo_filtered',
    'output_format': 'yolo',
    'classes': ['person', 'car'],
    'split': True,
})
print(f'Exported {result.stats.num_samples} samples')
if result.splits:
    for name, stats in result.splits.items():
        print(f'  {name}: {stats.num_samples}')
"
```

## Notes

- This is the most comprehensive export tool
- Combines filtering, splitting, and conversion
- Use convert_format for simple conversions
- Use split_dataset for splitting without filtering
- copy_images=False useful for large datasets (references paths)
