# Convert Format Agent Tool

> **Parent:** 06-data-pipeline
> **Depends on:** All loaders (03-09), all exporters (10-16)
> **Blocks:** None

## Objective

Implement the `convert_format` LangGraph agent tool for converting datasets between annotation formats.

## Acceptance Criteria

- [ ] Tool callable from LangGraph agent
- [ ] Auto-detect source format
- [ ] Convert to any supported target format
- [ ] Report conversion statistics
- [ ] Handle errors gracefully
- [ ] Return structured result

## Implementation Steps

### 1. Create convert.py

Create `backend/agent/tools/convert.py`:

```python
"""Convert format agent tool.

Converts datasets between annotation formats.
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ConvertResult(BaseModel):
    """Result of format conversion."""
    success: bool
    source_format: str
    target_format: str
    source_path: str
    output_path: str
    num_samples: int
    num_labels: int
    warnings: list[str] = []
    error: Optional[str] = None


@tool
def convert_format(
    source_path: str = Field(description="Path to source dataset"),
    output_path: str = Field(description="Path for converted output"),
    target_format: str = Field(description="Target format (yolo, coco, kitti, voc, cvat, nuscenes, openlabel)"),
    source_format: Optional[str] = Field(default=None, description="Source format (auto-detect if not provided)"),
    copy_images: bool = Field(default=True, description="Whether to copy images to output"),
) -> ConvertResult:
    """Convert a dataset from one annotation format to another.

    Supports: YOLO, COCO, KITTI, Pascal VOC, CVAT, nuScenes, OpenLABEL.
    Auto-detects source format if not specified.

    Example:
        convert_format("/data/coco", "/output/yolo", "yolo")
    """
    from backend.data.formats import convert, detect_format, get_exporter, get_loader

    source = Path(source_path)
    output = Path(output_path)

    try:
        # Detect source format if not provided
        if source_format is None:
            source_format = detect_format(source)
            if source_format is None:
                return ConvertResult(
                    success=False,
                    source_format="unknown",
                    target_format=target_format,
                    source_path=str(source),
                    output_path=str(output),
                    num_samples=0,
                    num_labels=0,
                    error="Could not detect source format",
                )

        # Load source dataset
        loader = get_loader(source, format_name=source_format)
        warnings = loader.validate()
        dataset = loader.load()

        # Export to target format
        exporter = get_exporter(output, target_format)
        exporter.export(dataset, copy_images=copy_images)
        warnings.extend(exporter.validate_export())

        return ConvertResult(
            success=True,
            source_format=source_format,
            target_format=target_format,
            source_path=str(source),
            output_path=str(output),
            num_samples=len(dataset),
            num_labels=dataset.total_labels(),
            warnings=warnings,
        )

    except Exception as e:
        return ConvertResult(
            success=False,
            source_format=source_format or "unknown",
            target_format=target_format,
            source_path=str(source),
            output_path=str(output),
            num_samples=0,
            num_labels=0,
            error=str(e),
        )
```

### 2. Register tool

Add to `backend/agent/tools/__init__.py`:

```python
from backend.agent.tools.convert import convert_format

DATA_TOOLS = [
    convert_format,
]
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/agent/tools/convert.py` | Create | Convert tool implementation |
| `backend/agent/tools/__init__.py` | Modify | Register tool |

## Verification

```bash
# Test tool directly
python -c "
from backend.agent.tools.convert import convert_format
result = convert_format.invoke({
    'source_path': '/data/coco',
    'output_path': '/tmp/yolo_out',
    'target_format': 'yolo',
})
print(result)
"
```

## Notes

- Tool returns structured result for agent parsing
- Errors returned in result, not raised
- Warnings collected from both loader and exporter
- copy_images=False useful for testing/previewing
