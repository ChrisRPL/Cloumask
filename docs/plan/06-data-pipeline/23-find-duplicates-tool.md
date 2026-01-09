# Find Duplicates Agent Tool

> **Parent:** 06-data-pipeline
> **Depends on:** 17-duplicate-detection
> **Blocks:** None

## Objective

Implement the `find_duplicates` LangGraph agent tool for finding duplicate and near-duplicate images in datasets.

## Acceptance Criteria

- [ ] Tool callable from LangGraph agent
- [ ] Support multiple detection methods
- [ ] Configurable similarity threshold
- [ ] Return duplicate groups with paths
- [ ] Option to auto-remove duplicates
- [ ] Return structured result

## Implementation Steps

### 1. Create duplicates.py

Create `backend/agent/tools/duplicates.py`:

```python
"""Find duplicates agent tool.

Finds duplicate and near-duplicate images in datasets.
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class DuplicateGroupInfo(BaseModel):
    """Info about a duplicate group."""
    representative: str
    duplicates: list[str]
    count: int


class FindDuplicatesResult(BaseModel):
    """Result of duplicate detection."""
    success: bool
    method: str
    threshold: float
    total_images: int
    num_groups: int
    num_duplicates: int
    groups: list[DuplicateGroupInfo]
    removed: list[str] = []
    error: Optional[str] = None


@tool
def find_duplicates(
    path: str = Field(description="Path to dataset or image directory"),
    method: str = Field(default="phash", description="Detection method: phash, dhash, ahash, or clip"),
    threshold: float = Field(default=0.9, description="Similarity threshold (0-1, higher = more similar)"),
    auto_remove: bool = Field(default=False, description="Automatically remove duplicates (keep representatives)"),
    max_groups: int = Field(default=50, description="Maximum number of groups to return"),
) -> FindDuplicatesResult:
    """Find duplicate and near-duplicate images in a directory.

    Methods:
    - phash: Perceptual hash (good for resized/compressed duplicates)
    - dhash: Difference hash (fast, good for exact duplicates)
    - ahash: Average hash (simplest, fastest)
    - clip: CLIP embeddings (finds semantically similar images)

    Example:
        find_duplicates("/data/images", method="phash", threshold=0.95)
    """
    from backend.data.duplicates import find_duplicates as detect_duplicates

    dataset_path = Path(path)

    try:
        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(dataset_path.rglob(f"*{ext}"))

        if not image_paths:
            return FindDuplicatesResult(
                success=True,
                method=method,
                threshold=threshold,
                total_images=0,
                num_groups=0,
                num_duplicates=0,
                groups=[],
            )

        # Run detection
        result = detect_duplicates(image_paths, method=method, threshold=threshold)

        # Build group info
        groups = []
        for group in result.groups[:max_groups]:
            groups.append(DuplicateGroupInfo(
                representative=str(group.representative),
                duplicates=[str(p) for p in group.duplicates],
                count=group.count,
            ))

        # Auto-remove if requested
        removed = []
        if auto_remove:
            for path in result.get_duplicates_to_remove():
                try:
                    path.unlink()
                    removed.append(str(path))
                except Exception:
                    pass

        return FindDuplicatesResult(
            success=True,
            method=method,
            threshold=threshold,
            total_images=result.total_images,
            num_groups=result.num_groups,
            num_duplicates=result.num_duplicates,
            groups=groups,
            removed=removed,
        )

    except Exception as e:
        return FindDuplicatesResult(
            success=False,
            method=method,
            threshold=threshold,
            total_images=0,
            num_groups=0,
            num_duplicates=0,
            groups=[],
            error=str(e),
        )
```

### 2. Register tool

Add to `backend/agent/tools/__init__.py`:

```python
from backend.agent.tools.duplicates import find_duplicates

DATA_TOOLS = [
    convert_format,
    find_duplicates,
]
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/agent/tools/duplicates.py` | Create | Find duplicates tool |
| `backend/agent/tools/__init__.py` | Modify | Register tool |

## Verification

```bash
python -c "
from backend.agent.tools.duplicates import find_duplicates
result = find_duplicates.invoke({
    'path': '/data/images',
    'method': 'phash',
    'threshold': 0.9,
})
print(f'Found {result.num_duplicates} duplicates in {result.num_groups} groups')
"
```

## Notes

- CLIP method requires CLIP model (slower but finds semantic duplicates)
- auto_remove should be used carefully - permanently deletes files
- Groups limited to max_groups to avoid huge responses
- Returns representative for each group (file to keep)
