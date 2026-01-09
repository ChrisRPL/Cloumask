# Tool Implementations

> **Status:** 🔴 Not Started
> **Priority:** P0 (Critical)
> **Dependencies:** 06-tool-system
> **Estimated Complexity:** Medium

## Overview

Implement the initial set of tools for the agent: `scan_directory` (fully functional) and stub implementations for `anonymize` and `export` that return mock data. These stubs allow end-to-end testing before CV models are integrated.

## Goals

- [ ] `scan_directory` tool: Analyze folder contents
- [ ] `anonymize` stub: Mock face/plate blurring
- [ ] `export` stub: Mock format conversion
- [ ] Realistic mock data for testing
- [ ] Clear integration points for real implementations

## Technical Design

### Scan Directory Tool

```python
import os
from pathlib import Path
from collections import Counter
from typing import Optional

from agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    success_result,
    error_result,
)
from agent.tools.registry import register_tool


# Supported file extensions by type
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
POINTCLOUD_EXTENSIONS = {".las", ".laz", ".pcd", ".ply", ".xyz"}
ANNOTATION_EXTENSIONS = {".json", ".xml", ".txt"}


@register_tool
class ScanDirectoryTool(BaseTool):
    """Scan a directory to analyze its contents."""

    name = "scan_directory"
    description = """Scan a directory to count files, detect formats, and analyze dataset structure.
Use this as the first step to understand what data you're working with."""
    category = ToolCategory.SCAN

    parameters = [
        ToolParameter(
            "path",
            str,
            "Path to the directory to scan",
            required=True,
        ),
        ToolParameter(
            "recursive",
            bool,
            "Whether to scan subdirectories",
            required=False,
            default=True,
        ),
        ToolParameter(
            "max_depth",
            int,
            "Maximum directory depth (0 = unlimited)",
            required=False,
            default=0,
        ),
    ]

    async def execute(
        self,
        path: str,
        recursive: bool = True,
        max_depth: int = 0,
    ) -> ToolResult:
        """Execute directory scan."""

        dir_path = Path(path)

        # Validate path
        if not dir_path.exists():
            return error_result(f"Directory not found: {path}")

        if not dir_path.is_dir():
            return error_result(f"Not a directory: {path}")

        # Scan files
        try:
            scan_result = await self._scan_directory(dir_path, recursive, max_depth)
            return success_result(scan_result)
        except PermissionError:
            return error_result(f"Permission denied: {path}")
        except Exception as e:
            return error_result(f"Scan failed: {e}")

    async def _scan_directory(
        self,
        root: Path,
        recursive: bool,
        max_depth: int,
    ) -> dict:
        """Perform the actual directory scan."""

        files = []
        extension_counts = Counter()
        type_counts = {"images": 0, "videos": 0, "pointclouds": 0, "annotations": 0, "other": 0}
        subdirs = []
        total_size = 0

        def scan_path(path: Path, current_depth: int = 0):
            nonlocal total_size

            if max_depth > 0 and current_depth > max_depth:
                return

            try:
                for entry in path.iterdir():
                    if entry.is_file():
                        files.append(str(entry))
                        ext = entry.suffix.lower()
                        extension_counts[ext] += 1
                        total_size += entry.stat().st_size

                        # Categorize by type
                        if ext in IMAGE_EXTENSIONS:
                            type_counts["images"] += 1
                        elif ext in VIDEO_EXTENSIONS:
                            type_counts["videos"] += 1
                        elif ext in POINTCLOUD_EXTENSIONS:
                            type_counts["pointclouds"] += 1
                        elif ext in ANNOTATION_EXTENSIONS:
                            type_counts["annotations"] += 1
                        else:
                            type_counts["other"] += 1

                        # Report progress
                        self.report_progress(len(files), 0, f"Scanned {len(files)} files")

                    elif entry.is_dir() and recursive:
                        subdirs.append(str(entry))
                        scan_path(entry, current_depth + 1)

            except PermissionError:
                pass  # Skip inaccessible directories

        scan_path(root)

        # Determine primary data type
        primary_type = "mixed"
        if type_counts["images"] > 0 and type_counts["videos"] == 0 and type_counts["pointclouds"] == 0:
            primary_type = "images"
        elif type_counts["videos"] > 0 and type_counts["images"] == 0 and type_counts["pointclouds"] == 0:
            primary_type = "video"
        elif type_counts["pointclouds"] > 0 and type_counts["images"] == 0 and type_counts["videos"] == 0:
            primary_type = "pointcloud"

        return {
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_human": self._format_size(total_size),
            "subdirectories": len(subdirs),
            "type_counts": type_counts,
            "primary_type": primary_type,
            "formats": dict(extension_counts.most_common(10)),
            "sample_files": files[:5],  # First 5 files as sample
            "has_annotations": type_counts["annotations"] > 0,
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"
```

### Anonymize Tool (Stub)

```python
import random
from pathlib import Path

from agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    success_result,
    error_result,
)
from agent.tools.registry import register_tool


@register_tool
class AnonymizeTool(BaseTool):
    """Anonymize faces and license plates in images/videos."""

    name = "anonymize"
    description = """Detect and blur faces and license plates in images or videos.
This protects privacy by anonymizing identifiable information."""
    category = ToolCategory.ANONYMIZATION

    parameters = [
        ToolParameter(
            "input_path",
            str,
            "Path to input file or directory",
            required=True,
        ),
        ToolParameter(
            "output_path",
            str,
            "Path to save anonymized output",
            required=True,
        ),
        ToolParameter(
            "target",
            str,
            "What to anonymize: faces, plates, or all",
            required=False,
            default="all",
            enum_values=["faces", "plates", "all"],
        ),
        ToolParameter(
            "blur_strength",
            int,
            "Blur intensity (1-10)",
            required=False,
            default=5,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        output_path: str,
        target: str = "all",
        blur_strength: int = 5,
    ) -> ToolResult:
        """
        STUB: Returns mock anonymization results.

        TODO: Replace with actual SCRFD face detection and blurring.
        Integration point: backend/cv/anonymize.py
        """

        input_p = Path(input_path)
        output_p = Path(output_path)

        # Validate input exists
        if not input_p.exists():
            return error_result(f"Input not found: {input_path}")

        # Count files to process
        if input_p.is_file():
            file_count = 1
        else:
            file_count = sum(1 for _ in input_p.glob("**/*.jpg"))
            file_count += sum(1 for _ in input_p.glob("**/*.png"))

        if file_count == 0:
            return error_result("No image files found in input path")

        # Generate mock results
        # In reality, this would:
        # 1. Load SCRFD model
        # 2. Detect faces/plates in each image
        # 3. Apply Gaussian blur to detected regions
        # 4. Save output files

        faces_detected = random.randint(file_count * 2, file_count * 5)
        plates_detected = random.randint(0, file_count * 2)

        faces_blurred = faces_detected if target in ["faces", "all"] else 0
        plates_blurred = plates_detected if target in ["plates", "all"] else 0

        # Simulate processing time
        for i in range(file_count):
            self.report_progress(i + 1, file_count, f"Processing file {i + 1}/{file_count}")

        return success_result({
            "files_processed": file_count,
            "faces_detected": faces_detected,
            "faces_blurred": faces_blurred,
            "plates_detected": plates_detected,
            "plates_blurred": plates_blurred,
            "output_path": str(output_p),
            "confidence": 0.87 + random.uniform(-0.1, 0.1),
            "_stub": True,  # Marker that this is mock data
            "_integration_point": "backend/cv/anonymize.py",
        })
```

### Export Tool (Stub)

```python
from pathlib import Path

from agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    success_result,
    error_result,
)
from agent.tools.registry import register_tool


SUPPORTED_FORMATS = ["yolo", "coco", "pascal", "labelme", "cvat"]


@register_tool
class ExportTool(BaseTool):
    """Export annotations to different formats."""

    name = "export"
    description = """Convert annotations to a specific format like YOLO, COCO, or Pascal VOC.
Use after detection/segmentation to create training data."""
    category = ToolCategory.EXPORT

    parameters = [
        ToolParameter(
            "input_path",
            str,
            "Path to input annotations",
            required=True,
        ),
        ToolParameter(
            "output_path",
            str,
            "Path to save exported annotations",
            required=True,
        ),
        ToolParameter(
            "format",
            str,
            "Output format",
            required=True,
            enum_values=SUPPORTED_FORMATS,
        ),
        ToolParameter(
            "split_ratio",
            float,
            "Train/val split ratio (0-1, portion for training)",
            required=False,
            default=0.8,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        output_path: str,
        format: str,
        split_ratio: float = 0.8,
    ) -> ToolResult:
        """
        STUB: Returns mock export results.

        TODO: Replace with actual format conversion.
        Integration point: backend/data/exporters/
        """

        input_p = Path(input_path)
        output_p = Path(output_path)

        if not input_p.exists():
            return error_result(f"Input not found: {input_path}")

        if format not in SUPPORTED_FORMATS:
            return error_result(f"Unsupported format: {format}. Use one of {SUPPORTED_FORMATS}")

        # Mock file counts
        if input_p.is_file():
            annotation_count = 1
        else:
            annotation_count = sum(1 for _ in input_p.glob("**/*.json"))

        if annotation_count == 0:
            return error_result("No annotation files found")

        # Calculate split
        train_count = int(annotation_count * split_ratio)
        val_count = annotation_count - train_count

        # Mock export structure based on format
        export_structure = self._get_format_structure(format, output_p)

        return success_result({
            "annotations_processed": annotation_count,
            "train_count": train_count,
            "val_count": val_count,
            "output_path": str(output_p),
            "format": format,
            "structure": export_structure,
            "_stub": True,
            "_integration_point": f"backend/data/exporters/{format}.py",
        })

    def _get_format_structure(self, format: str, output_path: Path) -> dict:
        """Get expected output structure for format."""

        if format == "yolo":
            return {
                "images/train": "Training images",
                "images/val": "Validation images",
                "labels/train": "Training labels (.txt)",
                "labels/val": "Validation labels (.txt)",
                "data.yaml": "Dataset configuration",
            }
        elif format == "coco":
            return {
                "train/": "Training images",
                "val/": "Validation images",
                "annotations/instances_train.json": "Training annotations",
                "annotations/instances_val.json": "Validation annotations",
            }
        elif format == "pascal":
            return {
                "JPEGImages/": "All images",
                "Annotations/": "XML annotations",
                "ImageSets/Main/train.txt": "Training image list",
                "ImageSets/Main/val.txt": "Validation image list",
            }
        else:
            return {"output/": "Exported annotations"}
```

### Detect Tool (Stub)

```python
import random
from pathlib import Path

from agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    success_result,
    error_result,
)
from agent.tools.registry import register_tool


DEFAULT_CLASSES = ["person", "car", "truck", "bus", "bicycle", "motorcycle"]


@register_tool
class DetectTool(BaseTool):
    """Detect objects in images using YOLO."""

    name = "detect"
    description = """Run object detection on images to find and label objects.
Supports common classes like vehicles, people, and can use custom prompts."""
    category = ToolCategory.DETECTION

    parameters = [
        ToolParameter(
            "input_path",
            str,
            "Path to input image or directory",
            required=True,
        ),
        ToolParameter(
            "classes",
            list,
            "List of object classes to detect",
            required=False,
            default=DEFAULT_CLASSES,
        ),
        ToolParameter(
            "confidence",
            float,
            "Minimum confidence threshold (0-1)",
            required=False,
            default=0.5,
        ),
        ToolParameter(
            "save_annotations",
            bool,
            "Whether to save detection annotations",
            required=False,
            default=True,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        classes: list = None,
        confidence: float = 0.5,
        save_annotations: bool = True,
    ) -> ToolResult:
        """
        STUB: Returns mock detection results.

        TODO: Replace with YOLO11 inference.
        Integration point: backend/cv/detection.py
        """

        classes = classes or DEFAULT_CLASSES
        input_p = Path(input_path)

        if not input_p.exists():
            return error_result(f"Input not found: {input_path}")

        # Count files
        if input_p.is_file():
            file_count = 1
        else:
            file_count = sum(1 for _ in input_p.glob("**/*.jpg"))
            file_count += sum(1 for _ in input_p.glob("**/*.png"))

        if file_count == 0:
            return error_result("No image files found")

        # Generate mock detections
        class_counts = {}
        total_detections = 0

        for cls in classes[:5]:  # Limit to 5 classes for mock
            count = random.randint(file_count, file_count * 10)
            class_counts[cls] = count
            total_detections += count

        # Simulate processing
        for i in range(file_count):
            self.report_progress(i + 1, file_count, f"Detecting in file {i + 1}/{file_count}")

        return success_result({
            "files_processed": file_count,
            "count": total_detections,
            "classes": class_counts,
            "confidence_threshold": confidence,
            "confidence": 0.82 + random.uniform(-0.05, 0.1),
            "annotations_saved": save_annotations,
            "_stub": True,
            "_integration_point": "backend/cv/detection.py",
        })
```

## Implementation Tasks

- [ ] Create `backend/agent/tools/scan.py`
  - [ ] Implement `ScanDirectoryTool`
  - [ ] Handle file type detection
  - [ ] Handle recursive scanning
  - [ ] Add progress reporting
- [ ] Create `backend/agent/tools/anonymize.py`
  - [ ] Implement `AnonymizeTool` stub
  - [ ] Generate realistic mock data
  - [ ] Document integration points
- [ ] Create `backend/agent/tools/export.py`
  - [ ] Implement `ExportTool` stub
  - [ ] Define format structures
  - [ ] Document integration points
- [ ] Create `backend/agent/tools/detect.py`
  - [ ] Implement `DetectTool` stub
  - [ ] Generate realistic mock data
  - [ ] Document integration points
- [ ] Update `backend/agent/tools/__init__.py` to export tools

## Testing

### Unit Tests

```python
# tests/agent/tools/test_scan.py

@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset directory."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(10):
        (img_dir / f"image_{i}.jpg").touch()

    vid_dir = tmp_path / "videos"
    vid_dir.mkdir()
    for i in range(3):
        (vid_dir / f"video_{i}.mp4").touch()

    return tmp_path


@pytest.mark.asyncio
async def test_scan_directory_basic(temp_dataset):
    """Should scan and categorize files."""
    tool = ScanDirectoryTool()
    result = await tool.run(path=str(temp_dataset))

    assert result.success == True
    assert result.data["total_files"] == 13
    assert result.data["type_counts"]["images"] == 10
    assert result.data["type_counts"]["videos"] == 3


@pytest.mark.asyncio
async def test_scan_directory_not_found():
    """Should return error for missing directory."""
    tool = ScanDirectoryTool()
    result = await tool.run(path="/nonexistent/path")

    assert result.success == False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_scan_directory_detects_primary_type(tmp_path):
    """Should detect primary data type."""
    for i in range(5):
        (tmp_path / f"image_{i}.jpg").touch()

    tool = ScanDirectoryTool()
    result = await tool.run(path=str(tmp_path))

    assert result.data["primary_type"] == "images"
```

### Stub Tests

```python
# tests/agent/tools/test_stubs.py

@pytest.mark.asyncio
async def test_anonymize_stub_returns_mock_data(temp_dataset):
    """Anonymize stub should return realistic mock data."""
    tool = AnonymizeTool()
    result = await tool.run(
        input_path=str(temp_dataset / "images"),
        output_path=str(temp_dataset / "output"),
    )

    assert result.success == True
    assert result.data["_stub"] == True
    assert result.data["files_processed"] > 0
    assert "faces_blurred" in result.data
    assert "plates_blurred" in result.data


@pytest.mark.asyncio
async def test_export_stub_yolo_format(temp_dataset):
    """Export stub should show YOLO structure."""
    # Create mock annotation
    (temp_dataset / "anno.json").touch()

    tool = ExportTool()
    result = await tool.run(
        input_path=str(temp_dataset),
        output_path=str(temp_dataset / "export"),
        format="yolo",
    )

    assert result.success == True
    assert "data.yaml" in result.data["structure"]


@pytest.mark.asyncio
async def test_detect_stub_returns_class_counts(temp_dataset):
    """Detect stub should return per-class counts."""
    tool = DetectTool()
    result = await tool.run(
        input_path=str(temp_dataset / "images"),
        classes=["car", "person"],
    )

    assert result.success == True
    assert "car" in result.data["classes"]
    assert "person" in result.data["classes"]
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_pipeline_with_stubs(temp_dataset):
    """Test scan -> detect -> export flow with stubs."""
    # Scan
    scan_tool = ScanDirectoryTool()
    scan_result = await scan_tool.run(path=str(temp_dataset))
    assert scan_result.success

    # Detect (stub)
    detect_tool = DetectTool()
    detect_result = await detect_tool.run(
        input_path=str(temp_dataset / "images"),
    )
    assert detect_result.success
    assert detect_result.data["_stub"]

    # Export (stub)
    export_tool = ExportTool()
    export_result = await export_tool.run(
        input_path=str(temp_dataset),
        output_path=str(temp_dataset / "export"),
        format="yolo",
    )
    assert export_result.success
```

## Acceptance Criteria

- [ ] `scan_directory` correctly counts files by type
- [ ] `scan_directory` handles missing directories gracefully
- [ ] `scan_directory` respects recursive and max_depth options
- [ ] Stub tools return realistic mock data
- [ ] Stub tools mark data with `_stub: True`
- [ ] Stub tools document integration points
- [ ] All tools report progress via callback
- [ ] Tools are auto-registered via decorator

## Files to Create/Modify

```
backend/
├── agent/
│   └── tools/
│       ├── __init__.py      # Export all tools
│       ├── scan.py          # scan_directory (real)
│       ├── anonymize.py     # anonymize (stub)
│       ├── detect.py        # detect (stub)
│       └── export.py        # export (stub)
└── tests/
    └── agent/
        └── tools/
            ├── test_scan.py
            └── test_stubs.py
```

## Notes

- Stubs should generate consistent mock data for testing
- Random values in stubs should be deterministic (seeded) for tests
- Integration points are clearly documented for 03-cv-models implementation
- Consider adding a `segment` stub tool as well
