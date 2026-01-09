# CV Agent Tools

> **Status:** Not Started
> **Priority:** P0 (Critical)
> **Dependencies:** [01-yolo11-detection.md](./01-yolo11-detection.md), [02-sam3-segmentation.md](./02-sam3-segmentation.md), [03-scrfd-faces.md](./03-scrfd-faces.md), [04-yolo-world-openvocab.md](./04-yolo-world-openvocab.md), [05-plate-detection.md](./05-plate-detection.md), [06-anonymization.md](./06-anonymization.md), [07-3d-detection.md](./07-3d-detection.md)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

LangGraph agent tools that expose all CV functionality to the conversational AI. These tools bridge the gap between natural language instructions and CV model execution. Each tool returns structured Pydantic models and supports progress streaming via SSE for long-running operations.

## Goals

- [ ] Create `detect_objects` tool (YOLO11 or YOLO-World based on prompt)
- [ ] Create `segment_sam3` tool (text-prompted segmentation)
- [ ] Create `anonymize` tool (full anonymization pipeline)
- [ ] Create `detect_faces` tool (face detection wrapper)
- [ ] Create `detect_3d` tool (3D object detection on point clouds)
- [ ] All tools return structured Pydantic models
- [ ] All tools stream progress via SSE

## Technical Design

### Tool Registration Pattern

```python
from langchain_core.tools import tool
from typing import Optional, List

# All tools follow this pattern:
# 1. Decorated with @tool
# 2. Comprehensive docstring (becomes LLM description)
# 3. Typed parameters with defaults
# 4. Return structured Pydantic model
# 5. Handle errors gracefully
```

### detect_objects Tool

```python
# backend/agent/tools/detect.py

from langchain_core.tools import tool
from typing import Optional, List
from pydantic import BaseModel, Field

class ObjectDetection(BaseModel):
    """Single object detection result."""
    class_name: str = Field(description="Detected class name")
    confidence: float = Field(description="Detection confidence (0-1)")
    bbox: dict = Field(description="Bounding box {x, y, width, height} normalized")

class DetectObjectsResult(BaseModel):
    """Result of object detection."""
    detections: List[ObjectDetection]
    image_path: str
    model_used: str
    processing_time_ms: float

@tool
def detect_objects(
    image_path: str,
    prompt: str = "all objects",
    confidence: float = 0.5,
) -> DetectObjectsResult:
    """
    Detect objects in an image.

    Use this tool when you need to find and locate objects in an image.
    The prompt can be:
    - A comma-separated list of class names: "car, person, dog"
    - A description for open-vocabulary detection: "red car, person wearing helmet"
    - "all objects" to use standard COCO classes

    Args:
        image_path: Path to the input image file
        prompt: What objects to detect (comma-separated classes or descriptions)
        confidence: Minimum confidence threshold (0.0 to 1.0)

    Returns:
        DetectObjectsResult with list of detections, each containing:
        - class_name: The detected object class
        - confidence: How confident the model is (0-1)
        - bbox: Bounding box location {x, y, width, height}

    Example:
        >>> detect_objects("/data/image.jpg", "car, person")
        >>> detect_objects("/data/image.jpg", "red car", confidence=0.7)
    """
    from backend.cv.detection import YOLO11Wrapper, COCO_CLASSES
    from backend.cv.openvocab import YOLOWorldWrapper
    from backend.cv.manager import ModelManager

    # Determine which detector to use
    classes = [c.strip() for c in prompt.split(",")]
    use_openvocab = any(
        c not in COCO_CLASSES and c != "all objects"
        for c in classes
    )

    manager = ModelManager()

    if use_openvocab:
        detector = manager.get("yolo-world")
        result = detector.predict(image_path, prompt=prompt, confidence=confidence)
    else:
        detector = manager.get("yolo11")
        # Convert "all objects" to None (all classes)
        class_filter = None if "all objects" in prompt else classes
        result = detector.predict(image_path, classes=class_filter, confidence=confidence)

    return DetectObjectsResult(
        detections=[
            ObjectDetection(
                class_name=d.class_name,
                confidence=d.confidence,
                bbox={"x": d.bbox.x, "y": d.bbox.y, "width": d.bbox.width, "height": d.bbox.height}
            )
            for d in result.detections
        ],
        image_path=result.image_path,
        model_used=result.model_name,
        processing_time_ms=result.processing_time_ms,
    )
```

### segment_sam3 Tool

```python
# backend/agent/tools/segment.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, List
import base64

class SegmentationMask(BaseModel):
    """Single segmentation mask."""
    confidence: float = Field(description="Mask confidence score")
    mask_base64: str = Field(description="Base64-encoded PNG mask")
    bbox: dict = Field(description="Bounding box of mask region")

class SegmentResult(BaseModel):
    """Result of segmentation."""
    masks: List[SegmentationMask]
    image_path: str
    prompt_used: str
    model_used: str
    processing_time_ms: float

@tool
def segment_sam3(
    image_path: str,
    prompt: str,
    multimask: bool = True,
) -> SegmentResult:
    """
    Segment objects in an image using text prompts.

    Use this tool to create precise pixel-level masks for objects matching
    a text description. SAM3 can understand natural language prompts like
    "the red car on the left" or "person wearing blue shirt".

    Args:
        image_path: Path to the input image file
        prompt: Text description of what to segment (e.g., "red car", "person on left")
        multimask: If True, return multiple mask candidates ranked by confidence

    Returns:
        SegmentResult containing:
        - masks: List of masks, each with confidence and base64-encoded PNG
        - prompt_used: The text prompt that was used
        - model_used: Which SAM model was used

    Example:
        >>> segment_sam3("/data/image.jpg", "red car")
        >>> segment_sam3("/data/image.jpg", "the person on the left", multimask=False)
    """
    from backend.cv.segmentation import get_segmenter
    from backend.cv.manager import ModelManager
    import cv2
    import numpy as np

    manager = ModelManager()
    segmenter = manager.get("sam3")

    result = segmenter.predict(
        image_path,
        prompt=prompt,
        multimask_output=multimask,
    )

    # Convert masks to base64 PNGs
    masks_out = []
    for mask in result.masks:
        mask_np = mask.to_numpy()

        # Encode as PNG
        _, buffer = cv2.imencode(".png", mask_np * 255)
        mask_b64 = base64.b64encode(buffer).decode("utf-8")

        # Calculate bounding box
        coords = np.where(mask_np > 0)
        if len(coords[0]) > 0:
            y1, y2 = coords[0].min(), coords[0].max()
            x1, x2 = coords[1].min(), coords[1].max()
            h, w = mask_np.shape
            bbox = {
                "x": (x1 + x2) / 2 / w,
                "y": (y1 + y2) / 2 / h,
                "width": (x2 - x1) / w,
                "height": (y2 - y1) / h,
            }
        else:
            bbox = {"x": 0, "y": 0, "width": 0, "height": 0}

        masks_out.append(SegmentationMask(
            confidence=mask.confidence,
            mask_base64=mask_b64,
            bbox=bbox,
        ))

    return SegmentResult(
        masks=masks_out,
        image_path=image_path,
        prompt_used=prompt,
        model_used=result.model_name,
        processing_time_ms=result.processing_time_ms,
    )
```

### anonymize Tool

```python
# backend/agent/tools/anonymize.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

class AnonymizeResult(BaseModel):
    """Result of anonymization."""
    output_path: str = Field(description="Path to anonymized image")
    faces_anonymized: int = Field(description="Number of faces anonymized")
    plates_anonymized: int = Field(description="Number of license plates anonymized")
    processing_time_ms: float

@tool
def anonymize(
    image_path: str,
    faces: bool = True,
    plates: bool = True,
    mode: Literal["blur", "blackbox", "pixelate", "mask"] = "blur",
    output_path: str = None,
) -> AnonymizeResult:
    """
    Anonymize faces and/or license plates in an image.

    Use this tool to protect privacy by obscuring identifying information
    in images. Supports multiple anonymization styles.

    Args:
        image_path: Path to the input image file
        faces: If True, detect and anonymize faces
        plates: If True, detect and anonymize license plates
        mode: Anonymization style:
            - "blur": Gaussian blur (default, natural look)
            - "blackbox": Solid black fill
            - "pixelate": Mosaic/pixelation effect
            - "mask": Precise mask using SAM3 (slower but better edges)
        output_path: Where to save the result (auto-generated if not provided)

    Returns:
        AnonymizeResult containing:
        - output_path: Path to the anonymized image
        - faces_anonymized: Count of faces processed
        - plates_anonymized: Count of plates processed

    Example:
        >>> anonymize("/data/street.jpg", faces=True, plates=True, mode="blur")
        >>> anonymize("/data/group.jpg", faces=True, plates=False, mode="pixelate")
    """
    from backend.cv.anonymization import AnonymizationPipeline, AnonymizationConfig

    config = AnonymizationConfig(
        faces=faces,
        plates=plates,
        mode=mode,
    )

    pipeline = AnonymizationPipeline(config)
    pipeline.load("auto")

    try:
        result = pipeline.process(image_path, output_path)
        return AnonymizeResult(
            output_path=result.output_path,
            faces_anonymized=result.faces_anonymized,
            plates_anonymized=result.plates_anonymized,
            processing_time_ms=result.processing_time_ms,
        )
    finally:
        pipeline.unload()
```

### detect_faces Tool

```python
# backend/agent/tools/faces.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional

class FaceResult(BaseModel):
    """Single face detection result."""
    confidence: float
    bbox: dict
    landmarks: Optional[List[dict]] = Field(
        default=None,
        description="5-point facial landmarks if requested"
    )

class DetectFacesResult(BaseModel):
    """Result of face detection."""
    faces: List[FaceResult]
    face_count: int
    image_path: str
    processing_time_ms: float

@tool
def detect_faces(
    image_path: str,
    include_landmarks: bool = False,
    confidence: float = 0.5,
) -> DetectFacesResult:
    """
    Detect all faces in an image.

    Use this tool when you need to find faces for analysis, cropping,
    or further processing. Can optionally include facial landmarks
    (eyes, nose, mouth positions).

    Args:
        image_path: Path to the input image file
        include_landmarks: If True, include 5-point facial landmarks
        confidence: Minimum confidence threshold (0.0 to 1.0)

    Returns:
        DetectFacesResult containing:
        - faces: List of detected faces with bbox and optional landmarks
        - face_count: Total number of faces found

    Example:
        >>> detect_faces("/data/group.jpg")
        >>> detect_faces("/data/portrait.jpg", include_landmarks=True)
    """
    from backend.cv.faces import get_face_detector
    from backend.cv.manager import ModelManager

    manager = ModelManager()
    detector = manager.get("scrfd")

    faces = detector.predict(
        image_path,
        confidence=confidence,
        include_landmarks=include_landmarks,
    )

    return DetectFacesResult(
        faces=[
            FaceResult(
                confidence=f.confidence,
                bbox={"x": f.bbox.x, "y": f.bbox.y, "width": f.bbox.width, "height": f.bbox.height},
                landmarks=[
                    {"x": lm[0], "y": lm[1]}
                    for lm in f.landmarks
                ] if f.landmarks else None,
            )
            for f in faces
        ],
        face_count=len(faces),
        image_path=image_path,
        processing_time_ms=0,  # TODO: Add timing
    )
```

### detect_3d Tool

```python
# backend/agent/tools/detect_3d.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional

class Detection3DResult(BaseModel):
    """Single 3D detection result."""
    class_name: str
    confidence: float
    center: dict = Field(description="Center position {x, y, z} in meters")
    dimensions: dict = Field(description="Size {length, width, height} in meters")
    rotation: float = Field(description="Yaw rotation in radians")

class Detect3DResult(BaseModel):
    """Result of 3D object detection."""
    detections: List[Detection3DResult]
    object_count: int
    pointcloud_path: str
    model_used: str
    processing_time_ms: float

@tool
def detect_3d(
    pointcloud_path: str,
    classes: List[str] = None,
    confidence: float = 0.3,
) -> Detect3DResult:
    """
    Detect 3D objects in a point cloud.

    Use this tool to detect vehicles, pedestrians, cyclists, and other
    objects in LiDAR or other 3D point cloud data. Returns 3D bounding
    boxes with position, size, and orientation.

    Args:
        pointcloud_path: Path to point cloud file (PCD, PLY, LAS, or KITTI BIN)
        classes: Which object classes to detect. Options:
            - ["Car"] - vehicles only
            - ["Pedestrian"] - people only
            - ["Cyclist"] - cyclists only
            - None or ["Car", "Pedestrian", "Cyclist"] - all classes
        confidence: Minimum confidence threshold (0.0 to 1.0)

    Returns:
        Detect3DResult containing:
        - detections: List of 3D detections with position, size, rotation
        - object_count: Total objects detected

    Example:
        >>> detect_3d("/data/lidar/scan_001.pcd")
        >>> detect_3d("/data/kitti/000001.bin", classes=["Car"])
    """
    from backend.cv.detection_3d import get_3d_detector
    from backend.cv.manager import ModelManager
    import time

    manager = ModelManager()
    detector = manager.get("pvrcnn")

    start = time.perf_counter()
    detections = detector.predict(
        pointcloud_path,
        classes=classes,
        confidence=confidence,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    return Detect3DResult(
        detections=[
            Detection3DResult(
                class_name=d.class_name,
                confidence=d.confidence,
                center={"x": d.center[0], "y": d.center[1], "z": d.center[2]},
                dimensions={"length": d.dimensions[0], "width": d.dimensions[1], "height": d.dimensions[2]},
                rotation=d.rotation,
            )
            for d in detections
        ],
        object_count=len(detections),
        pointcloud_path=pointcloud_path,
        model_used="pvrcnn++",
        processing_time_ms=elapsed_ms,
    )
```

### Tool Registry

```python
# backend/agent/tools/__init__.py

from backend.agent.tools.detect import detect_objects
from backend.agent.tools.segment import segment_sam3
from backend.agent.tools.anonymize import anonymize
from backend.agent.tools.faces import detect_faces
from backend.agent.tools.detect_3d import detect_3d

# All CV tools for agent registration
CV_TOOLS = [
    detect_objects,
    segment_sam3,
    anonymize,
    detect_faces,
    detect_3d,
]

__all__ = [
    "detect_objects",
    "segment_sam3",
    "anonymize",
    "detect_faces",
    "detect_3d",
    "CV_TOOLS",
]
```

## Implementation Tasks

- [ ] **Tool Creation**
  - [ ] Create `backend/agent/tools/detect.py`
  - [ ] Create `backend/agent/tools/segment.py`
  - [ ] Create `backend/agent/tools/anonymize.py`
  - [ ] Create `backend/agent/tools/faces.py`
  - [ ] Create `backend/agent/tools/detect_3d.py`
  - [ ] Create `backend/agent/tools/__init__.py` with registry

- [ ] **Model Integration**
  - [ ] Use ModelManager for all model access
  - [ ] Handle model loading/unloading efficiently
  - [ ] Share models between related tools

- [ ] **Progress Streaming**
  - [ ] Add progress callbacks to batch operations
  - [ ] SSE integration for frontend updates
  - [ ] Checkpoint integration for long operations

- [ ] **Error Handling**
  - [ ] Graceful error messages for invalid inputs
  - [ ] Model unavailable fallbacks
  - [ ] VRAM exhaustion handling

- [ ] **Testing**
  - [ ] Unit tests with mock models
  - [ ] Integration tests with real models
  - [ ] LangGraph agent integration tests

## Acceptance Criteria

- [ ] `detect_objects("car, person", image_path)` returns bounding boxes
- [ ] `segment_sam3("red car", image_path)` returns masks
- [ ] `anonymize(image_path, faces=True, plates=True)` produces anonymized image
- [ ] `detect_faces(image_path)` returns face bounding boxes and landmarks
- [ ] `detect_3d(pointcloud_path)` returns 3D bounding boxes
- [ ] All tools work within LangGraph execution flow
- [ ] Tool docstrings provide clear guidance to LLM
- [ ] Progress streams to frontend during long operations
- [ ] **Performance:** Tool overhead <10ms (excluding model inference)

## Files to Create

```
backend/agent/tools/
├── __init__.py       # CV_TOOLS registry
├── detect.py         # detect_objects tool
├── segment.py        # segment_sam3 tool
├── anonymize.py      # anonymize tool
├── faces.py          # detect_faces tool
└── detect_3d.py      # detect_3d tool
```

## Testing

```python
# test_tools.py
import pytest
from backend.agent.tools import (
    detect_objects, segment_sam3, anonymize, detect_faces, detect_3d, CV_TOOLS
)

def test_all_tools_registered():
    assert len(CV_TOOLS) == 5
    assert detect_objects in CV_TOOLS
    assert segment_sam3 in CV_TOOLS

def test_detect_objects_returns_structured(sample_image):
    result = detect_objects.invoke({
        "image_path": sample_image,
        "prompt": "car, person",
    })
    assert hasattr(result, "detections")
    assert hasattr(result, "model_used")

def test_detect_objects_uses_yolo11_for_coco(sample_image):
    result = detect_objects.invoke({
        "image_path": sample_image,
        "prompt": "car",  # COCO class
    })
    assert result.model_used == "yolo11m"

def test_detect_objects_uses_yoloworld_for_custom(sample_image):
    result = detect_objects.invoke({
        "image_path": sample_image,
        "prompt": "red car",  # Custom class
    })
    assert result.model_used == "yolo-world-l"

def test_segment_returns_base64_mask(sample_image):
    result = segment_sam3.invoke({
        "image_path": sample_image,
        "prompt": "main object",
    })
    assert len(result.masks) > 0
    # Should be valid base64
    import base64
    base64.b64decode(result.masks[0].mask_base64)

def test_anonymize_creates_output(sample_image, tmp_path):
    output = str(tmp_path / "anon.jpg")
    result = anonymize.invoke({
        "image_path": sample_image,
        "output_path": output,
    })
    assert Path(result.output_path).exists()

def test_tools_have_docstrings():
    for tool in CV_TOOLS:
        assert tool.description
        assert len(tool.description) > 50

@pytest.mark.gpu
def test_tool_overhead(sample_image):
    """Tool wrapper overhead should be minimal."""
    import time

    # Warm up
    detect_objects.invoke({"image_path": sample_image, "prompt": "car"})

    # Measure overhead
    times = []
    for _ in range(10):
        start = time.perf_counter()
        detect_objects.invoke({"image_path": sample_image, "prompt": "car"})
        times.append((time.perf_counter() - start) * 1000)

    # Check that tool adds minimal overhead
    # (Most time should be model inference, not tool wrapper)
    pass  # Actual benchmark depends on model speed
```
