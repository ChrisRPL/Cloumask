# YOLO11 Object Detection

> **Status:** Completed
> **Priority:** P0 (Critical)
> **Dependencies:** [00-infrastructure.md](./00-infrastructure.md)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Object detection using YOLO11m as the primary model with RT-DETR as a fallback for higher accuracy requirements. YOLO11 provides excellent speed-accuracy trade-off for real-time detection, while RT-DETR offers transformer-based detection with better accuracy for precision-critical tasks.

## Goals

- [ ] Integrate YOLO11m via ultralytics library
- [ ] Implement RT-DETR fallback (transformer-based, higher accuracy)
- [ ] Add configurable confidence threshold and NMS settings
- [ ] Implement class filtering (include/exclude specific COCO classes)
- [ ] Support batch inference with progress callbacks

## Technical Design

### Model Specifications

| Model | Size | VRAM | Inference | mAP (COCO) |
|-------|------|------|-----------|------------|
| YOLO11m | ~40MB | ~2GB | 2.4ms | 51.5% |
| RT-DETR-l | ~150MB | ~3GB | 5-10ms | 53.0% |

### YOLO11 Wrapper

```python
from ultralytics import YOLO
from backend.cv.base import BaseModelWrapper
from backend.cv.types import Detection, DetectionResult, BBox
import torch

class YOLO11Wrapper(BaseModelWrapper):
    """YOLO11m object detection wrapper."""

    model_name = "yolo11m"
    vram_required_mb = 2500
    supports_batching = True

    def __init__(self):
        self._model: Optional[YOLO] = None
        self._device: str = "cpu"

    def load(self, device: str = "cuda") -> None:
        """Load YOLO11m model."""
        from backend.cv.download import get_model_path, download_model

        model_path = get_model_path("yolo11m")
        if not model_path.exists():
            download_model("yolo11m")

        self._model = YOLO(model_path)
        self._device = device

        # Warm up model
        if device == "cuda":
            self._model.to(device)
            self._model.predict(torch.zeros(1, 3, 640, 640).to(device), verbose=False)

    def unload(self) -> None:
        """Unload model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        image_path: str,
        confidence: float = 0.5,
        classes: Optional[list[str]] = None,
        iou_threshold: float = 0.45,
    ) -> DetectionResult:
        """
        Run object detection on an image.

        Args:
            image_path: Path to input image
            confidence: Minimum confidence threshold (0-1)
            classes: List of class names to detect (None = all COCO classes)
            iou_threshold: IoU threshold for NMS

        Returns:
            DetectionResult with list of detections
        """
        import time

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert class names to indices if provided
        class_indices = None
        if classes:
            class_indices = [COCO_CLASSES.index(c) for c in classes if c in COCO_CLASSES]

        start = time.perf_counter()
        results = self._model.predict(
            image_path,
            conf=confidence,
            iou=iou_threshold,
            classes=class_indices,
            device=self._device,
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Convert to our types
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append(Detection(
                class_id=cls_id,
                class_name=COCO_CLASSES[cls_id],
                bbox=BBox(
                    x=float(box.xywhn[0][0]),
                    y=float(box.xywhn[0][1]),
                    width=float(box.xywhn[0][2]),
                    height=float(box.xywhn[0][3]),
                ),
                confidence=float(box.conf[0]),
            ))

        return DetectionResult(
            detections=detections,
            image_path=image_path,
            processing_time_ms=elapsed_ms,
            model_name=self.model_name,
        )

    def predict_batch(
        self,
        image_paths: list[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> list[DetectionResult]:
        """Batch inference with progress tracking."""
        results = []

        # Process in mini-batches for memory efficiency
        batch_size = kwargs.pop("batch_size", 8)

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            batch_results = self._model.predict(
                batch,
                conf=kwargs.get("confidence", 0.5),
                iou=kwargs.get("iou_threshold", 0.45),
                classes=kwargs.get("class_indices"),
                device=self._device,
                verbose=False,
            )

            for j, res in enumerate(batch_results):
                # Convert each result...
                results.append(self._convert_result(res, batch[j]))

            if progress_callback:
                progress_callback(min(i + batch_size, len(image_paths)), len(image_paths))

        return results


# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
```

### RT-DETR Fallback Wrapper

```python
class RTDETRWrapper(BaseModelWrapper):
    """RT-DETR transformer-based detection wrapper."""

    model_name = "rtdetr-l"
    vram_required_mb = 3500
    supports_batching = True

    def __init__(self):
        self._model: Optional[YOLO] = None  # Uses same ultralytics interface
        self._device: str = "cpu"

    def load(self, device: str = "cuda") -> None:
        """Load RT-DETR model."""
        from ultralytics import RTDETR

        self._model = RTDETR("rtdetr-l.pt")
        self._device = device

        if device == "cuda":
            self._model.to(device)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        image_path: str,
        confidence: float = 0.5,
        classes: Optional[list[str]] = None,
        **kwargs
    ) -> DetectionResult:
        """RT-DETR inference (same interface as YOLO11)."""
        # Similar implementation to YOLO11Wrapper
        ...
```

### Detector Factory

```python
from backend.cv.device import get_available_vram_mb

def get_detector(
    prefer_accuracy: bool = False,
    force_model: Optional[str] = None
) -> BaseModelWrapper:
    """
    Get appropriate detector based on requirements and available resources.

    Args:
        prefer_accuracy: If True, prefer RT-DETR over YOLO11
        force_model: Force specific model ("yolo11" or "rtdetr")

    Returns:
        Appropriate detector wrapper
    """
    if force_model == "yolo11":
        return YOLO11Wrapper()
    elif force_model == "rtdetr":
        return RTDETRWrapper()

    if prefer_accuracy:
        available = get_available_vram_mb()
        if available >= 3500:  # RT-DETR requires ~3.5GB
            return RTDETRWrapper()

    return YOLO11Wrapper()
```

## Implementation Tasks

- [ ] **YOLO11 Integration**
  - [ ] Create `backend/cv/detection.py`
  - [ ] Implement YOLO11Wrapper with ultralytics
  - [ ] Add model warm-up on load
  - [ ] Implement class filtering by name
  - [ ] Add NMS configuration

- [ ] **RT-DETR Fallback**
  - [ ] Implement RTDETRWrapper
  - [ ] Verify ultralytics RT-DETR interface
  - [ ] Add automatic fallback logic

- [ ] **Batch Processing**
  - [ ] Implement efficient batch inference
  - [ ] Add progress callback support
  - [ ] Optimize memory for large batches

- [ ] **Model Registration**
  - [ ] Register both models with ModelManager
  - [ ] Add to download registry
  - [ ] Update models/README.md

- [ ] **Testing**
  - [ ] Unit tests with mock model
  - [ ] Integration tests with real model (GPU)
  - [ ] Benchmark inference speed
  - [ ] Test class filtering

## Acceptance Criteria

- [ ] `detect(image_path, classes=["car", "person"])` returns list of DetectionResult
- [ ] Confidence threshold correctly filters low-confidence detections
- [ ] Class filtering only returns requested classes
- [ ] Batch of 100 images processes with progress updates
- [ ] Fallback to RT-DETR triggers when prefer_accuracy=True and VRAM available
- [ ] **VRAM Budget:** YOLO11m <2.5GB, RT-DETR <3.5GB peak
- [ ] **Performance:** YOLO11m <5ms/image, RT-DETR <15ms/image on GPU

## Files to Create

```
backend/cv/
└── detection.py   # YOLO11Wrapper, RTDETRWrapper, get_detector()
```

## Testing

```python
# test_detection.py
import pytest
from backend.cv.detection import YOLO11Wrapper, get_detector, COCO_CLASSES

@pytest.fixture
def detector():
    d = YOLO11Wrapper()
    d.load("cuda" if torch.cuda.is_available() else "cpu")
    yield d
    d.unload()

def test_detect_returns_detections(detector, sample_image):
    result = detector.predict(sample_image)
    assert isinstance(result, DetectionResult)
    assert len(result.detections) >= 0

def test_class_filtering(detector, sample_image_with_cars):
    result = detector.predict(sample_image_with_cars, classes=["car"])
    for det in result.detections:
        assert det.class_name == "car"

def test_confidence_threshold(detector, sample_image):
    high_conf = detector.predict(sample_image, confidence=0.9)
    low_conf = detector.predict(sample_image, confidence=0.1)
    assert len(low_conf.detections) >= len(high_conf.detections)

@pytest.mark.gpu
def test_vram_budget(detector):
    from backend.cv.device import get_vram_usage
    initial = get_vram_usage()[0]
    detector.load("cuda")
    loaded = get_vram_usage()[0]
    assert loaded - initial < 2500  # <2.5GB

@pytest.mark.gpu
def test_inference_speed(detector, sample_image):
    import time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        detector.predict(sample_image)
        times.append((time.perf_counter() - start) * 1000)

    avg_ms = sum(times) / len(times)
    assert avg_ms < 5  # <5ms average
```
