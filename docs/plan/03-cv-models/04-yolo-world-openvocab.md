# YOLO-World Open-Vocabulary Detection

> **Status:** Not Started
> **Priority:** P1 (High)
> **Dependencies:** [00-infrastructure.md](./00-infrastructure.md)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Open-vocabulary object detection using YOLO-World as the primary model with GroundingDINO as a fallback. Unlike traditional detectors limited to fixed classes (COCO), open-vocabulary detection supports arbitrary text prompts like "red car", "person wearing helmet", or "license plate". Essential for flexible auto-labeling workflows.

## Goals

- [ ] Integrate YOLO-World from ultralytics
- [ ] Support arbitrary text prompts ("red car", "person wearing helmet")
- [ ] Implement GroundingDINO fallback (more accurate, slower)
- [ ] Cache text embeddings for repeated prompts (LRU cache)
- [ ] Map detected classes back to user-provided names

## Technical Design

### Model Specifications

| Model | Size | VRAM | Inference | Features |
|-------|------|------|-----------|----------|
| YOLO-World-l | ~100MB | ~4GB | 15-20ms | 50+ FPS, ultralytics API |
| GroundingDINO | ~700MB | ~4GB | 100-150ms | Higher accuracy, better grounding |

### YOLO-World Wrapper

```python
from typing import Optional, List, Callable
from functools import lru_cache
from backend.cv.base import BaseModelWrapper
from backend.cv.types import Detection, DetectionResult, BBox
import torch

class YOLOWorldWrapper(BaseModelWrapper):
    """YOLO-World open-vocabulary detection wrapper."""

    model_name = "yolo-world-l"
    vram_required_mb = 4500
    supports_batching = True

    def __init__(self):
        self._model = None
        self._device: str = "cpu"
        self._current_classes: List[str] = []

    def load(self, device: str = "cuda") -> None:
        """Load YOLO-World model."""
        from ultralytics import YOLOWorld

        self._model = YOLOWorld("yolov8l-world.pt")
        self._device = device

        if device == "cuda":
            self._model.to(device)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._current_classes = []
            if self._device == "cuda":
                torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def set_classes(self, classes: List[str]) -> None:
        """
        Set custom classes for detection.
        Caches embeddings for efficiency with repeated prompts.
        """
        if classes != self._current_classes:
            self._model.set_classes(classes)
            self._current_classes = classes

    def predict(
        self,
        image_path: str,
        prompt: str,
        confidence: float = 0.3,
        iou_threshold: float = 0.45,
    ) -> DetectionResult:
        """
        Detect objects matching text prompt.

        Args:
            image_path: Path to input image
            prompt: Comma-separated class descriptions (e.g., "red car, person, dog")
            confidence: Minimum confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            DetectionResult with detections
        """
        import time

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Parse prompt into classes
        classes = [c.strip() for c in prompt.split(",")]
        self.set_classes(classes)

        start = time.perf_counter()
        results = self._model.predict(
            image_path,
            conf=confidence,
            iou=iou_threshold,
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
                class_name=classes[cls_id],  # Map back to user's class name
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
        image_paths: List[str],
        prompt: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[DetectionResult]:
        """Batch detection with same prompt."""
        # Set classes once for entire batch
        classes = [c.strip() for c in prompt.split(",")]
        self.set_classes(classes)

        results = []
        batch_size = kwargs.pop("batch_size", 8)

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            batch_results = self._model.predict(
                batch,
                conf=kwargs.get("confidence", 0.3),
                iou=kwargs.get("iou_threshold", 0.45),
                device=self._device,
                verbose=False,
            )

            for j, res in enumerate(batch_results):
                results.append(self._convert_result(res, batch[j], classes))

            if progress_callback:
                progress_callback(min(i + batch_size, len(image_paths)), len(image_paths))

        return results
```

### GroundingDINO Fallback Wrapper

```python
class GroundingDINOWrapper(BaseModelWrapper):
    """GroundingDINO open-vocabulary detection wrapper."""

    model_name = "groundingdino"
    vram_required_mb = 5000
    supports_batching = False  # GroundingDINO processes one at a time

    def __init__(self):
        self._model = None
        self._processor = None
        self._device: str = "cpu"

    def load(self, device: str = "cuda") -> None:
        """Load GroundingDINO model."""
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        model_id = "IDEA-Research/grounding-dino-base"

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

        self._device = device
        if device == "cuda":
            self._model.to(device)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            if self._device == "cuda":
                torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        image_path: str,
        prompt: str,
        confidence: float = 0.3,
        **kwargs
    ) -> DetectionResult:
        """
        Detect objects matching text prompt using GroundingDINO.

        Args:
            image_path: Path to input image
            prompt: Text description (can be natural language)
            confidence: Minimum confidence threshold

        Returns:
            DetectionResult with detections
        """
        import time
        from PIL import Image

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # GroundingDINO expects period-separated classes
        text = prompt.replace(",", " . ") + " ."

        start = time.perf_counter()

        inputs = self._processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=confidence,
            text_threshold=confidence,
            target_sizes=[(h, w)],
        )[0]

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Convert to our types
        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            detections.append(Detection(
                class_id=0,  # GroundingDINO doesn't use class IDs
                class_name=label,
                bbox=BBox(
                    x=(x1 + x2) / 2 / w,
                    y=(y1 + y2) / 2 / h,
                    width=(x2 - x1) / w,
                    height=(y2 - y1) / h,
                ),
                confidence=float(score),
            ))

        return DetectionResult(
            detections=detections,
            image_path=image_path,
            processing_time_ms=elapsed_ms,
            model_name=self.model_name,
        )
```

### Open-Vocab Detector Factory

```python
@lru_cache(maxsize=32)
def _get_cached_embeddings(prompt: str) -> str:
    """Cache prompt embeddings for efficiency."""
    return prompt

def get_openvocab_detector(
    prefer_accuracy: bool = False,
    force_model: Optional[str] = None
) -> BaseModelWrapper:
    """
    Get open-vocabulary detector.

    Args:
        prefer_accuracy: If True, prefer GroundingDINO over YOLO-World
        force_model: Force specific model

    Returns:
        Open-vocab detector wrapper
    """
    from backend.cv.device import get_available_vram_mb

    if force_model == "yoloworld":
        return YOLOWorldWrapper()
    elif force_model == "groundingdino":
        return GroundingDINOWrapper()

    if prefer_accuracy:
        available = get_available_vram_mb()
        if available >= 5000:
            return GroundingDINOWrapper()

    return YOLOWorldWrapper()
```

## Implementation Tasks

- [ ] **YOLO-World Integration**
  - [ ] Create `backend/cv/openvocab.py`
  - [ ] Implement YOLOWorldWrapper with ultralytics
  - [ ] Add class embedding caching
  - [ ] Support comma-separated prompts

- [ ] **GroundingDINO Fallback**
  - [ ] Implement GroundingDINOWrapper with transformers
  - [ ] Handle natural language prompts
  - [ ] Post-processing for detection boxes

- [ ] **Embedding Cache**
  - [ ] LRU cache for repeated prompts
  - [ ] Cache invalidation on class change

- [ ] **Testing**
  - [ ] Unit tests with mock
  - [ ] Open-vocab accuracy tests
  - [ ] Performance benchmarks
  - [ ] Cache effectiveness tests

## Acceptance Criteria

- [ ] `detect("red car, person", image)` returns detections with class names
- [ ] Custom text prompts work (not limited to COCO classes)
- [ ] Embedding cache improves repeated prompt performance (>2x speedup)
- [ ] Natural language prompts work with GroundingDINO ("a red car parked on the street")
- [ ] Fallback to GroundingDINO when prefer_accuracy=True
- [ ] **VRAM Budget:** YOLO-World <4.5GB, GroundingDINO <5GB
- [ ] **Performance:** YOLO-World <20ms/image, GroundingDINO <150ms/image on GPU

## Files to Create

```
backend/cv/
└── openvocab.py   # YOLOWorldWrapper, GroundingDINOWrapper, get_openvocab_detector()
```

## Testing

```python
# test_openvocab.py
import pytest
from backend.cv.openvocab import YOLOWorldWrapper, GroundingDINOWrapper

@pytest.fixture
def yolo_world():
    y = YOLOWorldWrapper()
    y.load("cuda" if torch.cuda.is_available() else "cpu")
    yield y
    y.unload()

@pytest.mark.gpu
def test_custom_class_detection(yolo_world, sample_image_with_car):
    result = yolo_world.predict(sample_image_with_car, prompt="red car")
    assert len(result.detections) > 0
    assert result.detections[0].class_name == "red car"

def test_multiple_classes(yolo_world, sample_image):
    result = yolo_world.predict(sample_image, prompt="car, person, bicycle")
    # Check that class names match input
    for det in result.detections:
        assert det.class_name in ["car", "person", "bicycle"]

@pytest.mark.gpu
def test_embedding_cache_speedup(yolo_world, sample_image):
    import time

    prompt = "car, person, bicycle"

    # First call - creates embeddings
    start = time.perf_counter()
    yolo_world.predict(sample_image, prompt=prompt)
    first_time = time.perf_counter() - start

    # Second call - uses cached embeddings
    start = time.perf_counter()
    yolo_world.predict(sample_image, prompt=prompt)
    second_time = time.perf_counter() - start

    assert second_time < first_time * 0.8  # At least 20% faster

@pytest.mark.gpu
def test_groundingdino_natural_language():
    gdino = GroundingDINOWrapper()
    gdino.load("cuda")
    result = gdino.predict("image.jpg", prompt="a red car parked on the street")
    assert len(result.detections) >= 0  # May or may not find matches
    gdino.unload()

@pytest.mark.gpu
def test_vram_budget(yolo_world):
    from backend.cv.device import get_vram_usage
    used, _ = get_vram_usage()
    assert used < 4500  # <4.5GB
```
