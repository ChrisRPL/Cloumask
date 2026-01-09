# License Plate Detection

> **Status:** Not Started
> **Priority:** P1 (High)
> **Dependencies:** [00-infrastructure.md](./00-infrastructure.md), [04-yolo-world-openvocab.md](./04-yolo-world-openvocab.md)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

License plate detection using YOLO-World with "license plate" prompt as the primary approach, with optional support for specialized fine-tuned models. This module focuses on detecting plate locations for anonymization; OCR/reading is out of scope. For MVP, the YOLO-World approach is sufficient.

## Goals

- [ ] Primary: Use YOLO-World with "license plate" prompt
- [ ] Fallback: Support specialized fine-tuned YOLO if available
- [ ] Handle different plate regions/formats (EU, US, etc.)
- [ ] Return plate bounding boxes with confidence
- [ ] Aspect ratio validation to filter false positives

## Technical Design

### Strategy

For MVP, we leverage YOLO-World's open-vocabulary capabilities with "license plate" as the prompt. This approach:
- **Pros:** No additional model download, reuses existing YOLO-World, works across regions
- **Cons:** Slightly lower accuracy than specialized models

Specialized models can be added later for production use cases requiring higher accuracy.

### Model Specifications

| Approach | VRAM | Inference | Notes |
|----------|------|-----------|-------|
| YOLO-World (prompt) | ~4.5GB (shared) | 15-20ms | Reuses loaded model |
| Specialized YOLO | ~2GB | 5-10ms | Higher accuracy, separate model |

### Plate Detector Wrapper

```python
from typing import Optional, List
from backend.cv.base import BaseModelWrapper
from backend.cv.types import Detection, BBox
from backend.cv.openvocab import YOLOWorldWrapper

class PlateDetectorWrapper(BaseModelWrapper):
    """License plate detection wrapper."""

    model_name = "plate-detector"
    vram_required_mb = 4500  # Uses YOLO-World
    supports_batching = True

    # Valid aspect ratios for license plates (width/height)
    MIN_ASPECT_RATIO = 1.5  # Minimum plate w/h ratio
    MAX_ASPECT_RATIO = 6.0  # Maximum plate w/h ratio

    def __init__(self, use_specialized: bool = False):
        """
        Initialize plate detector.

        Args:
            use_specialized: If True, use specialized model if available
        """
        self._use_specialized = use_specialized
        self._yolo_world: Optional[YOLOWorldWrapper] = None
        self._specialized_model = None
        self._device: str = "cpu"

    def load(self, device: str = "cuda") -> None:
        """Load detector."""
        if self._use_specialized and self._check_specialized_model():
            self._load_specialized(device)
        else:
            self._yolo_world = YOLOWorldWrapper()
            self._yolo_world.load(device)
        self._device = device

    def _check_specialized_model(self) -> bool:
        """Check if specialized model is available."""
        from backend.cv.download import MODELS_DIR
        return (MODELS_DIR / "plate_detector" / "best.pt").exists()

    def _load_specialized(self, device: str) -> None:
        """Load specialized plate detection model."""
        from ultralytics import YOLO
        from backend.cv.download import MODELS_DIR

        model_path = MODELS_DIR / "plate_detector" / "best.pt"
        self._specialized_model = YOLO(model_path)
        if device == "cuda":
            self._specialized_model.to(device)

    def unload(self) -> None:
        if self._yolo_world is not None:
            self._yolo_world.unload()
            self._yolo_world = None
        if self._specialized_model is not None:
            del self._specialized_model
            self._specialized_model = None
        if self._device == "cuda":
            import torch
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._yolo_world is not None or self._specialized_model is not None

    def predict(
        self,
        image_path: str,
        confidence: float = 0.3,
        validate_aspect_ratio: bool = True,
    ) -> List[Detection]:
        """
        Detect license plates in an image.

        Args:
            image_path: Path to input image
            confidence: Minimum confidence threshold
            validate_aspect_ratio: Filter detections by plate aspect ratio

        Returns:
            List of Detection objects for plates
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self._specialized_model is not None:
            detections = self._predict_specialized(image_path, confidence)
        else:
            detections = self._predict_yoloworld(image_path, confidence)

        # Filter by aspect ratio
        if validate_aspect_ratio:
            detections = self._filter_by_aspect_ratio(detections)

        return detections

    def _predict_yoloworld(self, image_path: str, confidence: float) -> List[Detection]:
        """Predict using YOLO-World with license plate prompt."""
        result = self._yolo_world.predict(
            image_path,
            prompt="license plate",
            confidence=confidence,
        )
        return result.detections

    def _predict_specialized(self, image_path: str, confidence: float) -> List[Detection]:
        """Predict using specialized model."""
        results = self._specialized_model.predict(
            image_path,
            conf=confidence,
            device=self._device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            detections.append(Detection(
                class_id=0,
                class_name="license_plate",
                bbox=BBox(
                    x=float(box.xywhn[0][0]),
                    y=float(box.xywhn[0][1]),
                    width=float(box.xywhn[0][2]),
                    height=float(box.xywhn[0][3]),
                ),
                confidence=float(box.conf[0]),
            ))

        return detections

    def _filter_by_aspect_ratio(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections by typical license plate aspect ratio."""
        filtered = []
        for det in detections:
            if det.bbox.height > 0:
                aspect_ratio = det.bbox.width / det.bbox.height
                if self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                    filtered.append(det)
        return filtered

    def predict_batch(
        self,
        image_paths: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[List[Detection]]:
        """Batch plate detection."""
        results = []
        for i, path in enumerate(image_paths):
            results.append(self.predict(path, **kwargs))
            if progress_callback:
                progress_callback(i + 1, len(image_paths))
        return results
```

### Region-Specific Configuration

```python
# Plate region configurations for future enhancement
PLATE_REGIONS = {
    "eu": {
        "aspect_ratio_range": (3.0, 5.5),
        "typical_size": (520, 110),  # mm
    },
    "us": {
        "aspect_ratio_range": (1.8, 2.5),
        "typical_size": (305, 152),  # mm
    },
    "china": {
        "aspect_ratio_range": (2.5, 4.0),
        "typical_size": (440, 140),  # mm
    },
}

def get_plate_detector(
    region: Optional[str] = None,
    use_specialized: bool = False
) -> PlateDetectorWrapper:
    """
    Get plate detector configured for region.

    Args:
        region: Plate region ("eu", "us", "china", None for auto)
        use_specialized: Use specialized model if available

    Returns:
        Configured plate detector
    """
    detector = PlateDetectorWrapper(use_specialized=use_specialized)

    if region and region in PLATE_REGIONS:
        config = PLATE_REGIONS[region]
        detector.MIN_ASPECT_RATIO = config["aspect_ratio_range"][0]
        detector.MAX_ASPECT_RATIO = config["aspect_ratio_range"][1]

    return detector
```

## Implementation Tasks

- [ ] **YOLO-World Integration**
  - [ ] Create `backend/cv/plates.py`
  - [ ] Implement PlateDetectorWrapper
  - [ ] Add aspect ratio validation
  - [ ] Reuse YOLOWorldWrapper from openvocab

- [ ] **Specialized Model Support**
  - [ ] Add check for specialized model
  - [ ] Load specialized YOLO if available
  - [ ] Document model placement in models/

- [ ] **Region Support**
  - [ ] Add region configurations
  - [ ] Configurable aspect ratio ranges
  - [ ] Factory function with region selection

- [ ] **Testing**
  - [ ] Unit tests with mock
  - [ ] Accuracy tests on plate images
  - [ ] Aspect ratio filtering tests
  - [ ] Performance benchmarks

## Acceptance Criteria

- [ ] `detect_plates(image)` returns plate bounding boxes
- [ ] Works with YOLO-World prompt approach
- [ ] Aspect ratio validation filters false positives (signs, windows)
- [ ] Reasonable accuracy on common license plate formats (EU, US)
- [ ] Specialized model used when available
- [ ] **VRAM Budget:** Uses YOLO-World (~4.5GB) or specialized (<2GB)
- [ ] **Performance:** <30ms/image (reuses YOLO-World embeddings)

## Files to Create

```
backend/cv/
└── plates.py   # PlateDetectorWrapper, get_plate_detector()
```

## Testing

```python
# test_plates.py
import pytest
from backend.cv.plates import PlateDetectorWrapper, get_plate_detector

@pytest.fixture
def plate_detector():
    p = PlateDetectorWrapper()
    p.load("cuda" if torch.cuda.is_available() else "cpu")
    yield p
    p.unload()

def test_detect_plates(plate_detector, image_with_car_and_plate):
    plates = plate_detector.predict(image_with_car_and_plate)
    assert len(plates) >= 1
    assert plates[0].class_name == "license plate"

def test_no_plates_in_landscape(plate_detector, landscape_image):
    plates = plate_detector.predict(landscape_image)
    assert len(plates) == 0

def test_aspect_ratio_filtering(plate_detector, image_with_signs):
    """Signs should be filtered out by aspect ratio."""
    plates_filtered = plate_detector.predict(image_with_signs, validate_aspect_ratio=True)
    plates_unfiltered = plate_detector.predict(image_with_signs, validate_aspect_ratio=False)

    # Filtered should have fewer false positives
    assert len(plates_filtered) <= len(plates_unfiltered)

def test_region_config():
    detector = get_plate_detector(region="eu")
    assert detector.MIN_ASPECT_RATIO == 3.0
    assert detector.MAX_ASPECT_RATIO == 5.5

@pytest.mark.gpu
def test_performance(plate_detector, sample_image):
    import time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        plate_detector.predict(sample_image)
        times.append((time.perf_counter() - start) * 1000)

    assert sum(times) / len(times) < 30  # <30ms average
```
