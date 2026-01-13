# Anonymization Pipeline

> **Status:** Completed
> **Priority:** P0 (Critical)
> **Dependencies:** [02-sam3-segmentation.md](./02-sam3-segmentation.md), [03-scrfd-faces.md](./03-scrfd-faces.md), [05-plate-detection.md](./05-plate-detection.md)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Complete image anonymization pipeline combining face detection (SCRFD), license plate detection (YOLO-World), and optional SAM3-based precise masking. Supports four anonymization modes: blur, blackbox, pixelate, and mask. Used as the primary CV feature for privacy-compliant dataset preparation.

## Goals

- [ ] Implement blur mode (Gaussian, configurable kernel size)
- [ ] Implement blackbox mode (solid color fill, configurable)
- [ ] Implement pixelate mode (mosaic effect, configurable block size)
- [ ] Implement mask mode (SAM3 precise boundary + effect)
- [ ] Combine face and plate detection into single pipeline
- [ ] Preserve image quality outside anonymized regions

## Technical Design

### Anonymization Modes

| Mode | Effect | VRAM Impact | Speed |
|------|--------|-------------|-------|
| blur | Gaussian blur | None | <10ms |
| blackbox | Solid fill | None | <5ms |
| pixelate | Mosaic | None | <10ms |
| mask | SAM3 boundary + effect | +8GB | +500ms |

### Anonymization Pipeline

```python
from typing import Optional, List, Callable, Literal
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

from backend.cv.faces import SCRFDWrapper, get_face_detector
from backend.cv.plates import PlateDetectorWrapper, get_plate_detector
from backend.cv.segmentation import SAM3Wrapper, get_segmenter
from backend.cv.types import Detection, BBox

AnonymizationMode = Literal["blur", "blackbox", "pixelate", "mask"]

@dataclass
class AnonymizationConfig:
    """Configuration for anonymization pipeline."""
    faces: bool = True
    plates: bool = True
    mode: AnonymizationMode = "blur"

    # Blur settings
    blur_kernel_size: int = 51  # Must be odd

    # Blackbox settings
    blackbox_color: tuple[int, int, int] = (0, 0, 0)  # RGB

    # Pixelate settings
    pixelate_block_size: int = 10

    # Mask mode settings
    mask_feather_radius: int = 3  # Smooth mask edges

    # Detection settings
    face_confidence: float = 0.5
    plate_confidence: float = 0.3

@dataclass
class AnonymizationResult:
    """Result of anonymization."""
    output_path: str
    faces_anonymized: int
    plates_anonymized: int
    processing_time_ms: float

class AnonymizationPipeline:
    """Complete anonymization pipeline."""

    def __init__(self, config: Optional[AnonymizationConfig] = None):
        self.config = config or AnonymizationConfig()
        self._face_detector: Optional[SCRFDWrapper] = None
        self._plate_detector: Optional[PlateDetectorWrapper] = None
        self._segmenter: Optional[SAM3Wrapper] = None
        self._device: str = "cpu"

    def load(self, device: str = "cuda") -> None:
        """Load all required models based on config."""
        self._device = device

        if self.config.faces:
            self._face_detector = get_face_detector()
            self._face_detector.load(device)

        if self.config.plates:
            self._plate_detector = get_plate_detector()
            self._plate_detector.load(device)

        if self.config.mode == "mask":
            self._segmenter = get_segmenter(prompt_type="point")
            self._segmenter.load(device)

    def unload(self) -> None:
        """Unload all models."""
        if self._face_detector:
            self._face_detector.unload()
        if self._plate_detector:
            self._plate_detector.unload()
        if self._segmenter:
            self._segmenter.unload()

    def process(
        self,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> AnonymizationResult:
        """
        Anonymize an image.

        Args:
            image_path: Path to input image
            output_path: Output path (default: adds _anon suffix)

        Returns:
            AnonymizationResult with stats
        """
        import time

        start = time.perf_counter()

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w = image.shape[:2]

        # Collect detections
        detections = []
        face_count = 0
        plate_count = 0

        if self.config.faces and self._face_detector:
            faces = self._face_detector.predict(
                image_path,
                confidence=self.config.face_confidence
            )
            for face in faces:
                detections.append(("face", face.bbox))
            face_count = len(faces)

        if self.config.plates and self._plate_detector:
            plates = self._plate_detector.predict(
                image_path,
                confidence=self.config.plate_confidence
            )
            for plate in plates:
                detections.append(("plate", plate.bbox))
            plate_count = len(plates)

        # Anonymize each detection
        for det_type, bbox in detections:
            image = self._anonymize_region(image, bbox, w, h, det_type, image_path)

        # Save output
        if output_path is None:
            path = Path(image_path)
            output_path = str(path.parent / f"{path.stem}_anon{path.suffix}")

        cv2.imwrite(output_path, image)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnonymizationResult(
            output_path=output_path,
            faces_anonymized=face_count,
            plates_anonymized=plate_count,
            processing_time_ms=elapsed_ms,
        )

    def _anonymize_region(
        self,
        image: np.ndarray,
        bbox: BBox,
        img_w: int,
        img_h: int,
        det_type: str,
        image_path: str,
    ) -> np.ndarray:
        """Apply anonymization effect to a region."""
        # Convert normalized bbox to pixel coordinates
        x1, y1, x2, y2 = bbox.to_xyxy(img_w, img_h)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        if x2 <= x1 or y2 <= y1:
            return image  # Invalid region

        if self.config.mode == "mask" and self._segmenter:
            # Get precise mask from SAM
            mask = self._get_sam_mask(image_path, (x1, y1, x2, y2))
            image = self._apply_effect_with_mask(image, mask, x1, y1, x2, y2)
        else:
            # Apply effect to rectangular region
            roi = image[y1:y2, x1:x2]
            roi = self._apply_effect(roi)
            image[y1:y2, x1:x2] = roi

        return image

    def _get_sam_mask(
        self,
        image_path: str,
        bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Get precise segmentation mask using SAM."""
        result = self._segmenter.predict(
            image_path,
            box=bbox,
            multimask_output=False,
        )

        if result.masks:
            return result.masks[0].to_numpy()

        # Fallback to rectangular mask
        x1, y1, x2, y2 = bbox
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        mask[:] = 255
        return mask

    def _apply_effect(self, roi: np.ndarray) -> np.ndarray:
        """Apply anonymization effect to ROI."""
        mode = self.config.mode

        if mode == "blur":
            k = self.config.blur_kernel_size
            return cv2.GaussianBlur(roi, (k, k), 0)

        elif mode == "blackbox":
            color = self.config.blackbox_color
            roi[:] = color[::-1]  # RGB to BGR
            return roi

        elif mode == "pixelate":
            block = self.config.pixelate_block_size
            h, w = roi.shape[:2]
            # Downsample then upsample
            small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                             interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        elif mode == "mask":
            # Should use _apply_effect_with_mask instead
            return self._apply_effect_blur(roi)

        return roi

    def _apply_effect_with_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """Apply effect within mask boundary."""
        roi = image[y1:y2, x1:x2].copy()

        # Apply effect to entire ROI
        roi_effect = self._apply_effect_blur(roi)

        # Feather mask edges
        if self.config.mask_feather_radius > 0:
            mask = cv2.GaussianBlur(
                mask,
                (self.config.mask_feather_radius * 2 + 1,) * 2,
                0
            )

        # Normalize mask to [0, 1]
        mask_norm = mask.astype(np.float32) / 255.0
        mask_norm = mask_norm[:, :, np.newaxis]

        # Blend: effect where mask=1, original where mask=0
        blended = (roi_effect * mask_norm + roi * (1 - mask_norm)).astype(np.uint8)
        image[y1:y2, x1:x2] = blended

        return image

    def _apply_effect_blur(self, roi: np.ndarray) -> np.ndarray:
        """Apply blur effect (used by mask mode)."""
        k = self.config.blur_kernel_size
        return cv2.GaussianBlur(roi, (k, k), 0)

    def process_batch(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[AnonymizationResult]:
        """
        Batch anonymization with progress tracking.

        Args:
            image_paths: List of input image paths
            output_dir: Output directory (default: same as input with _anon suffix)
            progress_callback: Called with (processed, total)

        Returns:
            List of AnonymizationResult
        """
        results = []

        for i, image_path in enumerate(image_paths):
            if output_dir:
                output_path = str(Path(output_dir) / Path(image_path).name)
            else:
                output_path = None

            result = self.process(image_path, output_path)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(image_paths))

        return results
```

### Quick API

```python
def anonymize(
    image_path: str,
    output_path: Optional[str] = None,
    faces: bool = True,
    plates: bool = True,
    mode: AnonymizationMode = "blur",
    device: str = "auto",
) -> AnonymizationResult:
    """
    Quick anonymization function.

    Args:
        image_path: Path to input image
        output_path: Output path (optional)
        faces: Anonymize faces
        plates: Anonymize license plates
        mode: "blur", "blackbox", "pixelate", or "mask"
        device: "cuda", "cpu", or "auto"

    Returns:
        AnonymizationResult
    """
    from backend.cv.device import get_device_info

    if device == "auto":
        device = "cuda" if get_device_info()["cuda_available"] else "cpu"

    config = AnonymizationConfig(faces=faces, plates=plates, mode=mode)
    pipeline = AnonymizationPipeline(config)
    pipeline.load(device)

    try:
        return pipeline.process(image_path, output_path)
    finally:
        pipeline.unload()
```

## Implementation Tasks

- [ ] **Effect Implementation**
  - [ ] Create `backend/cv/anonymization.py`
  - [ ] Implement blur effect (Gaussian)
  - [ ] Implement blackbox effect (solid fill)
  - [ ] Implement pixelate effect (mosaic)
  - [ ] Implement mask effect (SAM + blur)

- [ ] **Pipeline Integration**
  - [ ] Combine face and plate detectors
  - [ ] Coordinate VRAM for multiple models
  - [ ] Handle detection overlap

- [ ] **Mask Mode**
  - [ ] SAM3 integration for precise masks
  - [ ] Mask feathering for smooth edges
  - [ ] Fallback to rectangular mask

- [ ] **Batch Processing**
  - [ ] Efficient batch processing
  - [ ] Progress callbacks
  - [ ] Output directory handling

- [ ] **Testing**
  - [ ] Visual quality tests
  - [ ] Performance benchmarks
  - [ ] Edge case handling

## Acceptance Criteria

- [ ] `anonymize(image, faces=True, plates=True, mode="blur")` works
- [ ] All four modes produce visually correct output
- [ ] Mask mode creates clean edges around faces (no hard boundaries)
- [ ] Batch processing with progress callbacks
- [ ] Original image regions outside detections unchanged
- [ ] Works on images with 0-100 faces/plates
- [ ] **VRAM Budget:** Peak <10GB (faces + plates + SAM3 mask mode)
- [ ] **Performance:** Blur/blackbox/pixelate <50ms/image, mask mode <600ms/image

## Files to Create

```
backend/cv/
└── anonymization.py   # AnonymizationPipeline, AnonymizationConfig, anonymize()
```

## Testing

```python
# test_anonymization.py
import pytest
from backend.cv.anonymization import AnonymizationPipeline, AnonymizationConfig, anonymize

@pytest.fixture
def pipeline():
    config = AnonymizationConfig(faces=True, plates=True, mode="blur")
    p = AnonymizationPipeline(config)
    p.load("cuda" if torch.cuda.is_available() else "cpu")
    yield p
    p.unload()

def test_blur_mode(pipeline, image_with_face, tmp_path):
    output = tmp_path / "output.jpg"
    result = pipeline.process(image_with_face, str(output))
    assert output.exists()
    assert result.faces_anonymized >= 1

def test_blackbox_mode(image_with_face, tmp_path):
    config = AnonymizationConfig(mode="blackbox", blackbox_color=(255, 0, 0))
    p = AnonymizationPipeline(config)
    p.load("cpu")

    result = p.process(image_with_face, str(tmp_path / "output.jpg"))
    p.unload()

    # Verify red fill in face region
    import cv2
    output = cv2.imread(str(tmp_path / "output.jpg"))
    # ... visual verification

def test_pixelate_mode(image_with_face, tmp_path):
    config = AnonymizationConfig(mode="pixelate", pixelate_block_size=20)
    p = AnonymizationPipeline(config)
    p.load("cpu")

    result = p.process(image_with_face, str(tmp_path / "output.jpg"))
    p.unload()

    assert result.faces_anonymized >= 1

@pytest.mark.gpu
@pytest.mark.slow
def test_mask_mode(image_with_face, tmp_path):
    config = AnonymizationConfig(mode="mask")
    p = AnonymizationPipeline(config)
    p.load("cuda")

    result = p.process(image_with_face, str(tmp_path / "output.jpg"))
    p.unload()

    assert result.faces_anonymized >= 1
    assert result.processing_time_ms < 600

def test_quick_api(image_with_face, tmp_path):
    output = tmp_path / "anon.jpg"
    result = anonymize(image_with_face, str(output))
    assert output.exists()

def test_no_modification_outside_detections(pipeline, image_with_face, tmp_path):
    """Verify regions outside detections are unchanged."""
    import cv2
    original = cv2.imread(image_with_face)
    output_path = str(tmp_path / "output.jpg")

    result = pipeline.process(image_with_face, output_path)
    modified = cv2.imread(output_path)

    # Compare corner pixels (unlikely to contain faces)
    assert np.array_equal(original[0:10, 0:10], modified[0:10, 0:10])

@pytest.mark.gpu
def test_vram_budget_mask_mode():
    from backend.cv.device import get_vram_usage
    config = AnonymizationConfig(mode="mask")
    p = AnonymizationPipeline(config)
    p.load("cuda")

    used, _ = get_vram_usage()
    assert used < 10000  # <10GB

    p.unload()
```
