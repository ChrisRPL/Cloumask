# SAM3 Text-Prompted Segmentation

> **Status:** Completed
> **Priority:** P0 (Critical)
> **Dependencies:** [00-infrastructure.md](./00-infrastructure.md)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Text-prompted segmentation using SAM3 (Segment Anything Model 3) as the primary model, with SAM2 and MobileSAM as fallbacks. SAM3 supports natural language prompts like "red car" or "person on left", enabling open-vocabulary segmentation without predefined classes. Fallbacks support point/box prompts for compatibility.

## Goals

- [ ] Integrate SAM3 from HuggingFace (requires access approval)
- [ ] Implement text-prompted segmentation ("red car", "person on left")
- [ ] Support point/box prompts as alternative input modes
- [ ] Add SAM2 fallback (6x faster than SAM1, video-native)
- [ ] Add MobileSAM fallback (real-time capable, ~1GB VRAM)
- [ ] Handle multiple mask outputs with confidence scores

## Technical Design

### Model Specifications

| Model | Size | VRAM | Inference | Features |
|-------|------|------|-----------|----------|
| SAM3 | ~3.5GB | ~8GB | 300-500ms | Text prompts, 4M+ concepts |
| SAM2 | ~2.4GB | ~6GB | 100-200ms | Point/box prompts, video |
| MobileSAM | ~40MB | ~1GB | 50-100ms | Point prompts, real-time |

### SAM3 Wrapper

```python
from typing import Optional, Union, Callable
from backend.cv.base import BaseModelWrapper
from backend.cv.types import Mask, SegmentationResult
import numpy as np
import torch

class SAM3Wrapper(BaseModelWrapper):
    """SAM3 text-prompted segmentation wrapper."""

    model_name = "sam3"
    vram_required_mb = 8000
    supports_batching = False  # SAM3 processes one image at a time

    def __init__(self):
        self._model = None
        self._processor = None
        self._device: str = "cpu"

    def load(self, device: str = "cuda") -> None:
        """Load SAM3 model from HuggingFace."""
        from transformers import Sam3ForSegmentation, Sam3Processor
        import os

        model_id = "facebook/sam3-hiera-large"
        token = os.getenv("HF_TOKEN")

        self._processor = Sam3Processor.from_pretrained(model_id, token=token)
        self._model = Sam3ForSegmentation.from_pretrained(model_id, token=token)

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
        prompt: Optional[str] = None,
        point: Optional[tuple[int, int]] = None,
        box: Optional[tuple[int, int, int, int]] = None,
        multimask_output: bool = True,
    ) -> SegmentationResult:
        """
        Segment image based on prompt.

        Args:
            image_path: Path to input image
            prompt: Text prompt (e.g., "red car", "person on left")
            point: Point prompt as (x, y) pixel coordinates
            box: Box prompt as (x1, y1, x2, y2) pixel coordinates
            multimask_output: Return multiple mask candidates

        Returns:
            SegmentationResult with masks and confidences
        """
        import time
        from PIL import Image

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if prompt is None and point is None and box is None:
            raise ValueError("Must provide prompt, point, or box")

        image = Image.open(image_path).convert("RGB")

        start = time.perf_counter()

        # Prepare inputs based on prompt type
        if prompt:
            inputs = self._processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self._device)
        elif point:
            inputs = self._processor(
                images=image,
                input_points=[[[point[0], point[1]]]],
                return_tensors="pt"
            ).to(self._device)
        elif box:
            inputs = self._processor(
                images=image,
                input_boxes=[[list(box)]],
                return_tensors="pt"
            ).to(self._device)

        # Generate masks
        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=multimask_output)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Process outputs
        masks_tensor = outputs.pred_masks.squeeze().cpu()
        scores = outputs.iou_scores.squeeze().cpu().numpy()

        # Convert to our Mask type
        masks = []
        if multimask_output:
            for i, (mask_tensor, score) in enumerate(zip(masks_tensor, scores)):
                mask_np = (mask_tensor.numpy() > 0.5).astype(np.uint8)
                masks.append(Mask.from_numpy(mask_np, float(score)))
        else:
            mask_np = (masks_tensor.numpy() > 0.5).astype(np.uint8)
            masks.append(Mask.from_numpy(mask_np, float(scores)))

        # Sort by confidence
        masks.sort(key=lambda m: m.confidence, reverse=True)

        return SegmentationResult(
            masks=masks,
            image_path=image_path,
            processing_time_ms=elapsed_ms,
            model_name=self.model_name,
        )
```

### SAM2 Fallback Wrapper

```python
class SAM2Wrapper(BaseModelWrapper):
    """SAM2 point/box-prompted segmentation wrapper."""

    model_name = "sam2"
    vram_required_mb = 6000
    supports_batching = False

    def __init__(self):
        self._model = None
        self._predictor = None
        self._device: str = "cpu"

    def load(self, device: str = "cuda") -> None:
        """Load SAM2 model."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = "sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        self._model = build_sam2(model_cfg, checkpoint, device=device)
        self._predictor = SAM2ImagePredictor(self._model)
        self._device = device

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            del self._predictor
            self._model = None
            self._predictor = None
            if self._device == "cuda":
                torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        point: Optional[tuple[int, int]] = None,
        box: Optional[tuple[int, int, int, int]] = None,
        multimask_output: bool = True,
    ) -> SegmentationResult:
        """
        SAM2 segmentation with point/box prompts.

        Note: SAM2 does not support text prompts. If text prompt provided,
        raises NotImplementedError suggesting fallback to SAM3.
        """
        import time
        from PIL import Image
        import numpy as np

        if prompt is not None:
            raise NotImplementedError(
                "SAM2 does not support text prompts. Use SAM3 or provide point/box."
            )

        if point is None and box is None:
            raise ValueError("SAM2 requires point or box prompt")

        image = np.array(Image.open(image_path).convert("RGB"))

        start = time.perf_counter()

        self._predictor.set_image(image)

        # Prepare prompts
        point_coords = None
        point_labels = None
        box_input = None

        if point:
            point_coords = np.array([[point[0], point[1]]])
            point_labels = np.array([1])  # 1 = foreground

        if box:
            box_input = np.array([list(box)])

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_input,
            multimask_output=multimask_output,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Convert to our types
        mask_results = []
        for mask, score in zip(masks, scores):
            mask_results.append(Mask.from_numpy(mask.astype(np.uint8), float(score)))

        mask_results.sort(key=lambda m: m.confidence, reverse=True)

        return SegmentationResult(
            masks=mask_results,
            image_path=image_path,
            processing_time_ms=elapsed_ms,
            model_name=self.model_name,
        )
```

### MobileSAM Fallback Wrapper

```python
class MobileSAMWrapper(BaseModelWrapper):
    """MobileSAM lightweight segmentation wrapper."""

    model_name = "mobilesam"
    vram_required_mb = 1500
    supports_batching = False

    def load(self, device: str = "cuda") -> None:
        """Load MobileSAM model."""
        from mobile_sam import sam_model_registry, SamPredictor

        model_type = "vit_t"
        checkpoint = "mobile_sam.pt"

        self._model = sam_model_registry[model_type](checkpoint=checkpoint)
        self._model.to(device)
        self._predictor = SamPredictor(self._model)
        self._device = device

    # Similar unload/predict implementation as SAM2
    # Only supports point prompts for maximum speed
```

### Segmentation Factory

```python
def get_segmenter(
    prompt_type: str = "text",
    prefer_speed: bool = False,
    force_model: Optional[str] = None
) -> BaseModelWrapper:
    """
    Get appropriate segmentation model.

    Args:
        prompt_type: "text", "point", or "box"
        prefer_speed: If True, prefer MobileSAM over SAM2
        force_model: Force specific model

    Returns:
        Appropriate segmenter wrapper
    """
    from backend.cv.device import get_available_vram_mb

    if force_model:
        return {"sam3": SAM3Wrapper, "sam2": SAM2Wrapper, "mobilesam": MobileSAMWrapper}[force_model]()

    if prompt_type == "text":
        available = get_available_vram_mb()
        if available >= 8000:
            return SAM3Wrapper()
        # Text prompts require SAM3
        raise RuntimeError("SAM3 requires 8GB VRAM for text prompts. Use point/box prompts instead.")

    if prefer_speed or get_available_vram_mb() < 6000:
        return MobileSAMWrapper()

    return SAM2Wrapper()
```

## Implementation Tasks

- [ ] **SAM3 Integration**
  - [ ] Create `backend/cv/segmentation.py`
  - [ ] Implement SAM3Wrapper with HuggingFace transformers
  - [ ] Handle HF_TOKEN authentication for gated model
  - [ ] Support text, point, and box prompts
  - [ ] Implement multimask output handling

- [ ] **SAM2 Fallback**
  - [ ] Implement SAM2Wrapper
  - [ ] Handle video frames (future enhancement)
  - [ ] Add point/box prompt support

- [ ] **MobileSAM Fallback**
  - [ ] Implement MobileSAMWrapper
  - [ ] Optimize for real-time usage
  - [ ] Point-only prompt support

- [ ] **Factory & Fallback Logic**
  - [ ] Implement get_segmenter() with smart selection
  - [ ] Add fallback chain: SAM3 -> SAM2 -> MobileSAM
  - [ ] VRAM-based automatic model selection

- [ ] **Testing**
  - [ ] Unit tests with mock models
  - [ ] Integration tests (GPU, gated model access)
  - [ ] Text prompt accuracy tests
  - [ ] Performance benchmarks

## Acceptance Criteria

- [ ] `segment("red car", image_path)` returns mask array with SAM3
- [ ] `segment(image_path, point=(x,y))` works with SAM2/MobileSAM
- [ ] `segment(image_path, box=(x1,y1,x2,y2))` works with SAM2
- [ ] Multiple masks returned ranked by confidence
- [ ] Fallback chain works when primary model unavailable
- [ ] HF_TOKEN environment variable enables SAM3 access
- [ ] **VRAM Budget:** SAM3 <8GB, SAM2 <6GB, MobileSAM <1.5GB
- [ ] **Performance:** SAM3 <500ms/image, MobileSAM <100ms/image on GPU

## Files to Create

```
backend/cv/
└── segmentation.py   # SAM3Wrapper, SAM2Wrapper, MobileSAMWrapper, get_segmenter()
```

## Testing

```python
# test_segmentation.py
import pytest
from backend.cv.segmentation import SAM3Wrapper, SAM2Wrapper, MobileSAMWrapper

@pytest.fixture
def sam3():
    s = SAM3Wrapper()
    s.load("cuda")
    yield s
    s.unload()

@pytest.mark.gpu
@pytest.mark.slow
def test_text_prompt_segmentation(sam3, sample_image_with_car):
    result = sam3.predict(sample_image_with_car, prompt="red car")
    assert len(result.masks) > 0
    assert result.masks[0].confidence > 0.5

@pytest.mark.gpu
def test_point_prompt_sam2(sample_image):
    sam2 = SAM2Wrapper()
    sam2.load("cuda")
    result = sam2.predict(sample_image, point=(100, 100))
    assert len(result.masks) > 0
    sam2.unload()

def test_sam2_rejects_text_prompt():
    sam2 = SAM2Wrapper()
    sam2.load("cpu")
    with pytest.raises(NotImplementedError):
        sam2.predict("image.jpg", prompt="car")

@pytest.mark.gpu
def test_vram_budget_sam3(sam3):
    from backend.cv.device import get_vram_usage
    used, _ = get_vram_usage()
    assert used < 8000  # <8GB

@pytest.mark.gpu
def test_inference_speed_mobilesam(sample_image):
    import time
    mobile = MobileSAMWrapper()
    mobile.load("cuda")

    times = []
    for _ in range(10):
        start = time.perf_counter()
        mobile.predict(sample_image, point=(100, 100))
        times.append((time.perf_counter() - start) * 1000)

    mobile.unload()
    assert sum(times) / len(times) < 100  # <100ms average
```
