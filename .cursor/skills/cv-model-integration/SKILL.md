---
name: cv-model-integration
description: Guide for integrating new computer vision models into Cloumask following the BaseModelWrapper pattern. Use when adding new CV models, creating model wrappers, or selecting appropriate models for tasks.
---

# CV Model Integration

## Quick Start

When adding a new CV model to Cloumask:

1. Create a wrapper class inheriting from `BaseModelWrapper[T]`
2. Define `ModelInfo` with VRAM requirements
3. Implement `_load_model()`, `_unload_model()`, and `predict()`
4. Register with `ModelManager` or use `@register_model` decorator

## BaseModelWrapper Pattern

All CV models must inherit from `BaseModelWrapper[T]` where `T` is the result type:

```python
from backend.cv.base import BaseModelWrapper, ModelInfo, register_model
from backend.cv.types import DetectionResult

@register_model
class MyModelWrapper(BaseModelWrapper[DetectionResult]):
    # Required: Define model info
    info = ModelInfo(
        name="my_model",
        description="Description of what this model does",
        vram_required_mb=2000,  # VRAM needed in MB
        supports_batching=True,  # Can process multiple inputs
        supports_gpu=True,       # GPU acceleration available
        source="huggingface",    # Where model comes from
        version="1.0.0",
    )
    
    def _load_model(self, device: str) -> None:
        """Load model weights to device."""
        # Load your model here
        self._model = load_your_model()
        self._model.to(device)
    
    def _unload_model(self) -> None:
        """Free model memory."""
        del self._model
        self._model = None
        # Optionally: clear GPU cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def predict(self, input_path: str, **kwargs) -> DetectionResult:
        """Run inference and return structured result."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Run inference
        results = self._model(input_path, **kwargs)
        
        # Convert to DetectionResult
        return self._convert_results(results)
```

## ModelInfo Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | str | Yes | Unique model identifier |
| `description` | str | No | Human-readable description |
| `vram_required_mb` | int | Yes | VRAM needed in MB (0 for CPU-only) |
| `supports_batching` | bool | No | Can process batches (default: False) |
| `supports_gpu` | bool | No | GPU acceleration (default: True) |
| `default_device` | str | No | Preferred device (default: "auto") |
| `version` | str | No | Model version (default: "1.0.0") |
| `source` | str | No | Model source (e.g., "ultralytics", "huggingface") |

## Model Manager Usage

Models are lazy-loaded through `ModelManager`:

```python
from backend.cv import ModelManager

manager = ModelManager()

# Get model (loads on first access)
model = manager.get("yolo11m")

# Run inference
result = model.predict("image.jpg")

# Unload when done (frees VRAM)
manager.unload("yolo11m")
```

## Result Types

Use appropriate result types from `backend.cv.types`:

- `DetectionResult` - Object detection (bounding boxes)
- `SegmentationResult` - Instance/panoptic segmentation (masks)
- `AnonymizationResult` - Anonymization results

## Device Selection

Models automatically select device based on VRAM:

- `device="auto"` - Auto-selects CUDA/MPS/CPU based on availability
- `device="cuda"` - Force GPU (NVIDIA)
- `device="mps"` - Force GPU (Apple Silicon)
- `device="cpu"` - Force CPU

ModelManager handles VRAM eviction automatically when needed.

## Error Handling

Always handle GPU OOM errors:

```python
def _load_model(self, device: str) -> None:
    try:
        self._model = load_model()
        self._model.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Fallback to CPU
            logger.warning("GPU OOM, falling back to CPU")
            self._model.to("cpu")
        else:
            raise
```

## Model Selection Reference

| Task | Primary Model | Fallback | VRAM | Notes |
|------|---------------|----------|------|-------|
| Detection (COCO) | YOLO11m | RT-DETR | 2GB | Fast, accurate |
| Detection (open-vocab) | YOLO-World | GroundingDINO | 3GB | Text prompts |
| Segmentation (text) | SAM3 | SAM2 | 8GB | 4M+ concepts |
| Segmentation (point) | SAM2 | MobileSAM | 6GB | 6x faster |
| Face Detection | SCRFD-10G | YuNet | 1GB | 95%+ accuracy |
| 3D Detection | PV-RCNN++ | CenterPoint | 4GB | Point clouds |

## Common Patterns

### Batch Processing

```python
def predict_batch(self, input_paths: list[str], **kwargs) -> list[DetectionResult]:
    """Process multiple inputs efficiently."""
    if not self.info.supports_batching:
        # Fallback to sequential
        return [self.predict(path, **kwargs) for path in input_paths]
    
    # Batch inference
    results = self._model(input_paths, **kwargs)
    return [self._convert_results(r) for r in results]
```

### Progress Callbacks

```python
def predict_with_progress(
    self,
    input_path: str,
    callback: ProgressCallback | None = None,
    **kwargs
) -> DetectionResult:
    """Run inference with progress updates."""
    if callback:
        callback(0, 100, "Loading model...")
    
    # ... inference steps with callback updates ...
    
    if callback:
        callback(100, 100, "Complete")
    
    return result
```

## Testing

Always test model loading and inference:

```python
def test_model():
    manager = ModelManager()
    model = manager.get("my_model")
    
    # Test inference
    result = model.predict("test_image.jpg")
    assert result.count > 0
    
    # Cleanup
    manager.unload("my_model")
```

## Registration

Models are auto-registered with `@register_model` decorator. For manual registration:

```python
from backend.cv import ModelManager

manager = ModelManager()
manager.register("my_model", MyModelWrapper)
```

## Additional Resources

- See `backend/src/backend/cv/detection.py` for YOLO11 example
- See `backend/src/backend/cv/segmentation.py` for SAM2 example
- See `backend/src/backend/cv/base.py` for BaseModelWrapper API
