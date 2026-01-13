# CV Infrastructure

> **Status:** Completed
> **Priority:** P0 (Critical)
> **Dependencies:** 01-foundation (Python sidecar setup)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Foundation layer that all CV models depend on. Establishes base types, model management patterns, device detection, VRAM monitoring, and model download infrastructure. This module must be completed before any model integration can begin.

## Goals

- [ ] Define core Pydantic types for detection and segmentation results
- [ ] Implement ModelManager singleton with lazy loading
- [ ] Add VRAM monitoring via pynvml (fallback to nvidia-smi subprocess)
- [ ] Create device management (GPU/CPU detection, CUDA OOM handling)
- [ ] Build model download infrastructure (HuggingFace Hub integration)
- [ ] Define BaseModelWrapper abstract class with standard interface

## Technical Design

### Core Types

```python
from pydantic import BaseModel
from typing import Optional
import numpy as np

class BBox(BaseModel):
    """Bounding box in normalized coordinates [0-1]."""
    x: float      # Center x
    y: float      # Center y
    width: float
    height: float

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Convert to absolute pixel coordinates (x1, y1, x2, y2)."""
        ...

class Detection(BaseModel):
    """Single object detection result."""
    class_id: int
    class_name: str
    bbox: BBox
    confidence: float

class DetectionResult(BaseModel):
    """Complete detection result for an image."""
    detections: list[Detection]
    image_path: str
    processing_time_ms: float
    model_name: str

class Mask(BaseModel):
    """Segmentation mask."""
    data: bytes  # Compressed numpy array
    width: int
    height: int
    confidence: float

    @classmethod
    def from_numpy(cls, arr: np.ndarray, confidence: float) -> "Mask":
        ...

    def to_numpy(self) -> np.ndarray:
        ...

class SegmentationResult(BaseModel):
    """Complete segmentation result."""
    masks: list[Mask]
    image_path: str
    processing_time_ms: float
    model_name: str

class FaceDetection(BaseModel):
    """Face detection with optional landmarks."""
    bbox: BBox
    confidence: float
    landmarks: Optional[list[tuple[float, float]]] = None  # 5-point landmarks

class Detection3D(BaseModel):
    """3D bounding box detection."""
    class_id: int
    class_name: str
    center: tuple[float, float, float]  # x, y, z
    dimensions: tuple[float, float, float]  # length, width, height
    rotation: float  # yaw angle in radians
    confidence: float
```

### ModelManager Singleton

```python
from typing import Dict, Type, Optional
from abc import ABC, abstractmethod
import torch
import threading

class BaseModelWrapper(ABC):
    """Abstract base class for all model wrappers."""

    model_name: str
    vram_required_mb: int
    supports_batching: bool = False

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load model weights to device."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free memory."""
        ...

    @abstractmethod
    def predict(self, input_path: str, **kwargs) -> Any:
        """Run inference on single input."""
        ...

    def predict_batch(
        self,
        input_paths: list[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> list[Any]:
        """Run inference on batch of inputs. Default: iterate predict()."""
        results = []
        for i, path in enumerate(input_paths):
            results.append(self.predict(path, **kwargs))
            if progress_callback:
                progress_callback(i + 1, len(input_paths))
        return results

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        ...

class ModelManager:
    """Singleton manager for lazy-loading CV models."""

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models: Dict[str, BaseModelWrapper] = {}
                    cls._instance._registry: Dict[str, Type[BaseModelWrapper]] = {}
        return cls._instance

    def register(self, name: str, wrapper_class: Type[BaseModelWrapper]) -> None:
        """Register a model wrapper class."""
        self._registry[name] = wrapper_class

    def get(self, name: str, device: str = "auto") -> BaseModelWrapper:
        """Get or load a model by name."""
        if name not in self._models:
            if name not in self._registry:
                raise ValueError(f"Unknown model: {name}")

            # Check VRAM before loading
            wrapper_class = self._registry[name]
            if device == "auto":
                device = self._select_device(wrapper_class.vram_required_mb)

            wrapper = wrapper_class()
            wrapper.load(device)
            self._models[name] = wrapper

        return self._models[name]

    def unload(self, name: str) -> None:
        """Unload a model and free memory."""
        if name in self._models:
            self._models[name].unload()
            del self._models[name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload_all(self) -> None:
        """Unload all models."""
        for name in list(self._models.keys()):
            self.unload(name)

    def _select_device(self, required_mb: int) -> str:
        """Select best available device based on VRAM requirements."""
        if not torch.cuda.is_available():
            return "cpu"

        available = get_available_vram_mb()
        if available >= required_mb:
            return "cuda"

        # Try to free memory by unloading least recently used model
        # ... LRU logic ...

        return "cpu"  # Fallback

    @property
    def loaded_models(self) -> list[str]:
        """List of currently loaded model names."""
        return list(self._models.keys())
```

### Device Management

```python
import subprocess
from typing import Tuple, Optional

def get_vram_usage() -> Tuple[int, int]:
    """
    Get current and total VRAM in MB.
    Returns: (used_mb, total_mb)
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used / 1024 / 1024), int(info.total / 1024 / 1024)
    except ImportError:
        # Fallback to nvidia-smi
        return _get_vram_nvidia_smi()

def _get_vram_nvidia_smi() -> Tuple[int, int]:
    """Fallback VRAM detection via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        used, total = result.stdout.strip().split(",")
        return int(used), int(total)
    except Exception:
        return 0, 0

def get_available_vram_mb() -> int:
    """Get available VRAM in MB."""
    used, total = get_vram_usage()
    return total - used

def get_device_info() -> dict:
    """Get comprehensive device information."""
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": None,
        "vram_total_mb": 0,
        "vram_used_mb": 0,
        "cpu_count": os.cpu_count(),
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["vram_used_mb"], info["vram_total_mb"] = get_vram_usage()

    return info

class CUDAOOMHandler:
    """Context manager for handling CUDA out-of-memory errors."""

    def __init__(self, fallback_device: str = "cpu", callback: Optional[Callable] = None):
        self.fallback_device = fallback_device
        self.callback = callback
        self.used_fallback = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.used_fallback = True
            if self.callback:
                self.callback(self.fallback_device)
            return True  # Suppress exception
        return False
```

### Model Download Infrastructure

```python
from pathlib import Path
from typing import Optional, Callable
import os

MODELS_DIR = Path(os.getenv("CLOUMASK_MODELS_DIR", "models"))

MODEL_REGISTRY = {
    "yolo11m": {
        "source": "ultralytics",
        "filename": "yolo11m.pt",
        "size_mb": 40,
    },
    "sam3": {
        "source": "huggingface",
        "repo_id": "facebook/sam3-hiera-large",
        "size_mb": 3500,
        "requires_auth": True,
    },
    "scrfd": {
        "source": "huggingface",
        "repo_id": "insightface/scrfd_10g_bnkps",
        "size_mb": 16,
    },
    # ... more models
}

def get_model_path(name: str) -> Path:
    """Get local path for a model."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    info = MODEL_REGISTRY[name]
    return MODELS_DIR / info.get("filename", name)

def is_model_downloaded(name: str) -> bool:
    """Check if model is already downloaded."""
    return get_model_path(name).exists()

def download_model(
    name: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    force: bool = False
) -> Path:
    """
    Download a model if not already present.

    Args:
        name: Model name from registry
        progress_callback: Called with (downloaded_mb, total_mb)
        force: Re-download even if exists

    Returns:
        Path to downloaded model
    """
    if not force and is_model_downloaded(name):
        return get_model_path(name)

    info = MODEL_REGISTRY[name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if info["source"] == "ultralytics":
        return _download_ultralytics(name, info, progress_callback)
    elif info["source"] == "huggingface":
        return _download_huggingface(name, info, progress_callback)
    else:
        raise ValueError(f"Unknown source: {info['source']}")

def _download_huggingface(
    name: str,
    info: dict,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Path:
    """Download from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download, snapshot_download

    repo_id = info["repo_id"]
    token = os.getenv("HF_TOKEN") if info.get("requires_auth") else None

    # Download to local models directory
    local_dir = MODELS_DIR / name
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,
    )

    return local_dir

def _download_ultralytics(
    name: str,
    info: dict,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Path:
    """Download ultralytics model (auto-downloads on first use)."""
    # ultralytics handles its own downloads
    # Just return the expected path, model loads on first inference
    return MODELS_DIR / info["filename"]

def list_available_models() -> list[dict]:
    """List all models in registry with download status."""
    return [
        {
            "name": name,
            "downloaded": is_model_downloaded(name),
            "size_mb": info["size_mb"],
            "source": info["source"],
        }
        for name, info in MODEL_REGISTRY.items()
    ]
```

## Implementation Tasks

- [ ] **Core Types**
  - [ ] Create `backend/cv/types.py` with Pydantic models
  - [ ] Add BBox coordinate conversion utilities
  - [ ] Add Mask numpy serialization/deserialization
  - [ ] Add type exports to `backend/cv/__init__.py`

- [ ] **Model Manager**
  - [ ] Create `backend/cv/manager.py` with singleton
  - [ ] Implement lazy loading with device selection
  - [ ] Add LRU-based model eviction when VRAM full
  - [ ] Add thread safety for async usage
  - [ ] Implement model registration decorator

- [ ] **Device Management**
  - [ ] Create `backend/cv/device.py`
  - [ ] Implement pynvml-based VRAM monitoring
  - [ ] Add nvidia-smi subprocess fallback
  - [ ] Create CUDAOOMHandler context manager
  - [ ] Add comprehensive device info function

- [ ] **Model Downloads**
  - [ ] Create `backend/cv/download.py`
  - [ ] Implement HuggingFace Hub integration
  - [ ] Add progress callback support
  - [ ] Create model registry with metadata
  - [ ] Add download verification (checksums)

- [ ] **Base Wrapper**
  - [ ] Create `backend/cv/base.py` with abstract class
  - [ ] Define standard interface (load/unload/predict)
  - [ ] Add batch processing with progress callbacks
  - [ ] Add model state tracking

- [ ] **Models Directory Setup**
  - [ ] Create `models/README.md` with download instructions
  - [ ] Add `models/.gitignore` to exclude weights
  - [ ] Document HF_TOKEN setup for gated models

## Acceptance Criteria

- [ ] `ModelManager.get("yolo11")` returns cached or newly loaded model
- [ ] `ModelManager.unload("yolo11")` frees VRAM and shows decrease in `get_vram_usage()`
- [ ] `get_vram_usage()` returns (used_mb, total_mb) tuple
- [ ] `download_model("sam3", progress_callback)` downloads with progress updates
- [ ] GPU OOM triggers CPU fallback gracefully (no crash)
- [ ] All Pydantic types validate correctly and serialize to JSON
- [ ] **VRAM Budget:** Infrastructure overhead <100MB
- [ ] **Performance:** Model load time <5s, type operations <1ms

## Files to Create

```
backend/cv/
├── __init__.py           # Package exports
├── types.py              # Pydantic models (Detection, Mask, etc.)
├── manager.py            # ModelManager singleton
├── device.py             # GPU/CPU detection, VRAM monitoring
├── download.py           # Model download utilities
└── base.py               # BaseModelWrapper abstract class

models/
├── .gitignore            # Ignore model weights
└── README.md             # Download instructions
```

## Testing

```python
# test_manager.py
def test_model_manager_singleton():
    m1 = ModelManager()
    m2 = ModelManager()
    assert m1 is m2

def test_lazy_loading(mock_model):
    manager = ModelManager()
    manager.register("test", mock_model)

    assert "test" not in manager.loaded_models
    model = manager.get("test")
    assert "test" in manager.loaded_models

def test_unload_frees_memory(mock_model):
    manager = ModelManager()
    manager.register("test", mock_model)

    initial_vram = get_vram_usage()[0]
    manager.get("test")
    loaded_vram = get_vram_usage()[0]
    manager.unload("test")
    final_vram = get_vram_usage()[0]

    assert loaded_vram > initial_vram
    assert final_vram <= initial_vram

# test_device.py
def test_vram_usage_returns_tuple():
    used, total = get_vram_usage()
    assert isinstance(used, int)
    assert isinstance(total, int)
    assert total >= used

def test_cuda_oom_handler_fallback():
    handler = CUDAOOMHandler(fallback_device="cpu")
    # ... test OOM handling
```
