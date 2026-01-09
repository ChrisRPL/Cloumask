# SCRFD Face Detection

> **Status:** Not Started
> **Priority:** P0 (Critical)
> **Dependencies:** [00-infrastructure.md](./00-infrastructure.md)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Face detection using SCRFD-10G (Sample and Computation Redistribution for Face Detection) as the primary model, with YuNet as a lightweight fallback. SCRFD provides state-of-the-art accuracy on WIDER FACE benchmark while maintaining fast inference. Used primarily for face anonymization pipeline.

## Goals

- [ ] Integrate SCRFD-10G from InsightFace package
- [ ] Support 5-point facial landmark detection
- [ ] Implement YuNet fallback (OpenCV built-in, real-time capable)
- [ ] Handle multiple faces with confidence scores
- [ ] Optional face embedding extraction for deduplication

## Technical Design

### Model Specifications

| Model | Size | VRAM | Inference | Accuracy |
|-------|------|------|-----------|----------|
| SCRFD-10G | ~16MB | ~1GB | 5-10ms | 95%+ WIDER FACE |
| YuNet | Built-in | ~100MB (CPU) | 1-5ms | ~90% WIDER FACE |

### SCRFD Wrapper

```python
from typing import Optional, Callable, List
from backend.cv.base import BaseModelWrapper
from backend.cv.types import FaceDetection, BBox
import numpy as np
import torch

class SCRFDWrapper(BaseModelWrapper):
    """SCRFD-10G face detection wrapper using InsightFace."""

    model_name = "scrfd-10g"
    vram_required_mb = 1500
    supports_batching = True

    def __init__(self):
        self._model = None
        self._device: str = "cpu"

    def load(self, device: str = "cuda") -> None:
        """Load SCRFD model from InsightFace."""
        from insightface.app import FaceAnalysis

        # FaceAnalysis loads SCRFD by default
        self._model = FaceAnalysis(
            name="buffalo_sc",  # SCRFD-based model pack
            providers=["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        )
        self._model.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
        self._device = device

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
        include_landmarks: bool = True,
        include_embedding: bool = False,
    ) -> List[FaceDetection]:
        """
        Detect faces in an image.

        Args:
            image_path: Path to input image
            confidence: Minimum confidence threshold
            include_landmarks: Include 5-point facial landmarks
            include_embedding: Include face embedding (for dedup)

        Returns:
            List of FaceDetection results
        """
        import cv2

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        faces = self._model.get(image_rgb)

        results = []
        for face in faces:
            if face.det_score < confidence:
                continue

            # Convert bbox to normalized coordinates
            x1, y1, x2, y2 = face.bbox
            bbox = BBox(
                x=(x1 + x2) / 2 / w,
                y=(y1 + y2) / 2 / h,
                width=(x2 - x1) / w,
                height=(y2 - y1) / h,
            )

            # Extract landmarks (5-point: left_eye, right_eye, nose, left_mouth, right_mouth)
            landmarks = None
            if include_landmarks and face.kps is not None:
                landmarks = [
                    (float(kp[0]) / w, float(kp[1]) / h)
                    for kp in face.kps
                ]

            detection = FaceDetection(
                bbox=bbox,
                confidence=float(face.det_score),
                landmarks=landmarks,
            )

            # Optionally add embedding as extra field
            if include_embedding and face.embedding is not None:
                detection.embedding = face.embedding.tolist()

            results.append(detection)

        # Sort by confidence descending
        results.sort(key=lambda f: f.confidence, reverse=True)

        return results

    def predict_batch(
        self,
        image_paths: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[List[FaceDetection]]:
        """Batch face detection with progress tracking."""
        results = []
        for i, path in enumerate(image_paths):
            results.append(self.predict(path, **kwargs))
            if progress_callback:
                progress_callback(i + 1, len(image_paths))
        return results
```

### YuNet Fallback Wrapper

```python
class YuNetWrapper(BaseModelWrapper):
    """YuNet face detection wrapper using OpenCV DNN."""

    model_name = "yunet"
    vram_required_mb = 200  # Runs on CPU
    supports_batching = False

    def __init__(self):
        self._model = None

    def load(self, device: str = "cpu") -> None:
        """
        Load YuNet model from OpenCV.
        Note: YuNet always runs on CPU via OpenCV DNN.
        """
        import cv2

        # YuNet model is bundled with recent OpenCV versions
        model_path = cv2.data.haarcascades + "../face_detection_yunet_2023mar.onnx"

        # If not found, download from OpenCV Zoo
        if not os.path.exists(model_path):
            model_path = self._download_yunet_model()

        self._model = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )

    def _download_yunet_model(self) -> str:
        """Download YuNet model from OpenCV Zoo."""
        import urllib.request
        from backend.cv.download import MODELS_DIR

        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        model_path = MODELS_DIR / "yunet" / "face_detection_yunet_2023mar.onnx"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        urllib.request.urlretrieve(url, model_path)
        return str(model_path)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        image_path: str,
        confidence: float = 0.5,
        include_landmarks: bool = True,
        **kwargs
    ) -> List[FaceDetection]:
        """
        Detect faces using YuNet.

        YuNet returns 15 values per face:
        - bbox: x, y, w, h
        - landmarks: 5 points (10 values)
        - confidence: 1 value
        """
        import cv2

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w = image.shape[:2]
        self._model.setInputSize((w, h))

        _, faces = self._model.detect(image)

        if faces is None:
            return []

        results = []
        for face in faces:
            if face[14] < confidence:
                continue

            # bbox: x, y, w, h
            x, y, fw, fh = face[0:4]
            bbox = BBox(
                x=(x + fw / 2) / w,
                y=(y + fh / 2) / h,
                width=fw / w,
                height=fh / h,
            )

            # Landmarks: 5 points
            landmarks = None
            if include_landmarks:
                landmarks = [
                    (float(face[4 + i * 2]) / w, float(face[5 + i * 2]) / h)
                    for i in range(5)
                ]

            results.append(FaceDetection(
                bbox=bbox,
                confidence=float(face[14]),
                landmarks=landmarks,
            ))

        results.sort(key=lambda f: f.confidence, reverse=True)
        return results
```

### Face Detector Factory

```python
def get_face_detector(
    realtime: bool = False,
    force_model: Optional[str] = None
) -> BaseModelWrapper:
    """
    Get appropriate face detector.

    Args:
        realtime: If True, prefer YuNet for speed
        force_model: Force specific model ("scrfd" or "yunet")

    Returns:
        Face detector wrapper
    """
    from backend.cv.device import get_available_vram_mb

    if force_model == "yunet":
        return YuNetWrapper()
    elif force_model == "scrfd":
        return SCRFDWrapper()

    if realtime:
        return YuNetWrapper()

    # Prefer SCRFD if GPU available
    if get_available_vram_mb() >= 1500:
        return SCRFDWrapper()

    return YuNetWrapper()
```

## Implementation Tasks

- [ ] **SCRFD Integration**
  - [ ] Create `backend/cv/faces.py`
  - [ ] Implement SCRFDWrapper with InsightFace
  - [ ] Add 5-point landmark extraction
  - [ ] Optional face embedding support
  - [ ] Batch processing with progress

- [ ] **YuNet Fallback**
  - [ ] Implement YuNetWrapper with OpenCV DNN
  - [ ] Handle model download from OpenCV Zoo
  - [ ] CPU-only operation support

- [ ] **Factory & Selection**
  - [ ] Implement get_face_detector()
  - [ ] Add realtime mode selection
  - [ ] GPU availability check

- [ ] **Testing**
  - [ ] Unit tests with mock
  - [ ] Integration tests (SCRFD on GPU)
  - [ ] Accuracy tests on sample faces
  - [ ] Performance benchmarks

## Acceptance Criteria

- [ ] `detect_faces(image)` returns List[FaceDetection] with bbox, confidence, landmarks
- [ ] Works on images with 0-100+ faces
- [ ] YuNet fallback works without GPU
- [ ] Landmarks accurate enough for blur alignment (within 5px of true position)
- [ ] Face embedding extraction works (for future deduplication)
- [ ] **VRAM Budget:** SCRFD-10G <1.5GB, YuNet <200MB (CPU)
- [ ] **Performance:** SCRFD <10ms/image, YuNet <5ms/image

## Files to Create

```
backend/cv/
└── faces.py   # SCRFDWrapper, YuNetWrapper, get_face_detector()
```

## Testing

```python
# test_faces.py
import pytest
from backend.cv.faces import SCRFDWrapper, YuNetWrapper, get_face_detector

@pytest.fixture
def scrfd():
    s = SCRFDWrapper()
    s.load("cuda" if torch.cuda.is_available() else "cpu")
    yield s
    s.unload()

def test_detect_single_face(scrfd, single_face_image):
    faces = scrfd.predict(single_face_image)
    assert len(faces) == 1
    assert faces[0].confidence > 0.9

def test_detect_multiple_faces(scrfd, group_photo):
    faces = scrfd.predict(group_photo)
    assert len(faces) >= 3

def test_no_faces_empty_result(scrfd, landscape_image):
    faces = scrfd.predict(landscape_image)
    assert len(faces) == 0

def test_landmarks_five_points(scrfd, single_face_image):
    faces = scrfd.predict(single_face_image, include_landmarks=True)
    assert faces[0].landmarks is not None
    assert len(faces[0].landmarks) == 5

def test_yunet_cpu_only():
    yunet = YuNetWrapper()
    yunet.load("cpu")
    assert yunet.is_loaded
    yunet.unload()

@pytest.mark.gpu
def test_vram_budget(scrfd):
    from backend.cv.device import get_vram_usage
    used, _ = get_vram_usage()
    assert used < 1500  # <1.5GB

@pytest.mark.gpu
def test_inference_speed(scrfd, single_face_image):
    import time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        scrfd.predict(single_face_image)
        times.append((time.perf_counter() - start) * 1000)

    assert sum(times) / len(times) < 10  # <10ms average
```
