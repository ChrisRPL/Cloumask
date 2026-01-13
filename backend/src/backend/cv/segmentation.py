"""
SAM3, SAM2, and MobileSAM segmentation wrappers.

This module provides SAM3 as the primary segmenter with text prompt support,
SAM2 as a faster fallback with point/box prompts, and MobileSAM for real-time
lightweight segmentation.

Implements spec: 03-cv-models/02-sam3-segmentation

Model Hierarchy:
- SAM3: Text prompts + visual prompts (~8GB VRAM, 300-500ms)
- SAM2: Point/box prompts only (~6GB VRAM, 100-200ms)
- MobileSAM: Point prompts only (~1GB VRAM, 50-100ms)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.cv.base import BaseModelWrapper, ModelInfo, register_model
from backend.cv.device import CUDAOOMHandler
from backend.cv.types import Mask, SegmentationResult

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ultralytics import SAM

logger = logging.getLogger(__name__)


def _convert_sam_masks_to_result(
    masks: NDArray[np.uint8] | Any,
    scores: NDArray[np.float32] | Any,
    image_path: str,
    elapsed_ms: float,
    model_name: str,
    prompt: str | None = None,
) -> SegmentationResult:
    """
    Convert SAM output masks to SegmentationResult.

    Args:
        masks: Binary masks array [N, H, W] or ultralytics results.
        scores: Confidence scores array [N].
        image_path: Path to input image.
        elapsed_ms: Processing time in milliseconds.
        model_name: Name of the model used.
        prompt: Optional text prompt used.

    Returns:
        SegmentationResult with converted masks.
    """
    mask_list: list[Mask] = []

    # Handle case where masks/scores are tensors
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()

    # Ensure masks is iterable along first dimension
    if masks is not None and len(masks) > 0:
        for i in range(len(masks)):
            mask_data = masks[i]
            # Handle 3D masks (C, H, W) by taking first channel
            if mask_data.ndim == 3:
                mask_data = mask_data[0]
            # Ensure binary uint8
            mask_binary = (mask_data > 0.5).astype(np.uint8)
            confidence = float(scores[i]) if i < len(scores) else 0.5
            mask_list.append(Mask.from_numpy(mask_binary, confidence))

    # Sort by confidence (highest first)
    mask_list.sort(key=lambda m: m.confidence, reverse=True)

    return SegmentationResult(
        masks=mask_list,
        image_path=image_path,
        processing_time_ms=elapsed_ms,
        model_name=model_name,
        prompt=prompt,
    )


@register_model
class SAM3Wrapper(BaseModelWrapper[SegmentationResult]):
    """
    SAM3 text-prompted segmentation wrapper.

    Primary segmentation model supporting natural language prompts like
    "red car" or "person on left". Also supports point/box prompts for
    compatibility with other SAM models.

    Uses Ultralytics SAM3SemanticPredictor for text prompts which enables
    open-vocabulary segmentation with 4M+ concepts.

    Attributes:
        info: Model metadata with VRAM requirements and capabilities.

    Example:
        sam3 = SAM3Wrapper()
        sam3.load("cuda")
        result = sam3.predict("image.jpg", prompt="red car")
        sam3.unload()
    """

    info = ModelInfo(
        name="sam3",
        description="SAM3 text-prompted segmentation (4M+ concepts)",
        vram_required_mb=8000,
        supports_batching=False,
        supports_gpu=True,
        source="ultralytics",
        version="3.0",
        extra={"supports_text_prompts": True},
    )

    def __init__(self) -> None:
        """Initialize SAM3 wrapper."""
        super().__init__()
        self._predictor: Any = None

    def _load_model(self, device: str) -> None:
        """
        Load SAM3 model with semantic predictor for text prompts.

        Args:
            device: Target device ("cuda", "cpu", "mps").

        Note:
            SAM3 weights must be downloaded from HuggingFace (gated model).
            Set HF_TOKEN environment variable if required.
            Falls back to CPU if GPU runs out of memory.
        """
        from ultralytics.models.sam import SAM3SemanticPredictor  # type: ignore[attr-defined]

        from backend.cv.download import get_model_path

        model_path = get_model_path("sam3")

        logger.info("Loading SAM3 from %s", model_path)

        def create_predictor(dev: str) -> None:
            # Configure predictor overrides
            overrides = {
                "conf": 0.25,
                "task": "segment",
                "mode": "predict",
                "model": str(model_path),
                "half": dev == "cuda",  # FP16 for GPU
                "save": False,
                "verbose": False,
                "device": dev if dev != "cuda" else 0,
            }
            self._predictor = SAM3SemanticPredictor(overrides=overrides)
            self._model = self._predictor

        # Use OOM handler to catch GPU memory errors and fallback to CPU
        oom_handler = CUDAOOMHandler(fallback_device="cpu", callback=create_predictor)
        with oom_handler:
            create_predictor(device)

        if oom_handler.used_fallback:
            logger.warning("SAM3 fell back to CPU due to GPU OOM")
            self._device = "cpu"

        logger.debug("SAM3 loaded on %s", self._device or device)

    def _unload_model(self) -> None:
        """Unload model and free memory."""
        if self._predictor is not None:
            del self._predictor
            self._predictor = None

    def predict(
        self,
        input_path: str,
        *,
        prompt: str | None = None,
        point: tuple[int, int] | None = None,
        box: tuple[int, int, int, int] | None = None,
        confidence: float = 0.25,
        **kwargs: Any,
    ) -> SegmentationResult:
        """
        Segment image using text, point, or box prompt.

        SAM3 supports natural language prompts for open-vocabulary segmentation.
        Point and box prompts are also supported for compatibility.

        Args:
            input_path: Path to input image.
            prompt: Text prompt (e.g., "red car", "person on left").
            point: Point prompt as (x, y) pixel coordinates.
            box: Box prompt as (x1, y1, x2, y2) pixel coordinates.
            confidence: Minimum confidence threshold (0-1) for filtering masks.
            **kwargs: Additional arguments (ignored).

        Returns:
            SegmentationResult with masks ranked by confidence, filtered by threshold.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If no prompt type provided.
        """
        if not self.is_loaded or self._predictor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if prompt is None and point is None and box is None:
            raise ValueError("Must provide prompt, point, or box")

        start = time.perf_counter()

        # Set image for processing
        self._predictor.set_image(input_path)

        # Run inference based on prompt type
        if prompt is not None:
            # Text prompt - SAM3's unique capability
            results = self._predictor(text=[prompt])
        elif box is not None:
            # Box prompt
            results = self._predictor(bboxes=[list(box)])
        elif point is not None:
            # Point prompt
            results = self._predictor(points=[[point[0], point[1]]], labels=[1])

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract masks and scores from results
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None:
                masks = result.masks.data
                # Get confidence scores - use masks.conf if available, else use 1.0
                if hasattr(result.masks, "conf") and result.masks.conf is not None:
                    scores = result.masks.conf
                elif hasattr(result, "boxes") and result.boxes is not None:
                    scores = result.boxes.conf
                else:
                    scores = np.ones(len(masks))

                seg_result = _convert_sam_masks_to_result(
                    masks=masks,
                    scores=scores,
                    image_path=input_path,
                    elapsed_ms=elapsed_ms,
                    model_name=self.info.name,
                    prompt=prompt,
                )

                # Filter masks by confidence threshold
                if confidence > 0:
                    filtered_masks = [m for m in seg_result.masks if m.confidence >= confidence]
                    return SegmentationResult(
                        masks=filtered_masks,
                        image_path=input_path,
                        processing_time_ms=elapsed_ms,
                        model_name=self.info.name,
                        prompt=prompt,
                    )
                return seg_result

        # No masks found
        return SegmentationResult(
            masks=[],
            image_path=input_path,
            processing_time_ms=elapsed_ms,
            model_name=self.info.name,
            prompt=prompt,
        )


@register_model
class SAM2Wrapper(BaseModelWrapper[SegmentationResult]):
    """
    SAM2 point/box-prompted segmentation wrapper.

    Fallback segmenter that is 6x faster than original SAM. Supports point
    and box prompts but NOT text prompts. Use SAM3 for text-based segmentation.

    Attributes:
        info: Model metadata with VRAM requirements and capabilities.

    Example:
        sam2 = SAM2Wrapper()
        sam2.load("cuda")
        result = sam2.predict("image.jpg", point=(100, 200))
        sam2.unload()
    """

    info = ModelInfo(
        name="sam2",
        description="SAM2 fast segmentation (point/box prompts, 6x faster)",
        vram_required_mb=6000,
        supports_batching=False,
        supports_gpu=True,
        source="ultralytics",
        version="2.1",
        extra={"supports_text_prompts": False},
    )

    def __init__(self) -> None:
        """Initialize SAM2 wrapper."""
        super().__init__()
        self._sam: SAM | None = None

    def _load_model(self, device: str) -> None:
        """
        Load SAM2 model.

        Args:
            device: Target device ("cuda", "cpu", "mps").
            Falls back to CPU if GPU runs out of memory.
        """
        from ultralytics import SAM

        from backend.cv.download import get_model_path

        model_path = get_model_path("sam2")

        logger.info("Loading SAM2 from %s", model_path)
        self._sam = SAM(str(model_path))

        def move_to_device(dev: str) -> None:
            if dev == "cuda":
                import torch

                if torch.cuda.is_available():
                    self._sam.to(dev)  # type: ignore[union-attr]
                else:
                    logger.warning("CUDA requested but not available, using CPU")
            elif dev != "cpu":
                self._sam.to(dev)  # type: ignore[union-attr]

        # Use OOM handler to catch GPU memory errors and fallback to CPU
        oom_handler = CUDAOOMHandler(fallback_device="cpu", callback=move_to_device)
        with oom_handler:
            move_to_device(device)

        if oom_handler.used_fallback:
            logger.warning("SAM2 fell back to CPU due to GPU OOM")
            self._device = "cpu"

        self._model = self._sam
        logger.debug("SAM2 loaded on %s", self._device or device)

    def _unload_model(self) -> None:
        """Unload model and free memory."""
        if self._sam is not None:
            del self._sam
            self._sam = None

    def predict(
        self,
        input_path: str,
        *,
        prompt: str | None = None,
        point: tuple[int, int] | None = None,
        box: tuple[int, int, int, int] | None = None,
        **kwargs: Any,
    ) -> SegmentationResult:
        """
        Segment image using point or box prompt.

        SAM2 does NOT support text prompts. Use SAM3 for text-based segmentation.

        Args:
            input_path: Path to input image.
            prompt: Text prompt (NOT SUPPORTED - raises NotImplementedError).
            point: Point prompt as (x, y) pixel coordinates.
            box: Box prompt as (x1, y1, x2, y2) pixel coordinates.
            **kwargs: Additional arguments (ignored).

        Returns:
            SegmentationResult with masks ranked by confidence.

        Raises:
            RuntimeError: If model not loaded.
            NotImplementedError: If text prompt provided.
            ValueError: If no valid prompt provided.
        """
        if not self.is_loaded or self._sam is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if prompt is not None:
            raise NotImplementedError(
                "SAM2 does not support text prompts. Use SAM3 or provide point/box."
            )

        if point is None and box is None:
            raise ValueError("SAM2 requires point or box prompt")

        start = time.perf_counter()

        # Run inference based on prompt type
        if box is not None:
            results = self._sam.predict(
                input_path,
                bboxes=[list(box)],
                device=self._device,
                verbose=False,
            )
        elif point is not None:
            results = self._sam.predict(
                input_path,
                points=[[point[0], point[1]]],
                labels=[1],  # 1 = foreground point
                device=self._device,
                verbose=False,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract masks and scores
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None:
                masks = result.masks.data
                # Get confidence scores
                if hasattr(result, "boxes") and result.boxes is not None and result.boxes.conf is not None:
                    scores = result.boxes.conf
                else:
                    scores = np.ones(len(masks))

                return _convert_sam_masks_to_result(
                    masks=masks,
                    scores=scores,
                    image_path=input_path,
                    elapsed_ms=elapsed_ms,
                    model_name=self.info.name,
                )

        # No masks found
        return SegmentationResult(
            masks=[],
            image_path=input_path,
            processing_time_ms=elapsed_ms,
            model_name=self.info.name,
            prompt=None,
        )


@register_model
class MobileSAMWrapper(BaseModelWrapper[SegmentationResult]):
    """
    MobileSAM lightweight segmentation wrapper.

    Ultra-fast segmenter optimized for real-time use. Supports only point
    prompts for maximum speed. Uses ~1GB VRAM and achieves <100ms inference.

    Attributes:
        info: Model metadata with VRAM requirements and capabilities.

    Example:
        mobile = MobileSAMWrapper()
        mobile.load("cuda")
        result = mobile.predict("image.jpg", point=(100, 200))
        mobile.unload()
    """

    info = ModelInfo(
        name="mobilesam",
        description="MobileSAM real-time segmentation (point prompts, <100ms)",
        vram_required_mb=1500,
        supports_batching=False,
        supports_gpu=True,
        source="ultralytics",
        version="1.0",
        extra={"supports_text_prompts": False},
    )

    def __init__(self) -> None:
        """Initialize MobileSAM wrapper."""
        super().__init__()
        self._sam: SAM | None = None

    def _load_model(self, device: str) -> None:
        """
        Load MobileSAM model.

        Args:
            device: Target device ("cuda", "cpu", "mps").
            Falls back to CPU if GPU runs out of memory.
        """
        from ultralytics import SAM

        from backend.cv.download import get_model_path

        model_path = get_model_path("mobilesam")

        logger.info("Loading MobileSAM from %s", model_path)
        self._sam = SAM(str(model_path))

        def move_to_device(dev: str) -> None:
            if dev == "cuda":
                import torch

                if torch.cuda.is_available():
                    self._sam.to(dev)  # type: ignore[union-attr]
                else:
                    logger.warning("CUDA requested but not available, using CPU")
            elif dev != "cpu":
                self._sam.to(dev)  # type: ignore[union-attr]

        # Use OOM handler to catch GPU memory errors and fallback to CPU
        oom_handler = CUDAOOMHandler(fallback_device="cpu", callback=move_to_device)
        with oom_handler:
            move_to_device(device)

        if oom_handler.used_fallback:
            logger.warning("MobileSAM fell back to CPU due to GPU OOM")
            self._device = "cpu"

        self._model = self._sam
        logger.debug("MobileSAM loaded on %s", self._device or device)

    def _unload_model(self) -> None:
        """Unload model and free memory."""
        if self._sam is not None:
            del self._sam
            self._sam = None

    def predict(
        self,
        input_path: str,
        *,
        prompt: str | None = None,
        point: tuple[int, int] | None = None,
        box: tuple[int, int, int, int] | None = None,
        **kwargs: Any,
    ) -> SegmentationResult:
        """
        Segment image using point prompt.

        MobileSAM supports only point prompts for maximum speed.
        Use SAM2 for box prompts or SAM3 for text prompts.

        Args:
            input_path: Path to input image.
            prompt: Text prompt (NOT SUPPORTED - raises NotImplementedError).
            point: Point prompt as (x, y) pixel coordinates.
            box: Box prompt (NOT SUPPORTED by MobileSAM - raises NotImplementedError).
            **kwargs: Additional arguments (ignored).

        Returns:
            SegmentationResult with masks ranked by confidence.

        Raises:
            RuntimeError: If model not loaded.
            NotImplementedError: If text or box prompt provided.
            ValueError: If no point prompt provided.
        """
        if not self.is_loaded or self._sam is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if prompt is not None:
            raise NotImplementedError(
                "MobileSAM does not support text prompts. Use SAM3."
            )

        if box is not None:
            raise NotImplementedError(
                "MobileSAM does not support box prompts for speed. Use SAM2."
            )

        if point is None:
            raise ValueError("MobileSAM requires point prompt")

        start = time.perf_counter()

        results = self._sam.predict(
            input_path,
            points=[[point[0], point[1]]],
            labels=[1],  # 1 = foreground point
            device=self._device,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract masks and scores
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None:
                masks = result.masks.data
                # Get confidence scores
                if hasattr(result, "boxes") and result.boxes is not None and result.boxes.conf is not None:
                    scores = result.boxes.conf
                else:
                    scores = np.ones(len(masks))

                return _convert_sam_masks_to_result(
                    masks=masks,
                    scores=scores,
                    image_path=input_path,
                    elapsed_ms=elapsed_ms,
                    model_name=self.info.name,
                )

        # No masks found
        return SegmentationResult(
            masks=[],
            image_path=input_path,
            processing_time_ms=elapsed_ms,
            model_name=self.info.name,
            prompt=None,
        )


def get_segmenter(
    prompt_type: str = "text",
    prefer_speed: bool = False,
    force_model: str | None = None,
) -> BaseModelWrapper[SegmentationResult]:
    """
    Get appropriate segmentation model based on requirements.

    Factory function that selects the best segmenter for the given prompt type
    and speed preference. Automatically considers available VRAM.

    Args:
        prompt_type: Type of prompt to use ("text", "point", or "box").
        prefer_speed: If True, prefer MobileSAM over SAM2 for point prompts.
        force_model: Force specific model ("sam3", "sam2", or "mobilesam").

    Returns:
        Appropriate segmenter wrapper (unloaded - call load() before use).

    Raises:
        RuntimeError: If text prompts requested but insufficient VRAM for SAM3.
        ValueError: If force_model is unknown.

    Example:
        # Text-prompted segmentation
        seg = get_segmenter(prompt_type="text")
        seg.load()
        result = seg.predict("image.jpg", prompt="red car")
        seg.unload()

        # Fast point-based segmentation
        seg = get_segmenter(prompt_type="point", prefer_speed=True)
        seg.load()
        result = seg.predict("image.jpg", point=(100, 200))
        seg.unload()
    """
    from backend.cv.device import get_available_vram_mb

    # Force specific model if requested
    if force_model:
        if force_model == "sam3":
            logger.info("Returning SAM3 segmenter (forced)")
            return SAM3Wrapper()
        elif force_model == "sam2":
            logger.info("Returning SAM2 segmenter (forced)")
            return SAM2Wrapper()
        elif force_model == "mobilesam":
            logger.info("Returning MobileSAM segmenter (forced)")
            return MobileSAMWrapper()
        else:
            raise ValueError(
                f"Unknown model: {force_model}. Available: sam3, sam2, mobilesam"
            )

    # Text prompts require SAM3
    if prompt_type == "text":
        available = get_available_vram_mb()
        if available >= SAM3Wrapper.info.vram_required_mb:
            logger.info(
                "Selecting SAM3 for text prompts (VRAM available: %dMB)",
                available,
            )
            return SAM3Wrapper()
        raise RuntimeError(
            f"SAM3 requires {SAM3Wrapper.info.vram_required_mb}MB VRAM for text prompts, "
            f"only {available}MB available. Use point/box prompts instead."
        )

    # Point prompts - prefer speed or check VRAM
    if prompt_type == "point":
        available = get_available_vram_mb()
        if prefer_speed or available < SAM2Wrapper.info.vram_required_mb:
            logger.info(
                "Selecting MobileSAM for speed (VRAM available: %dMB)",
                available,
            )
            return MobileSAMWrapper()
        logger.info(
            "Selecting SAM2 for point prompts (VRAM available: %dMB)",
            available,
        )
        return SAM2Wrapper()

    # Box prompts - need SAM2 or SAM3 (MobileSAM doesn't support box prompts)
    if prompt_type == "box":
        available = get_available_vram_mb()
        if available >= SAM2Wrapper.info.vram_required_mb:
            logger.info(
                "Selecting SAM2 for box prompts (VRAM available: %dMB)",
                available,
            )
            return SAM2Wrapper()
        raise RuntimeError(
            f"SAM2 requires {SAM2Wrapper.info.vram_required_mb}MB VRAM for box prompts, "
            f"only {available}MB available. MobileSAM does not support box prompts. "
            "Use point prompts instead or free up GPU memory."
        )

    # Default to SAM2
    logger.info("Returning SAM2 segmenter (default)")
    return SAM2Wrapper()
