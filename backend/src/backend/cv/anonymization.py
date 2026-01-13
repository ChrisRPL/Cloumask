"""
Anonymization pipeline for faces and license plates.

This module provides a complete image anonymization pipeline combining face
detection (SCRFD), license plate detection (YOLO-World), and optional SAM3-based
precise masking. Supports four anonymization modes: blur, blackbox, pixelate,
and mask.

Implements spec: 03-cv-models/06-anonymization

VRAM Budget (mask mode with sequential loading):
- Detection phase: SCRFD (1.5GB) + PlateDetector (4.5GB) = 6GB
- Masking phase: SCRFD (1.5GB) + SAM3 (8GB) = 9.5GB (under 10GB budget)

Example:
    from backend.cv.anonymization import anonymize

    # Quick API
    result = anonymize("photo.jpg", mode="blur")
    print(f"Anonymized {result.faces_anonymized} faces")

    # Full pipeline control
    config = AnonymizationConfig(faces=True, plates=True, mode="mask")
    pipeline = AnonymizationPipeline(config)
    pipeline.load("cuda")
    result = pipeline.process("photo.jpg", "output.jpg")
    pipeline.unload()
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

from backend.cv.device import clear_gpu_memory, get_device_info, select_device
from backend.cv.faces import get_face_detector
from backend.cv.plates import get_plate_detector
from backend.cv.types import BBox, FaceDetection, PlateDetection

if TYPE_CHECKING:
    from backend.cv.base import BaseModelWrapper
    from backend.cv.segmentation import SAM3Wrapper
    from backend.cv.types import FaceDetectionResult, PlateDetectionResult

# Type alias for numpy arrays (permissive for cv2 compatibility)
ImageArray = np.ndarray  # cv2 returns generic ndarray

logger = logging.getLogger(__name__)

# Type alias for anonymization modes
AnonymizationMode = Literal["blur", "blackbox", "pixelate", "mask"]


@dataclass
class AnonymizationConfig:
    """
    Configuration for anonymization pipeline.

    Attributes:
        faces: Whether to anonymize detected faces.
        plates: Whether to anonymize detected license plates.
        mode: Anonymization effect to apply.
        blur_kernel_size: Gaussian blur kernel size (must be odd).
        blackbox_color: RGB color for blackbox mode.
        pixelate_block_size: Block size for mosaic effect.
        mask_feather_radius: Radius for feathering mask edges (0 = hard edges).
        mask_effect: Effect to apply within mask boundaries.
        face_confidence: Minimum confidence for face detections.
        plate_confidence: Minimum confidence for plate detections.
        bbox_expansion: Expand bounding boxes by this fraction (0.1 = 10% each side).
    """

    faces: bool = True
    plates: bool = True
    mode: AnonymizationMode = "blur"

    # Blur settings
    blur_kernel_size: int = 51

    # Blackbox settings
    blackbox_color: tuple[int, int, int] = (0, 0, 0)

    # Pixelate settings
    pixelate_block_size: int = 10

    # Mask mode settings
    mask_feather_radius: int = 3
    mask_effect: Literal["blur", "blackbox", "pixelate"] = "blur"

    # Detection thresholds
    face_confidence: float = 0.5
    plate_confidence: float = 0.3

    # Bounding box expansion for better coverage
    bbox_expansion: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be odd")
        if self.blur_kernel_size < 1:
            raise ValueError("blur_kernel_size must be >= 1")
        if self.pixelate_block_size < 1:
            raise ValueError("pixelate_block_size must be >= 1")
        if self.mask_feather_radius < 0:
            raise ValueError("mask_feather_radius must be >= 0")
        if not 0 <= self.face_confidence <= 1:
            raise ValueError("face_confidence must be between 0 and 1")
        if not 0 <= self.plate_confidence <= 1:
            raise ValueError("plate_confidence must be between 0 and 1")
        if self.bbox_expansion < 0:
            raise ValueError("bbox_expansion must be >= 0")


@dataclass
class AnonymizationResult:
    """
    Result of an anonymization operation.

    Attributes:
        output_path: Path to the output image.
        faces_anonymized: Number of faces anonymized.
        plates_anonymized: Number of license plates anonymized.
        processing_time_ms: Total processing time in milliseconds.
        mode_used: Anonymization mode that was applied.
        face_detections: Optional list of face detections (for debugging).
        plate_detections: Optional list of plate detections (for debugging).
    """

    output_path: str
    faces_anonymized: int
    plates_anonymized: int
    processing_time_ms: float
    mode_used: AnonymizationMode
    face_detections: list[FaceDetection] | None = None
    plate_detections: list[PlateDetection] | None = None


class AnonymizationPipeline:
    """
    Complete anonymization pipeline combining face and plate detection.

    This is a composite pipeline that orchestrates multiple CV models:
    - Face detection: SCRFD (GPU) or YuNet (CPU fallback)
    - Plate detection: YOLO-World with "license plate" prompt
    - Segmentation (mask mode): SAM3 for precise boundaries

    VRAM is managed through sequential loading to stay under 10GB:
    1. Load face + plate detectors for detection phase
    2. Unload plate detector before loading SAM3 for masking phase

    Example:
        config = AnonymizationConfig(faces=True, plates=True, mode="blur")
        pipeline = AnonymizationPipeline(config)
        pipeline.load("cuda")

        result = pipeline.process("image.jpg", "output.jpg")
        print(f"Anonymized {result.faces_anonymized} faces")

        pipeline.unload()
    """

    def __init__(self, config: AnonymizationConfig | None = None) -> None:
        """
        Initialize anonymization pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or AnonymizationConfig()
        self._face_detector: BaseModelWrapper[FaceDetectionResult] | None = None
        self._plate_detector: BaseModelWrapper[PlateDetectionResult] | None = None
        self._segmenter: SAM3Wrapper | None = None
        self._device: str = "cpu"
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Check if pipeline models are loaded."""
        return self._is_loaded

    @property
    def device(self) -> str:
        """Get the device models are loaded on."""
        return self._device

    def load(self, device: str = "auto") -> None:
        """
        Load required models based on configuration.

        Models are loaded lazily based on config:
        - Face detector loaded if config.faces=True
        - Plate detector loaded if config.plates=True
        - SAM3 is loaded on-demand during masking to manage VRAM

        Args:
            device: Target device ("cuda", "cpu", "mps", or "auto").
        """
        if self._is_loaded:
            logger.warning("Pipeline already loaded, skipping")
            return

        if device == "auto":
            device = select_device()

        self._device = device
        logger.info("Loading anonymization pipeline on device: %s", device)

        # Load face detector with OOM fallback
        if self.config.faces:
            logger.info("Loading face detector...")
            self._face_detector = get_face_detector(realtime=False)
            try:
                self._face_detector.load(device)
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    logger.warning("GPU OOM loading face detector, falling back to CPU")
                    self._face_detector.load("cpu")
                    self._device = "cpu"
                else:
                    raise

        # Load plate detector with OOM fallback
        if self.config.plates:
            logger.info("Loading plate detector...")
            self._plate_detector = get_plate_detector()
            try:
                self._plate_detector.load(self._device)
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    logger.warning("GPU OOM loading plate detector, falling back to CPU")
                    self._plate_detector.load("cpu")
                    self._device = "cpu"
                else:
                    raise

        self._is_loaded = True
        logger.info("Anonymization pipeline loaded successfully on %s", self._device)

    def unload(self) -> None:
        """Unload all models and free VRAM."""
        if self._face_detector is not None:
            self._face_detector.unload()
            self._face_detector = None

        if self._plate_detector is not None:
            self._plate_detector.unload()
            self._plate_detector = None

        if self._segmenter is not None:
            self._segmenter.unload()
            self._segmenter = None

        self._is_loaded = False
        clear_gpu_memory()
        logger.info("Anonymization pipeline unloaded")

    def process(
        self,
        image_path: str,
        output_path: str | None = None,
        *,
        return_detections: bool = False,
    ) -> AnonymizationResult:
        """
        Anonymize a single image.

        Args:
            image_path: Path to input image.
            output_path: Path for output image. If None, adds "_anon" suffix.
            return_detections: Include detection lists in result for debugging.

        Returns:
            AnonymizationResult with statistics.

        Raises:
            RuntimeError: If pipeline not loaded.
            ValueError: If image cannot be read.
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        start = time.perf_counter()

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        img_h, img_w = image.shape[:2]

        # Collect detections
        detections: list[tuple[str, BBox]] = []
        face_list: list[FaceDetection] = []
        plate_list: list[PlateDetection] = []

        # Detect faces
        if self.config.faces and self._face_detector is not None:
            face_result = self._face_detector.predict(
                image_path,
                confidence=self.config.face_confidence,
            )
            for face in face_result.faces:
                detections.append(("face", face.bbox))
                face_list.append(face)

        # Detect plates (reload if unloaded during mask mode)
        if self.config.plates:
            self._ensure_plate_detector_loaded()
        if self.config.plates and self._plate_detector is not None:
            plate_result = self._plate_detector.predict(
                image_path,
                confidence=self.config.plate_confidence,
            )
            for plate in plate_result.plates:
                detections.append(("plate", plate.bbox))
                plate_list.append(plate)

        # Apply anonymization based on mode
        if self.config.mode == "mask":
            image = self._process_mask_mode(image, detections, img_w, img_h, image_path)
        else:
            for _det_type, bbox in detections:
                image = self._anonymize_region(image, bbox, img_w, img_h)

        # Generate output path if not provided
        if output_path is None:
            path = Path(image_path)
            output_path = str(path.parent / f"{path.stem}_anon{path.suffix}")

        # Save output
        success = cv2.imwrite(output_path, image)
        if not success:
            raise OSError(f"Failed to write output image: {output_path}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnonymizationResult(
            output_path=output_path,
            faces_anonymized=len(face_list),
            plates_anonymized=len(plate_list),
            processing_time_ms=elapsed_ms,
            mode_used=self.config.mode,
            face_detections=face_list if return_detections else None,
            plate_detections=plate_list if return_detections else None,
        )

    def process_batch(
        self,
        image_paths: list[str],
        output_dir: str | None = None,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        error_callback: Callable[[str, Exception], None] | None = None,
    ) -> list[AnonymizationResult]:
        """
        Process multiple images with progress tracking.

        Args:
            image_paths: List of input image paths.
            output_dir: Output directory (default: same dir with _anon suffix).
            progress_callback: Called with (processed_count, total_count).
            error_callback: Called with (image_path, exception) on errors.

        Returns:
            List of AnonymizationResult for successful images.
        """
        results: list[AnonymizationResult] = []
        total = len(image_paths)

        for i, image_path in enumerate(image_paths):
            try:
                # Determine output path
                if output_dir is not None:
                    out_path = str(Path(output_dir) / Path(image_path).name)
                else:
                    out_path = None

                result = self.process(image_path, out_path)
                results.append(result)

            except Exception as e:
                logger.error("Failed to process %s: %s", image_path, e)
                if error_callback is not None:
                    error_callback(image_path, e)

            # Report progress
            if progress_callback is not None:
                progress_callback(i + 1, total)

        return results

    def _anonymize_region(
        self,
        image: ImageArray,
        bbox: BBox,
        img_w: int,
        img_h: int,
    ) -> ImageArray:
        """
        Apply anonymization effect to a rectangular region.

        Args:
            image: Input image array (modified in place).
            bbox: Normalized bounding box.
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            Modified image array.
        """
        # Convert normalized bbox to pixel coordinates
        x1, y1, x2, y2 = bbox.to_xyxy(img_w, img_h)

        # Expand bbox if configured
        if self.config.bbox_expansion > 0:
            expand_w = int((x2 - x1) * self.config.bbox_expansion)
            expand_h = int((y2 - y1) * self.config.bbox_expansion)
            x1 -= expand_w
            y1 -= expand_h
            x2 += expand_w
            y2 += expand_h

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        # Skip invalid regions
        if x2 <= x1 or y2 <= y1:
            return image

        # Extract ROI and apply effect
        roi = image[y1:y2, x1:x2]
        roi = self._apply_effect(roi)
        image[y1:y2, x1:x2] = roi

        return image

    def _apply_effect(
        self,
        roi: ImageArray,
        effect: str | None = None,
    ) -> ImageArray:
        """
        Apply anonymization effect to region of interest.

        Args:
            roi: Region of interest array.
            effect: Effect to apply (default: config.mode).

        Returns:
            Modified ROI array.
        """
        mode = effect or self.config.mode

        if mode == "blur":
            return self._apply_blur(roi)
        elif mode == "blackbox":
            return self._apply_blackbox(roi)
        elif mode == "pixelate":
            return self._apply_pixelate(roi)
        elif mode == "mask":
            # For mask mode without actual mask, fall back to blur
            return self._apply_blur(roi)

        return roi

    def _apply_blur(self, roi: ImageArray) -> ImageArray:
        """Apply Gaussian blur effect."""
        k = self.config.blur_kernel_size
        return cv2.GaussianBlur(roi, (k, k), 0)

    def _apply_blackbox(self, roi: ImageArray) -> ImageArray:
        """Apply solid color fill effect."""
        color = self.config.blackbox_color
        roi[:] = color[::-1]  # RGB to BGR for OpenCV
        return roi

    def _apply_pixelate(self, roi: ImageArray) -> ImageArray:
        """Apply mosaic/pixelate effect."""
        block = self.config.pixelate_block_size
        h, w = roi.shape[:2]

        # Downsample
        small_w = max(1, w // block)
        small_h = max(1, h // block)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # Upsample with nearest neighbor for mosaic effect
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _process_mask_mode(
        self,
        image: ImageArray,
        detections: list[tuple[str, BBox]],
        img_w: int,
        img_h: int,
        image_path: str,
    ) -> ImageArray:
        """
        Process mask mode with SAM3 for precise boundaries.

        Uses sequential loading to stay under VRAM budget:
        1. Unload plate detector to free VRAM
        2. Load SAM3 for precise masking
        3. Apply masks with feathering
        4. Unload SAM3 (plate detector reloaded on next process call if needed)

        Args:
            image: Input image array.
            detections: List of (type, bbox) tuples.
            img_w: Image width.
            img_h: Image height.
            image_path: Path to image for SAM3.

        Returns:
            Modified image array.
        """
        if not detections:
            return image

        # Convert all bboxes to pixel coordinates first
        pixel_boxes: list[tuple[int, int, int, int]] = []
        for _, bbox in detections:
            x1, y1, x2, y2 = bbox.to_xyxy(img_w, img_h)
            # Expand if configured
            if self.config.bbox_expansion > 0:
                expand_w = int((x2 - x1) * self.config.bbox_expansion)
                expand_h = int((y2 - y1) * self.config.bbox_expansion)
                x1 -= expand_w
                y1 -= expand_h
                x2 += expand_w
                y2 += expand_h
            # Clamp to bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            if x2 > x1 and y2 > y1:
                pixel_boxes.append((x1, y1, x2, y2))

        if not pixel_boxes:
            return image

        # Unload plate detector to free VRAM for SAM3
        if self._plate_detector is not None:
            logger.info("Unloading plate detector for SAM3 VRAM management")
            self._plate_detector.unload()
            self._plate_detector = None
            clear_gpu_memory()

        # Load SAM3 for precise masking
        self._ensure_sam_loaded()

        # Apply mask effect to each detection
        for box in pixel_boxes:
            image = self._apply_mask_effect(image, box, image_path)

        # Unload SAM3 to free VRAM
        if self._segmenter is not None:
            self._segmenter.unload()
            self._segmenter = None
            clear_gpu_memory()

        return image

    def _ensure_plate_detector_loaded(self) -> None:
        """Reload plate detector if it was unloaded for VRAM management."""
        if self._plate_detector is None and self.config.plates:
            logger.info("Reloading plate detector...")
            self._plate_detector = get_plate_detector()
            try:
                self._plate_detector.load(self._device)
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    logger.warning("GPU OOM reloading plate detector, using CPU")
                    self._plate_detector.load("cpu")
                else:
                    raise

    def _ensure_sam_loaded(self) -> None:
        """Ensure SAM3 is loaded for mask mode."""
        if self._segmenter is None:
            from backend.cv.segmentation import SAM3Wrapper

            logger.info("Loading SAM3 for precise masking...")
            self._segmenter = SAM3Wrapper()
            try:
                self._segmenter.load(self._device)
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    logger.warning("GPU OOM loading SAM3, falling back to CPU")
                    self._segmenter.load("cpu")
                else:
                    raise

    def _apply_mask_effect(
        self,
        image: ImageArray,
        bbox_pixels: tuple[int, int, int, int],
        image_path: str,
    ) -> ImageArray:
        """
        Apply effect within SAM3 mask boundary.

        Args:
            image: Input image array.
            bbox_pixels: Bounding box as (x1, y1, x2, y2) pixels.
            image_path: Path to image for SAM3.

        Returns:
            Modified image array.
        """
        x1, y1, x2, y2 = bbox_pixels

        # Get SAM3 mask using box prompt
        mask = self._get_sam_mask(image_path, bbox_pixels)

        if mask is None:
            # Fallback to rectangular region
            roi = image[y1:y2, x1:x2]
            roi = self._apply_effect(roi, self.config.mask_effect)
            image[y1:y2, x1:x2] = roi
            return image

        # Resize mask to match ROI if needed
        roi_h, roi_w = y2 - y1, x2 - x1
        if mask.shape[0] != roi_h or mask.shape[1] != roi_w:
            mask = cv2.resize(mask, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

        # Extract ROI
        roi = image[y1:y2, x1:x2].copy()

        # Apply effect to entire ROI
        roi_effected = self._apply_effect(roi.copy(), self.config.mask_effect)

        # Feather mask edges if configured
        if self.config.mask_feather_radius > 0:
            kernel_size = self.config.mask_feather_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

        # Normalize mask to [0, 1] for blending
        mask_norm = mask.astype(np.float32) / 255.0
        if mask_norm.ndim == 2:
            mask_norm = mask_norm[:, :, np.newaxis]

        # Blend: effect where mask=1, original where mask=0
        blended = (roi_effected * mask_norm + roi * (1 - mask_norm)).astype(np.uint8)
        image[y1:y2, x1:x2] = blended

        return image

    def _get_sam_mask(
        self,
        image_path: str,
        bbox_pixels: tuple[int, int, int, int],
    ) -> ImageArray | None:
        """
        Get precise segmentation mask from SAM3.

        Args:
            image_path: Path to image.
            bbox_pixels: Bounding box as (x1, y1, x2, y2) pixels.

        Returns:
            Binary mask array (uint8, 0-255), or None on failure.
        """
        if self._segmenter is None:
            return None

        try:
            result = self._segmenter.predict(
                image_path,
                box=bbox_pixels,
            )

            if result.masks:
                # Get highest confidence mask
                mask_data = result.masks[0].to_numpy()
                # Convert to uint8 (0-255)
                return (mask_data * 255).astype(np.uint8)

        except Exception as e:
            logger.warning("SAM3 mask failed for box %s: %s", bbox_pixels, e)

        return None


def anonymize(
    image_path: str,
    output_path: str | None = None,
    *,
    faces: bool = True,
    plates: bool = True,
    mode: AnonymizationMode = "blur",
    device: str = "auto",
    **kwargs: Any,
) -> AnonymizationResult:
    """
    Quick anonymization function for single images.

    Convenience function that creates, loads, processes, and unloads
    the pipeline automatically. For batch processing or repeated use,
    create an AnonymizationPipeline instance directly.

    Args:
        image_path: Path to input image.
        output_path: Output path (default: adds "_anon" suffix).
        faces: Whether to anonymize faces.
        plates: Whether to anonymize license plates.
        mode: Anonymization mode ("blur", "blackbox", "pixelate", "mask").
        device: Target device ("cuda", "cpu", "mps", or "auto").
        **kwargs: Additional AnonymizationConfig parameters.

    Returns:
        AnonymizationResult with statistics.

    Example:
        result = anonymize("photo.jpg", mode="blur")
        print(f"Anonymized {result.faces_anonymized} faces")

        result = anonymize("street.jpg", mode="pixelate", pixelate_block_size=20)
    """
    if device == "auto":
        info = get_device_info()
        device = info.best_device

    config = AnonymizationConfig(
        faces=faces,
        plates=plates,
        mode=mode,
        **kwargs,
    )

    pipeline = AnonymizationPipeline(config)
    pipeline.load(device)

    try:
        return pipeline.process(image_path, output_path)
    finally:
        pipeline.unload()
