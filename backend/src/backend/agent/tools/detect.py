"""Detection tool for object detection in images.

Uses YOLO11 (primary) or RT-DETR (fallback) for real inference.
Supports 80 COCO classes with configurable confidence threshold.
Auto-selects YOLO-World for open-vocabulary detection (non-COCO classes).
Quality mode routes through SAM3 for superior detection-via-segmentation.

Implements spec: 03-cv-models/01-yolo11-detection, 03-cv-models/08-cv-tools
Integration points: backend/cv/detection.py, backend/cv/openvocab.py, backend/cv/segmentation.py
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.constants import IMAGE_EXTENSIONS
from backend.agent.tools.registry import register_tool

if TYPE_CHECKING:
    from backend.cv.types import SegmentationResult

logger = logging.getLogger(__name__)

MAX_PREVIEW_ITEMS = 6
MAX_PREVIEW_ANNOTATIONS = 50


def _is_mps_runtime_error(error: Exception) -> bool:
    """Return True when inference fails due to known Apple MPS runtime issues."""
    message = str(error).lower()
    return (
        "placeholder storage" in message and "mps" in message
    ) or "mps backend out of memory" in message


def _needs_openvocab(classes: list[str] | None, coco_classes: set[str]) -> bool:
    """
    Check if any requested class requires open-vocabulary detection.

    Args:
        classes: List of class names to detect.
        coco_classes: Set of lowercase COCO class names.

    Returns:
        True if any class is not in COCO classes.
    """
    if not classes:
        return False
    return any(c.lower() not in coco_classes for c in classes)


def _masks_to_detections(
    seg_result: SegmentationResult,
    classes: list[str],
) -> tuple[list[dict], dict[str, int]]:
    """
    Convert segmentation masks to detection-style bounding boxes.

    Args:
        seg_result: Segmentation result with masks.
        classes: Class names used in the prompt.

    Returns:
        Tuple of (list of detection dicts, class counts dict).
    """
    import numpy as np

    detections = []
    class_counts: dict[str, int] = {}

    for i, mask in enumerate(seg_result.masks):
        # Get mask data as numpy array
        mask_np = mask.to_numpy()

        # Find bounding box from mask
        coords = np.where(mask_np > 0)
        if len(coords[0]) == 0:
            continue

        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        h, w = mask_np.shape

        # Normalized center format
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # Assign class name (cycle through if more masks than classes)
        class_name = classes[i % len(classes)] if classes and len(classes) > 0 else "object"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        detections.append({
            "class_name": class_name,
            "confidence": mask.confidence,
            "bbox": {"x": cx, "y": cy, "width": bw, "height": bh},
        })

    return detections, class_counts


def _clamp_unit(value: float) -> float:
    """Clamp numeric values to [0, 1] for normalized geometry payloads."""
    return max(0.0, min(1.0, value))


def _center_bbox_to_top_left(
    bbox: dict[str, Any] | None,
) -> dict[str, float]:
    """
    Convert normalized center-format bbox to normalized top-left format.

    Input keys are expected as:
    - x: center x
    - y: center y
    - width: box width
    - height: box height
    """
    if not isinstance(bbox, dict):
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}

    cx = _clamp_unit(float(bbox.get("x", 0.0)))
    cy = _clamp_unit(float(bbox.get("y", 0.0)))
    width = _clamp_unit(float(bbox.get("width", 0.0)))
    height = _clamp_unit(float(bbox.get("height", 0.0)))

    x = _clamp_unit(cx - width / 2.0)
    y = _clamp_unit(cy - height / 2.0)

    width = min(width, 1.0 - x)
    height = min(height, 1.0 - y)

    return {
        "x": round(x, 6),
        "y": round(y, 6),
        "width": round(max(0.0, width), 6),
        "height": round(max(0.0, height), 6),
    }


def _build_preview_items_from_results(results: list[Any]) -> list[dict[str, Any]]:
    """Build compact preview payload for frontend overlay rendering."""
    preview_items: list[dict[str, Any]] = []

    for result in results[:MAX_PREVIEW_ITEMS]:
        annotations: list[dict[str, Any]] = []
        detections = getattr(result, "detections", []) or []

        for detection in detections[:MAX_PREVIEW_ANNOTATIONS]:
            annotations.append({
                "label": str(getattr(detection, "class_name", "object")),
                "confidence": round(float(getattr(detection, "confidence", 1.0)), 4),
                "bbox": _center_bbox_to_top_left({
                    "x": getattr(getattr(detection, "bbox", None), "x", 0.0),
                    "y": getattr(getattr(detection, "bbox", None), "y", 0.0),
                    "width": getattr(getattr(detection, "bbox", None), "width", 0.0),
                    "height": getattr(getattr(detection, "bbox", None), "height", 0.0),
                }),
            })

        preview_items.append({
            "image_path": str(getattr(result, "image_path", "")),
            "annotations": annotations,
        })

    return preview_items


def _build_preview_items_from_annotation_records(
    records: list[tuple[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Build preview payload from detection-like annotation records."""
    preview_items: list[dict[str, Any]] = []

    for image_path, detections in records[:MAX_PREVIEW_ITEMS]:
        annotations: list[dict[str, Any]] = []
        for detection in detections[:MAX_PREVIEW_ANNOTATIONS]:
            confidence_raw = detection.get("confidence", 1.0)
            confidence = float(confidence_raw) if isinstance(confidence_raw, int | float) else 1.0
            annotations.append({
                "label": str(detection.get("class_name", "object")),
                "confidence": round(_clamp_unit(confidence), 4),
                "bbox": _center_bbox_to_top_left(detection.get("bbox")),  # type: ignore[arg-type]
            })

        preview_items.append({
            "image_path": str(image_path),
            "annotations": annotations,
        })

    return preview_items


@register_tool
class DetectTool(BaseTool):
    """Detect objects in images using YOLO11, YOLO-World, or SAM3."""

    name = "detect"
    description = """Run object detection on images to find and label objects.

Model Selection:
- COCO classes (person, car, etc.): Uses YOLO11m (fast) or RT-DETR (accurate)
- Custom classes (red car, delivery truck): Auto-selects YOLO-World (open-vocab)
- quality=True: Routes through SAM3 for superior detection (~8GB VRAM)

Examples:
- detect(path, classes=["person", "car"]) → YOLO11
- detect(path, classes=["red car", "delivery truck"]) → YOLO-World (auto)
- detect(path, classes=["person"], quality=True) → SAM3"""
    category = ToolCategory.DETECTION

    parameters = [
        ToolParameter(
            name="input_path",
            type=str,
            description="Path to input image or directory",
            required=True,
        ),
        ToolParameter(
            name="classes",
            type=list,
            description="List of class names to detect (e.g., ['person', 'car']). "
            "Supports COCO classes + any custom descriptions for open-vocab detection.",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="confidence",
            type=float,
            description="Minimum confidence threshold (0-1)",
            required=False,
            default=0.5,
        ),
        ToolParameter(
            name="prefer_accuracy",
            type=bool,
            description="Use RT-DETR/GroundingDINO for higher accuracy (slower, more VRAM)",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="quality",
            type=bool,
            description="Use SAM3 for superior detection-via-segmentation (~8GB VRAM, slower but best results)",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="save_annotations",
            type=bool,
            description="Whether to save detection annotations",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="output_path",
            type=str,
            description=(
                "Optional output directory for generated YOLO annotations "
                "(defaults to <input>_detections_yolo)"
            ),
            required=False,
            default=None,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        classes: list[str] | None = None,
        confidence: float = 0.5,
        prefer_accuracy: bool = False,
        quality: bool = False,
        save_annotations: bool = True,
        output_path: str | None = None,
    ) -> ToolResult:
        """
        Execute object detection using CV models.

        Automatically selects the best model based on parameters:
        - quality=True: SAM3 (detection-via-segmentation, best results)
        - Non-COCO classes: YOLO-World (open-vocabulary)
        - COCO classes: YOLO11 (fast) or RT-DETR (accurate)

        Args:
            input_path: Path to input image or directory.
            classes: List of class names to detect.
            confidence: Minimum confidence threshold (0-1).
            prefer_accuracy: Use RT-DETR/GroundingDINO instead of YOLO11/YOLO-World.
            quality: Use SAM3 for superior detection-via-segmentation.
            save_annotations: Whether to save detection annotations.

        Returns:
            ToolResult with detection statistics.
        """
        from backend.cv.detection import COCO_CLASSES

        input_p = Path(input_path)

        if not input_p.exists():
            return error_result(f"Input not found: {input_path}")

        if confidence < 0 or confidence > 1:
            return error_result(
                f"Invalid confidence: {confidence}. Must be between 0 and 1."
            )

        # Collect image files
        image_paths = self._collect_image_files(input_p)
        if not image_paths:
            return error_result("No image files found")

        # Build lowercase COCO classes set for comparison
        coco_lower = {c.lower() for c in COCO_CLASSES}

        # Determine detection mode
        use_quality = quality
        use_openvocab = _needs_openvocab(classes, coco_lower) if not quality else False

        try:
            if use_quality:
                # SAM3 quality mode: detection-via-segmentation
                return await self._execute_sam3(
                    image_paths, classes, confidence, save_annotations, input_path, output_path
                )
            elif use_openvocab:
                # Open-vocabulary mode: YOLO-World or GroundingDINO
                return await self._execute_openvocab(
                    image_paths,
                    classes,
                    confidence,
                    prefer_accuracy,
                    save_annotations,
                    input_path,
                    output_path,
                )
            else:
                # Standard COCO mode: YOLO11 or RT-DETR
                return await self._execute_coco(
                    image_paths,
                    classes,
                    confidence,
                    prefer_accuracy,
                    save_annotations,
                    input_path,
                    output_path,
                )

        except ImportError as e:
            logger.exception("CV dependencies not installed")
            return error_result(
                f"CV dependencies not installed: {e}. "
                "Install with: pip install -r requirements-cv.txt"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory: %s", e)
                return error_result(
                    f"GPU out of memory. Try: reduce batch size, disable quality mode, "
                    f"or close other GPU applications. Error: {e}"
                )
            logger.exception("Detection failed")
            return error_result(f"Detection failed: {e}")
        except Exception as e:
            logger.exception("Detection failed")
            return error_result(f"Detection failed: {e}")

    async def _execute_coco(
        self,
        image_paths: list[str],
        classes: list[str] | None,
        confidence: float,
        prefer_accuracy: bool,
        save_annotations: bool,
        input_path: str,
        output_path: str | None,
    ) -> ToolResult:
        """Execute detection using YOLO11 or RT-DETR for COCO classes."""
        from backend.cv.detection import COCO_CLASSES, get_detector

        # Validate COCO class names
        if classes:
            coco_lower = {c.lower() for c in COCO_CLASSES}
            invalid = [c for c in classes if c.lower() not in coco_lower]
            if invalid:
                return error_result(
                    f"Invalid COCO class names: {invalid}. "
                    f"Valid classes include: {COCO_CLASSES[:10]}... (80 total). "
                    "For custom classes, they will auto-route to YOLO-World."
                )

        # Get and load detector
        detector = get_detector(prefer_accuracy=prefer_accuracy)
        detector.load()

        try:
            # Run detection with progress reporting
            try:
                results = detector.predict_batch(
                    image_paths,
                    progress_callback=lambda curr, total: self.report_progress(
                        curr, total, f"Detecting in image {curr}/{total}"
                    ),
                    confidence=confidence,
                    classes=classes,
                )
            except RuntimeError as e:
                if not _is_mps_runtime_error(e) or detector.device == "cpu":
                    raise
                logger.warning(
                    "Detection failed on %s with MPS runtime error; retrying on CPU: %s",
                    detector.device,
                    e,
                )
                detector.unload()
                detector.load(device="cpu")
                results = detector.predict_batch(
                    image_paths,
                    progress_callback=lambda curr, total: self.report_progress(
                        curr, total, f"Detecting in image {curr}/{total} (cpu fallback)"
                    ),
                    confidence=confidence,
                    classes=classes,
                )

            # Aggregate results
            total_detections = 0
            class_counts: dict[str, int] = {}
            total_confidence = 0.0

            for result in results:
                total_detections += result.count
                for det in result.detections:
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                    total_confidence += det.confidence

            avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0

            preview_items = _build_preview_items_from_results(results)
            annotations_path: str | None = None
            if save_annotations:
                annotations_root = self._resolve_annotations_output_path(input_path, output_path)
                records = [
                    (
                        result.image_path,
                        [
                            {
                                "class_name": det.class_name,
                                "bbox": {
                                    "x": det.bbox.x,
                                    "y": det.bbox.y,
                                    "width": det.bbox.width,
                                    "height": det.bbox.height,
                                },
                            }
                            for det in result.detections
                        ],
                    )
                    for result in results
                ]
                annotations_path = self._save_yolo_annotations(records, annotations_root)

            return success_result(
                {
                    "files_processed": len(image_paths),
                    "count": total_detections,
                    "classes": class_counts,
                    "sample_images": image_paths[:6],
                    "preview_items": preview_items,
                    "confidence_threshold": confidence,
                    "confidence": round(avg_confidence, 3),
                    "model": detector.info.name,
                    "mode": "coco",
                    "annotations_saved": save_annotations,
                    "annotations_path": annotations_path,
                    "annotation_format": "yolo" if annotations_path else None,
                }
            )
        finally:
            detector.unload()

    async def _execute_openvocab(
        self,
        image_paths: list[str],
        classes: list[str] | None,
        confidence: float,
        prefer_accuracy: bool,
        save_annotations: bool,
        input_path: str,
        output_path: str | None,
    ) -> ToolResult:
        """Execute detection using YOLO-World or GroundingDINO for open-vocabulary."""
        from backend.cv.openvocab import get_openvocab_detector

        # Get and load open-vocab detector
        detector = get_openvocab_detector(prefer_accuracy=prefer_accuracy)
        detector.load()

        # Build prompt from classes
        prompt = ", ".join(classes) if classes else "object"

        try:
            # Run detection with progress reporting
            try:
                results = detector.predict_batch(
                    image_paths,
                    progress_callback=lambda curr, total: self.report_progress(
                        curr, total, f"Detecting (open-vocab) {curr}/{total}"
                    ),
                    prompt=prompt,
                    confidence=confidence,
                )
            except RuntimeError as e:
                if not _is_mps_runtime_error(e) or detector.device == "cpu":
                    raise
                logger.warning(
                    "Open-vocab detection failed on %s with MPS runtime error; retrying on CPU: %s",
                    detector.device,
                    e,
                )
                detector.unload()
                detector.load(device="cpu")
                results = detector.predict_batch(
                    image_paths,
                    progress_callback=lambda curr, total: self.report_progress(
                        curr, total, f"Detecting (open-vocab) {curr}/{total} (cpu fallback)"
                    ),
                    prompt=prompt,
                    confidence=confidence,
                )

            # Aggregate results
            total_detections = 0
            class_counts: dict[str, int] = {}
            total_confidence = 0.0

            for result in results:
                total_detections += result.count
                for det in result.detections:
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                    total_confidence += det.confidence

            avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0

            preview_items = _build_preview_items_from_results(results)
            annotations_path: str | None = None
            if save_annotations:
                annotations_root = self._resolve_annotations_output_path(input_path, output_path)
                records = [
                    (
                        result.image_path,
                        [
                            {
                                "class_name": det.class_name,
                                "bbox": {
                                    "x": det.bbox.x,
                                    "y": det.bbox.y,
                                    "width": det.bbox.width,
                                    "height": det.bbox.height,
                                },
                            }
                            for det in result.detections
                        ],
                    )
                    for result in results
                ]
                annotations_path = self._save_yolo_annotations(records, annotations_root)

            return success_result(
                {
                    "files_processed": len(image_paths),
                    "count": total_detections,
                    "classes": class_counts,
                    "sample_images": image_paths[:6],
                    "preview_items": preview_items,
                    "confidence_threshold": confidence,
                    "confidence": round(avg_confidence, 3),
                    "model": detector.info.name,
                    "mode": "openvocab",
                    "prompt": prompt,
                    "annotations_saved": save_annotations,
                    "annotations_path": annotations_path,
                    "annotation_format": "yolo" if annotations_path else None,
                }
            )
        finally:
            detector.unload()

    async def _execute_sam3(
        self,
        image_paths: list[str],
        classes: list[str] | None,
        confidence: float,
        save_annotations: bool,
        input_path: str,
        output_path: str | None,
    ) -> ToolResult:
        """Execute detection using SAM3 (detection-via-segmentation)."""
        from backend.cv.segmentation import get_segmenter

        # Get and load SAM3 segmenter
        segmenter = get_segmenter(prompt_type="text")
        segmenter.load()

        # Build prompt from classes
        prompt = ", ".join(classes) if classes else "object"

        try:
            total_detections = 0
            class_counts: dict[str, int] = {}
            total_confidence = 0.0
            total = len(image_paths)
            annotation_records: list[tuple[str, list[dict[str, Any]]]] = []

            for i, path in enumerate(image_paths):
                self.report_progress(i + 1, total, f"Segmenting (SAM3) {i + 1}/{total}")

                result = segmenter.predict(path, prompt=prompt, confidence=confidence)

                # Convert masks to detection format
                detections, counts = _masks_to_detections(
                    result, classes if classes else ["object"]
                )
                annotation_records.append((path, detections))

                total_detections += len(detections)
                for class_name, count in counts.items():
                    class_counts[class_name] = class_counts.get(class_name, 0) + count
                for det in detections:
                    total_confidence += det["confidence"]

            avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0

            preview_items = _build_preview_items_from_annotation_records(annotation_records)
            annotations_path: str | None = None
            if save_annotations:
                annotations_root = self._resolve_annotations_output_path(input_path, output_path)
                annotations_path = self._save_yolo_annotations(annotation_records, annotations_root)

            return success_result(
                {
                    "files_processed": len(image_paths),
                    "count": total_detections,
                    "classes": class_counts,
                    "sample_images": image_paths[:6],
                    "preview_items": preview_items,
                    "confidence_threshold": confidence,
                    "confidence": round(avg_confidence, 3),
                    "model": segmenter.info.name,
                    "mode": "sam3",
                    "prompt": prompt,
                    "annotations_saved": save_annotations,
                    "annotations_path": annotations_path,
                    "annotation_format": "yolo" if annotations_path else None,
                }
            )
        finally:
            segmenter.unload()

    def _resolve_annotations_output_path(
        self,
        input_path: str,
        output_path: str | None,
    ) -> Path:
        """Resolve where generated YOLO annotations should be written."""
        if output_path:
            candidate = Path(output_path)
            return candidate if not candidate.suffix else candidate.with_suffix("")

        source = Path(input_path)
        if source.is_file():
            return source.parent / f"{source.stem}_detections_yolo"
        return source.parent / f"{source.name}_detections_yolo"

    def _save_yolo_annotations(
        self,
        records: list[tuple[str, list[dict[str, Any]]]],
        output_root: Path,
    ) -> str:
        """
        Save detections as a minimal YOLO dataset.

        Creates:
        - data.yaml
        - train/images/*
        - train/labels/*.txt
        """
        import yaml

        if output_root.exists():
            shutil.rmtree(output_root)

        images_dir = output_root / "train" / "images"
        labels_dir = output_root / "train" / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        class_names: list[str] = []
        for _, detections in records:
            for detection in detections:
                class_name = str(detection.get("class_name", "object")).strip() or "object"
                if class_name not in class_names:
                    class_names.append(class_name)

        if not class_names:
            class_names = ["object"]
        class_to_id = {name: idx for idx, name in enumerate(class_names)}

        for index, (image_path, detections) in enumerate(records):
            source_image = Path(image_path)
            image_name = f"{index:05d}_{source_image.name}"
            target_image = images_dir / image_name

            try:
                target_image.symlink_to(source_image.resolve())
            except Exception:
                shutil.copy2(source_image, target_image)

            label_path = labels_dir / f"{Path(image_name).stem}.txt"
            lines: list[str] = []
            for detection in detections:
                class_name = str(detection.get("class_name", "object")).strip() or "object"
                bbox = detection.get("bbox") or {}
                x = float(bbox.get("x", 0.0))
                y = float(bbox.get("y", 0.0))
                w = float(bbox.get("width", 0.0))
                h = float(bbox.get("height", 0.0))

                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                lines.append(f"{class_to_id[class_name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            label_path.write_text("\n".join(lines), encoding="utf-8")

        data_yaml = {
            "path": str(output_root),
            "train": "train/images",
            "val": "train/images",
            "test": "train/images",
            "names": class_names,
        }
        (output_root / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False),
            encoding="utf-8",
        )

        return str(output_root)

    def _collect_image_files(self, path: Path) -> list[str]:
        """
        Collect all image file paths.

        Args:
            path: Input path (file or directory).

        Returns:
            List of image file paths as strings.
        """
        if path.is_file():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                return [str(path)]
            return []

        files: list[str] = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(str(f) for f in path.glob(f"**/*{ext}"))
        return files
