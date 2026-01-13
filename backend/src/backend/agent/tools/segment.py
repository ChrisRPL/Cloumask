"""Segmentation tool using SAM3, SAM2, or MobileSAM.

SAM3 supports natural language text prompts (e.g., "red car", "person on left").
SAM2 and MobileSAM support point/box prompts for faster inference.
Auto-selects the appropriate model based on prompt type.

Implements spec: 03-cv-models/02-sam3-segmentation, 03-cv-models/08-cv-tools
Integration point: backend/cv/segmentation.py
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

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

logger = logging.getLogger(__name__)


def _serialize_mask(mask: Any) -> dict[str, Any]:
    """
    Serialize a Mask object for JSON transport.

    Args:
        mask: Mask object from cv/types.py with data (zlib compressed), width, height, confidence.

    Returns:
        Dict with mask metadata and base64-encoded compressed data.
    """
    import numpy as np

    mask_np = mask.to_numpy()

    # Calculate bounding box from mask
    coords = np.where(mask_np > 0)
    if len(coords[0]) > 0:
        y1, y2 = int(coords[0].min()), int(coords[0].max())
        x1, x2 = int(coords[1].min()), int(coords[1].max())
        h, w = mask_np.shape
        bbox = {
            "x": (x1 + x2) / 2 / w,
            "y": (y1 + y2) / 2 / h,
            "width": (x2 - x1) / w,
            "height": (y2 - y1) / h,
        }
        area_pixels = int(np.sum(mask_np > 0))
    else:
        bbox = {"x": 0, "y": 0, "width": 0, "height": 0}
        area_pixels = 0

    return {
        "width": mask.width,
        "height": mask.height,
        "confidence": round(mask.confidence, 4),
        "area_pixels": area_pixels,
        "bbox": bbox,
        # Data is already zlib compressed in Mask type
        "data_base64": base64.b64encode(mask.data).decode("utf-8"),
    }


@register_tool
class SegmentTool(BaseTool):
    """Segment objects in images using SAM3, SAM2, or MobileSAM."""

    name = "segment"
    description = """Segment objects in images to create pixel-level masks.

Model Selection:
- Text prompt (e.g., "red car"): Uses SAM3 (~8GB VRAM, 300-500ms)
- Point prompt [x, y]: Uses SAM2 or MobileSAM (~1-6GB VRAM, 50-200ms)
- Box prompt [x1, y1, x2, y2]: Uses SAM2 (~6GB VRAM, 100-200ms)

SAM3 understands natural language and can segment based on descriptions like
"the person on the left" or "red car in background".

Examples:
- segment(path, prompt="red car")  # SAM3 text prompt
- segment(path, point=[320, 240])  # Point in image center
- segment(path, box=[100, 100, 400, 300])  # Box region"""
    category = ToolCategory.SEGMENTATION

    parameters = [
        ToolParameter(
            name="input_path",
            type=str,
            description="Path to input image",
            required=True,
        ),
        ToolParameter(
            name="prompt",
            type=str,
            description="Text description of what to segment (SAM3 only, e.g., 'red car', 'person')",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="point",
            type=list,
            description="Point prompt as [x, y] pixel coordinates (SAM2/MobileSAM)",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="box",
            type=list,
            description="Box prompt as [x1, y1, x2, y2] pixel coordinates (SAM2)",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="confidence",
            type=float,
            description="Minimum confidence threshold (0-1) for filtering masks",
            required=False,
            default=0.25,
        ),
        ToolParameter(
            name="return_masks",
            type=bool,
            description="Include mask data (base64 compressed) in result",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="prefer_speed",
            type=bool,
            description="Use MobileSAM for faster inference (point prompts only)",
            required=False,
            default=False,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        prompt: str | None = None,
        point: list[int] | None = None,
        box: list[int] | None = None,
        confidence: float = 0.25,
        return_masks: bool = False,
        prefer_speed: bool = False,
    ) -> ToolResult:
        """
        Execute segmentation using SAM models.

        Args:
            input_path: Path to input image.
            prompt: Text prompt for SAM3 (e.g., "red car").
            point: Point prompt as [x, y] pixels.
            box: Box prompt as [x1, y1, x2, y2] pixels.
            confidence: Minimum confidence threshold.
            return_masks: Include mask data in result.
            prefer_speed: Use MobileSAM for speed (point prompts only).

        Returns:
            ToolResult with segmentation statistics and optional mask data.
        """
        from backend.cv.segmentation import get_segmenter

        input_p = Path(input_path)

        # Validate input
        if not input_p.exists():
            return error_result(f"Input not found: {input_path}")

        if not input_p.is_file():
            return error_result("Segmentation requires a single image file, not a directory")

        if input_p.suffix.lower() not in IMAGE_EXTENSIONS:
            return error_result(f"Unsupported image format: {input_p.suffix}")

        if confidence < 0 or confidence > 1:
            return error_result(
                f"Invalid confidence: {confidence}. Must be between 0 and 1."
            )

        # Require at least one prompt type
        if prompt is None and point is None and box is None:
            return error_result(
                "Must provide at least one prompt type: prompt (text), point, or box"
            )

        # Validate point format
        if point is not None:
            if len(point) != 2:
                return error_result(f"Point must have 2 coordinates [x, y], got {len(point)}")
            point_tuple: tuple[int, int] = (int(point[0]), int(point[1]))
        else:
            point_tuple = None  # type: ignore[assignment]

        # Validate box format
        if box is not None:
            if len(box) != 4:
                return error_result(
                    f"Box must have 4 coordinates [x1, y1, x2, y2], got {len(box)}"
                )
            box_tuple: tuple[int, int, int, int] = (
                int(box[0]), int(box[1]), int(box[2]), int(box[3])
            )
        else:
            box_tuple = None  # type: ignore[assignment]

        # Determine prompt type for model selection
        if prompt is not None:
            prompt_type = "text"
        elif box is not None:
            prompt_type = "box"
        else:
            prompt_type = "point"

        try:
            # Get appropriate segmenter
            segmenter = get_segmenter(
                prompt_type=prompt_type,
                prefer_speed=prefer_speed,
            )
            segmenter.load()

            try:
                self.report_progress(1, 1, f"Segmenting with {segmenter.info.name}...")

                result = segmenter.predict(
                    str(input_p),
                    prompt=prompt,
                    point=point_tuple,
                    box=box_tuple,
                    confidence=confidence,
                )

                # Build response
                mask_count = len(result.masks)
                total_area = sum(
                    m.width * m.height for m in result.masks
                )

                response_data: dict[str, Any] = {
                    "image_path": str(input_p),
                    "mask_count": mask_count,
                    "total_area_pixels": total_area,
                    "model": result.model_name,
                    "prompt_type": prompt_type,
                    "processing_time_ms": round(result.processing_time_ms, 2),
                }

                if prompt is not None:
                    response_data["prompt"] = prompt

                if return_masks and result.masks:
                    response_data["masks"] = [
                        _serialize_mask(m) for m in result.masks
                    ]

                return success_result(response_data)

            finally:
                segmenter.unload()

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
                    f"GPU out of memory. SAM3 requires ~8GB VRAM. "
                    f"Try: use point/box prompts instead of text, or use prefer_speed=True. "
                    f"Error: {e}"
                )
            if "text prompts" in str(e).lower():
                return error_result(
                    f"Text prompts require SAM3 with sufficient VRAM. "
                    f"Try using point or box prompts instead. Error: {e}"
                )
            logger.exception("Segmentation failed")
            return error_result(f"Segmentation failed: {e}")
        except NotImplementedError as e:
            logger.warning("Unsupported prompt type: %s", e)
            return error_result(str(e))
        except Exception as e:
            logger.exception("Segmentation failed")
            return error_result(f"Segmentation failed: {e}")
