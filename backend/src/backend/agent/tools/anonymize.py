"""Anonymize tool for face and license plate anonymization.

Integrates with AnonymizationPipeline for SCRFD face detection,
YOLO-World plate detection, and optional SAM3 precise masking.
Supports four modes: blur, blackbox, pixelate, mask.

Implements spec: 03-cv-models/06-anonymization, 03-cv-models/08-cv-tools
Integration point: backend/cv/anonymization.py
"""

from __future__ import annotations

import logging
from pathlib import Path

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


def _blur_strength_to_kernel(blur_strength: int) -> int:
    """
    Convert blur_strength (1-10) to Gaussian kernel size.

    Maps 1-10 to kernel sizes 11-91 (must be odd for Gaussian blur).

    Args:
        blur_strength: User-friendly blur intensity (1-10).

    Returns:
        Odd kernel size for cv2.GaussianBlur.
    """
    # Map 1-10 to 11-91 (step of 8)
    kernel_size = 11 + (blur_strength - 1) * 8
    # Ensure odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


@register_tool
class AnonymizeTool(BaseTool):
    """Anonymize faces and license plates in images."""

    name = "anonymize"
    description = """Anonymize faces and license plates in images to protect privacy.

Modes:
- blur: Gaussian blur (default, natural look)
- blackbox: Solid black fill (strong privacy)
- pixelate: Mosaic effect (recognizable but anonymous)
- mask: SAM3 precise boundaries + blur (~8GB VRAM, best edges)

Quality mode uses SAM3 for precise segmentation on ALL modes (not just mask).

Examples:
- anonymize(path, output, target="all", mode="blur")
- anonymize(path, output, target="faces", mode="pixelate")
- anonymize(path, output, quality=True)  # SAM3 for all detections"""
    category = ToolCategory.ANONYMIZATION

    parameters = [
        ToolParameter(
            name="input_path",
            type=str,
            description="Path to input image or directory",
            required=True,
        ),
        ToolParameter(
            name="output_path",
            type=str,
            description="Path for output (file or directory). If directory, maintains filenames.",
            required=True,
        ),
        ToolParameter(
            name="target",
            type=str,
            description="What to anonymize: faces, plates, or all",
            required=False,
            default="all",
            enum_values=["faces", "plates", "all"],
        ),
        ToolParameter(
            name="mode",
            type=str,
            description="Anonymization style: blur, blackbox, pixelate, or mask",
            required=False,
            default="blur",
            enum_values=["blur", "blackbox", "pixelate", "mask"],
        ),
        ToolParameter(
            name="blur_strength",
            type=int,
            description="Blur intensity (1-10, only applies to blur mode)",
            required=False,
            default=5,
        ),
        ToolParameter(
            name="quality",
            type=bool,
            description="Use SAM3 for precise masks on ALL modes (~8GB VRAM, slower but best edges)",
            required=False,
            default=False,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        output_path: str,
        target: str = "all",
        mode: str = "blur",
        blur_strength: int = 5,
        quality: bool = False,
    ) -> ToolResult:
        """
        Execute anonymization using AnonymizationPipeline.

        Args:
            input_path: Path to input image or directory.
            output_path: Path for output file or directory.
            target: What to anonymize (faces, plates, all).
            mode: Anonymization style (blur, blackbox, pixelate, mask).
            blur_strength: Blur intensity (1-10).
            quality: Use SAM3 for precise masks on all modes.

        Returns:
            ToolResult with anonymization statistics.
        """
        from backend.cv.anonymization import AnonymizationConfig, AnonymizationPipeline

        input_p = Path(input_path)
        output_p = Path(output_path)

        # Validate input exists
        if not input_p.exists():
            return error_result(f"Input not found: {input_path}")

        # Validate blur strength
        if blur_strength < 1 or blur_strength > 10:
            return error_result(
                f"Invalid blur_strength: {blur_strength}. Must be between 1 and 10."
            )

        # Collect image files
        image_paths = self._collect_image_files(input_p)
        if not image_paths:
            return error_result("No image files found in input path")

        # Determine output handling
        is_batch = len(image_paths) > 1 or input_p.is_dir()
        if is_batch and output_p.suffix:
            # Output looks like a file but we have multiple inputs
            output_dir = output_p.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        elif is_batch:
            output_dir = output_p
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Single file - ensure parent exists
            output_p.parent.mkdir(parents=True, exist_ok=True)

        # Map target to config
        detect_faces = target in ("faces", "all")
        detect_plates = target in ("plates", "all")

        # If quality=True, use mask mode for precise SAM3 boundaries
        actual_mode = "mask" if quality else mode

        # Convert blur_strength to kernel size
        kernel_size = _blur_strength_to_kernel(blur_strength)

        try:
            # Configure pipeline
            config = AnonymizationConfig(
                faces=detect_faces,
                plates=detect_plates,
                mode=actual_mode,  # type: ignore[arg-type]
                blur_kernel_size=kernel_size,
                mask_effect="blur" if mode == "blur" else mode,  # type: ignore[arg-type]
            )

            pipeline = AnonymizationPipeline(config)
            pipeline.load("auto")

            try:
                total_faces = 0
                total_plates = 0
                total_time_ms = 0.0
                files_processed = 0
                sample_outputs: list[str] = []

                if is_batch:
                    # Batch processing
                    results = pipeline.process_batch(
                        image_paths,
                        output_dir=str(output_dir) if is_batch else None,
                        progress_callback=lambda curr, total: self.report_progress(
                            curr, total, f"Anonymizing image {curr}/{total}"
                        ),
                    )

                    for result in results:
                        total_faces += result.faces_anonymized
                        total_plates += result.plates_anonymized
                        total_time_ms += result.processing_time_ms
                        files_processed += 1
                        if len(sample_outputs) < 6:
                            sample_outputs.append(result.output_path)

                    final_output = str(output_dir)
                else:
                    # Single file
                    self.report_progress(1, 1, "Anonymizing image...")
                    result = pipeline.process(image_paths[0], str(output_p))

                    total_faces = result.faces_anonymized
                    total_plates = result.plates_anonymized
                    total_time_ms = result.processing_time_ms
                    files_processed = 1
                    final_output = result.output_path
                    sample_outputs = [result.output_path]

                return success_result(
                    {
                        "files_processed": files_processed,
                        "faces_anonymized": total_faces,
                        "plates_anonymized": total_plates,
                        "sample_images": sample_outputs,
                        "output_path": final_output,
                        "mode": actual_mode,
                        "target": target,
                        "quality": quality,
                        "blur_strength": blur_strength,
                        "processing_time_ms": round(total_time_ms, 2),
                    }
                )

            finally:
                pipeline.unload()

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
                    f"GPU out of memory. Try: disable quality mode, use 'blur' instead of 'mask', "
                    f"or close other GPU applications. Error: {e}"
                )
            logger.exception("Anonymization failed")
            return error_result(f"Anonymization failed: {e}")
        except Exception as e:
            logger.exception("Anonymization failed")
            return error_result(f"Anonymization failed: {e}")

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
