"""Anonymize tool for face and license plate blurring.

This is a STUB implementation that returns mock data for testing.
Real implementation will integrate with SCRFD face detection.

Integration point: backend/cv/anonymize.py
"""

import random
from pathlib import Path

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.registry import register_tool

# Supported image extensions for anonymization
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@register_tool
class AnonymizeTool(BaseTool):
    """Anonymize faces and license plates in images/videos."""

    name = "anonymize"
    description = """Detect and blur faces and license plates in images or videos.
This protects privacy by anonymizing identifiable information."""
    category = ToolCategory.ANONYMIZATION

    parameters = [
        ToolParameter(
            name="input_path",
            type=str,
            description="Path to input file or directory",
            required=True,
        ),
        ToolParameter(
            name="output_path",
            type=str,
            description="Path to save anonymized output",
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
            name="blur_strength",
            type=int,
            description="Blur intensity (1-10)",
            required=False,
            default=5,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        output_path: str,
        target: str = "all",
        blur_strength: int = 5,
    ) -> ToolResult:
        """
        STUB: Returns mock anonymization results.

        TODO: Replace with actual SCRFD face detection and blurring.
        Integration point: backend/cv/anonymize.py
        """
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

        # Count files to process
        file_count = self._count_image_files(input_p)

        if file_count == 0:
            return error_result("No image files found in input path")

        # Generate mock results
        # In reality, this would:
        # 1. Load SCRFD model
        # 2. Detect faces/plates in each image
        # 3. Apply Gaussian blur to detected regions
        # 4. Save output files

        # Use deterministic seed based on input for consistent mock data
        seed = hash(input_path) % (2**32)
        rng = random.Random(seed)

        faces_detected = rng.randint(file_count * 2, file_count * 5)
        plates_detected = rng.randint(0, file_count * 2)

        faces_blurred = faces_detected if target in ("faces", "all") else 0
        plates_blurred = plates_detected if target in ("plates", "all") else 0

        # Simulate processing with progress reporting
        for i in range(file_count):
            self.report_progress(
                i + 1, file_count, f"Processing file {i + 1}/{file_count}"
            )

        return success_result(
            {
                "files_processed": file_count,
                "faces_detected": faces_detected,
                "faces_blurred": faces_blurred,
                "plates_detected": plates_detected,
                "plates_blurred": plates_blurred,
                "output_path": str(output_p),
                "blur_strength": blur_strength,
                "target": target,
                "confidence": round(0.87 + rng.uniform(-0.1, 0.1), 3),
                "_stub": True,  # Marker that this is mock data
                "_integration_point": "backend/cv/anonymize.py",
            }
        )

    def _count_image_files(self, path: Path) -> int:
        """Count image files in path."""
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                return 1
            return 0

        count = 0
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            count += sum(1 for _ in path.glob(f"**/*{ext}"))
        return count
