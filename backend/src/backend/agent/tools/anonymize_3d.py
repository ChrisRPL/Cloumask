"""3D point cloud anonymization tool for face privacy compliance.

Anonymizes faces in 3D point clouds by detecting face regions via multi-view
projection and either removing or adding noise to the corresponding points.
Supports verification pass to confirm no detectable faces remain.

Implements spec: 05-point-cloud/07-anonymization-3d, 05-point-cloud/08-agent-tools
Integration point: backend/cv/anonymization_3d.py
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
from backend.agent.tools.constants import POINTCLOUD_EXTENSIONS
from backend.agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

# Output formats that Open3D can write to
SUPPORTED_OUTPUT_FORMATS: frozenset[str] = frozenset({".pcd", ".ply"})


@register_tool
class AnonymizePointCloudTool(BaseTool):
    """Anonymize faces in 3D point clouds for privacy compliance."""

    name = "anonymize_pointcloud"
    description = """Anonymize faces in a 3D point cloud for privacy compliance (GDPR).

Modes:
- remove: Completely removes points in face regions (cleaner but sparser)
- noise: Adds random Gaussian noise to face points (preserves density)

Pipeline:
1. Generates virtual cameras around the scene
2. Renders depth images from each viewpoint
3. Runs face detection (SCRFD) on each view
4. Lifts face regions back to 3D point indices
5. Applies anonymization (remove or noise)
6. Optionally verifies no faces remain

Supported Formats:
- Input: PCD, PLY, LAS/LAZ, KITTI BIN
- Output: PCD, PLY

Examples:
- anonymize_pointcloud(input, output)  # Remove faces (default)
- anonymize_pointcloud(input, output, mode="noise")  # Add noise instead
- anonymize_pointcloud(input, output, verify=False)  # Skip verification
- anonymize_pointcloud(input, output, num_views=12)  # More viewpoints"""
    category = ToolCategory.ANONYMIZATION

    parameters = [
        ToolParameter(
            name="input_path",
            type=str,
            description="Path to input point cloud file (.pcd, .ply, .las, .laz, .bin)",
            required=True,
        ),
        ToolParameter(
            name="output_path",
            type=str,
            description="Path for anonymized output (.pcd or .ply)",
            required=True,
        ),
        ToolParameter(
            name="mode",
            type=str,
            description="Anonymization mode: 'remove' deletes face points, 'noise' adds displacement",
            required=False,
            default="remove",
            enum_values=["remove", "noise"],
        ),
        ToolParameter(
            name="verify",
            type=bool,
            description="Re-check output to confirm no detectable faces remain",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="num_views",
            type=int,
            description="Number of virtual camera viewpoints (more = thorough, slower)",
            required=False,
            default=8,
        ),
        ToolParameter(
            name="face_confidence",
            type=float,
            description="Minimum face detection confidence (0-1)",
            required=False,
            default=0.4,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        output_path: str,
        mode: str = "remove",
        verify: bool = True,
        num_views: int = 8,
        face_confidence: float = 0.4,
    ) -> ToolResult:
        """
        Execute 3D point cloud anonymization.

        Args:
            input_path: Path to input point cloud file.
            output_path: Path for anonymized output file.
            mode: Anonymization mode ('remove' or 'noise').
            verify: Whether to verify no faces remain after anonymization.
            num_views: Number of virtual camera viewpoints for face detection.
            face_confidence: Minimum confidence for face detection.

        Returns:
            ToolResult with anonymization statistics.
        """
        from backend.cv.anonymization_3d import PointCloudAnonymizer

        input_p = Path(input_path)
        output_p = Path(output_path)

        # Validate input file
        if not input_p.exists():
            return error_result(f"Input file not found: {input_path}")

        if not input_p.is_file():
            return error_result("Input must be a file, not a directory")

        if input_p.suffix.lower() not in POINTCLOUD_EXTENSIONS:
            return error_result(
                f"Unsupported input format: {input_p.suffix}. "
                f"Supported: {', '.join(sorted(POINTCLOUD_EXTENSIONS))}"
            )

        # Validate output format
        if output_p.suffix.lower() not in SUPPORTED_OUTPUT_FORMATS:
            return error_result(
                f"Unsupported output format: {output_p.suffix}. "
                f"Supported for writing: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}"
            )

        # Validate parameters
        if face_confidence < 0 or face_confidence > 1:
            return error_result(
                f"Invalid face_confidence: {face_confidence}. Must be between 0 and 1."
            )

        if num_views < 1:
            return error_result("num_views must be at least 1")

        try:
            self.report_progress(0, 3, "Loading face detection model...")

            anonymizer = PointCloudAnonymizer()
            anonymizer.load("auto")

            try:
                self.report_progress(1, 3, f"Anonymizing faces ({mode} mode)...")

                result = anonymizer.anonymize(
                    pcd_path=input_path,
                    output_path=output_path,
                    mode=mode,  # type: ignore[arg-type]
                    num_views=num_views,
                    face_confidence=face_confidence,
                    verify=verify,
                )

                self.report_progress(3, 3, "Anonymization complete")

                return success_result(
                    {
                        "input_path": input_path,
                        "output_path": result.output_path,
                        "mode": result.mode,
                        "original_points": result.original_point_count,
                        "anonymized_points": result.anonymized_point_count,
                        "faces_found": result.face_regions_found,
                        "points_removed": result.points_removed,
                        "points_noised": result.points_noised,
                        "verified": result.verification_passed,
                        "views_processed": result.views_processed,
                        "processing_time_ms": round(result.processing_time_ms, 2),
                    }
                )

            finally:
                anonymizer.unload()

        except ImportError as e:
            logger.exception("CV dependencies not installed")
            return error_result(
                f"CV dependencies not installed: {e}. "
                "3D anonymization requires Open3D and face detection models. "
                "Install with: pip install -r requirements-cv.txt"
            )
        except FileNotFoundError as e:
            return error_result(str(e))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory: %s", e)
                return error_result(
                    f"GPU out of memory during face detection. "
                    f"Try reducing num_views or closing other GPU applications. Error: {e}"
                )
            logger.exception("3D anonymization failed")
            return error_result(f"3D anonymization failed: {e}")
        except Exception as e:
            logger.exception("3D anonymization failed")
            return error_result(f"3D anonymization failed: {e}")
