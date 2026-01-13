"""Face detection tool using SCRFD, YuNet, or SAM3 quality mode.

SCRFD provides high accuracy (95%+ on WIDER FACE easy, ~87% on hard cases).
YuNet provides fast CPU-based detection for real-time applications.
SAM3 quality mode handles challenging cases (distant, crowded, occluded).

Implements spec: 03-cv-models/03-scrfd-faces, 03-cv-models/08-cv-tools
Integration point: backend/cv/faces.py
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


@register_tool
class FaceDetectTool(BaseTool):
    """Detect faces in images using SCRFD, YuNet, or SAM3."""

    name = "detect_faces"
    description = """Detect faces and facial landmarks in images.

Model Selection:
- SCRFD (default): High accuracy (95% easy / 87% hard cases), GPU
- YuNet (realtime=True): Fast CPU-based, 90% accuracy
- SAM3 (quality=True): Best for hard cases (distant, crowds, occlusion), ~8GB VRAM

Landmarks: 5-point (eyes, nose, mouth corners) available with SCRFD.

Examples:
- detect_faces(path)  # SCRFD with landmarks
- detect_faces(path, realtime=True)  # Fast YuNet
- detect_faces(path, quality=True)  # SAM3 for difficult cases"""
    category = ToolCategory.DETECTION

    parameters = [
        ToolParameter(
            name="input_path",
            type=str,
            description="Path to input image or directory",
            required=True,
        ),
        ToolParameter(
            name="confidence",
            type=float,
            description="Minimum confidence threshold (0-1)",
            required=False,
            default=0.5,
        ),
        ToolParameter(
            name="realtime",
            type=bool,
            description="Use fast YuNet for real-time detection (CPU, lower accuracy)",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="include_landmarks",
            type=bool,
            description="Include 5-point facial landmarks (SCRFD only)",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="quality",
            type=bool,
            description="Use SAM3 for hard cases (distant, crowds, occlusion) - ~8GB VRAM",
            required=False,
            default=False,
        ),
    ]

    async def execute(
        self,
        input_path: str,
        confidence: float = 0.5,
        realtime: bool = False,
        include_landmarks: bool = True,
        quality: bool = False,
    ) -> ToolResult:
        """
        Execute face detection.

        Args:
            input_path: Path to input image or directory.
            confidence: Minimum confidence threshold (0-1).
            realtime: Use YuNet for fast CPU-based detection.
            include_landmarks: Include 5-point facial landmarks.
            quality: Use SAM3 for challenging cases.

        Returns:
            ToolResult with face detection statistics.
        """
        input_p = Path(input_path)

        # Validate input
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

        try:
            if quality:
                # SAM3 quality mode for challenging cases
                return await self._execute_sam3(image_paths, confidence)
            else:
                # Standard face detection (SCRFD or YuNet)
                return await self._execute_standard(
                    image_paths, confidence, realtime, include_landmarks
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
                    f"GPU out of memory. Try: use realtime=True for CPU detection, "
                    f"or close other GPU applications. Error: {e}"
                )
            logger.exception("Face detection failed")
            return error_result(f"Face detection failed: {e}")
        except Exception as e:
            logger.exception("Face detection failed")
            return error_result(f"Face detection failed: {e}")

    async def _execute_standard(
        self,
        image_paths: list[str],
        confidence: float,
        realtime: bool,
        include_landmarks: bool,
    ) -> ToolResult:
        """Execute face detection using SCRFD or YuNet."""
        from backend.cv.faces import get_face_detector

        # Get and load detector
        detector = get_face_detector(realtime=realtime)
        detector.load()

        try:
            total_faces = 0
            total_confidence = 0.0
            faces_per_image: dict[str, int] = {}
            total = len(image_paths)

            for i, path in enumerate(image_paths):
                self.report_progress(i + 1, total, f"Detecting faces {i + 1}/{total}")

                result = detector.predict(
                    path,
                    confidence=confidence,
                    include_landmarks=include_landmarks,
                )

                face_count = len(result.faces)
                total_faces += face_count
                faces_per_image[Path(path).name] = face_count

                for face in result.faces:
                    total_confidence += face.confidence

            avg_confidence = total_confidence / total_faces if total_faces > 0 else 0.0

            return success_result(
                {
                    "files_processed": len(image_paths),
                    "total_faces": total_faces,
                    "faces_per_image": faces_per_image,
                    "average_confidence": round(avg_confidence, 3),
                    "model": detector.info.name,
                    "mode": "realtime" if realtime else "accuracy",
                    "include_landmarks": include_landmarks,
                }
            )

        finally:
            detector.unload()

    async def _execute_sam3(
        self,
        image_paths: list[str],
        confidence: float,
    ) -> ToolResult:
        """Execute face detection using SAM3 with 'face' prompt."""

        from backend.cv.segmentation import get_segmenter

        # Get and load SAM3 segmenter
        segmenter = get_segmenter(prompt_type="text")
        segmenter.load()

        try:
            total_faces = 0
            total_confidence = 0.0
            faces_per_image: dict[str, int] = {}
            total = len(image_paths)

            for i, path in enumerate(image_paths):
                self.report_progress(i + 1, total, f"Detecting faces (SAM3) {i + 1}/{total}")

                result = segmenter.predict(
                    path,
                    prompt="face, human face",
                    confidence=confidence,
                )

                face_count = len(result.masks)
                total_faces += face_count
                faces_per_image[Path(path).name] = face_count

                for mask in result.masks:
                    total_confidence += mask.confidence

            avg_confidence = total_confidence / total_faces if total_faces > 0 else 0.0

            return success_result(
                {
                    "files_processed": len(image_paths),
                    "total_faces": total_faces,
                    "faces_per_image": faces_per_image,
                    "average_confidence": round(avg_confidence, 3),
                    "model": segmenter.info.name,
                    "mode": "sam3",
                    "include_landmarks": False,  # SAM3 doesn't provide landmarks
                }
            )

        finally:
            segmenter.unload()

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
