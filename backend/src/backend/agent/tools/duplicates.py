"""Find duplicate and near-duplicate images in datasets.

Implements spec: 06-data-pipeline/23-find-duplicates-tool
Integration point: backend/data/duplicates.py
"""

from __future__ import annotations

import asyncio
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
from backend.data.duplicates import find_duplicates as detect_duplicates

logger = logging.getLogger(__name__)

SUPPORTED_DUPLICATE_METHODS = ["phash", "dhash", "ahash", "clip"]


@register_tool
class FindDuplicatesTool(BaseTool):
    """Find duplicate and near-duplicate images in a directory or dataset."""

    name = "find_duplicates"
    description = """Find duplicate and near-duplicate images in a dataset.

Supports perceptual hashing (phash, dhash, ahash) and CLIP similarity.
Can optionally remove duplicate files while keeping one representative per group."""
    category = ToolCategory.UTILITY

    parameters = [
        ToolParameter(
            name="path",
            type=str,
            description="Path to an image directory or single image file",
            required=True,
        ),
        ToolParameter(
            name="method",
            type=str,
            description="Duplicate detection method (phash, dhash, ahash, clip)",
            required=False,
            default="phash",
            enum_values=SUPPORTED_DUPLICATE_METHODS,
        ),
        ToolParameter(
            name="threshold",
            type=float,
            description="Similarity threshold in range [0, 1] (higher = stricter)",
            required=False,
            default=0.9,
        ),
        ToolParameter(
            name="auto_remove",
            type=bool,
            description="Delete duplicates automatically while keeping group representatives",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="max_groups",
            type=int,
            description="Maximum number of duplicate groups returned in the response",
            required=False,
            default=50,
        ),
    ]

    async def execute(
        self,
        path: str,
        method: str = "phash",
        threshold: float = 0.9,
        auto_remove: bool = False,
        max_groups: int = 50,
    ) -> ToolResult:
        """Run duplicate detection and optionally remove duplicate files."""
        dataset_path = Path(path)
        normalized_method = method.lower()

        if not dataset_path.exists():
            return error_result(f"Input path not found: {path}")

        if normalized_method not in SUPPORTED_DUPLICATE_METHODS:
            return error_result(
                f"Unsupported method '{method}'. Use one of {SUPPORTED_DUPLICATE_METHODS}."
            )

        if not 0 <= threshold <= 1:
            return error_result(
                f"Invalid threshold: {threshold}. Threshold must be between 0 and 1."
            )

        if max_groups < 1:
            return error_result(
                f"Invalid max_groups: {max_groups}. max_groups must be >= 1."
            )

        try:
            result_data = await asyncio.to_thread(
                self._run_detection,
                dataset_path,
                normalized_method,
                threshold,
                auto_remove,
                max_groups,
            )
            return success_result(result_data)
        except RuntimeError as exc:
            logger.warning("Duplicate detection runtime error: %s", exc)
            return error_result(f"Duplicate detection failed: {exc}", path=str(dataset_path))
        except Exception as exc:
            logger.exception("Duplicate detection failed unexpectedly")
            return error_result(f"Duplicate detection failed: {exc}", path=str(dataset_path))

    def _run_detection(
        self,
        dataset_path: Path,
        method: str,
        threshold: float,
        auto_remove: bool,
        max_groups: int,
    ) -> dict[str, Any]:
        """Execute duplicate detection synchronously in a worker thread."""
        image_paths = self._collect_images(dataset_path)
        total_images = len(image_paths)

        if total_images == 0:
            return {
                "path": str(dataset_path),
                "method": method,
                "threshold": threshold,
                "total_images": 0,
                "num_groups": 0,
                "num_duplicates": 0,
                "groups": [],
                "max_groups": max_groups,
                "truncated": False,
                "auto_remove": auto_remove,
                "removed": [],
                "remove_errors": [],
            }

        result = detect_duplicates(
            image_paths,
            method=method,
            threshold=threshold,
            progress_callback=self.report_progress,
        )

        groups: list[dict[str, Any]] = []
        for group in result.groups[:max_groups]:
            groups.append({
                "representative": str(group.representative),
                "duplicates": [str(image_path) for image_path in group.duplicates],
                "count": group.count,
                "similarity_scores": [round(score, 6) for score in group.similarity_scores],
            })

        removed: list[str] = []
        remove_errors: list[dict[str, str]] = []
        if auto_remove:
            for duplicate_path in result.get_duplicates_to_remove():
                try:
                    duplicate_path.unlink()
                    removed.append(str(duplicate_path))
                except OSError as exc:
                    remove_errors.append({
                        "path": str(duplicate_path),
                        "error": str(exc),
                    })

        return {
            "path": str(dataset_path),
            "method": result.method,
            "threshold": result.threshold,
            "total_images": result.total_images,
            "num_groups": result.num_groups,
            "num_duplicates": result.num_duplicates,
            "groups": groups,
            "max_groups": max_groups,
            "truncated": result.num_groups > max_groups,
            "auto_remove": auto_remove,
            "removed": removed,
            "remove_errors": remove_errors,
        }

    @staticmethod
    def _collect_images(path: Path) -> list[Path]:
        """Collect image files from a file or directory input."""
        if path.is_file():
            return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []

        images = [
            image_path
            for image_path in path.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        images.sort()
        return images
