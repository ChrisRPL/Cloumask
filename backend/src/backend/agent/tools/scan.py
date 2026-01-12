"""Scan directory tool for analyzing dataset contents.

This tool provides functionality to scan directories and analyze their contents,
categorizing files by type (images, videos, point clouds, annotations).
"""

import asyncio
import contextlib
from collections import Counter
from pathlib import Path

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.constants import (
    ANNOTATION_EXTENSIONS,
    IMAGE_EXTENSIONS,
    POINTCLOUD_EXTENSIONS,
    PROGRESS_THROTTLE_INTERVAL,
    VIDEO_EXTENSIONS,
)
from backend.agent.tools.registry import register_tool


@register_tool
class ScanDirectoryTool(BaseTool):
    """Scan a directory to analyze its contents."""

    name = "scan_directory"
    description = """Scan a directory to count files, detect formats, and analyze dataset structure.
Use this as the first step to understand what data you're working with."""
    category = ToolCategory.SCAN

    parameters = [
        ToolParameter(
            name="path",
            type=str,
            description="Path to the directory to scan",
            required=True,
        ),
        ToolParameter(
            name="recursive",
            type=bool,
            description="Whether to scan subdirectories",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="max_depth",
            type=int,
            description="Maximum directory depth (0 = unlimited)",
            required=False,
            default=0,
        ),
    ]

    async def execute(
        self,
        path: str,
        recursive: bool = True,
        max_depth: int = 0,
    ) -> ToolResult:
        """Execute directory scan."""
        dir_path = Path(path)

        # Validate path
        if not dir_path.exists():
            return error_result(f"Directory not found: {path}")

        if not dir_path.is_dir():
            return error_result(f"Not a directory: {path}")

        # Scan files in thread pool to avoid blocking event loop
        try:
            scan_result = await asyncio.to_thread(
                self._scan_directory, dir_path, recursive, max_depth
            )
            return success_result(scan_result)
        except PermissionError:
            return error_result(f"Permission denied: {path}")
        except Exception as e:
            return error_result(f"Scan failed: {e}")

    def _scan_directory(
        self,
        root: Path,
        recursive: bool,
        max_depth: int,
    ) -> dict:
        """Perform the actual directory scan."""
        files: list[str] = []
        extension_counts: Counter[str] = Counter()
        type_counts = {
            "images": 0,
            "videos": 0,
            "pointclouds": 0,
            "annotations": 0,
            "other": 0,
        }
        subdirs: list[str] = []
        total_size = 0

        def scan_path(path: Path, current_depth: int = 0) -> None:
            nonlocal total_size

            if max_depth > 0 and current_depth > max_depth:
                return

            try:
                for entry in path.iterdir():
                    if entry.is_file():
                        files.append(str(entry))
                        ext = entry.suffix.lower()
                        extension_counts[ext] += 1

                        with contextlib.suppress(OSError):
                            total_size += entry.stat().st_size

                        # Categorize by type
                        if ext in IMAGE_EXTENSIONS:
                            type_counts["images"] += 1
                        elif ext in VIDEO_EXTENSIONS:
                            type_counts["videos"] += 1
                        elif ext in POINTCLOUD_EXTENSIONS:
                            type_counts["pointclouds"] += 1
                        elif ext in ANNOTATION_EXTENSIONS:
                            type_counts["annotations"] += 1
                        else:
                            type_counts["other"] += 1

                        # Report progress with throttling to avoid performance issues
                        if len(files) % PROGRESS_THROTTLE_INTERVAL == 0:
                            self.report_progress(
                                len(files), 0, f"Scanned {len(files)} files"
                            )

                    elif entry.is_dir() and recursive:
                        subdirs.append(str(entry))
                        scan_path(entry, current_depth + 1)

            except PermissionError:
                pass  # Skip inaccessible directories

        scan_path(root)

        # Report final progress
        if len(files) > 0:
            self.report_progress(len(files), 0, f"Scanned {len(files)} files")

        # Determine primary data type
        primary_type = self._determine_primary_type(type_counts)

        return {
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_human": self._format_size(total_size),
            "subdirectories": len(subdirs),
            "type_counts": type_counts,
            "primary_type": primary_type,
            "formats": dict(extension_counts.most_common(10)),
            "sample_files": files[:5],  # First 5 files as sample
            "has_annotations": type_counts["annotations"] > 0,
        }

    @staticmethod
    def _determine_primary_type(type_counts: dict[str, int]) -> str:
        """Determine the primary data type in the directory."""
        images = type_counts.get("images", 0)
        videos = type_counts.get("videos", 0)
        pointclouds = type_counts.get("pointclouds", 0)

        # Check for single-type datasets
        if images > 0 and videos == 0 and pointclouds == 0:
            return "images"
        elif videos > 0 and images == 0 and pointclouds == 0:
            return "video"
        elif pointclouds > 0 and images == 0 and videos == 0:
            return "pointcloud"

        # Mixed or empty
        return "mixed"

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
