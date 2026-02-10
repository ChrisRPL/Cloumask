"""Dataset conversion tool for annotation formats.

Converts datasets between supported annotation formats using the
shared data format loaders/exporters.
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
from backend.agent.tools.registry import register_tool
from backend.data.formats import detect_format, get_exporter, get_loader, list_formats

logger = logging.getLogger(__name__)


SUPPORTED_CONVERT_FORMATS = [
    "yolo",
    "coco",
    "kitti",
    "voc",
    "cvat",
    "nuscenes",
    "openlabel",
]


@register_tool
class ConvertFormatTool(BaseTool):
    """Convert datasets between supported annotation formats."""

    name = "convert_format"
    description = """Convert a labeled dataset from one annotation format to another.
Supports YOLO, COCO, KITTI, Pascal VOC, CVAT, nuScenes, and OpenLABEL.
Auto-detects source format when source_format is omitted."""
    category = ToolCategory.EXPORT

    parameters = [
        ToolParameter(
            name="source_path",
            type=str,
            description="Path to source dataset root directory",
            required=True,
        ),
        ToolParameter(
            name="output_path",
            type=str,
            description="Path to output directory for converted dataset",
            required=True,
        ),
        ToolParameter(
            name="target_format",
            type=str,
            description="Target format: yolo | coco | kitti | voc | cvat | nuscenes | openlabel",
            required=True,
            enum_values=SUPPORTED_CONVERT_FORMATS,
        ),
        ToolParameter(
            name="source_format",
            type=str,
            description=(
                "Optional source format override. "
                "If omitted, format is auto-detected from dataset structure."
            ),
            required=False,
            default=None,
            enum_values=SUPPORTED_CONVERT_FORMATS,
        ),
        ToolParameter(
            name="copy_images",
            type=bool,
            description="Copy images into the output dataset structure",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="overwrite",
            type=bool,
            description="Allow writing into a non-empty output directory",
            required=False,
            default=True,
        ),
    ]

    async def execute(
        self,
        source_path: str,
        output_path: str,
        target_format: str,
        source_format: str | None = None,
        copy_images: bool = True,
        overwrite: bool = True,
    ) -> ToolResult:
        """Convert dataset format and return conversion statistics."""
        source = Path(source_path)
        output = Path(output_path)
        normalized_target = target_format.lower()
        normalized_source = source_format.lower() if source_format else None

        if not source.exists():
            return error_result(f"Source dataset not found: {source_path}")

        if not source.is_dir():
            return error_result(f"Source path must be a directory: {source_path}")

        available_formats = list_formats()
        loader_formats = sorted(
            name for name, details in available_formats.items() if details.get("loader")
        )
        exporter_formats = sorted(
            name for name, details in available_formats.items() if details.get("exporter")
        )

        if normalized_target not in exporter_formats:
            return error_result(
                f"Unsupported target_format '{target_format}'. "
                f"Available export formats: {exporter_formats}"
            )

        if normalized_source and normalized_source not in loader_formats:
            return error_result(
                f"Unsupported source_format '{source_format}'. "
                f"Available source formats: {loader_formats}"
            )

        if normalized_source is None:
            normalized_source = detect_format(source)
            if normalized_source is None:
                return error_result(
                    "Could not detect source format. "
                    f"Please provide source_format explicitly ({loader_formats})."
                )

        try:
            result_data = await asyncio.to_thread(
                self._convert_dataset,
                source,
                output,
                normalized_source,
                normalized_target,
                copy_images,
                overwrite,
            )
            return success_result(result_data)
        except Exception as exc:
            logger.exception("Dataset conversion failed")
            return error_result(
                f"Conversion failed: {exc}",
                source_path=str(source),
                output_path=str(output),
                source_format=normalized_source,
                target_format=normalized_target,
            )

    def _convert_dataset(
        self,
        source: Path,
        output: Path,
        source_format: str,
        target_format: str,
        copy_images: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        """Run synchronous conversion logic in a worker thread."""
        warnings: list[str] = []

        self.report_progress(0, 3, "Initializing conversion")

        loader = get_loader(
            source,
            format_name=source_format,
            progress_callback=self.report_progress,
        )
        warnings.extend(loader.validate())
        dataset = loader.load()
        self.report_progress(1, 3, f"Loaded {len(dataset)} samples from {source_format}")

        exporter = get_exporter(
            output,
            target_format,
            overwrite=overwrite,
            progress_callback=self.report_progress,
        )
        exported_path = exporter.export(dataset, copy_images=copy_images)
        warnings.extend(exporter.validate_export())
        self.report_progress(2, 3, f"Exported dataset to {target_format}")

        self.report_progress(3, 3, "Conversion complete")

        return {
            "source_format": source_format,
            "target_format": target_format,
            "source_path": str(source),
            "output_path": str(exported_path),
            "num_samples": len(dataset),
            "num_labels": dataset.total_labels(),
            "num_classes": dataset.num_classes,
            "class_names": dataset.class_names,
            "copy_images": copy_images,
            "overwrite": overwrite,
            "warnings": warnings,
        }
