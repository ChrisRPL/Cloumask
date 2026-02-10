"""Split datasets into train/val/test subsets and export each split.

Implements spec: 06-data-pipeline/25-split-dataset-tool
Integration points: backend/data/splitting.py and backend/data/formats/*
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
from backend.data.splitting import split_dataset as split_data

logger = logging.getLogger(__name__)


def _normalize_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, float]:
    """Validate and normalize train/val/test ratios."""
    raw_ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }

    for split_name, ratio in raw_ratios.items():
        if ratio < 0:
            raise ValueError(f"{split_name}_ratio must be >= 0, got {ratio}")

    total_ratio = sum(raw_ratios.values())
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be > 0")

    return {
        split_name: ratio / total_ratio for split_name, ratio in raw_ratios.items()
    }


@register_tool
class SplitDatasetTool(BaseTool):
    """Split datasets into train/val/test subsets and export each split."""

    name = "split_dataset"
    description = """Split an annotated dataset into train/val/test subsets and export each split.
Supports custom ratios, optional stratified splitting, and optional output format conversion."""
    category = ToolCategory.UTILITY

    parameters = [
        ToolParameter(
            name="path",
            type=str,
            description="Path to source dataset root directory",
            required=True,
        ),
        ToolParameter(
            name="output_path",
            type=str,
            description="Path to output directory where split datasets are written",
            required=True,
        ),
        ToolParameter(
            name="format",
            type=str,
            description="Optional source format override (auto-detected if omitted)",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="train_ratio",
            type=float,
            description="Training split ratio (normalized with val_ratio/test_ratio)",
            required=False,
            default=0.8,
        ),
        ToolParameter(
            name="val_ratio",
            type=float,
            description="Validation split ratio (normalized with train_ratio/test_ratio)",
            required=False,
            default=0.1,
        ),
        ToolParameter(
            name="test_ratio",
            type=float,
            description="Test split ratio (normalized with train_ratio/val_ratio)",
            required=False,
            default=0.1,
        ),
        ToolParameter(
            name="stratify",
            type=bool,
            description="Use class-aware stratified splitting",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="seed",
            type=int,
            description="Random seed for reproducible splits",
            required=False,
            default=42,
        ),
        ToolParameter(
            name="output_format",
            type=str,
            description="Optional output format override (defaults to source format)",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="copy_images",
            type=bool,
            description="Copy images into each split output dataset",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="overwrite",
            type=bool,
            description="Allow writing into non-empty split output directories",
            required=False,
            default=True,
        ),
    ]

    async def execute(
        self,
        path: str,
        output_path: str,
        format: str | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify: bool = True,
        seed: int = 42,
        output_format: str | None = None,
        copy_images: bool = True,
        overwrite: bool = True,
    ) -> ToolResult:
        """Split a dataset and export train/val/test subsets."""
        source = Path(path)
        output = Path(output_path)
        normalized_source_format = format.lower() if format else None
        normalized_output_format = output_format.lower() if output_format else None

        if not source.exists():
            return error_result(f"Input path not found: {path}")

        if not source.is_dir():
            return error_result(f"Input path must be a directory: {path}")

        if output.exists() and not output.is_dir():
            return error_result(f"Output path must be a directory: {output_path}")

        available_formats = list_formats()
        loader_formats = sorted(
            name for name, details in available_formats.items() if details.get("loader")
        )
        exporter_formats = sorted(
            name for name, details in available_formats.items() if details.get("exporter")
        )

        if normalized_source_format and normalized_source_format not in loader_formats:
            return error_result(
                f"Unsupported format '{format}'. Available source formats: {loader_formats}"
            )

        if normalized_output_format and normalized_output_format not in exporter_formats:
            return error_result(
                f"Unsupported output_format '{output_format}'. "
                f"Available output formats: {exporter_formats}"
            )

        if normalized_source_format is None:
            normalized_source_format = detect_format(source)
            if normalized_source_format is None:
                return error_result(
                    "Could not detect dataset format. "
                    f"Please provide format explicitly ({loader_formats})."
                )

        target_format = normalized_output_format or normalized_source_format
        if target_format not in exporter_formats:
            return error_result(
                f"Source format '{normalized_source_format}' does not support export. "
                f"Available output formats: {exporter_formats}"
            )

        try:
            ratios = _normalize_split_ratios(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
        except ValueError as exc:
            return error_result(str(exc))

        try:
            result_data = await asyncio.to_thread(
                self._split_and_export,
                source,
                output,
                normalized_source_format,
                target_format,
                ratios,
                stratify,
                seed,
                copy_images,
                overwrite,
            )
            return success_result(result_data)
        except Exception as exc:
            logger.exception("Dataset split failed")
            return error_result(
                f"Dataset split failed: {exc}",
                path=str(source),
                output_path=str(output),
                format=normalized_source_format,
                output_format=target_format,
            )

    def _split_and_export(
        self,
        source: Path,
        output: Path,
        source_format: str,
        output_format: str,
        ratios: dict[str, float],
        stratify: bool,
        seed: int,
        copy_images: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        """Load, split, and export dataset synchronously in a worker thread."""
        self.report_progress(0, 4, "Initializing dataset split")

        loader = get_loader(
            source,
            format_name=source_format,
            progress_callback=self.report_progress,
        )
        dataset_warnings = loader.validate()
        dataset = loader.load()
        self.report_progress(1, 4, f"Loaded {len(dataset)} samples from {source_format}")

        split_result = split_data(
            dataset,
            ratios=ratios,
            stratify=stratify,
            seed=seed,
        )
        self.report_progress(2, 4, "Dataset split complete")

        split_items = list(split_result.splits.items())
        split_stats: list[dict[str, Any]] = []
        export_warnings: list[str] = []

        export_total = max(1, len(split_items))
        for idx, (split_name, split_dataset) in enumerate(split_items, start=1):
            split_output = output / split_name
            exporter = get_exporter(
                split_output,
                output_format,
                overwrite=overwrite,
                progress_callback=self.report_progress,
            )
            exported_path = exporter.export(split_dataset, copy_images=copy_images)
            warnings = exporter.validate_export()
            export_warnings.extend(f"{split_name}: {warning}" for warning in warnings)

            split_stats.append({
                "name": split_name,
                "num_samples": len(split_dataset),
                "num_labels": split_dataset.total_labels(),
                "ratio": split_result.ratios.get(split_name, 0.0),
                "class_distribution": split_dataset.class_distribution(),
                "output_path": str(exported_path),
            })

            self.report_progress(
                idx,
                export_total,
                f"Exported {split_name} split ({len(split_dataset)} samples)",
            )

        self.report_progress(4, 4, "Split and export complete")

        return {
            "source_path": str(source),
            "output_path": str(output),
            "source_format": source_format,
            "output_format": output_format,
            "stratified": stratify,
            "seed": seed,
            "requested_ratios": ratios,
            "actual_ratios": split_result.ratios,
            "total_samples": len(dataset),
            "total_labels": dataset.total_labels(),
            "splits": split_stats,
            "dataset_warnings": dataset_warnings,
            "export_warnings": export_warnings,
            "copy_images": copy_images,
            "overwrite": overwrite,
        }
