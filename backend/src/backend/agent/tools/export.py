"""Comprehensive dataset export tool with filtering and splitting support.

Implements spec: 06-data-pipeline/26-export-tool
Integration points: backend/data/formats/* and backend/data/splitting.py
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
from backend.agent.tools.constants import SUPPORTED_EXPORT_FORMATS
from backend.agent.tools.registry import register_tool
from backend.data.formats import detect_format, get_exporter, get_loader, list_formats
from backend.data.models import Dataset
from backend.data.splitting import split_dataset as split_data

logger = logging.getLogger(__name__)

FORMAT_ALIASES = {
    "pascal": "voc",
    "pascal_voc": "voc",
}


def _normalize_format_name(format_name: str) -> str:
    normalized = format_name.strip().lower()
    return FORMAT_ALIASES.get(normalized, normalized)


def _normalize_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, float]:
    raw = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }

    for split_name, ratio in raw.items():
        if ratio < 0:
            raise ValueError(f"{split_name}_ratio must be >= 0, got {ratio}")

    total = sum(raw.values())
    if total <= 0:
        raise ValueError("At least one split ratio must be > 0")

    return {split_name: ratio / total for split_name, ratio in raw.items()}


def _dataset_stats(dataset: Dataset) -> dict[str, Any]:
    return {
        "num_samples": len(dataset),
        "num_labels": dataset.total_labels(),
        "num_classes": dataset.num_classes,
        "class_distribution": dataset.class_distribution(),
        "class_names": dataset.class_names,
    }


def _normalize_class_filter(classes: list[str] | None) -> list[str] | None:
    if classes is None:
        return None

    normalized: list[str] = []
    seen: set[str] = set()

    for class_name in classes:
        value = str(class_name).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    return normalized


@register_tool
class ExportTool(BaseTool):
    """Export datasets with optional class/confidence filtering and splitting."""

    name = "export"
    description = """Export an annotated dataset to any supported format.
Supports format auto-detection, optional class filtering, confidence thresholding,
optional train/val/test splitting, and image copy/link behavior."""
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
            description="Path to output directory for exported dataset",
            required=True,
        ),
        ToolParameter(
            name="output_format",
            type=str,
            description=(
                "Output format: yolo | coco | kitti | voc (pascal) | cvat | "
                "nuscenes | openlabel"
            ),
            required=True,
            enum_values=SUPPORTED_EXPORT_FORMATS,
        ),
        ToolParameter(
            name="source_format",
            type=str,
            description="Optional source format override (auto-detected if omitted)",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="classes",
            type=list,
            description="Optional class names to include (filter); others are excluded",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="min_confidence",
            type=float,
            description="Optional minimum confidence threshold in range [0, 1]",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="split",
            type=bool,
            description="Split dataset into train/val/test and export each split",
            required=False,
            default=False,
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
            description="Use class-aware stratified splitting when split=true",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="seed",
            type=int,
            description="Random seed for reproducible split assignment when split=true",
            required=False,
            default=42,
        ),
        ToolParameter(
            name="copy_images",
            type=bool,
            description="Copy images into exported dataset (false keeps path references)",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="overwrite",
            type=bool,
            description="Allow writing into non-empty output directories",
            required=False,
            default=True,
        ),
    ]

    async def execute(
        self,
        source_path: str,
        output_path: str,
        output_format: str,
        source_format: str | None = None,
        classes: list[str] | None = None,
        min_confidence: float | None = None,
        split: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify: bool = True,
        seed: int = 42,
        copy_images: bool = True,
        overwrite: bool = True,
    ) -> ToolResult:
        """Export dataset with filtering and optional splitting."""
        source = Path(source_path)
        output = Path(output_path)
        normalized_output_format = _normalize_format_name(output_format)
        normalized_source_format = (
            _normalize_format_name(source_format) if source_format else None
        )
        normalized_classes = _normalize_class_filter(classes)

        if not source.exists():
            return error_result(f"Source dataset not found: {source_path}")

        if not source.is_dir():
            return error_result(f"Source path must be a directory: {source_path}")

        if output.exists() and not output.is_dir():
            return error_result(f"Output path must be a directory: {output_path}")

        if min_confidence is not None and not 0.0 <= min_confidence <= 1.0:
            return error_result(
                f"Invalid min_confidence: {min_confidence}. Must be between 0 and 1."
            )

        if classes is not None and not normalized_classes:
            return error_result(
                "Invalid classes filter: provide at least one non-empty class name."
            )

        available_formats = list_formats()
        loader_formats = sorted(
            name for name, details in available_formats.items() if details.get("loader")
        )
        exporter_formats = sorted(
            name for name, details in available_formats.items() if details.get("exporter")
        )

        if normalized_output_format not in exporter_formats:
            return error_result(
                f"Unsupported output_format '{output_format}'. "
                f"Available output formats: {exporter_formats}"
            )

        if (
            normalized_source_format is not None
            and normalized_source_format not in loader_formats
        ):
            return error_result(
                f"Unsupported source_format '{source_format}'. "
                f"Available source formats: {loader_formats}"
            )

        if normalized_source_format is None:
            normalized_source_format = detect_format(source)
            if normalized_source_format is None:
                return error_result(
                    "Could not detect source format. "
                    f"Please provide source_format explicitly ({loader_formats})."
                )

        split_ratios: dict[str, float] | None = None
        if split:
            try:
                split_ratios = _normalize_split_ratios(
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                )
            except ValueError as exc:
                return error_result(str(exc))

        try:
            result_data = await asyncio.to_thread(
                self._export_dataset,
                source,
                output,
                normalized_source_format,
                normalized_output_format,
                normalized_classes,
                min_confidence,
                split,
                split_ratios,
                stratify,
                seed,
                copy_images,
                overwrite,
            )
            return success_result(result_data)
        except Exception as exc:
            logger.exception("Dataset export failed")
            return error_result(
                f"Dataset export failed: {exc}",
                source_path=str(source),
                output_path=str(output),
                source_format=normalized_source_format,
                output_format=normalized_output_format,
            )

    def _export_dataset(
        self,
        source: Path,
        output: Path,
        source_format: str,
        output_format: str,
        classes: list[str] | None,
        min_confidence: float | None,
        split: bool,
        split_ratios: dict[str, float] | None,
        stratify: bool,
        seed: int,
        copy_images: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        """Run export logic in a worker thread."""
        self.report_progress(0, 4, "Initializing dataset export")

        loader = get_loader(
            source,
            format_name=source_format,
            progress_callback=self.report_progress,
        )
        dataset_warnings = loader.validate()
        dataset = loader.load()
        source_stats = _dataset_stats(dataset)
        self.report_progress(1, 4, f"Loaded {len(dataset)} samples from {source_format}")

        filter_warnings: list[str] = []
        if classes:
            source_classes = set(dataset.class_names)
            missing_classes = sorted(set(classes) - source_classes)
            if missing_classes:
                filter_warnings.append(
                    f"Requested classes not found in source dataset: {missing_classes}"
                )

            dataset = dataset.filter_by_class(classes)

        if min_confidence is not None:
            filtered_samples = [sample.filter_by_confidence(min_confidence) for sample in dataset]
            dataset = Dataset(
                filtered_samples,
                name=dataset.name,
                class_names=list(dataset.class_names),
            )

        export_stats = _dataset_stats(dataset)
        self.report_progress(2, 4, "Applied dataset filters")

        export_warnings: list[str] = []
        split_stats: dict[str, dict[str, Any]] | None = None
        split_output_paths: dict[str, str] | None = None
        actual_split_ratios: dict[str, float] | None = None

        if split:
            split_result = split_data(
                dataset,
                ratios=split_ratios,
                stratify=stratify,
                seed=seed,
            )

            split_stats = {}
            split_output_paths = {}
            split_items = list(split_result.splits.items())
            export_total = max(1, len(split_items))

            for index, (split_name, split_dataset) in enumerate(split_items, start=1):
                split_output = output / split_name
                exporter = get_exporter(
                    split_output,
                    output_format,
                    overwrite=overwrite,
                    progress_callback=self.report_progress,
                )
                exported_path = exporter.export(split_dataset, copy_images=copy_images)
                warnings = exporter.validate_export()
                export_warnings.extend(
                    f"{split_name}: {warning}" for warning in warnings
                )

                split_output_paths[split_name] = str(exported_path)
                split_stats[split_name] = {
                    **_dataset_stats(split_dataset),
                    "ratio": split_result.ratios.get(split_name, 0.0),
                    "output_path": str(exported_path),
                }

                self.report_progress(
                    index,
                    export_total,
                    f"Exported {split_name} split ({len(split_dataset)} samples)",
                )

            actual_split_ratios = split_result.ratios
            output_location = str(output)
        else:
            exporter = get_exporter(
                output,
                output_format,
                overwrite=overwrite,
                progress_callback=self.report_progress,
            )
            exported_path = exporter.export(dataset, copy_images=copy_images)
            export_warnings.extend(exporter.validate_export())
            output_location = str(exported_path)

        self.report_progress(4, 4, "Dataset export complete")

        return {
            "source_path": str(source),
            "source_format": source_format,
            "output_path": output_location,
            "output_format": output_format,
            "format": output_format,
            "source_stats": source_stats,
            "stats": export_stats,
            "num_samples": export_stats["num_samples"],
            "num_labels": export_stats["num_labels"],
            "num_classes": export_stats["num_classes"],
            "class_distribution": export_stats["class_distribution"],
            "splits": split_stats,
            "split_output_paths": split_output_paths,
            "split_requested": split,
            "requested_split_ratios": split_ratios,
            "actual_split_ratios": actual_split_ratios,
            "stratified": stratify if split else None,
            "seed": seed if split else None,
            "filtered_classes": classes,
            "confidence_threshold": min_confidence,
            "images_copied": copy_images,
            "overwrite": overwrite,
            "dataset_warnings": dataset_warnings,
            "filter_warnings": filter_warnings,
            "export_warnings": export_warnings,
            "files_processed": export_stats["num_samples"],
        }
