# YOLO Format Exporter

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base, 03-yolo-loader
> **Blocks:** 22-convert-format-tool, 26-export-tool

## Objective

Implement an exporter for YOLO format. Generates `data.yaml` config and one `.txt` label file per image with normalized bounding boxes.

## Acceptance Criteria

- [ ] Generate valid `data.yaml` with class names and paths
- [ ] Write labels in YOLO format (class cx cy w h)
- [ ] Support segmentation polygon export
- [ ] Copy or link images to output directory
- [ ] Create train/val/test splits if provided
- [ ] Validate exported dataset
- [ ] Unit tests with roundtrip conversion

## Implementation Steps

### 1. Add YoloExporter to yolo.py

Add to `backend/data/formats/yolo.py`:

```python
import shutil
from typing import Sequence

import yaml

from backend.data.formats.base import FormatExporter, FormatRegistry


@FormatRegistry.register_exporter
class YoloExporter(FormatExporter):
    """Export datasets to YOLO format.

    Creates:
    - data.yaml with class names and paths
    - images/ directory with images
    - labels/ directory with .txt label files

    Example:
        exporter = YoloExporter(Path("/output/yolo"))
        exporter.export(dataset)
    """

    format_name = "yolo"
    description = "YOLO v5/v8/v11 format"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        split_ratios: Optional[dict[str, float]] = None,
    ) -> None:
        """Initialize YOLO exporter.

        Args:
            output_path: Output directory
            overwrite: Whether to overwrite existing files
            progress_callback: Progress callback
            split_ratios: Split ratios e.g. {"train": 0.8, "val": 0.1, "test": 0.1}
        """
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.split_ratios = split_ratios

    def _write_label_file(self, sample: "Sample", label_path: Path) -> None:
        """Write labels to YOLO format file.

        Args:
            sample: Sample with labels
            label_path: Output path for label file
        """
        lines = []
        for label in sample.labels:
            cx, cy, w, h = label.bbox.to_cxcywh()

            # Check for polygon in attributes
            polygon = label.attributes.get("polygon")
            if polygon:
                # Segmentation format: class x1 y1 x2 y2 ...
                points_str = " ".join(f"{p:.6f}" for p in polygon)
                lines.append(f"{label.class_id} {points_str}")
            else:
                # Detection format: class cx cy w h
                lines.append(f"{label.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + "\n" if lines else "")

    def _create_data_yaml(
        self,
        class_names: list[str],
        splits: dict[str, str],
    ) -> None:
        """Create data.yaml file.

        Args:
            class_names: List of class names
            splits: Dict of split name to relative path
        """
        data = {
            "path": ".",
            "names": {i: name for i, name in enumerate(class_names)},
        }
        data.update(splits)

        yaml_path = self.output_path / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def export(
        self,
        dataset: "Dataset",
        *,
        copy_images: bool = True,
        image_subdir: str = "images",
    ) -> Path:
        """Export dataset to YOLO format.

        Args:
            dataset: Dataset to export
            copy_images: Whether to copy images
            image_subdir: Subdirectory for images

        Returns:
            Path to exported dataset
        """
        self._ensure_output_dir()

        class_names = dataset.class_names
        samples = list(dataset)
        total = len(samples)

        # Determine splits
        if self.split_ratios:
            from backend.data.splitting import split_indices
            splits_data = split_indices(len(samples), self.split_ratios)
        else:
            # Use metadata split if available, otherwise all to train
            splits_data = {"train": list(range(len(samples)))}
            for i, sample in enumerate(samples):
                split = sample.metadata.get("split", "train")
                if split not in splits_data:
                    splits_data[split] = []
                splits_data[split].append(i)

        # Create directories and export
        split_paths = {}
        processed = 0

        for split_name, indices in splits_data.items():
            images_dir = self.output_path / split_name / "images"
            labels_dir = self.output_path / split_name / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            split_paths[split_name] = f"{split_name}/images"

            for idx in indices:
                sample = samples[idx]

                # Copy/link image
                if copy_images and sample.image_path.exists():
                    dest_image = images_dir / sample.image_path.name
                    if not dest_image.exists():
                        shutil.copy2(sample.image_path, dest_image)

                # Write label file
                label_path = labels_dir / (sample.image_path.stem + ".txt")
                self._write_label_file(sample, label_path)

                processed += 1
                self._report_progress(processed, total, f"Exporting {split_name}")

        # Create data.yaml
        self._create_data_yaml(class_names, split_paths)

        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported dataset.

        Returns:
            List of warning messages
        """
        warnings: list[str] = []

        # Check data.yaml
        yaml_path = self.output_path / "data.yaml"
        if not yaml_path.exists():
            warnings.append("data.yaml not found")
            return warnings

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Check splits
        for split in ["train", "val", "test"]:
            if split in data:
                split_path = self.output_path / data[split]
                if not split_path.exists():
                    warnings.append(f"Split directory not found: {data[split]}")

        return warnings
```

### 2. Create unit tests

Add to `backend/tests/data/test_yolo_loader.py`:

```python
class TestYoloExporter:
    """Tests for YoloExporter."""

    def test_export_basic(self, yolo_dataset, tmp_path):
        """Test basic export."""
        from backend.data.formats.yolo import YoloLoader, YoloExporter

        # Load dataset
        loader = YoloLoader(yolo_dataset)
        ds = loader.load()

        # Export
        output = tmp_path / "export"
        exporter = YoloExporter(output)
        exporter.export(ds)

        # Verify structure
        assert (output / "data.yaml").exists()
        assert (output / "train" / "images").exists()
        assert (output / "train" / "labels").exists()

    def test_export_roundtrip(self, yolo_dataset, tmp_path):
        """Test load -> export -> load roundtrip."""
        from backend.data.formats.yolo import YoloLoader, YoloExporter

        # Load original
        loader = YoloLoader(yolo_dataset)
        ds_original = loader.load()

        # Export
        output = tmp_path / "export"
        exporter = YoloExporter(output)
        exporter.export(ds_original)

        # Load exported
        loader2 = YoloLoader(output)
        ds_exported = loader2.load()

        # Compare
        assert len(ds_exported) == len(ds_original)
        assert ds_exported.class_names == ds_original.class_names

    def test_export_with_splits(self, yolo_dataset, tmp_path):
        """Test export with custom splits."""
        from backend.data.formats.yolo import YoloLoader, YoloExporter

        loader = YoloLoader(yolo_dataset)
        ds = loader.load()

        output = tmp_path / "export"
        exporter = YoloExporter(output, split_ratios={"train": 0.5, "val": 0.5})
        exporter.export(ds)

        # Check both splits exist
        assert (output / "train").exists()
        assert (output / "val").exists()

    def test_validate_export(self, yolo_dataset, tmp_path):
        """Test export validation."""
        from backend.data.formats.yolo import YoloLoader, YoloExporter

        loader = YoloLoader(yolo_dataset)
        ds = loader.load()

        output = tmp_path / "export"
        exporter = YoloExporter(output)
        exporter.export(ds)

        warnings = exporter.validate_export()
        assert len(warnings) == 0
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/yolo.py` | Modify | Add YoloExporter class |
| `backend/tests/data/test_yolo_loader.py` | Modify | Add exporter tests |

## Verification

```bash
cd backend
pytest tests/data/test_yolo_loader.py -v -k exporter
```

## Notes

- Empty label file = image with no objects (valid in YOLO)
- Label files must have same name as images (different extension)
- data.yaml `names` can be dict or list format
- Segmentation uses polygon format: class x1 y1 x2 y2 ...
- Images are copied by default; use copy_images=False to skip
