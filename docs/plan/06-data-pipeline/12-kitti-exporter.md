# KITTI Format Exporter

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base, 05-kitti-loader
> **Blocks:** 22-convert-format-tool, 26-export-tool

## Objective

Implement an exporter for KITTI format. Generates one `.txt` label file per image with 15-field format including 3D information.

## Acceptance Criteria

- [ ] Generate valid KITTI label files (15 fields)
- [ ] Preserve 3D attributes if available
- [ ] Use default values for missing 3D data
- [ ] Copy images to output directory
- [ ] Validate exported dataset
- [ ] Unit tests with roundtrip conversion

## Implementation Steps

### 1. Add KittiExporter to kitti.py

Add to `backend/data/formats/kitti.py`:

```python
import shutil


@FormatRegistry.register_exporter
class KittiExporter(FormatExporter):
    """Export datasets to KITTI format.

    Creates:
    - training/image_2/ with images
    - training/label_2/ with labels

    Example:
        exporter = KittiExporter(Path("/output/kitti"))
        exporter.export(dataset)
    """

    format_name = "kitti"
    description = "KITTI format (autonomous driving)"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        split: str = "training",
    ) -> None:
        """Initialize KITTI exporter.

        Args:
            output_path: Output directory
            overwrite: Whether to overwrite existing files
            progress_callback: Progress callback
            split: Split name (training, testing)
        """
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.split = split

    def _format_label_line(self, label: "Label", img_width: int, img_height: int) -> str:
        """Format a label as KITTI line.

        Args:
            label: Label object
            img_width: Image width for denormalization
            img_height: Image height for denormalization

        Returns:
            KITTI format line (15 fields)
        """
        # Convert bbox to pixel coordinates
        x1, y1, x2, y2 = label.bbox.to_xyxy()
        left = x1 * img_width
        top = y1 * img_height
        right = x2 * img_width
        bottom = y2 * img_height

        # Get 3D attributes or use defaults
        attrs = label.attributes
        truncated = attrs.get("truncated", 0.0)
        occluded = attrs.get("occluded", 0)
        if isinstance(occluded, bool):
            occluded = 1 if occluded else 0
        alpha = attrs.get("alpha", -10.0)  # -10 = don't care

        dims_3d = attrs.get("dimensions_3d", {})
        height_3d = dims_3d.get("height", -1)
        width_3d = dims_3d.get("width", -1)
        length_3d = dims_3d.get("length", -1)

        loc_3d = attrs.get("location_3d", {})
        loc_x = loc_3d.get("x", -1000)
        loc_y = loc_3d.get("y", -1000)
        loc_z = loc_3d.get("z", -1000)

        rotation_y = attrs.get("rotation_y", -10.0)

        return (
            f"{label.class_name} "
            f"{truncated:.2f} "
            f"{occluded} "
            f"{alpha:.2f} "
            f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f} "
            f"{height_3d:.2f} {width_3d:.2f} {length_3d:.2f} "
            f"{loc_x:.2f} {loc_y:.2f} {loc_z:.2f} "
            f"{rotation_y:.2f}"
        )

    def export(
        self,
        dataset: "Dataset",
        *,
        copy_images: bool = True,
        image_subdir: str = "image_2",
    ) -> Path:
        """Export dataset to KITTI format.

        Args:
            dataset: Dataset to export
            copy_images: Whether to copy images
            image_subdir: Image subdirectory name

        Returns:
            Path to exported dataset
        """
        self._ensure_output_dir()

        # Create directories
        images_dir = self.output_path / self.split / image_subdir
        labels_dir = self.output_path / self.split / "label_2"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        samples = list(dataset)
        total = len(samples)

        for idx, sample in enumerate(samples):
            # Generate KITTI-style filename
            file_num = f"{idx:06d}"

            # Get image dimensions
            img_width = sample.image_width or 1242
            img_height = sample.image_height or 375

            # Copy image
            if copy_images and sample.image_path.exists():
                ext = sample.image_path.suffix
                dest = images_dir / f"{file_num}{ext}"
                if not dest.exists():
                    shutil.copy2(sample.image_path, dest)

            # Write label file
            lines = []
            for label in sample.labels:
                line = self._format_label_line(label, img_width, img_height)
                lines.append(line)

            label_path = labels_dir / f"{file_num}.txt"
            label_path.write_text("\n".join(lines) + "\n" if lines else "")

            self._report_progress(idx + 1, total, "Exporting KITTI")

        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported dataset.

        Returns:
            List of warning messages
        """
        warnings: list[str] = []

        images_dir = self.output_path / self.split / "image_2"
        labels_dir = self.output_path / self.split / "label_2"

        if not images_dir.exists():
            warnings.append("No image_2 directory")
        if not labels_dir.exists():
            warnings.append("No label_2 directory")

        if images_dir.exists() and labels_dir.exists():
            img_count = len(list(images_dir.glob("*")))
            lbl_count = len(list(labels_dir.glob("*.txt")))
            if img_count != lbl_count:
                warnings.append(f"Image/label count mismatch: {img_count} vs {lbl_count}")

        return warnings
```

### 2. Create unit tests

Add to `backend/tests/data/test_kitti_loader.py`:

```python
class TestKittiExporter:
    """Tests for KittiExporter."""

    def test_export_basic(self, kitti_dataset, tmp_path):
        """Test basic export."""
        from backend.data.formats.kitti import KittiLoader, KittiExporter

        loader = KittiLoader(kitti_dataset)
        ds = loader.load()

        output = tmp_path / "export"
        exporter = KittiExporter(output)
        exporter.export(ds)

        assert (output / "training" / "image_2").exists()
        assert (output / "training" / "label_2").exists()

    def test_export_preserves_3d(self, kitti_dataset, tmp_path):
        """Test 3D attributes are preserved."""
        from backend.data.formats.kitti import KittiLoader, KittiExporter

        loader = KittiLoader(kitti_dataset)
        ds = loader.load()

        output = tmp_path / "export"
        exporter = KittiExporter(output)
        exporter.export(ds)

        # Read back and check 3D values
        loader2 = KittiLoader(output)
        ds2 = loader2.load()

        # Compare attributes
        orig_label = ds[0].labels[0]
        new_label = ds2[0].labels[0]

        if "dimensions_3d" in orig_label.attributes:
            assert "dimensions_3d" in new_label.attributes

    def test_validate_export(self, kitti_dataset, tmp_path):
        """Test export validation."""
        from backend.data.formats.kitti import KittiLoader, KittiExporter

        loader = KittiLoader(kitti_dataset)
        ds = loader.load()

        output = tmp_path / "export"
        exporter = KittiExporter(output)
        exporter.export(ds)

        warnings = exporter.validate_export()
        assert len(warnings) == 0
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/kitti.py` | Modify | Add KittiExporter class |
| `backend/tests/data/test_kitti_loader.py` | Modify | Add exporter tests |

## Verification

```bash
cd backend
pytest tests/data/test_kitti_loader.py -v -k exporter
```

## Notes

- KITTI uses 0-indexed filenames (000000.txt)
- Missing 3D values use -1 or -1000 as placeholders
- Standard image size is 1242x375 pixels
- Truncated ranges 0-1, occluded is 0/1/2/3
- Alpha and rotation_y use -10 for "don't care"
