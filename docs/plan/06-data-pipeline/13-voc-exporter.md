# Pascal VOC Format Exporter

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base, 06-voc-loader
> **Blocks:** 22-convert-format-tool, 26-export-tool

## Objective

Implement an exporter for Pascal VOC XML format. Generates one XML annotation file per image.

## Acceptance Criteria

- [ ] Generate valid VOC XML structure
- [ ] Export bounding boxes in pixel coordinates
- [ ] Include difficult, truncated, occluded attributes
- [ ] Copy images to JPEGImages directory
- [ ] Generate ImageSets split files
- [ ] Unit tests with roundtrip conversion

## Implementation Steps

### 1. Add VocExporter to voc.py

Add to `backend/data/formats/voc.py`:

```python
import shutil


@FormatRegistry.register_exporter
class VocExporter(FormatExporter):
    """Export datasets to Pascal VOC format.

    Creates:
    - Annotations/ with XML files
    - JPEGImages/ with images
    - ImageSets/Main/ with split files

    Example:
        exporter = VocExporter(Path("/output/voc"))
        exporter.export(dataset)
    """

    format_name = "voc"
    description = "Pascal VOC XML format"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        split: str = "trainval",
    ) -> None:
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.split = split

    def _create_annotation_xml(
        self,
        sample: "Sample",
        filename: str,
    ) -> str:
        """Create VOC annotation XML string.

        Args:
            sample: Sample with labels
            filename: Image filename

        Returns:
            XML string
        """
        img_width = sample.image_width or 1920
        img_height = sample.image_height or 1080

        lines = [
            '<?xml version="1.0"?>',
            "<annotation>",
            f"  <filename>{filename}</filename>",
            "  <size>",
            f"    <width>{img_width}</width>",
            f"    <height>{img_height}</height>",
            "    <depth>3</depth>",
            "  </size>",
        ]

        for label in sample.labels:
            x1, y1, x2, y2 = label.bbox.to_xyxy()
            xmin = int(x1 * img_width)
            ymin = int(y1 * img_height)
            xmax = int(x2 * img_width)
            ymax = int(y2 * img_height)

            difficult = "1" if label.attributes.get("difficult") else "0"
            truncated = "1" if label.attributes.get("truncated") else "0"
            pose = label.attributes.get("pose", "Unspecified")

            lines.extend([
                "  <object>",
                f"    <name>{label.class_name}</name>",
                f"    <pose>{pose}</pose>",
                f"    <truncated>{truncated}</truncated>",
                f"    <difficult>{difficult}</difficult>",
                "    <bndbox>",
                f"      <xmin>{xmin}</xmin>",
                f"      <ymin>{ymin}</ymin>",
                f"      <xmax>{xmax}</xmax>",
                f"      <ymax>{ymax}</ymax>",
                "    </bndbox>",
                "  </object>",
            ])

        lines.append("</annotation>")
        return "\n".join(lines)

    def export(
        self,
        dataset: "Dataset",
        *,
        copy_images: bool = True,
        image_subdir: str = "JPEGImages",
    ) -> Path:
        """Export dataset to VOC format."""
        self._ensure_output_dir()

        ann_dir = self.output_path / "Annotations"
        images_dir = self.output_path / image_subdir
        sets_dir = self.output_path / "ImageSets" / "Main"

        ann_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        sets_dir.mkdir(parents=True, exist_ok=True)

        samples = list(dataset)
        total = len(samples)
        file_list = []

        for idx, sample in enumerate(samples):
            stem = sample.image_path.stem
            file_list.append(stem)

            # Copy image
            if copy_images and sample.image_path.exists():
                dest = images_dir / sample.image_path.name
                if not dest.exists():
                    shutil.copy2(sample.image_path, dest)

            # Write XML
            xml_content = self._create_annotation_xml(sample, sample.image_path.name)
            xml_path = ann_dir / f"{stem}.xml"
            xml_path.write_text(xml_content)

            self._report_progress(idx + 1, total, "Exporting VOC")

        # Write split file
        split_file = sets_dir / f"{self.split}.txt"
        split_file.write_text("\n".join(file_list) + "\n")

        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported dataset."""
        warnings: list[str] = []

        ann_dir = self.output_path / "Annotations"
        img_dir = self.output_path / "JPEGImages"

        if not ann_dir.exists():
            warnings.append("No Annotations directory")
        if not img_dir.exists():
            warnings.append("No JPEGImages directory")

        return warnings
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/voc.py` | Modify | Add VocExporter class |
| `backend/tests/data/test_voc_loader.py` | Modify | Add exporter tests |

## Verification

```bash
pytest tests/data/test_voc_loader.py -v -k exporter
```
