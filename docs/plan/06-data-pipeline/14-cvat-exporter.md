# CVAT Format Exporter

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base, 07-cvat-loader
> **Blocks:** 22-convert-format-tool, 26-export-tool

## Objective

Implement an exporter for CVAT XML format. Generates a single XML file with all annotations.

## Acceptance Criteria

- [ ] Generate valid CVAT for images 1.1 XML
- [ ] Export box and polygon annotations
- [ ] Preserve object attributes
- [ ] Support track annotations for video
- [ ] Copy images to output directory
- [ ] Unit tests with roundtrip conversion

## Implementation Steps

### 1. Add CvatExporter to cvat.py

Add to `backend/data/formats/cvat.py`:

```python
import shutil


@FormatRegistry.register_exporter
class CvatExporter(FormatExporter):
    """Export datasets to CVAT XML format.

    Creates:
    - annotations.xml
    - images/ directory

    Example:
        exporter = CvatExporter(Path("/output/cvat"))
        exporter.export(dataset)
    """

    format_name = "cvat"
    description = "CVAT XML format"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        task_name: str = "exported",
    ) -> None:
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.task_name = task_name

    def export(
        self,
        dataset: "Dataset",
        *,
        copy_images: bool = True,
        image_subdir: str = "images",
    ) -> Path:
        """Export dataset to CVAT format."""
        self._ensure_output_dir()

        images_dir = self.output_path / image_subdir
        images_dir.mkdir(parents=True, exist_ok=True)

        # Build XML
        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            "<annotations>",
            "  <version>1.1</version>",
            "  <meta>",
            "    <task>",
            f"      <name>{self.task_name}</name>",
            f"      <size>{len(dataset)}</size>",
            "      <labels>",
        ]

        # Add labels
        for class_name in dataset.class_names:
            lines.append(f"        <label><name>{class_name}</name></label>")

        lines.extend([
            "      </labels>",
            "    </task>",
            "  </meta>",
        ])

        # Add images
        samples = list(dataset)
        total = len(samples)

        for idx, sample in enumerate(samples):
            img_width = sample.image_width or 1920
            img_height = sample.image_height or 1080

            lines.append(
                f'  <image id="{idx}" name="{sample.image_path.name}" '
                f'width="{img_width}" height="{img_height}">'
            )

            for label in sample.labels:
                x1, y1, x2, y2 = label.bbox.to_xyxy()
                xtl = x1 * img_width
                ytl = y1 * img_height
                xbr = x2 * img_width
                ybr = y2 * img_height
                occluded = "1" if label.attributes.get("occluded") else "0"

                shape_type = label.attributes.get("shape_type", "box")

                if shape_type == "polygon" and "points" in label.attributes:
                    points = label.attributes["points"]
                    points_str = ";".join(
                        f"{points[i] * img_width},{points[i+1] * img_height}"
                        for i in range(0, len(points), 2)
                    )
                    lines.append(
                        f'    <polygon label="{label.class_name}" '
                        f'points="{points_str}" occluded="{occluded}" />'
                    )
                else:
                    lines.append(
                        f'    <box label="{label.class_name}" '
                        f'xtl="{xtl:.2f}" ytl="{ytl:.2f}" '
                        f'xbr="{xbr:.2f}" ybr="{ybr:.2f}" '
                        f'occluded="{occluded}" />'
                    )

            lines.append("  </image>")

            # Copy image
            if copy_images and sample.image_path.exists():
                dest = images_dir / sample.image_path.name
                if not dest.exists():
                    shutil.copy2(sample.image_path, dest)

            self._report_progress(idx + 1, total, "Exporting CVAT")

        lines.append("</annotations>")

        # Write XML
        xml_path = self.output_path / "annotations.xml"
        xml_path.write_text("\n".join(lines))

        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported dataset."""
        warnings: list[str] = []

        xml_path = self.output_path / "annotations.xml"
        if not xml_path.exists():
            warnings.append("annotations.xml not found")

        return warnings
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/cvat.py` | Modify | Add CvatExporter class |
| `backend/tests/data/test_cvat_loader.py` | Modify | Add exporter tests |

## Verification

```bash
pytest tests/data/test_cvat_loader.py -v -k exporter
```
