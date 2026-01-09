# COCO Format Exporter

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base, 04-coco-loader
> **Blocks:** 22-convert-format-tool, 26-export-tool

## Objective

Implement an exporter for COCO JSON format. Generates a single annotations JSON file with images, annotations, and categories.

## Acceptance Criteria

- [ ] Generate valid COCO JSON structure
- [ ] Export bounding boxes in COCO format (x, y, w, h in pixels)
- [ ] Support segmentation mask export (polygon or RLE)
- [ ] Generate unique annotation and image IDs
- [ ] Copy images to output directory
- [ ] Validate exported JSON
- [ ] Unit tests with roundtrip conversion

## Implementation Steps

### 1. Add CocoExporter to coco.py

Add to `backend/data/formats/coco.py`:

```python
import shutil
from datetime import datetime


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """Convert binary mask to polygon contours.

    Args:
        mask: Binary mask (H, W)

    Returns:
        List of polygon contours [[x1,y1,x2,y2,...], ...]
    """
    try:
        import cv2
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                poly = contour.flatten().tolist()
                polygons.append(poly)
        return polygons
    except ImportError:
        return []


@FormatRegistry.register_exporter
class CocoExporter(FormatExporter):
    """Export datasets to COCO JSON format.

    Creates:
    - annotations/instances_*.json
    - images/ directory with images

    Example:
        exporter = CocoExporter(Path("/output/coco"))
        exporter.export(dataset, split="train")
    """

    format_name = "coco"
    description = "COCO JSON format"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        split: str = "train",
        export_masks: bool = True,
    ) -> None:
        """Initialize COCO exporter.

        Args:
            output_path: Output directory
            overwrite: Whether to overwrite existing files
            progress_callback: Progress callback
            split: Split name for annotation file
            export_masks: Whether to export segmentation masks
        """
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.split = split
        self.export_masks = export_masks

    def export(
        self,
        dataset: "Dataset",
        *,
        copy_images: bool = True,
        image_subdir: str = "images",
    ) -> Path:
        """Export dataset to COCO format.

        Args:
            dataset: Dataset to export
            copy_images: Whether to copy images
            image_subdir: Subdirectory for images

        Returns:
            Path to exported dataset
        """
        self._ensure_output_dir()

        # Create directories
        images_dir = self.output_path / image_subdir
        ann_dir = self.output_path / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        # Build COCO structure
        coco_data = {
            "info": {
                "description": f"Exported from {dataset.name}",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # Add categories
        class_names = dataset.class_names
        for idx, name in enumerate(class_names):
            coco_data["categories"].append({
                "id": idx + 1,  # COCO uses 1-indexed
                "name": name,
                "supercategory": "object",
            })

        # Process samples
        ann_id = 1
        samples = list(dataset)
        total = len(samples)

        for img_id, sample in enumerate(samples, start=1):
            # Get image dimensions
            img_width = sample.image_width or 1920
            img_height = sample.image_height or 1080

            # Add image entry
            file_name = sample.image_path.name
            coco_data["images"].append({
                "id": img_id,
                "file_name": file_name,
                "width": img_width,
                "height": img_height,
            })

            # Copy image
            if copy_images and sample.image_path.exists():
                dest = images_dir / file_name
                if not dest.exists():
                    shutil.copy2(sample.image_path, dest)

            # Add annotations
            for label in sample.labels:
                # Convert bbox to COCO format (x, y, w, h in pixels)
                x, y, w, h = label.bbox.to_xywh()
                x_px = x * img_width
                y_px = y * img_height
                w_px = w * img_width
                h_px = h * img_height
                area = w_px * h_px

                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": label.class_id + 1,  # 1-indexed
                    "bbox": [x_px, y_px, w_px, h_px],
                    "area": area,
                    "iscrowd": 0,
                }

                # Add segmentation
                if self.export_masks and label.mask is not None:
                    polygons = mask_to_polygon(label.mask)
                    if polygons:
                        ann["segmentation"] = polygons
                elif "polygon" in label.attributes:
                    # Convert from normalized to pixel coordinates
                    poly = label.attributes["polygon"]
                    poly_px = []
                    for i, val in enumerate(poly):
                        if i % 2 == 0:
                            poly_px.append(val * img_width)
                        else:
                            poly_px.append(val * img_height)
                    ann["segmentation"] = [poly_px]
                else:
                    ann["segmentation"] = []

                # Add iscrowd from attributes
                if label.attributes.get("iscrowd"):
                    ann["iscrowd"] = 1

                coco_data["annotations"].append(ann)
                ann_id += 1

            self._report_progress(img_id, total, "Exporting COCO")

        # Write JSON
        ann_file = ann_dir / f"instances_{self.split}.json"
        with open(ann_file, "w") as f:
            json.dump(coco_data, f, indent=2)

        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported dataset.

        Returns:
            List of warning messages
        """
        warnings: list[str] = []

        ann_file = self.output_path / "annotations" / f"instances_{self.split}.json"
        if not ann_file.exists():
            warnings.append(f"Annotation file not found: {ann_file.name}")
            return warnings

        with open(ann_file) as f:
            data = json.load(f)

        if not data.get("images"):
            warnings.append("No images in annotation file")
        if not data.get("categories"):
            warnings.append("No categories in annotation file")

        # Check image files exist
        images_dir = self.output_path / "images"
        if images_dir.exists():
            for img in data.get("images", []):
                if not (images_dir / img["file_name"]).exists():
                    warnings.append(f"Missing image: {img['file_name']}")
                    break  # Just warn once

        return warnings
```

### 2. Create unit tests

Add to `backend/tests/data/test_coco_loader.py`:

```python
class TestCocoExporter:
    """Tests for CocoExporter."""

    def test_export_basic(self, coco_dataset, tmp_path):
        """Test basic export."""
        from backend.data.formats.coco import CocoLoader, CocoExporter

        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        output = tmp_path / "export"
        exporter = CocoExporter(output)
        exporter.export(ds)

        # Verify structure
        assert (output / "annotations" / "instances_train.json").exists()
        assert (output / "images").exists()

    def test_export_roundtrip(self, coco_dataset, tmp_path):
        """Test load -> export -> load roundtrip."""
        from backend.data.formats.coco import CocoLoader, CocoExporter

        loader = CocoLoader(coco_dataset)
        ds_original = loader.load()

        output = tmp_path / "export"
        exporter = CocoExporter(output)
        exporter.export(ds_original)

        loader2 = CocoLoader(output)
        ds_exported = loader2.load()

        assert len(ds_exported) == len(ds_original)

    def test_export_categories(self, coco_dataset, tmp_path):
        """Test categories are exported correctly."""
        from backend.data.formats.coco import CocoLoader, CocoExporter
        import json

        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        output = tmp_path / "export"
        exporter = CocoExporter(output)
        exporter.export(ds)

        with open(output / "annotations" / "instances_train.json") as f:
            data = json.load(f)

        assert len(data["categories"]) == len(ds.class_names)
        # COCO uses 1-indexed category IDs
        assert all(cat["id"] >= 1 for cat in data["categories"])

    def test_validate_export(self, coco_dataset, tmp_path):
        """Test export validation."""
        from backend.data.formats.coco import CocoLoader, CocoExporter

        loader = CocoLoader(coco_dataset)
        ds = loader.load()

        output = tmp_path / "export"
        exporter = CocoExporter(output)
        exporter.export(ds)

        warnings = exporter.validate_export()
        assert len(warnings) == 0
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/coco.py` | Modify | Add CocoExporter class |
| `backend/tests/data/test_coco_loader.py` | Modify | Add exporter tests |

## Verification

```bash
cd backend
pytest tests/data/test_coco_loader.py -v -k exporter
```

## Notes

- COCO uses 1-indexed category IDs (not 0-indexed)
- Bounding boxes are in absolute pixels, not normalized
- Segmentation can be polygon or RLE
- `iscrowd` indicates crowd regions
- Area is computed from bbox, not mask
