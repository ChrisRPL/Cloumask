# OpenLABEL Format Exporter

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base, 09-openlabel-loader
> **Blocks:** 22-convert-format-tool, 26-export-tool

## Objective

Implement an exporter for OpenLABEL (ASAM) format. Generates a single JSON file with the OpenLABEL schema structure.

## Acceptance Criteria

- [ ] Generate valid OpenLABEL 1.0 JSON
- [ ] Export 2D bounding boxes
- [ ] Support 3D cuboid export
- [ ] Preserve object tracking
- [ ] Copy images to output directory
- [ ] Unit tests with roundtrip conversion

## Implementation Steps

### 1. Add OpenlabelExporter to openlabel.py

Add to `backend/data/formats/openlabel.py`:

```python
import shutil
from datetime import datetime


@FormatRegistry.register_exporter
class OpenlabelExporter(FormatExporter):
    """Export datasets to OpenLABEL format.

    Creates:
    - annotations.json with OpenLABEL schema
    - images/ directory

    Example:
        exporter = OpenlabelExporter(Path("/output/openlabel"))
        exporter.export(dataset)
    """

    format_name = "openlabel"
    description = "OpenLABEL (ASAM) format"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        export_3d: bool = True,
    ) -> None:
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.export_3d = export_3d

    def export(
        self,
        dataset: "Dataset",
        *,
        copy_images: bool = True,
        image_subdir: str = "images",
    ) -> Path:
        """Export dataset to OpenLABEL format."""
        self._ensure_output_dir()

        images_dir = self.output_path / image_subdir
        images_dir.mkdir(parents=True, exist_ok=True)

        # Build OpenLABEL structure
        openlabel = {
            "openlabel": {
                "metadata": {
                    "schema_version": "1.0.0",
                    "exporter": "cloumask",
                    "date_created": datetime.now().isoformat(),
                },
                "coordinate_systems": {},
                "streams": {},
                "objects": {},
                "frames": {},
            }
        }

        # Track objects across frames
        objects = openlabel["openlabel"]["objects"]
        frames = openlabel["openlabel"]["frames"]
        uid_map = {}  # track_id -> object_uid

        samples = list(dataset)
        total = len(samples)

        for frame_idx, sample in enumerate(samples):
            frame_id = str(frame_idx)
            img_width = sample.image_width or 1920
            img_height = sample.image_height or 1080

            frames[frame_id] = {
                "frame_properties": {
                    "timestamp": frame_idx * 100,
                    "width": img_width,
                    "height": img_height,
                },
                "objects": {},
            }

            for label in sample.labels:
                # Get or create object UID
                if label.track_id is not None:
                    if label.track_id not in uid_map:
                        uid = f"obj_{len(uid_map)}"
                        uid_map[label.track_id] = uid
                        objects[uid] = {
                            "name": label.attributes.get("object_name", f"{label.class_name}_{len(uid_map)}"),
                            "type": label.class_name,
                        }
                    uid = uid_map[label.track_id]
                else:
                    uid = f"obj_{frame_idx}_{len(frames[frame_id]['objects'])}"
                    objects[uid] = {
                        "name": uid,
                        "type": label.class_name,
                    }

                # Build object_data
                x, y, w, h = label.bbox.to_xywh()
                x_px = x * img_width
                y_px = y * img_height
                w_px = w * img_width
                h_px = h * img_height

                object_data = {
                    "bbox": [{
                        "name": "bbox2d",
                        "val": [x_px, y_px, w_px, h_px],
                    }],
                }

                # Add 3D cuboid if available
                if self.export_3d and "cuboid_3d" in label.attributes:
                    cuboid = label.attributes["cuboid_3d"]
                    pos = cuboid.get("position", [0, 0, 0])
                    rot = cuboid.get("rotation", [1, 0, 0, 0])
                    size = cuboid.get("size", [1, 1, 1])
                    object_data["cuboid"] = [{
                        "name": "cuboid3d",
                        "val": pos + rot + size,
                    }]

                frames[frame_id]["objects"][uid] = {
                    "object_data": object_data,
                }

            # Copy image
            if copy_images and sample.image_path.exists():
                dest = images_dir / sample.image_path.name
                if not dest.exists():
                    shutil.copy2(sample.image_path, dest)

            self._report_progress(frame_idx + 1, total, "Exporting OpenLABEL")

        # Write JSON
        json_path = self.output_path / "annotations.json"
        with open(json_path, "w") as f:
            json.dump(openlabel, f, indent=2)

        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported dataset."""
        warnings: list[str] = []

        json_path = self.output_path / "annotations.json"
        if not json_path.exists():
            warnings.append("annotations.json not found")
            return warnings

        with open(json_path) as f:
            data = json.load(f)

        if "openlabel" not in data:
            warnings.append("Missing openlabel root key")

        return warnings
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/openlabel.py` | Modify | Add OpenlabelExporter class |
| `backend/tests/data/test_openlabel_loader.py` | Modify | Add exporter tests |

## Verification

```bash
pytest tests/data/test_openlabel_loader.py -v -k exporter
```

## Notes

- OpenLABEL uses objects for persistent entities
- Frames contain per-frame object_data
- 3D cuboid format: [x, y, z, qx, qy, qz, qw, sx, sy, sz]
- Schema version 1.0.0 is the ASAM standard
