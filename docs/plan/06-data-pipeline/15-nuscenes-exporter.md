# nuScenes Format Exporter

> **Parent:** 06-data-pipeline
> **Depends on:** 01-data-models, 02-format-base, 08-nuscenes-loader
> **Blocks:** 22-convert-format-tool, 26-export-tool

## Objective

Implement an exporter for nuScenes format. Generates the table-based JSON structure.

## Acceptance Criteria

- [ ] Generate valid nuScenes table JSON files
- [ ] Export 3D annotations with position, rotation, size
- [ ] Create proper token-based linking
- [ ] Generate sample_data for images
- [ ] Unit tests with basic export

## Implementation Steps

### 1. Add NuscenesExporter to nuscenes.py

Add to `backend/data/formats/nuscenes.py`:

```python
import shutil
import uuid


def generate_token() -> str:
    """Generate a nuScenes-style token."""
    return uuid.uuid4().hex


@FormatRegistry.register_exporter
class NuscenesExporter(FormatExporter):
    """Export datasets to nuScenes format.

    Creates:
    - v1.0-custom/ with JSON tables
    - samples/ with images

    Note: This creates a simplified nuScenes structure.
    Full nuScenes requires sensor calibration data.
    """

    format_name = "nuscenes"
    description = "nuScenes format (simplified)"

    def __init__(
        self,
        output_path: Path,
        *,
        overwrite: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        version: str = "v1.0-custom",
    ) -> None:
        super().__init__(output_path, overwrite=overwrite, progress_callback=progress_callback)
        self.version = version

    def export(
        self,
        dataset: "Dataset",
        *,
        copy_images: bool = True,
        image_subdir: str = "samples/CAM_FRONT",
    ) -> Path:
        """Export dataset to nuScenes format."""
        self._ensure_output_dir()

        version_dir = self.output_path / self.version
        samples_dir = self.output_path / image_subdir
        version_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Build tables
        categories = []
        samples = []
        sample_data = []
        sample_annotations = []
        instances = {}

        # Create categories
        cat_tokens = {}
        for idx, name in enumerate(dataset.class_names):
            token = generate_token()
            cat_tokens[name] = token
            categories.append({
                "token": token,
                "name": name,
                "description": "",
            })

        # Process samples
        dataset_samples = list(dataset)
        total = len(dataset_samples)

        for idx, sample in enumerate(dataset_samples):
            sample_token = generate_token()
            sd_token = generate_token()

            # Sample
            samples.append({
                "token": sample_token,
                "timestamp": idx * 100000,
                "scene_token": generate_token(),
            })

            # Sample data
            filename = f"{image_subdir}/{sample.image_path.name}"
            sample_data.append({
                "token": sd_token,
                "sample_token": sample_token,
                "filename": filename,
                "width": sample.image_width or 1600,
                "height": sample.image_height or 900,
            })

            # Annotations
            for label in sample.labels:
                # Get or create instance
                if label.track_id is not None:
                    if label.track_id not in instances:
                        instances[label.track_id] = {
                            "token": generate_token(),
                            "category_token": cat_tokens.get(label.class_name, ""),
                        }
                    inst_token = instances[label.track_id]["token"]
                else:
                    inst_token = generate_token()

                # Get 3D data
                attrs = label.attributes
                translation = attrs.get("translation_3d", [0, 0, 0])
                size = attrs.get("size_3d", [1, 1, 1])
                rotation = attrs.get("rotation_3d", [1, 0, 0, 0])

                sample_annotations.append({
                    "token": generate_token(),
                    "sample_token": sample_token,
                    "instance_token": inst_token,
                    "category_token": cat_tokens.get(label.class_name, ""),
                    "translation": translation,
                    "size": size,
                    "rotation": rotation,
                    "attribute_tokens": [],
                })

            # Copy image
            if copy_images and sample.image_path.exists():
                dest = samples_dir / sample.image_path.name
                if not dest.exists():
                    shutil.copy2(sample.image_path, dest)

            self._report_progress(idx + 1, total, "Exporting nuScenes")

        # Write tables
        def write_table(name: str, data: list) -> None:
            path = version_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        write_table("category", categories)
        write_table("sample", samples)
        write_table("sample_data", sample_data)
        write_table("sample_annotation", sample_annotations)
        write_table("instance", list(instances.values()))
        write_table("attribute", [])
        write_table("visibility", [])
        write_table("sensor", [{"token": generate_token(), "channel": "CAM_FRONT", "modality": "camera"}])
        write_table("calibrated_sensor", [])
        write_table("ego_pose", [])
        write_table("scene", [])
        write_table("log", [])
        write_table("map", [])

        return self.output_path

    def validate_export(self) -> list[str]:
        """Validate exported dataset."""
        warnings: list[str] = []

        version_dir = self.output_path / self.version
        if not version_dir.exists():
            warnings.append(f"Version directory not found: {self.version}")
            return warnings

        required = ["category.json", "sample.json", "sample_annotation.json"]
        for table in required:
            if not (version_dir / table).exists():
                warnings.append(f"Missing table: {table}")

        return warnings
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/data/formats/nuscenes.py` | Modify | Add NuscenesExporter class |
| `backend/tests/data/test_nuscenes_loader.py` | Modify | Add exporter tests |

## Verification

```bash
pytest tests/data/test_nuscenes_loader.py -v -k exporter
```

## Notes

- This is a simplified nuScenes export
- Full nuScenes requires sensor calibration and ego pose
- Token linking maintains referential integrity
- Instance tokens link annotations across frames
