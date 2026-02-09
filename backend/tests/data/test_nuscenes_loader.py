"""Tests for nuScenes format loader."""

import json
from pathlib import Path

import pytest

from backend.data.formats.nuscenes import NuscenesExporter, NuscenesLoader
from backend.data.models import Dataset


@pytest.fixture
def nuscenes_dataset(tmp_path: Path) -> Path:
    """Create a minimal nuScenes-like dataset fixture."""
    version_dir = tmp_path / "v1.0-mini"
    version_dir.mkdir()

    tables = {
        "category": [
            {"token": "cat_car", "name": "car"},
            {"token": "cat_pedestrian", "name": "pedestrian"},
        ],
        "attribute": [
            {"token": "attr_vehicle_parked", "name": "vehicle.parked"},
            {"token": "attr_pedestrian_moving", "name": "pedestrian.moving"},
        ],
        "visibility": [
            {"token": "v40", "level": "v40-60", "description": "40-60% visible"},
        ],
        "instance": [
            {"token": "inst_car_1", "category_token": "cat_car"},
            {"token": "inst_pedestrian_1", "category_token": "cat_pedestrian"},
        ],
        "scene": [
            {"token": "scene_1", "name": "scene-mini"},
        ],
        "sample": [
            {"token": "sample_1", "timestamp": 1_000_000, "scene_token": "scene_1"},
            {"token": "sample_2", "timestamp": 1_000_100, "scene_token": "scene_1"},
        ],
        "sensor": [
            {"token": "sensor_cam_front", "channel": "CAM_FRONT", "modality": "camera"},
            {"token": "sensor_cam_back", "channel": "CAM_BACK", "modality": "camera"},
            {"token": "sensor_lidar_top", "channel": "LIDAR_TOP", "modality": "lidar"},
        ],
        "calibrated_sensor": [
            {"token": "cal_cam_front", "sensor_token": "sensor_cam_front"},
            {"token": "cal_cam_back", "sensor_token": "sensor_cam_back"},
            {"token": "cal_lidar_top", "sensor_token": "sensor_lidar_top"},
        ],
        "ego_pose": [
            {"token": "pose_1", "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
            {"token": "pose_2", "translation": [1.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        ],
        "sample_data": [
            {
                "token": "sd_front_1",
                "sample_token": "sample_1",
                "ego_pose_token": "pose_1",
                "calibrated_sensor_token": "cal_cam_front",
                "filename": "samples/CAM_FRONT/frame_1.jpg",
                "width": 1600,
                "height": 900,
                "is_key_frame": True,
                "prev": "",
                "next": "sd_front_2",
            },
            {
                "token": "sd_back_1",
                "sample_token": "sample_1",
                "ego_pose_token": "pose_1",
                "calibrated_sensor_token": "cal_cam_back",
                "filename": "samples/CAM_BACK/frame_1.jpg",
                "width": 1600,
                "height": 900,
                "is_key_frame": True,
                "prev": "",
                "next": "sd_back_2",
            },
            {
                "token": "sd_lidar_1",
                "sample_token": "sample_1",
                "ego_pose_token": "pose_1",
                "calibrated_sensor_token": "cal_lidar_top",
                "filename": "samples/LIDAR_TOP/sweep_1.bin",
            },
            {
                "token": "sd_front_2",
                "sample_token": "sample_2",
                "ego_pose_token": "pose_2",
                "calibrated_sensor_token": "cal_cam_front",
                "filename": "samples/CAM_FRONT/frame_2.jpg",
                "width": 1600,
                "height": 900,
                "is_key_frame": True,
                "prev": "sd_front_1",
                "next": "",
            },
            {
                "token": "sd_back_2",
                "sample_token": "sample_2",
                "ego_pose_token": "pose_2",
                "calibrated_sensor_token": "cal_cam_back",
                "filename": "samples/CAM_BACK/frame_2.jpg",
                "width": 1600,
                "height": 900,
                "is_key_frame": True,
                "prev": "sd_back_1",
                "next": "",
            },
        ],
        "sample_annotation": [
            {
                "token": "ann_car_1",
                "sample_token": "sample_1",
                "instance_token": "inst_car_1",
                "attribute_tokens": ["attr_vehicle_parked"],
                "visibility_token": "v40",
                "translation": [10.0, 2.0, 0.5],
                "size": [4.2, 1.9, 1.6],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "num_lidar_pts": 15,
                "num_radar_pts": 2,
                "bbox": [100, 120, 240, 200],  # xywh in pixels
            },
            {
                "token": "ann_ped_1",
                "sample_token": "sample_1",
                "instance_token": "inst_pedestrian_1",
                "category_token": "cat_pedestrian",
                "attribute_tokens": ["attr_pedestrian_moving"],
                "translation": [12.0, 1.0, 0.3],
                "size": [0.6, 0.6, 1.7],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "bbox_2d": {"x1": 300, "y1": 200, "x2": 380, "y2": 460},  # xyxy in pixels
            },
            {
                "token": "ann_car_2",
                "sample_token": "sample_2",
                "instance_token": "inst_car_1",
                "attribute_tokens": [],
                "translation": [8.0, 4.0, 0.5],
                "size": [4.2, 1.9, 1.6],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
        ],
    }

    for table_name, rows in tables.items():
        (version_dir / f"{table_name}.json").write_text(json.dumps(rows))

    front_dir = tmp_path / "samples" / "CAM_FRONT"
    back_dir = tmp_path / "samples" / "CAM_BACK"
    front_dir.mkdir(parents=True)
    back_dir.mkdir(parents=True)
    (front_dir / "frame_1.jpg").touch()
    (front_dir / "frame_2.jpg").touch()
    (back_dir / "frame_1.jpg").touch()
    (back_dir / "frame_2.jpg").touch()

    return tmp_path


class TestNuscenesLoader:
    """Tests for NuscenesLoader."""

    def test_load_dataset(self, nuscenes_dataset: Path) -> None:
        """Test loading all configured camera samples."""
        loader = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT", "CAM_BACK"])
        dataset = loader.load()

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 4
        assert dataset.class_names == ["car", "pedestrian"]

    def test_class_names(self, nuscenes_dataset: Path) -> None:
        """Test class names are inferred from category table."""
        loader = NuscenesLoader(nuscenes_dataset)
        assert loader.get_class_names() == ["car", "pedestrian"]

    def test_annotations_and_attributes(self, nuscenes_dataset: Path) -> None:
        """Test labels include 2D+3D data and semantic attributes."""
        loader = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"])
        dataset = loader.load()

        sample = next(item for item in dataset if item.metadata["sample_token"] == "sample_1")
        assert len(sample.labels) == 2

        car_label = next(label for label in sample.labels if label.class_name == "car")
        assert car_label.track_id is not None
        assert car_label.attributes["vehicle.parked"] is True
        assert car_label.attributes["visibility"] == "v40-60"
        assert car_label.attributes["num_lidar_pts"] == 15
        assert car_label.attributes["bbox_source"] == "provided_2d"
        assert car_label.bbox.cx == pytest.approx(220 / 1600, rel=0.01)
        assert car_label.bbox.cy == pytest.approx(220 / 900, rel=0.01)
        assert "translation_3d" in car_label.attributes
        assert "size_3d" in car_label.attributes
        assert "rotation_3d" in car_label.attributes

    def test_without_3d_attributes(self, nuscenes_dataset: Path) -> None:
        """Test optional exclusion of 3D attributes."""
        loader = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"], with_3d=False)
        dataset = loader.load()

        first_label = dataset[0].labels[0]
        assert "translation_3d" not in first_label.attributes
        assert "size_3d" not in first_label.attributes
        assert "rotation_3d" not in first_label.attributes

    def test_placeholder_bbox_when_missing_2d(self, nuscenes_dataset: Path) -> None:
        """Test fallback bbox when 2D extension fields are missing."""
        loader = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"])
        dataset = loader.load()

        sample = next(item for item in dataset if item.metadata["sample_token"] == "sample_2")
        label = sample.labels[0]
        assert label.attributes["bbox_source"] == "placeholder"
        assert label.bbox.cx == pytest.approx(0.5)
        assert label.bbox.cy == pytest.approx(0.5)

    def test_camera_filter_and_metadata(self, nuscenes_dataset: Path) -> None:
        """Test camera filtering and sample linking metadata."""
        loader = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"])
        dataset = loader.load()

        assert len(dataset) == 2
        assert all(sample.metadata["camera"] == "CAM_FRONT" for sample in dataset)

        first = next(sample for sample in dataset if sample.metadata["sample_token"] == "sample_1")
        assert first.metadata["sample_data_token"] == "sd_front_1"
        assert first.metadata["sample_data_next"] == "sd_front_2"
        assert first.metadata["scene_token"] == "scene_1"
        assert first.metadata["ego_pose_token"] == "pose_1"

    def test_validate(self, nuscenes_dataset: Path) -> None:
        """Test validation for a valid mini fixture."""
        loader = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT", "CAM_BACK"])
        warnings = loader.validate()
        assert warnings == []

    def test_summary(self, nuscenes_dataset: Path) -> None:
        """Test summary contains nuScenes-specific counts."""
        loader = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT", "CAM_BACK"])
        summary = loader.summary()

        assert summary["format"] == "nuscenes"
        assert summary["requested_version"] == "v1.0-trainval"
        assert summary["version"] == "v1.0-mini"
        assert summary["num_samples"] == 2
        assert summary["num_sample_data"] == 5
        assert summary["num_annotations"] == 3
        assert summary["num_categories"] == 2


class TestNuscenesExporter:
    """Tests for NuscenesExporter."""

    def test_export_basic_tables_and_images(self, nuscenes_dataset: Path, tmp_path: Path) -> None:
        """Test exporter writes expected table files and copies images."""
        dataset = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"]).load()

        output = tmp_path / "export"
        exporter = NuscenesExporter(output)
        exported = exporter.export(dataset)

        assert exported == output

        version_dir = output / "v1.0-custom"
        assert version_dir.exists()
        assert (version_dir / "category.json").exists()
        assert (version_dir / "sample.json").exists()
        assert (version_dir / "sample_data.json").exists()
        assert (version_dir / "sample_annotation.json").exists()
        assert (version_dir / "instance.json").exists()

        exported_images = list((output / "samples" / "CAM_FRONT").glob("*.jpg"))
        assert len(exported_images) == len(dataset)

    def test_export_token_linking_and_3d_fields(
        self, nuscenes_dataset: Path, tmp_path: Path
    ) -> None:
        """Test token references are valid and 3D fields are exported."""
        dataset = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"]).load()
        output = tmp_path / "export"
        NuscenesExporter(output).export(dataset)

        version_dir = output / "v1.0-custom"
        with (version_dir / "category.json").open() as file:
            categories = json.load(file)
        with (version_dir / "sample.json").open() as file:
            samples = json.load(file)
        with (version_dir / "sample_data.json").open() as file:
            sample_data = json.load(file)
        with (version_dir / "sample_annotation.json").open() as file:
            annotations = json.load(file)
        with (version_dir / "instance.json").open() as file:
            instances = json.load(file)

        sample_tokens = {sample["token"] for sample in samples}
        instance_tokens = {instance["token"] for instance in instances}
        category_by_token = {category["token"]: category["name"] for category in categories}

        assert all(entry["sample_token"] in sample_tokens for entry in sample_data)
        assert all(annotation["sample_token"] in sample_tokens for annotation in annotations)
        assert all(annotation["instance_token"] in instance_tokens for annotation in annotations)
        assert all(len(annotation["translation"]) == 3 for annotation in annotations)
        assert all(len(annotation["size"]) == 3 for annotation in annotations)
        assert all(len(annotation["rotation"]) == 4 for annotation in annotations)

        car_instance_tokens = {
            annotation["instance_token"]
            for annotation in annotations
            if category_by_token.get(annotation["category_token"]) == "car"
        }
        assert len(car_instance_tokens) == 1

    def test_export_roundtrip(self, nuscenes_dataset: Path, tmp_path: Path) -> None:
        """Test nuScenes load -> export -> load roundtrip."""
        original = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"]).load()

        output = tmp_path / "export"
        NuscenesExporter(output).export(original)
        reloaded = NuscenesLoader(output, version="v1.0-custom", cameras=["CAM_FRONT"]).load()

        assert len(reloaded) == len(original)
        assert reloaded.total_labels() == original.total_labels()
        assert reloaded.class_names == original.class_names

    def test_validate_export(self, nuscenes_dataset: Path, tmp_path: Path) -> None:
        """Test exported nuScenes structure passes exporter validation."""
        dataset = NuscenesLoader(nuscenes_dataset, cameras=["CAM_FRONT"]).load()

        output = tmp_path / "export"
        exporter = NuscenesExporter(output)
        exporter.export(dataset)

        warnings = exporter.validate_export()
        assert warnings == []
