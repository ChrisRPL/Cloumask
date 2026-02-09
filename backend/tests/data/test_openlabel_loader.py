"""Tests for OpenLABEL format loader."""

import json
from pathlib import Path

import pytest

from backend.data.formats.openlabel import OpenlabelLoader
from backend.data.models import Dataset


@pytest.fixture
def openlabel_dataset(tmp_path: Path) -> Path:
    """Create a sample OpenLABEL dataset."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "frame_000000.jpg").touch()
    (images_dir / "frame_000001.jpg").touch()

    openlabel_data = {
        "openlabel": {
            "metadata": {
                "schema_version": "1.0.0",
                "annotator": "test-suite",
            },
            "coordinate_systems": {
                "world": {"type": "global_cs"},
                "vehicle_base": {"type": "local_cs", "parent": "world"},
                "cam_front": {"type": "sensor_cs", "parent": "vehicle_base"},
            },
            "streams": {
                "CAM_FRONT": {"description": "front"},
                "CAM_REAR": {"description": "rear"},
            },
            "objects": {
                "obj1": {
                    "name": "car_001",
                    "type": "car",
                },
                "obj2": {
                    "name": "ped_001",
                    "type": "pedestrian",
                },
                "obj3": {
                    "name": "cyclist_001",
                    "type": "cyclist",
                },
            },
            "actions": {
                "act_drive": {
                    "name": "driving",
                    "type": "motion",
                    "rdf_subjects": [{"type": "object", "uid": "obj1"}],
                    "frame_intervals": [{"frame_start": 0, "frame_end": 1}],
                }
            },
            "events": {
                "evt_brake": {
                    "name": "hard_brake",
                    "type": "event",
                    "objects": [{"type": "object", "uid": "obj1"}],
                    "frame_intervals": [{"frame_start": 1, "frame_end": 1}],
                }
            },
            "relations": {
                "rel_follow": {
                    "name": "following",
                    "type": "spatial",
                    "rdf_subjects": [{"type": "object", "uid": "obj1"}],
                    "rdf_objects": [{"type": "object", "uid": "obj2"}],
                    "frame_intervals": [{"frame_start": 1, "frame_end": 1}],
                }
            },
            "frames": {
                "0": {
                    "frame_properties": {
                        "width": 1920,
                        "height": 1080,
                        "timestamp": 100,
                        "streams": {
                            "CAM_FRONT": {"uri": "images/frame_000000.jpg"},
                        },
                    },
                    "transforms": {
                        "vehicle_base->world": {
                            "translation": [0.0, 0.0, 0.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0],
                        }
                    },
                    "objects": {
                        "obj1": {
                            "object_data": {
                                "bbox": [
                                    {
                                        "name": "bbox2d",
                                        "val": [100, 200, 300, 150],
                                    }
                                ],
                                "cuboid": [
                                    {
                                        "name": "cuboid3d",
                                        "val": [10.0, 5.0, 0.5, 0, 0, 0, 1, 4.0, 1.8, 1.5],
                                    }
                                ],
                                "num": [{"name": "speed", "val": 12.5}],
                            }
                        }
                    },
                },
                "1": {
                    "frame_properties": {
                        "width": 1920,
                        "height": 1080,
                        "timestamp": 200,
                        "streams": {
                            "CAM_FRONT": {"uri": "images/frame_000001.jpg"},
                            "CAM_REAR": {"uri": "images/frame_000001.jpg"},
                        },
                    },
                    "relations": {
                        "rel_adjacent": {
                            "name": "adjacent",
                            "type": "spatial",
                            "subject": "obj2",
                            "target": "obj3",
                        }
                    },
                    "objects": {
                        "obj1": {
                            "object_data": {
                                "bbox": [
                                    {
                                        "name": "bbox2d",
                                        "val": [110, 200, 300, 150],
                                    }
                                ]
                            }
                        },
                        "obj2": {
                            "object_data": {
                                "poly2d": [
                                    {
                                        "name": "ped_poly",
                                        "val": [500, 300, 580, 300, 580, 500, 500, 500],
                                    }
                                ],
                            }
                        },
                        "obj3": {
                            "object_data": {
                                "stream": "CAM_REAR",
                                "bbox": [{"name": "bbox2d", "val": [700, 300, 80, 160]}],
                            }
                        },
                    },
                },
            },
        }
    }

    (tmp_path / "annotations.json").write_text(json.dumps(openlabel_data))
    return tmp_path


class TestOpenlabelLoader:
    """Tests for OpenlabelLoader."""

    def test_load_dataset(self, openlabel_dataset: Path) -> None:
        """Test loading an OpenLABEL dataset."""
        loader = OpenlabelLoader(openlabel_dataset)
        dataset = loader.load()

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 2
        assert set(dataset.class_names) == {"car", "pedestrian", "cyclist"}

    def test_parse_bbox_and_poly2d(self, openlabel_dataset: Path) -> None:
        """Test bbox normalization and poly2d fallback."""
        loader = OpenlabelLoader(openlabel_dataset)
        dataset = loader.load()

        frame0 = dataset[0]
        assert len(frame0.labels) == 1
        car = frame0.labels[0]
        assert car.class_name == "car"
        assert car.bbox.w == pytest.approx(300 / 1920, rel=0.01)
        assert car.bbox.h == pytest.approx(150 / 1080, rel=0.01)

        frame1 = dataset[1]
        pedestrian = next(label for label in frame1.labels if label.class_name == "pedestrian")
        assert pedestrian.bbox.w == pytest.approx(80 / 1920, rel=0.01)
        assert pedestrian.bbox.h == pytest.approx(200 / 1080, rel=0.01)

    def test_parse_cuboid_and_named_attributes(self, openlabel_dataset: Path) -> None:
        """Test 3D cuboid extraction and object_data scalar attributes."""
        loader = OpenlabelLoader(openlabel_dataset)
        dataset = loader.load()

        label = dataset[0].labels[0]
        assert "cuboid_3d" in label.attributes
        assert "position" in label.attributes["cuboid_3d"]
        assert "rotation" in label.attributes["cuboid_3d"]
        assert "size" in label.attributes["cuboid_3d"]
        assert label.attributes["speed"] == 12.5

    def test_tracking_ids_are_stable(self, openlabel_dataset: Path) -> None:
        """Test that object UID maps to stable numeric track IDs."""
        loader = OpenlabelLoader(openlabel_dataset)
        dataset = loader.load()

        car_frame0 = dataset[0].labels[0]
        car_frame1 = next(label for label in dataset[1].labels if label.class_name == "car")
        assert car_frame0.track_id == car_frame1.track_id
        assert car_frame0.attributes["object_uid"] == car_frame1.attributes["object_uid"] == "obj1"

    def test_actions_events_relations(self, openlabel_dataset: Path) -> None:
        """Test frame metadata and label attributes for scene semantics."""
        loader = OpenlabelLoader(openlabel_dataset)
        dataset = loader.load()

        frame0 = dataset[0]
        frame1 = dataset[1]

        assert len(frame0.metadata["actions"]) == 1
        assert len(frame0.metadata["events"]) == 0
        assert len(frame0.metadata["relations"]) == 0

        assert len(frame1.metadata["actions"]) == 1
        assert len(frame1.metadata["events"]) == 1
        assert len(frame1.metadata["relations"]) == 2

        car = next(label for label in frame1.labels if label.class_name == "car")
        assert len(car.attributes["actions"]) == 1
        assert len(car.attributes["events"]) == 1
        assert any(item["uid"] == "rel_follow" for item in car.attributes["relations"])

    def test_coordinate_systems_and_transforms(self, openlabel_dataset: Path) -> None:
        """Test coordinate systems and frame transforms metadata."""
        loader = OpenlabelLoader(openlabel_dataset)
        samples = list(loader.iter_samples())

        assert "coordinate_systems" in samples[0].metadata
        assert "vehicle_base" in samples[0].metadata["coordinate_systems"]
        assert "transforms" in samples[0].metadata

    def test_stream_filter(self, openlabel_dataset: Path) -> None:
        """Test stream filter removes labels bound to excluded streams."""
        loader = OpenlabelLoader(openlabel_dataset, stream_filter=["CAM_FRONT"])
        dataset = loader.load()

        frame1_labels = {label.class_name for label in dataset[1].labels}
        assert "cyclist" not in frame1_labels
        assert "car" in frame1_labels
        assert "pedestrian" in frame1_labels

    def test_iter_samples(self, openlabel_dataset: Path) -> None:
        """Test lazy iteration."""
        loader = OpenlabelLoader(openlabel_dataset)
        samples = list(loader.iter_samples())
        assert len(samples) == 2

    def test_validate(self, openlabel_dataset: Path) -> None:
        """Test validation on a valid fixture."""
        loader = OpenlabelLoader(openlabel_dataset)
        warnings = loader.validate()
        assert warnings == []

    def test_validate_missing_json(self, tmp_path: Path) -> None:
        """Test validation for missing OpenLABEL JSON file."""
        loader = OpenlabelLoader(tmp_path)
        warnings = loader.validate()
        assert any("OpenLABEL JSON" in warning for warning in warnings)

    def test_summary(self, openlabel_dataset: Path) -> None:
        """Test summary method."""
        loader = OpenlabelLoader(openlabel_dataset)
        summary = loader.summary()

        assert summary["format"] == "openlabel"
        assert summary["num_frames"] == 2
        assert summary["num_objects"] == 3
        assert summary["num_streams"] == 2
        assert summary["num_actions"] == 1
        assert summary["num_events"] == 1
        assert summary["num_relations"] == 1
        assert summary["schema_version"] == "1.0.0"

    def test_explicit_json_file(self, openlabel_dataset: Path) -> None:
        """Test loading with explicit JSON filename."""
        source = openlabel_dataset / "annotations.json"
        target = openlabel_dataset / "custom_name.json"
        target.write_text(source.read_text())
        source.unlink()

        loader = OpenlabelLoader(openlabel_dataset, json_file="custom_name.json")
        dataset = loader.load()
        assert len(dataset) == 2
