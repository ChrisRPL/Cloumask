"""
API integration tests for 3D detection endpoints.

Tests the FastAPI routes for 3D object detection operations.
Uses mocked models to avoid requiring OpenPCDet installation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.cv.types import Detection3D, Detection3DResult


@pytest.fixture
def client() -> TestClient:
    """Create test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_bin_file(tmp_path: Path) -> Path:
    """Create sample KITTI binary point cloud file."""
    np.random.seed(42)
    points = np.random.rand(1000, 4).astype(np.float32)
    points[:, :3] = (points[:, :3] - 0.5) * 100  # Scale to -50 to 50 meters

    path = tmp_path / "sample.bin"
    points.tofile(str(path))
    return path


@pytest.fixture
def mock_detection_result() -> Detection3DResult:
    """Create mock detection result."""
    return Detection3DResult(
        detections=[
            Detection3D(
                class_id=0,
                class_name="Car",
                center=(10.0, 0.0, 0.5),
                dimensions=(4.5, 1.8, 1.5),
                rotation=0.1,
                confidence=0.95,
            ),
            Detection3D(
                class_id=1,
                class_name="Pedestrian",
                center=(15.0, 2.0, 0.8),
                dimensions=(0.6, 0.6, 1.7),
                rotation=0.0,
                confidence=0.87,
            ),
        ],
        pointcloud_path="/test/scan.bin",
        processing_time_ms=150.0,
        model_name="pvrcnn++",
    )


# -----------------------------------------------------------------------------
# GET /detect3d/models Tests
# -----------------------------------------------------------------------------


class TestModelsEndpoint:
    """Tests for GET /detect3d/models endpoint."""

    def test_list_models_success(self, client: TestClient) -> None:
        """Should return list of available models."""
        response = client.get("/detect3d/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 2

        # Check model info structure
        model_names = {m["name"] for m in data["models"]}
        assert "pvrcnn++" in model_names
        assert "centerpoint" in model_names

        # Check model has required fields
        pvrcnn = next(m for m in data["models"] if m["name"] == "pvrcnn++")
        assert "description" in pvrcnn
        assert "loaded" in pvrcnn
        assert "vram_required_mb" in pvrcnn
        assert "classes" in pvrcnn
        assert "benchmark" in pvrcnn

    def test_models_have_correct_vram(self, client: TestClient) -> None:
        """Should report correct VRAM requirements."""
        response = client.get("/detect3d/models")

        data = response.json()
        pvrcnn = next(m for m in data["models"] if m["name"] == "pvrcnn++")
        centerpoint = next(m for m in data["models"] if m["name"] == "centerpoint")

        assert pvrcnn["vram_required_mb"] == 4000
        assert centerpoint["vram_required_mb"] == 3000


# -----------------------------------------------------------------------------
# GET /detect3d/classes Tests
# -----------------------------------------------------------------------------


class TestClassesEndpoint:
    """Tests for GET /detect3d/classes endpoint."""

    def test_get_classes_success(self, client: TestClient) -> None:
        """Should return supported object classes."""
        response = client.get("/detect3d/classes")

        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert "Car" in data["classes"]
        assert "Pedestrian" in data["classes"]
        assert "Cyclist" in data["classes"]


# -----------------------------------------------------------------------------
# POST /detect3d/infer Tests
# -----------------------------------------------------------------------------


class TestInferEndpoint:
    """Tests for POST /detect3d/infer endpoint."""

    def test_infer_file_not_found(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Should return 404 for missing point cloud file."""
        nonexistent_file = tmp_path / "nonexistent.bin"
        response = client.post(
            "/detect3d/infer",
            json={
                "input_path": str(nonexistent_file),
                "model": "auto",
                "confidence": 0.3,
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_infer_invalid_file_extension(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Should return 400 for invalid file extension."""
        # Create a file with invalid extension
        invalid_file = tmp_path / "data.txt"
        invalid_file.write_text("test")

        response = client.post(
            "/detect3d/infer",
            json={
                "input_path": str(invalid_file),
                "model": "auto",
            },
        )

        assert response.status_code == 400
        assert "invalid file extension" in response.json()["detail"].lower()

    def test_infer_invalid_classes(
        self, client: TestClient, sample_bin_file: Path
    ) -> None:
        """Should return 400 for invalid class names."""
        response = client.post(
            "/detect3d/infer",
            json={
                "input_path": str(sample_bin_file),
                "model": "auto",
                "classes": ["InvalidClass"],
            },
        )

        assert response.status_code == 400
        assert "invalid classes" in response.json()["detail"].lower()

    def test_infer_invalid_confidence(self, client: TestClient) -> None:
        """Should return 422 for out-of-range confidence."""
        response = client.post(
            "/detect3d/infer",
            json={
                "input_path": "/some/path.bin",
                "confidence": 1.5,  # Invalid: > 1.0
            },
        )

        assert response.status_code == 422

    def test_infer_invalid_model(self, client: TestClient) -> None:
        """Should return 422 for invalid model name."""
        response = client.post(
            "/detect3d/infer",
            json={
                "input_path": "/some/path.bin",
                "model": "invalid_model",
            },
        )

        assert response.status_code == 422

    def test_infer_invalid_coordinate_system(self, client: TestClient) -> None:
        """Should return 422 for invalid coordinate system."""
        response = client.post(
            "/detect3d/infer",
            json={
                "input_path": "/some/path.bin",
                "coordinate_system": "invalid",
            },
        )

        assert response.status_code == 422

    @patch("backend.api.routes.detect3d.get_3d_detector")
    def test_infer_success_mocked(
        self,
        mock_get_detector: MagicMock,
        client: TestClient,
        sample_bin_file: Path,
        mock_detection_result: Detection3DResult,
    ) -> None:
        """Should return detection results with mocked detector."""
        # Setup mock detector
        mock_detector = MagicMock()
        mock_detector.info.name = "pvrcnn++"
        mock_detector.is_loaded = True
        mock_detector.predict.return_value = mock_detection_result
        mock_get_detector.return_value = mock_detector

        response = client.post(
            "/detect3d/infer",
            json={
                "input_path": str(sample_bin_file),
                "model": "auto",
                "confidence": 0.3,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["model_name"] == "pvrcnn++"
        assert len(data["detections"]) == 2

        # Check detection structure
        car = data["detections"][0]
        assert car["class_name"] == "Car"
        assert car["confidence"] == 0.95
        assert len(car["center"]) == 3
        assert len(car["dimensions"]) == 3

    @patch("backend.api.routes.detect3d.get_3d_detector")
    def test_infer_with_class_filter(
        self,
        mock_get_detector: MagicMock,
        client: TestClient,
        sample_bin_file: Path,
    ) -> None:
        """Should pass class filter to detector."""
        mock_detector = MagicMock()
        mock_detector.info.name = "pvrcnn++"
        mock_detector.is_loaded = True
        mock_detector.predict.return_value = Detection3DResult(
            detections=[],
            pointcloud_path=str(sample_bin_file),
            processing_time_ms=100.0,
            model_name="pvrcnn++",
        )
        mock_get_detector.return_value = mock_detector

        response = client.post(
            "/detect3d/infer",
            json={
                "input_path": str(sample_bin_file),
                "classes": ["Car"],
            },
        )

        assert response.status_code == 200
        mock_detector.predict.assert_called_once()
        call_kwargs = mock_detector.predict.call_args.kwargs
        assert call_kwargs["classes"] == ["Car"]


# -----------------------------------------------------------------------------
# POST /detect3d/load Tests
# -----------------------------------------------------------------------------


class TestLoadEndpoint:
    """Tests for POST /detect3d/load endpoint."""

    def test_load_invalid_model(self, client: TestClient) -> None:
        """Should return 422 for invalid model name."""
        response = client.post(
            "/detect3d/load",
            json={"model": "invalid"},
        )

        assert response.status_code == 422

    def test_load_invalid_device(self, client: TestClient) -> None:
        """Should return 422 for invalid device."""
        response = client.post(
            "/detect3d/load",
            json={"model": "pvrcnn++", "device": "tpu"},
        )

        assert response.status_code == 422

    @patch("backend.api.routes.detect3d.PVRCNNWrapper")
    def test_load_success_mocked(
        self, mock_wrapper_class: MagicMock, client: TestClient
    ) -> None:
        """Should load model successfully with mocked wrapper."""
        mock_wrapper = MagicMock()
        mock_wrapper.is_loaded = True
        mock_wrapper.device = "cpu"
        mock_wrapper_class.return_value = mock_wrapper

        response = client.post(
            "/detect3d/load",
            json={"model": "pvrcnn++", "device": "cpu"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model"] == "pvrcnn++"
        assert "loaded" in data["message"].lower()
        mock_wrapper.load.assert_called_once_with(device="cpu")

    @patch("backend.api.routes.detect3d.PVRCNNWrapper")
    def test_load_failure_runtime_error(
        self, mock_wrapper_class: MagicMock, client: TestClient
    ) -> None:
        """Should return 500 when model loading fails."""
        # Clear any cached models first
        from backend.api.routes import detect3d
        detect3d._loaded_models.clear()

        mock_wrapper = MagicMock()
        mock_wrapper.is_loaded = False  # Ensure not cached as loaded
        mock_wrapper.load.side_effect = RuntimeError("OpenPCDet not installed")
        mock_wrapper_class.return_value = mock_wrapper

        response = client.post(
            "/detect3d/load",
            json={"model": "pvrcnn++"},
        )

        assert response.status_code == 500
        assert "openpcdet" in response.json()["detail"].lower()


# -----------------------------------------------------------------------------
# POST /detect3d/unload Tests
# -----------------------------------------------------------------------------


class TestUnloadEndpoint:
    """Tests for POST /detect3d/unload endpoint."""

    def test_unload_not_loaded_model(self, client: TestClient) -> None:
        """Should succeed gracefully when model not loaded."""
        # Clear any loaded models from previous tests
        from backend.api.routes import detect3d
        detect3d._loaded_models.clear()

        response = client.post(
            "/detect3d/unload",
            json={"model": "pvrcnn++"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "not loaded" in data["message"].lower()

    def test_unload_invalid_model(self, client: TestClient) -> None:
        """Should return 422 for invalid model name."""
        response = client.post(
            "/detect3d/unload",
            json={"model": "invalid"},
        )

        assert response.status_code == 422

    @patch("backend.api.routes.detect3d._loaded_models", {"pvrcnn++": MagicMock()})
    def test_unload_loaded_model(self, client: TestClient) -> None:
        """Should unload a loaded model."""
        from backend.api.routes import detect3d

        mock_detector = MagicMock()
        detect3d._loaded_models["pvrcnn++"] = mock_detector

        response = client.post(
            "/detect3d/unload",
            json={"model": "pvrcnn++"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "unloaded" in data["message"].lower()
        mock_detector.unload.assert_called_once()


# -----------------------------------------------------------------------------
# Request Validation Tests
# -----------------------------------------------------------------------------


class TestRequestValidation:
    """Tests for request validation."""

    def test_infer_missing_input_path(self, client: TestClient) -> None:
        """Should return 422 when input_path is missing."""
        response = client.post(
            "/detect3d/infer",
            json={"model": "auto"},
        )

        assert response.status_code == 422

    def test_load_missing_model(self, client: TestClient) -> None:
        """Should return 422 when model is missing."""
        response = client.post(
            "/detect3d/load",
            json={"device": "cpu"},
        )

        assert response.status_code == 422

    def test_unload_missing_model(self, client: TestClient) -> None:
        """Should return 422 when model is missing."""
        response = client.post(
            "/detect3d/unload",
            json={},
        )

        assert response.status_code == 422
