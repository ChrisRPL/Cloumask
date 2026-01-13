"""Tests for CV agent tools.

These tests use mocked CV models to test tool logic without requiring
actual model weights or GPU resources.

Tests cover:
- DetectTool (COCO, open-vocab, SAM3 quality mode)
- SegmentTool (text, point, box prompts)
- AnonymizeTool (blur, blackbox, pixelate, mask modes)
- FaceDetectTool (SCRFD, YuNet, SAM3 quality mode)
- Detect3DTool (PV-RCNN++, CenterPoint)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.agent.tools import (
    AnonymizeTool,
    Detect3DTool,
    DetectTool,
    FaceDetectTool,
    SegmentTool,
)


# =============================================================================
# Test Fixtures (Mock CV Models)
# =============================================================================


@pytest.fixture
def mock_detection_result():
    """Create a mock DetectionResult."""
    from backend.cv.types import BBox, Detection, DetectionResult

    return DetectionResult(
        detections=[
            Detection(
                class_id=0,
                class_name="person",
                bbox=BBox(x=0.5, y=0.5, width=0.2, height=0.4),
                confidence=0.95,
            ),
            Detection(
                class_id=2,
                class_name="car",
                bbox=BBox(x=0.3, y=0.6, width=0.3, height=0.2),
                confidence=0.88,
            ),
        ],
        image_path="test.jpg",
        processing_time_ms=50.0,
        model_name="yolo11m",
    )


@pytest.fixture
def mock_segmentation_result():
    """Create a mock SegmentationResult."""
    from backend.cv.types import Mask, SegmentationResult

    import numpy as np

    # Create a simple binary mask
    mask_data = np.zeros((480, 640), dtype=np.uint8)
    mask_data[100:200, 150:350] = 1

    return SegmentationResult(
        masks=[Mask.from_numpy(mask_data, confidence=0.92)],
        image_path="test.jpg",
        processing_time_ms=350.0,
        model_name="sam3",
        prompt="red car",
    )


@pytest.fixture
def mock_face_detection_result():
    """Create a mock FaceDetectionResult."""
    from backend.cv.types import BBox, FaceDetection, FaceDetectionResult

    return FaceDetectionResult(
        faces=[
            FaceDetection(
                bbox=BBox(x=0.4, y=0.3, width=0.1, height=0.15),
                confidence=0.97,
                landmarks=[
                    (0.38, 0.28),  # left eye
                    (0.42, 0.28),  # right eye
                    (0.40, 0.32),  # nose
                    (0.37, 0.36),  # left mouth
                    (0.43, 0.36),  # right mouth
                ],
            ),
        ],
        image_path="test.jpg",
        processing_time_ms=25.0,
        model_name="scrfd-10g",
    )


@pytest.fixture
def mock_3d_detection_result():
    """Create a mock Detection3DResult."""
    from backend.cv.types import Detection3D, Detection3DResult

    return Detection3DResult(
        detections=[
            Detection3D(
                class_id=0,
                class_name="Car",
                center=(10.5, 2.3, 0.8),
                dimensions=(4.5, 1.8, 1.5),
                rotation=1.57,
                confidence=0.91,
            ),
            Detection3D(
                class_id=1,
                class_name="Pedestrian",
                center=(5.2, -1.1, 0.0),
                dimensions=(0.6, 0.6, 1.8),
                rotation=0.0,
                confidence=0.85,
            ),
        ],
        pointcloud_path="test.bin",
        processing_time_ms=180.0,
        model_name="pvrcnn++",
    )


@pytest.fixture
def mock_anonymization_result():
    """Create a mock AnonymizationResult."""
    from backend.cv.anonymization import AnonymizationResult

    return AnonymizationResult(
        output_path="/tmp/output.jpg",
        faces_anonymized=3,
        plates_anonymized=1,
        processing_time_ms=500.0,
        mode_used="blur",
    )


# =============================================================================
# DetectTool Tests
# =============================================================================


class TestDetectTool:
    """Tests for DetectTool with COCO, open-vocab, and SAM3 modes."""

    @pytest.mark.asyncio
    async def test_detect_coco_classes(self, temp_image, mock_detection_result):
        """Test detection with COCO classes uses YOLO11."""
        with patch("backend.cv.detection.get_detector") as mock_get:
            mock_detector = MagicMock()
            mock_detector.info.name = "yolo11m"
            mock_detector.predict_batch.return_value = [mock_detection_result]
            mock_get.return_value = mock_detector

            tool = DetectTool()
            result = await tool.run(
                input_path=str(temp_image),
                classes=["person", "car"],
            )

            assert result.success
            assert result.data["mode"] == "coco"
            assert result.data["model"] == "yolo11m"
            assert result.data["count"] == 2
            mock_detector.load.assert_called_once()
            mock_detector.unload.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_openvocab_auto_selection(self, temp_image, mock_detection_result):
        """Test that non-COCO classes auto-select YOLO-World."""
        with patch("backend.cv.openvocab.get_openvocab_detector") as mock_get:
            mock_detector = MagicMock()
            mock_detector.info.name = "yolo-world-l"
            mock_detector.predict_batch.return_value = [mock_detection_result]
            mock_get.return_value = mock_detector

            tool = DetectTool()
            result = await tool.run(
                input_path=str(temp_image),
                classes=["red car", "delivery truck"],  # Non-COCO
            )

            assert result.success
            assert result.data["mode"] == "openvocab"
            assert result.data["model"] == "yolo-world-l"

    @pytest.mark.asyncio
    async def test_detect_quality_mode(self, temp_image, mock_segmentation_result):
        """Test that quality=True uses SAM3."""
        with patch("backend.cv.segmentation.get_segmenter") as mock_get:
            mock_segmenter = MagicMock()
            mock_segmenter.info.name = "sam3"
            mock_segmenter.predict.return_value = mock_segmentation_result
            mock_get.return_value = mock_segmenter

            tool = DetectTool()
            result = await tool.run(
                input_path=str(temp_image),
                classes=["person"],
                quality=True,
            )

            assert result.success
            assert result.data["mode"] == "sam3"

    @pytest.mark.asyncio
    async def test_detect_invalid_path(self):
        """Test detection with non-existent path."""
        tool = DetectTool()
        result = await tool.run(input_path="/nonexistent/path.jpg")

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_detect_invalid_confidence(self, temp_image):
        """Test detection with invalid confidence value."""
        tool = DetectTool()
        result = await tool.run(input_path=str(temp_image), confidence=1.5)

        assert not result.success
        assert "confidence" in result.error.lower()


# =============================================================================
# SegmentTool Tests
# =============================================================================


class TestSegmentTool:
    """Tests for SegmentTool with text, point, and box prompts."""

    @pytest.mark.asyncio
    async def test_segment_text_prompt(self, temp_image, mock_segmentation_result):
        """Test segmentation with text prompt uses SAM3."""
        with patch("backend.cv.segmentation.get_segmenter") as mock_get:
            mock_segmenter = MagicMock()
            mock_segmenter.info.name = "sam3"
            mock_segmenter.predict.return_value = mock_segmentation_result
            mock_get.return_value = mock_segmenter

            tool = SegmentTool()
            result = await tool.run(
                input_path=str(temp_image),
                prompt="red car",
            )

            assert result.success
            assert result.data["prompt_type"] == "text"
            assert result.data["model"] == "sam3"
            mock_get.assert_called_with(prompt_type="text", prefer_speed=False)

    @pytest.mark.asyncio
    async def test_segment_point_prompt(self, temp_image, mock_segmentation_result):
        """Test segmentation with point prompt."""
        with patch("backend.cv.segmentation.get_segmenter") as mock_get:
            mock_segmenter = MagicMock()
            mock_segmenter.info.name = "sam2"
            mock_segmenter.predict.return_value = mock_segmentation_result
            mock_get.return_value = mock_segmenter

            tool = SegmentTool()
            result = await tool.run(
                input_path=str(temp_image),
                point=[320, 240],
            )

            assert result.success
            assert result.data["prompt_type"] == "point"
            mock_get.assert_called_with(prompt_type="point", prefer_speed=False)

    @pytest.mark.asyncio
    async def test_segment_box_prompt(self, temp_image, mock_segmentation_result):
        """Test segmentation with box prompt."""
        with patch("backend.cv.segmentation.get_segmenter") as mock_get:
            mock_segmenter = MagicMock()
            mock_segmenter.info.name = "sam2"
            mock_segmenter.predict.return_value = mock_segmentation_result
            mock_get.return_value = mock_segmenter

            tool = SegmentTool()
            result = await tool.run(
                input_path=str(temp_image),
                box=[100, 100, 400, 300],
            )

            assert result.success
            assert result.data["prompt_type"] == "box"

    @pytest.mark.asyncio
    async def test_segment_no_prompt(self, temp_image):
        """Test segmentation without any prompt fails."""
        tool = SegmentTool()
        result = await tool.run(input_path=str(temp_image))

        assert not result.success
        assert "prompt" in result.error.lower()

    @pytest.mark.asyncio
    async def test_segment_return_masks(self, temp_image, mock_segmentation_result):
        """Test that return_masks includes mask data."""
        with patch("backend.cv.segmentation.get_segmenter") as mock_get:
            mock_segmenter = MagicMock()
            mock_segmenter.info.name = "sam3"
            mock_segmenter.predict.return_value = mock_segmentation_result
            mock_get.return_value = mock_segmenter

            tool = SegmentTool()
            result = await tool.run(
                input_path=str(temp_image),
                prompt="object",
                return_masks=True,
            )

            assert result.success
            assert "masks" in result.data
            assert len(result.data["masks"]) == 1
            assert "data_base64" in result.data["masks"][0]


# =============================================================================
# AnonymizeTool Tests
# =============================================================================


class TestAnonymizeTool:
    """Tests for AnonymizeTool with different modes."""

    @pytest.mark.asyncio
    async def test_anonymize_blur_mode(self, temp_image, tmp_path, mock_anonymization_result):
        """Test anonymization with blur mode."""
        output = tmp_path / "output.jpg"

        with patch("backend.cv.anonymization.AnonymizationPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.process.return_value = mock_anonymization_result
            mock_pipeline_cls.return_value = mock_pipeline

            tool = AnonymizeTool()
            result = await tool.run(
                input_path=str(temp_image),
                output_path=str(output),
                mode="blur",
            )

            assert result.success
            assert result.data["mode"] == "blur"
            mock_pipeline.load.assert_called_once()
            mock_pipeline.unload.assert_called_once()

    @pytest.mark.asyncio
    async def test_anonymize_quality_mode(self, temp_image, tmp_path, mock_anonymization_result):
        """Test anonymization with quality=True uses mask mode."""
        output = tmp_path / "output.jpg"

        with patch("backend.cv.anonymization.AnonymizationPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.process.return_value = mock_anonymization_result
            mock_pipeline_cls.return_value = mock_pipeline

            tool = AnonymizeTool()
            result = await tool.run(
                input_path=str(temp_image),
                output_path=str(output),
                quality=True,
            )

            assert result.success
            assert result.data["mode"] == "mask"  # Quality mode uses mask

    @pytest.mark.asyncio
    async def test_anonymize_faces_only(self, temp_image, tmp_path, mock_anonymization_result):
        """Test anonymization targeting faces only."""
        output = tmp_path / "output.jpg"

        with patch("backend.cv.anonymization.AnonymizationPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.process.return_value = mock_anonymization_result
            mock_pipeline_cls.return_value = mock_pipeline

            tool = AnonymizeTool()
            result = await tool.run(
                input_path=str(temp_image),
                output_path=str(output),
                target="faces",
            )

            assert result.success
            assert result.data["target"] == "faces"


# =============================================================================
# FaceDetectTool Tests
# =============================================================================


class TestFaceDetectTool:
    """Tests for FaceDetectTool with different modes."""

    @pytest.mark.asyncio
    async def test_face_detect_scrfd(self, temp_image, mock_face_detection_result):
        """Test face detection with SCRFD (default)."""
        with patch("backend.cv.faces.get_face_detector") as mock_get:
            mock_detector = MagicMock()
            mock_detector.info.name = "scrfd-10g"
            mock_detector.predict.return_value = mock_face_detection_result
            mock_get.return_value = mock_detector

            tool = FaceDetectTool()
            result = await tool.run(input_path=str(temp_image))

            assert result.success
            assert result.data["model"] == "scrfd-10g"
            assert result.data["mode"] == "accuracy"
            mock_get.assert_called_with(realtime=False)

    @pytest.mark.asyncio
    async def test_face_detect_realtime(self, temp_image, mock_face_detection_result):
        """Test face detection with YuNet (realtime mode)."""
        with patch("backend.cv.faces.get_face_detector") as mock_get:
            mock_detector = MagicMock()
            mock_detector.info.name = "yunet"
            mock_detector.predict.return_value = mock_face_detection_result
            mock_get.return_value = mock_detector

            tool = FaceDetectTool()
            result = await tool.run(input_path=str(temp_image), realtime=True)

            assert result.success
            assert result.data["mode"] == "realtime"
            mock_get.assert_called_with(realtime=True)

    @pytest.mark.asyncio
    async def test_face_detect_quality_mode(self, temp_image, mock_segmentation_result):
        """Test face detection with SAM3 quality mode."""
        with patch("backend.cv.segmentation.get_segmenter") as mock_get:
            mock_segmenter = MagicMock()
            mock_segmenter.info.name = "sam3"
            mock_segmenter.predict.return_value = mock_segmentation_result
            mock_get.return_value = mock_segmenter

            tool = FaceDetectTool()
            result = await tool.run(input_path=str(temp_image), quality=True)

            assert result.success
            assert result.data["mode"] == "sam3"


# =============================================================================
# Detect3DTool Tests
# =============================================================================


class TestDetect3DTool:
    """Tests for Detect3DTool with point cloud detection."""

    @pytest.mark.asyncio
    async def test_detect_3d_pvrcnn(self, temp_pointcloud, mock_3d_detection_result):
        """Test 3D detection with PV-RCNN++ (default)."""
        with patch("backend.cv.detection_3d.get_3d_detector") as mock_get:
            mock_detector = MagicMock()
            mock_detector.info.name = "pvrcnn++"
            mock_detector.predict.return_value = mock_3d_detection_result
            mock_get.return_value = mock_detector

            tool = Detect3DTool()
            result = await tool.run(input_path=str(temp_pointcloud))

            assert result.success
            assert result.data["count"] == 2
            assert "Car" in result.data["classes_found"]
            mock_get.assert_called_with(prefer_accuracy=True)

    @pytest.mark.asyncio
    async def test_detect_3d_centerpoint(self, temp_pointcloud, mock_3d_detection_result):
        """Test 3D detection with CenterPoint (faster)."""
        with patch("backend.cv.detection_3d.get_3d_detector") as mock_get:
            mock_detector = MagicMock()
            mock_detector.info.name = "centerpoint"
            mock_detector.predict.return_value = mock_3d_detection_result
            mock_get.return_value = mock_detector

            tool = Detect3DTool()
            result = await tool.run(
                input_path=str(temp_pointcloud),
                prefer_accuracy=False,
            )

            assert result.success
            mock_get.assert_called_with(prefer_accuracy=False)

    @pytest.mark.asyncio
    async def test_detect_3d_class_filter(self, temp_pointcloud, mock_3d_detection_result):
        """Test 3D detection with class filtering."""
        with patch("backend.cv.detection_3d.get_3d_detector") as mock_get:
            mock_detector = MagicMock()
            mock_detector.info.name = "pvrcnn++"
            mock_detector.predict.return_value = mock_3d_detection_result
            mock_get.return_value = mock_detector

            tool = Detect3DTool()
            result = await tool.run(
                input_path=str(temp_pointcloud),
                classes=["Car"],
            )

            assert result.success

    @pytest.mark.asyncio
    async def test_detect_3d_invalid_class(self, temp_pointcloud):
        """Test 3D detection with invalid class name."""
        tool = Detect3DTool()
        result = await tool.run(
            input_path=str(temp_pointcloud),
            classes=["InvalidClass"],
        )

        assert not result.success
        assert "invalid" in result.error.lower()

    @pytest.mark.asyncio
    async def test_detect_3d_invalid_format(self, tmp_path):
        """Test 3D detection with unsupported file format."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.touch()

        tool = Detect3DTool()
        result = await tool.run(input_path=str(invalid_file))

        assert not result.success
        assert "unsupported" in result.error.lower()


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Test that all CV tools are properly registered."""

    def test_all_cv_tools_registered(self):
        """Verify all CV tools are in the registry."""
        from backend.agent.tools import get_tool_registry, initialize_tools

        initialize_tools()
        registry = get_tool_registry()

        expected_tools = [
            "detect",
            "segment",
            "anonymize",
            "detect_faces",
            "detect_3d",
        ]

        for tool_name in expected_tools:
            assert registry.has(tool_name), f"Tool '{tool_name}' not registered"

    def test_cv_tools_have_descriptions(self):
        """Verify all CV tools have proper descriptions."""
        from backend.agent.tools import (
            AnonymizeTool,
            Detect3DTool,
            DetectTool,
            FaceDetectTool,
            SegmentTool,
        )

        tools = [DetectTool, SegmentTool, AnonymizeTool, FaceDetectTool, Detect3DTool]

        for tool_cls in tools:
            assert len(tool_cls.description) > 50, f"{tool_cls.name} description too short"
