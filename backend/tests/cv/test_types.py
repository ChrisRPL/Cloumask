"""
Tests for CV types module.

Tests Pydantic models for detection, segmentation, and other CV primitives.

Implements spec: 03-cv-models/00-infrastructure (testing section)
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.cv.types import (
    BBox,
    Detection,
    Detection3D,
    DetectionResult,
    FaceDetection,
    Mask,
    PlateDetection,
    SegmentationResult,
)

# -----------------------------------------------------------------------------
# BBox Tests
# -----------------------------------------------------------------------------


class TestBBox:
    """Tests for BBox model."""

    def test_bbox_creation(self) -> None:
        """BBox should be created with normalized coordinates."""
        bbox = BBox(x=0.5, y=0.5, width=0.2, height=0.3)
        assert bbox.x == 0.5
        assert bbox.y == 0.5
        assert bbox.width == 0.2
        assert bbox.height == 0.3

    def test_bbox_to_xyxy(self) -> None:
        """to_xyxy should convert to pixel coordinates."""
        bbox = BBox(x=0.5, y=0.5, width=0.2, height=0.2)
        x1, y1, x2, y2 = bbox.to_xyxy(100, 100)

        assert x1 == 40
        assert y1 == 40
        assert x2 == 60
        assert y2 == 60

    def test_bbox_to_xyxy_clamping(self) -> None:
        """to_xyxy should clamp values to image bounds."""
        bbox = BBox(x=0.0, y=0.0, width=0.4, height=0.4)
        x1, y1, x2, y2 = bbox.to_xyxy(100, 100)

        # Should not go negative
        assert x1 >= 0
        assert y1 >= 0

    def test_bbox_to_xywh(self) -> None:
        """to_xywh should convert to x,y,width,height format."""
        bbox = BBox(x=0.5, y=0.5, width=0.2, height=0.2)
        x, y, w, h = bbox.to_xywh(100, 100)

        assert x == 40
        assert y == 40
        assert w == 20
        assert h == 20

    def test_bbox_from_xyxy(self) -> None:
        """from_xyxy should create BBox from pixel coordinates."""
        bbox = BBox.from_xyxy(40, 40, 60, 60, 100, 100)

        assert bbox.x == 0.5
        assert bbox.y == 0.5
        assert bbox.width == 0.2
        assert bbox.height == 0.2

    def test_bbox_area(self) -> None:
        """area property should compute normalized area."""
        bbox = BBox(x=0.5, y=0.5, width=0.2, height=0.3)
        assert bbox.area == pytest.approx(0.06)

    def test_bbox_validation(self) -> None:
        """BBox should validate coordinate ranges."""
        with pytest.raises(ValueError):
            BBox(x=1.5, y=0.5, width=0.2, height=0.2)  # x > 1

        with pytest.raises(ValueError):
            BBox(x=0.5, y=-0.1, width=0.2, height=0.2)  # y < 0

    def test_bbox_serialization(self) -> None:
        """BBox should serialize to JSON."""
        bbox = BBox(x=0.5, y=0.5, width=0.2, height=0.3)
        data = bbox.model_dump()

        assert data["x"] == 0.5
        assert data["y"] == 0.5
        assert data["width"] == 0.2
        assert data["height"] == 0.3


# -----------------------------------------------------------------------------
# Detection Tests
# -----------------------------------------------------------------------------


class TestDetection:
    """Tests for Detection model."""

    def test_detection_creation(self) -> None:
        """Detection should be created with class info and bbox."""
        bbox = BBox(x=0.5, y=0.5, width=0.2, height=0.2)
        detection = Detection(
            class_id=0,
            class_name="person",
            bbox=bbox,
            confidence=0.95,
        )

        assert detection.class_id == 0
        assert detection.class_name == "person"
        assert detection.confidence == 0.95

    def test_detection_confidence_validation(self) -> None:
        """Detection should validate confidence range."""
        bbox = BBox(x=0.5, y=0.5, width=0.2, height=0.2)

        with pytest.raises(ValueError):
            Detection(
                class_id=0,
                class_name="person",
                bbox=bbox,
                confidence=1.5,  # > 1
            )


class TestDetectionResult:
    """Tests for DetectionResult model."""

    def test_detection_result_creation(self) -> None:
        """DetectionResult should contain multiple detections."""
        bbox = BBox(x=0.5, y=0.5, width=0.2, height=0.2)
        detections = [
            Detection(class_id=0, class_name="person", bbox=bbox, confidence=0.9),
            Detection(class_id=1, class_name="car", bbox=bbox, confidence=0.8),
        ]

        result = DetectionResult(
            detections=detections,
            image_path="/test/image.jpg",
            processing_time_ms=50.0,
            model_name="yolo11m",
        )

        assert result.count == 2
        assert result.model_name == "yolo11m"

    def test_filter_by_confidence(self) -> None:
        """filter_by_confidence should filter detections."""
        bbox = BBox(x=0.5, y=0.5, width=0.2, height=0.2)
        detections = [
            Detection(class_id=0, class_name="person", bbox=bbox, confidence=0.9),
            Detection(class_id=1, class_name="car", bbox=bbox, confidence=0.5),
            Detection(class_id=2, class_name="dog", bbox=bbox, confidence=0.3),
        ]

        result = DetectionResult(
            detections=detections,
            image_path="/test/image.jpg",
            processing_time_ms=50.0,
            model_name="yolo11m",
        )

        filtered = result.filter_by_confidence(0.6)
        assert filtered.count == 1
        assert filtered.detections[0].class_name == "person"

    def test_filter_by_class(self) -> None:
        """filter_by_class should filter by class names."""
        bbox = BBox(x=0.5, y=0.5, width=0.2, height=0.2)
        detections = [
            Detection(class_id=0, class_name="person", bbox=bbox, confidence=0.9),
            Detection(class_id=1, class_name="car", bbox=bbox, confidence=0.8),
            Detection(class_id=2, class_name="dog", bbox=bbox, confidence=0.7),
        ]

        result = DetectionResult(
            detections=detections,
            image_path="/test/image.jpg",
            processing_time_ms=50.0,
            model_name="yolo11m",
        )

        filtered = result.filter_by_class(["person", "dog"])
        assert filtered.count == 2


# -----------------------------------------------------------------------------
# Mask Tests
# -----------------------------------------------------------------------------


class TestMask:
    """Tests for Mask model."""

    def test_mask_from_numpy(self) -> None:
        """from_numpy should create Mask from array."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[25:75, 25:75] = 1  # 50x50 square in center

        mask = Mask.from_numpy(arr, confidence=0.85)

        assert mask.width == 100
        assert mask.height == 100
        assert mask.confidence == 0.85
        assert len(mask.data) > 0  # Compressed data

    def test_mask_to_numpy(self) -> None:
        """to_numpy should reconstruct array."""
        original = np.zeros((50, 80), dtype=np.uint8)
        original[10:40, 20:60] = 1

        mask = Mask.from_numpy(original, confidence=0.9)
        reconstructed = mask.to_numpy()

        assert reconstructed.shape == (50, 80)
        np.testing.assert_array_equal(original, reconstructed)

    def test_mask_area_pixels(self) -> None:
        """area_pixels should count non-zero pixels."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[25:75, 25:75] = 1  # 50x50 = 2500 pixels

        mask = Mask.from_numpy(arr, confidence=0.9)
        assert mask.area_pixels == 2500


class TestSegmentationResult:
    """Tests for SegmentationResult model."""

    def test_segmentation_result_creation(self) -> None:
        """SegmentationResult should contain masks."""
        arr = np.ones((50, 50), dtype=np.uint8)
        masks = [
            Mask.from_numpy(arr, confidence=0.9),
            Mask.from_numpy(arr, confidence=0.8),
        ]

        result = SegmentationResult(
            masks=masks,
            image_path="/test/image.jpg",
            processing_time_ms=100.0,
            model_name="sam2",
            prompt="red car",
        )

        assert result.count == 2
        assert result.prompt == "red car"


# -----------------------------------------------------------------------------
# FaceDetection Tests
# -----------------------------------------------------------------------------


class TestFaceDetection:
    """Tests for FaceDetection model."""

    def test_face_detection_with_landmarks(self) -> None:
        """FaceDetection should support landmarks."""
        bbox = BBox(x=0.5, y=0.3, width=0.2, height=0.25)
        landmarks = [
            (0.45, 0.27),  # left eye
            (0.55, 0.27),  # right eye
            (0.50, 0.32),  # nose
            (0.47, 0.37),  # left mouth
            (0.53, 0.37),  # right mouth
        ]

        face = FaceDetection(
            bbox=bbox,
            confidence=0.98,
            landmarks=landmarks,
        )

        assert face.confidence == 0.98
        assert len(face.landmarks or []) == 5

    def test_face_detection_eye_distance(self) -> None:
        """get_eye_distance should compute inter-eye distance."""
        bbox = BBox(x=0.5, y=0.3, width=0.2, height=0.25)
        landmarks = [
            (0.4, 0.27),  # left eye at x=40 for 100px image
            (0.6, 0.27),  # right eye at x=60 for 100px image
            (0.5, 0.32),
            (0.47, 0.37),
            (0.53, 0.37),
        ]

        face = FaceDetection(bbox=bbox, confidence=0.98, landmarks=landmarks)
        distance = face.get_eye_distance(100, 100)

        assert distance == pytest.approx(20.0)

    def test_face_detection_no_landmarks(self) -> None:
        """get_eye_distance should return None without landmarks."""
        bbox = BBox(x=0.5, y=0.3, width=0.2, height=0.25)
        face = FaceDetection(bbox=bbox, confidence=0.98)

        assert face.get_eye_distance(100, 100) is None


# -----------------------------------------------------------------------------
# Detection3D Tests
# -----------------------------------------------------------------------------


class TestDetection3D:
    """Tests for Detection3D model."""

    def test_detection3d_creation(self) -> None:
        """Detection3D should store 3D bounding box info."""
        detection = Detection3D(
            class_id=0,
            class_name="car",
            center=(10.5, 5.0, 0.8),
            dimensions=(4.5, 1.8, 1.5),
            rotation=0.785,  # ~45 degrees
            confidence=0.92,
        )

        assert detection.class_name == "car"
        assert detection.center == (10.5, 5.0, 0.8)
        assert detection.rotation == pytest.approx(0.785)

    def test_detection3d_volume(self) -> None:
        """volume property should compute box volume."""
        detection = Detection3D(
            class_id=0,
            class_name="car",
            center=(0, 0, 0),
            dimensions=(4.0, 2.0, 1.5),  # 4 * 2 * 1.5 = 12 cubic meters
            rotation=0,
            confidence=0.9,
        )

        assert detection.volume == pytest.approx(12.0)


# -----------------------------------------------------------------------------
# PlateDetection Tests
# -----------------------------------------------------------------------------


class TestPlateDetection:
    """Tests for PlateDetection model."""

    def test_plate_detection_with_text(self) -> None:
        """PlateDetection should support OCR text."""
        bbox = BBox(x=0.5, y=0.7, width=0.15, height=0.05)
        plate = PlateDetection(
            bbox=bbox,
            confidence=0.95,
            text="ABC123",
            text_confidence=0.88,
        )

        assert plate.text == "ABC123"
        assert plate.text_confidence == 0.88

    def test_plate_detection_without_text(self) -> None:
        """PlateDetection should work without OCR text."""
        bbox = BBox(x=0.5, y=0.7, width=0.15, height=0.05)
        plate = PlateDetection(bbox=bbox, confidence=0.95)

        assert plate.text is None
        assert plate.text_confidence is None
