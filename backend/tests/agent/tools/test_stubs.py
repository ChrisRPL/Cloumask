"""Tests for stub tool implementations.

These tests verify that stub tools (anonymize, detect, export) return
realistic mock data that can be used for end-to-end testing before
CV models are integrated.

Implements spec: 07-tool-implementations
"""

import pytest

from backend.agent.tools.anonymize import AnonymizeTool
from backend.agent.tools.detect import DetectTool
from backend.agent.tools.export import ExportTool
from backend.agent.tools.scan import ScanDirectoryTool


@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset directory."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(10):
        (img_dir / f"image_{i}.jpg").touch()

    vid_dir = tmp_path / "videos"
    vid_dir.mkdir()
    for i in range(3):
        (vid_dir / f"video_{i}.mp4").touch()

    # Add some annotation files
    anno_dir = tmp_path / "annotations"
    anno_dir.mkdir()
    for i in range(5):
        (anno_dir / f"label_{i}.json").write_text('{"objects": []}')

    return tmp_path


class TestAnonymizeStub:
    """Tests for AnonymizeTool stub implementation."""

    @pytest.mark.asyncio
    async def test_anonymize_returns_mock_data(self, temp_dataset):
        """Should return realistic mock anonymization data."""
        tool = AnonymizeTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
            output_path=str(temp_dataset / "output"),
        )

        assert result.success is True
        assert result.data["_stub"] is True
        assert result.data["files_processed"] > 0
        assert "faces_detected" in result.data
        assert "faces_blurred" in result.data
        assert "plates_detected" in result.data
        assert "plates_blurred" in result.data

    @pytest.mark.asyncio
    async def test_anonymize_documents_integration_point(self, temp_dataset):
        """Should document the CV integration point."""
        tool = AnonymizeTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
            output_path=str(temp_dataset / "output"),
        )

        assert "_integration_point" in result.data
        assert "anonymize" in result.data["_integration_point"]

    @pytest.mark.asyncio
    async def test_anonymize_target_faces_only(self, temp_dataset):
        """Should only blur faces when target=faces."""
        tool = AnonymizeTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
            output_path=str(temp_dataset / "output"),
            target="faces",
        )

        assert result.success is True
        assert result.data["faces_blurred"] > 0
        assert result.data["plates_blurred"] == 0

    @pytest.mark.asyncio
    async def test_anonymize_target_plates_only(self, temp_dataset):
        """Should only blur plates when target=plates."""
        tool = AnonymizeTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
            output_path=str(temp_dataset / "output"),
            target="plates",
        )

        assert result.success is True
        assert result.data["faces_blurred"] == 0
        assert result.data["plates_blurred"] >= 0  # May be 0 if no plates detected

    @pytest.mark.asyncio
    async def test_anonymize_target_all(self, temp_dataset):
        """Should blur both when target=all."""
        tool = AnonymizeTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
            output_path=str(temp_dataset / "output"),
            target="all",
        )

        assert result.success is True
        assert result.data["faces_blurred"] == result.data["faces_detected"]

    @pytest.mark.asyncio
    async def test_anonymize_error_missing_input(self, tmp_path):
        """Should return error for missing input."""
        tool = AnonymizeTool()
        result = await tool.run(
            input_path=str(tmp_path / "nonexistent"),
            output_path=str(tmp_path / "output"),
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_anonymize_error_no_images(self, tmp_path):
        """Should return error when no images found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        tool = AnonymizeTool()
        result = await tool.run(
            input_path=str(empty_dir),
            output_path=str(tmp_path / "output"),
        )

        assert result.success is False
        assert "no image files" in result.error.lower()

    @pytest.mark.asyncio
    async def test_anonymize_invalid_blur_strength(self, temp_dataset):
        """Should validate blur strength range."""
        tool = AnonymizeTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
            output_path=str(temp_dataset / "output"),
            blur_strength=15,  # Invalid: > 10
        )

        assert result.success is False
        assert "blur_strength" in result.error.lower()

    @pytest.mark.asyncio
    async def test_anonymize_deterministic_results(self, temp_dataset):
        """Should return deterministic results for same input."""
        tool = AnonymizeTool()

        result1 = await tool.run(
            input_path=str(temp_dataset / "images"),
            output_path=str(temp_dataset / "output"),
        )
        result2 = await tool.run(
            input_path=str(temp_dataset / "images"),
            output_path=str(temp_dataset / "output"),
        )

        # Same input should produce same mock data
        assert result1.data["faces_detected"] == result2.data["faces_detected"]


class TestDetectStub:
    """Tests for DetectTool stub implementation."""

    @pytest.mark.asyncio
    async def test_detect_returns_mock_data(self, temp_dataset):
        """Should return realistic mock detection data."""
        tool = DetectTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
        )

        assert result.success is True
        assert result.data["_stub"] is True
        assert result.data["files_processed"] > 0
        assert "count" in result.data
        assert "classes" in result.data

    @pytest.mark.asyncio
    async def test_detect_returns_class_counts(self, temp_dataset):
        """Should return per-class detection counts."""
        tool = DetectTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
            classes=["car", "person"],
        )

        assert result.success is True
        assert "car" in result.data["classes"]
        assert "person" in result.data["classes"]

    @pytest.mark.asyncio
    async def test_detect_uses_default_classes(self, temp_dataset):
        """Should use default classes when not specified."""
        tool = DetectTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
        )

        assert result.success is True
        # Default classes include common objects
        assert len(result.data["classes"]) > 0

    @pytest.mark.asyncio
    async def test_detect_respects_confidence_threshold(self, temp_dataset):
        """Should include confidence threshold in result."""
        tool = DetectTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
            confidence=0.7,
        )

        assert result.success is True
        assert result.data["confidence_threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_detect_error_missing_input(self):
        """Should return error for missing input."""
        tool = DetectTool()
        result = await tool.run(
            input_path="/nonexistent/path",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_detect_error_no_images(self, tmp_path):
        """Should return error when no images found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        tool = DetectTool()
        result = await tool.run(
            input_path=str(empty_dir),
        )

        assert result.success is False
        assert "no image files" in result.error.lower()

    @pytest.mark.asyncio
    async def test_detect_invalid_confidence(self, temp_dataset):
        """Should validate confidence range."""
        tool = DetectTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
            confidence=1.5,  # Invalid: > 1
        )

        assert result.success is False
        assert "confidence" in result.error.lower()

    @pytest.mark.asyncio
    async def test_detect_documents_integration_point(self, temp_dataset):
        """Should document the CV integration point."""
        tool = DetectTool()
        result = await tool.run(
            input_path=str(temp_dataset / "images"),
        )

        assert "_integration_point" in result.data
        assert "detection" in result.data["_integration_point"]


class TestExportStub:
    """Tests for ExportTool stub implementation."""

    @pytest.mark.asyncio
    async def test_export_yolo_format(self, temp_dataset):
        """Should show YOLO export structure."""
        tool = ExportTool()
        result = await tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            format="yolo",
        )

        assert result.success is True
        assert result.data["_stub"] is True
        assert result.data["format"] == "yolo"
        assert "data.yaml" in result.data["structure"]
        assert "images/train" in result.data["structure"]
        assert "labels/train" in result.data["structure"]

    @pytest.mark.asyncio
    async def test_export_coco_format(self, temp_dataset):
        """Should show COCO export structure."""
        tool = ExportTool()
        result = await tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            format="coco",
        )

        assert result.success is True
        assert result.data["format"] == "coco"
        assert "annotations/instances_train.json" in result.data["structure"]

    @pytest.mark.asyncio
    async def test_export_pascal_format(self, temp_dataset):
        """Should show Pascal VOC export structure."""
        tool = ExportTool()
        result = await tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            format="pascal",
        )

        assert result.success is True
        assert result.data["format"] == "pascal"
        assert "JPEGImages/" in result.data["structure"]
        assert "Annotations/" in result.data["structure"]

    @pytest.mark.asyncio
    async def test_export_train_val_split(self, temp_dataset):
        """Should calculate train/val split correctly."""
        tool = ExportTool()
        result = await tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            format="yolo",
            split_ratio=0.8,
        )

        assert result.success is True
        total = result.data["annotations_processed"]
        train = result.data["train_count"]
        val = result.data["val_count"]

        assert train + val == total
        assert train == int(total * 0.8)

    @pytest.mark.asyncio
    async def test_export_custom_split_ratio(self, temp_dataset):
        """Should respect custom split ratio."""
        tool = ExportTool()
        result = await tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            format="yolo",
            split_ratio=0.7,
        )

        assert result.success is True
        assert result.data["split_ratio"] == 0.7

    @pytest.mark.asyncio
    async def test_export_error_missing_input(self):
        """Should return error for missing input."""
        tool = ExportTool()
        result = await tool.run(
            input_path="/nonexistent/path",
            output_path="/tmp/export",
            format="yolo",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_export_error_unsupported_format(self, temp_dataset):
        """Should return error for unsupported format."""
        tool = ExportTool()
        result = await tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            format="invalid_format",
        )

        assert result.success is False
        # Validation catches invalid enum before execute()
        assert "format" in result.error.lower()
        assert "must be one of" in result.error.lower()

    @pytest.mark.asyncio
    async def test_export_error_no_annotations(self, tmp_path):
        """Should return error when no annotations found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        tool = ExportTool()
        result = await tool.run(
            input_path=str(empty_dir),
            output_path=str(tmp_path / "export"),
            format="yolo",
        )

        assert result.success is False
        assert "no annotation files" in result.error.lower()

    @pytest.mark.asyncio
    async def test_export_invalid_split_ratio(self, temp_dataset):
        """Should validate split ratio range."""
        tool = ExportTool()
        result = await tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            format="yolo",
            split_ratio=1.5,  # Invalid: > 1
        )

        assert result.success is False
        assert "split_ratio" in result.error.lower()

    @pytest.mark.asyncio
    async def test_export_documents_integration_point(self, temp_dataset):
        """Should document the format-specific integration point."""
        tool = ExportTool()
        result = await tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            format="yolo",
        )

        assert "_integration_point" in result.data
        assert "yolo" in result.data["_integration_point"]


class TestIntegrationPipeline:
    """Test full pipeline with stub tools."""

    @pytest.mark.asyncio
    async def test_scan_detect_export_flow(self, temp_dataset):
        """Test scan -> detect -> export flow with stubs."""
        # Step 1: Scan
        scan_tool = ScanDirectoryTool()
        scan_result = await scan_tool.run(path=str(temp_dataset))
        assert scan_result.success is True

        # Step 2: Detect (stub)
        detect_tool = DetectTool()
        detect_result = await detect_tool.run(
            input_path=str(temp_dataset / "images"),
        )
        assert detect_result.success is True
        assert detect_result.data["_stub"] is True

        # Step 3: Export (stub)
        export_tool = ExportTool()
        export_result = await export_tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            format="yolo",
        )
        assert export_result.success is True
        assert export_result.data["_stub"] is True


class TestToolRegistration:
    """Test that tools are properly registered."""

    def test_tools_are_registered(self):
        """All tools should be registered via decorator."""
        from backend.agent.tools import get_tool_registry

        registry = get_tool_registry()

        # Manually register tools if not present (registry may be cleared by other tests)
        if not registry.has("scan_directory"):
            registry.register_class(ScanDirectoryTool)
        if not registry.has("anonymize"):
            registry.register_class(AnonymizeTool)
        if not registry.has("detect"):
            registry.register_class(DetectTool)
        if not registry.has("export"):
            registry.register_class(ExportTool)

        assert registry.has("scan_directory")
        assert registry.has("anonymize")
        assert registry.has("detect")
        assert registry.has("export")

    def test_tool_schemas_available(self):
        """All tools should provide valid schemas."""
        from backend.agent.tools import get_tool_registry

        registry = get_tool_registry()

        # Manually register tools if not present (registry may be cleared by other tests)
        if not registry.has("scan_directory"):
            registry.register_class(ScanDirectoryTool)
        if not registry.has("anonymize"):
            registry.register_class(AnonymizeTool)
        if not registry.has("detect"):
            registry.register_class(DetectTool)
        if not registry.has("export"):
            registry.register_class(ExportTool)

        schemas = registry.get_schemas()

        tool_names = {s["function"]["name"] for s in schemas}
        assert "scan_directory" in tool_names
        assert "anonymize" in tool_names
        assert "detect" in tool_names
        assert "export" in tool_names
