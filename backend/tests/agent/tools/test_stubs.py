"""Tests for stub tool implementations.

These tests verify that stub tools (export) return realistic mock data
that can be used for end-to-end testing before their CV models are integrated.

Note: AnonymizeTool and DetectTool have been upgraded to real implementations
with CV model integration. Their tests are in test_cv_tools.py.

Implements spec: 07-tool-implementations
"""

import pytest

from backend.agent.tools.export import ExportTool
from backend.agent.tools.scan import ScanDirectoryTool

# Fixtures temp_dataset, empty_dataset are defined in conftest.py


class TestExportStub:
    """Tests for ExportTool stub implementation."""

    @pytest.mark.asyncio
    async def test_export_yolo_format(self, temp_dataset):
        """Should show YOLO export structure."""
        tool = ExportTool()
        result = await tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            output_format="yolo",
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
            output_format="coco",
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
            output_format="pascal",
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
            output_format="yolo",
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
            output_format="yolo",
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
            output_format="yolo",
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
            output_format="invalid_format",
        )

        assert result.success is False
        # Validation catches invalid enum before execute()
        assert "output_format" in result.error.lower()
        assert "must be one of" in result.error.lower()

    @pytest.mark.asyncio
    async def test_export_error_no_annotations(self, empty_dataset):
        """Should return error when no annotations found."""
        tool = ExportTool()
        result = await tool.run(
            input_path=str(empty_dataset),
            output_path=str(empty_dataset.parent / "export"),
            output_format="yolo",
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
            output_format="yolo",
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
            output_format="yolo",
        )

        assert "_integration_point" in result.data
        assert "yolo" in result.data["_integration_point"]

    @pytest.mark.asyncio
    async def test_export_reports_progress(self, temp_dataset):
        """Should report progress during export."""
        progress_reports = []

        def capture_progress(current, total, message):
            progress_reports.append((current, total, message))

        tool = ExportTool()
        tool.set_progress_callback(capture_progress)
        result = await tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            output_format="yolo",
        )

        assert result.success is True
        # Should report progress for each annotation file
        assert len(progress_reports) == result.data["annotations_processed"]


class TestIntegrationPipeline:
    """Test pipeline with scan and export tools.

    Note: Full pipeline tests with CV tools (detect, anonymize, segment, etc.)
    are in test_cv_tools.py with mocked CV dependencies.
    """

    @pytest.mark.asyncio
    async def test_scan_export_flow(self, temp_dataset):
        """Test scan -> export flow."""
        # Step 1: Scan
        scan_tool = ScanDirectoryTool()
        scan_result = await scan_tool.run(path=str(temp_dataset))
        assert scan_result.success is True
        assert scan_result.data["total_files"] > 0

        # Step 2: Export (stub)
        export_tool = ExportTool()
        export_result = await export_tool.run(
            input_path=str(temp_dataset / "annotations"),
            output_path=str(temp_dataset / "export"),
            output_format="yolo",
        )
        assert export_result.success is True
        assert export_result.data["_stub"] is True


class TestToolRegistration:
    """Test that tools are properly registered."""

    def test_tools_are_registered(self):
        """All tools should be registered via decorator."""
        from backend.agent.tools import get_tool_registry, initialize_tools

        # Ensure tools are initialized
        initialize_tools()

        registry = get_tool_registry()

        assert registry.has("scan_directory")
        assert registry.has("anonymize")
        assert registry.has("detect")
        assert registry.has("export")
        assert registry.has("segment")
        assert registry.has("detect_faces")
        assert registry.has("detect_3d")

    def test_tool_schemas_available(self):
        """All tools should provide valid schemas."""
        from backend.agent.tools import get_tool_registry, initialize_tools

        # Ensure tools are initialized
        initialize_tools()

        registry = get_tool_registry()
        schemas = registry.get_schemas()

        tool_names = {s["function"]["name"] for s in schemas}
        assert "scan_directory" in tool_names
        assert "anonymize" in tool_names
        assert "detect" in tool_names
        assert "export" in tool_names
        assert "segment" in tool_names
        assert "detect_faces" in tool_names
        assert "detect_3d" in tool_names
