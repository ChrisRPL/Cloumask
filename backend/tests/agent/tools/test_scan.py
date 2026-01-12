"""Tests for ScanDirectoryTool.

Implements spec: 07-tool-implementations
"""

import pytest

from backend.agent.tools.scan import ScanDirectoryTool

# Fixtures temp_dataset, nested_dataset, pointcloud_dataset are defined in conftest.py


class TestScanDirectoryBasic:
    """Basic functionality tests for ScanDirectoryTool."""

    @pytest.mark.asyncio
    async def test_scan_directory_counts_files(self, temp_dataset):
        """Should correctly count all files in directory."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(temp_dataset))

        assert result.success is True
        # 10 images + 3 videos + 5 annotations = 18 files
        assert result.data["total_files"] == 18

    @pytest.mark.asyncio
    async def test_scan_directory_categorizes_types(self, temp_dataset):
        """Should correctly categorize files by type."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(temp_dataset))

        assert result.success is True
        assert result.data["type_counts"]["images"] == 10
        assert result.data["type_counts"]["videos"] == 3
        assert result.data["type_counts"]["annotations"] == 5
        assert result.data["type_counts"]["pointclouds"] == 0

    @pytest.mark.asyncio
    async def test_scan_directory_counts_subdirectories(self, temp_dataset):
        """Should count subdirectories."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(temp_dataset))

        assert result.success is True
        # images/, videos/, and annotations/ directories
        assert result.data["subdirectories"] == 3

    @pytest.mark.asyncio
    async def test_scan_directory_calculates_size(self, temp_dataset):
        """Should calculate total file size."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(temp_dataset))

        assert result.success is True
        assert result.data["total_size_bytes"] > 0
        assert "total_size_human" in result.data

    @pytest.mark.asyncio
    async def test_scan_directory_returns_formats(self, temp_dataset):
        """Should return extension counts."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(temp_dataset))

        assert result.success is True
        assert ".jpg" in result.data["formats"]
        assert result.data["formats"][".jpg"] == 10

    @pytest.mark.asyncio
    async def test_scan_directory_returns_sample_files(self, temp_dataset):
        """Should return sample file paths."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(temp_dataset))

        assert result.success is True
        assert "sample_files" in result.data
        assert len(result.data["sample_files"]) <= 5


class TestScanDirectoryPrimaryType:
    """Tests for primary type detection."""

    @pytest.mark.asyncio
    async def test_detects_images_primary_type(self, tmp_path):
        """Should detect images as primary type."""
        for i in range(5):
            (tmp_path / f"image_{i}.jpg").touch()

        tool = ScanDirectoryTool()
        result = await tool.run(path=str(tmp_path))

        assert result.data["primary_type"] == "images"

    @pytest.mark.asyncio
    async def test_detects_video_primary_type(self, tmp_path):
        """Should detect video as primary type."""
        for i in range(3):
            (tmp_path / f"video_{i}.mp4").touch()

        tool = ScanDirectoryTool()
        result = await tool.run(path=str(tmp_path))

        assert result.data["primary_type"] == "video"

    @pytest.mark.asyncio
    async def test_detects_pointcloud_primary_type(self, pointcloud_dataset):
        """Should detect point cloud as primary type."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(pointcloud_dataset / "pointclouds"))

        assert result.data["primary_type"] == "pointcloud"

    @pytest.mark.asyncio
    async def test_detects_mixed_primary_type(self, temp_dataset):
        """Should detect mixed type when multiple data types present."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(temp_dataset))

        assert result.data["primary_type"] == "mixed"


class TestScanDirectoryRecursive:
    """Tests for recursive scanning options."""

    @pytest.mark.asyncio
    async def test_recursive_scan_finds_all_files(self, nested_dataset):
        """Should find all files when recursive=True."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(nested_dataset), recursive=True)

        assert result.success is True
        assert result.data["total_files"] == 3

    @pytest.mark.asyncio
    async def test_non_recursive_scan_top_level_only(self, nested_dataset):
        """Should only scan top level when recursive=False."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(nested_dataset), recursive=False)

        assert result.success is True
        assert result.data["total_files"] == 0  # No files at top level
        assert result.data["subdirectories"] == 0  # Not counted without recursion

    @pytest.mark.asyncio
    async def test_max_depth_limits_recursion(self, nested_dataset):
        """Should respect max_depth parameter."""
        tool = ScanDirectoryTool()

        # max_depth=1 should find file in level1 only
        result = await tool.run(
            path=str(nested_dataset),
            recursive=True,
            max_depth=1,
        )

        assert result.success is True
        # Only level1/file1.jpg should be found
        assert result.data["total_files"] == 1

    @pytest.mark.asyncio
    async def test_max_depth_two_finds_more(self, nested_dataset):
        """Should find more files with higher max_depth."""
        tool = ScanDirectoryTool()

        result = await tool.run(
            path=str(nested_dataset),
            recursive=True,
            max_depth=2,
        )

        assert result.success is True
        # level1/file1.jpg and level1/level2/file2.jpg
        assert result.data["total_files"] == 2


class TestScanDirectoryErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_returns_error_for_missing_directory(self):
        """Should return error for non-existent directory."""
        tool = ScanDirectoryTool()
        result = await tool.run(path="/nonexistent/path/that/does/not/exist")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_returns_error_for_file_path(self, tmp_path):
        """Should return error when path is a file."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        tool = ScanDirectoryTool()
        result = await tool.run(path=str(test_file))

        assert result.success is False
        assert "not a directory" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_directory_returns_zero_files(self, empty_dataset):
        """Should handle empty directories gracefully."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(empty_dataset))

        assert result.success is True
        assert result.data["total_files"] == 0


class TestScanDirectoryAnnotations:
    """Tests for annotation detection."""

    @pytest.mark.asyncio
    async def test_detects_has_annotations(self, temp_dataset):
        """Should detect presence of annotation files."""
        tool = ScanDirectoryTool()
        result = await tool.run(path=str(temp_dataset))

        assert result.data["has_annotations"] is True

    @pytest.mark.asyncio
    async def test_detects_no_annotations(self, tmp_path):
        """Should detect absence of annotation files."""
        (tmp_path / "image.jpg").touch()

        tool = ScanDirectoryTool()
        result = await tool.run(path=str(tmp_path))

        assert result.data["has_annotations"] is False


class TestScanDirectoryProgress:
    """Tests for progress reporting."""

    @pytest.mark.asyncio
    async def test_reports_progress(self, temp_dataset):
        """Should report progress during scan."""
        progress_reports = []

        def capture_progress(current, total, message):
            progress_reports.append((current, total, message))

        tool = ScanDirectoryTool()
        tool.set_progress_callback(capture_progress)
        result = await tool.run(path=str(temp_dataset))

        assert result.success is True
        # Progress is reported at least once (final report)
        assert len(progress_reports) >= 1
        # Last progress should report all files
        last_current, _, _ = progress_reports[-1]
        assert last_current == result.data["total_files"]

    @pytest.mark.asyncio
    async def test_progress_throttling_with_many_files(self, tmp_path):
        """Should throttle progress for large number of files."""
        # Create 250 files to test throttling (threshold is 100)
        for i in range(250):
            (tmp_path / f"file_{i}.txt").touch()

        progress_reports = []

        def capture_progress(current, total, message):
            progress_reports.append((current, total, message))

        tool = ScanDirectoryTool()
        tool.set_progress_callback(capture_progress)
        result = await tool.run(path=str(tmp_path))

        assert result.success is True
        # Should have progress at 100, 200, and 250 (final)
        assert len(progress_reports) == 3


class TestScanDirectorySchema:
    """Tests for tool schema generation."""

    def test_schema_has_required_fields(self):
        """Should generate valid schema with required fields."""
        tool = ScanDirectoryTool()
        schema = tool.get_schema()

        # Schema has structure: {"type": "function", "function": {...}}
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "scan_directory"
        assert "description" in func
        assert "parameters" in func
        assert "path" in func["parameters"]["properties"]

    def test_path_is_required(self):
        """Path parameter should be required."""
        tool = ScanDirectoryTool()
        schema = tool.get_schema()

        func = schema["function"]
        assert "path" in func["parameters"]["required"]

    def test_optional_parameters_have_defaults(self):
        """Optional parameters should have default values."""
        tool = ScanDirectoryTool()
        schema = tool.get_schema()

        props = schema["function"]["parameters"]["properties"]
        assert props["recursive"]["default"] is True
        assert props["max_depth"]["default"] == 0
