"""Tests for point cloud agent tools.

These tests use mocked CV models and data parsers to test tool logic
without requiring actual model weights, GPU resources, or ROS bag files.

Tests cover:
- PointCloudStatsTool: metadata and statistics extraction
- ProcessPointCloudTool: downsampling, filtering, normals
- Detect3DTool: 3D object detection (already in test_cv_tools.py, extended here)
- Project3DTo2DTool: 3D to 2D projection
- AnonymizePointCloudTool: 3D face anonymization
- ExtractRosbagTool: ROS bag extraction
- Tool registration and docstring quality
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.agent.tools import (
    AnonymizePointCloudTool,
    Detect3DTool,
    ExtractRosbagTool,
    PointCloudStatsTool,
    ProcessPointCloudTool,
    Project3DTo2DTool,
)


# =============================================================================
# Test Fixtures (Mock Backend Results)
# =============================================================================


@pytest.fixture
def mock_pointcloud_stats():
    """Create a mock PointCloudStats result."""
    stats = MagicMock()
    stats.file_path = "/data/scan.pcd"
    stats.file_format = "pcd"
    stats.point_count = 100000
    stats.bounds_min = (-50.0, -50.0, -2.0)
    stats.bounds_max = (50.0, 50.0, 5.0)
    stats.extent = (100.0, 100.0, 7.0)
    stats.center = (0.0, 0.0, 1.5)
    stats.has_colors = True
    stats.has_normals = False
    stats.has_intensity = True
    return stats


@pytest.fixture
def mock_processing_result():
    """Create a mock ProcessingResult."""
    result = MagicMock()
    result.output_path = "/data/scan_downsampled.pcd"
    result.operation = "voxel"
    result.original_count = 100000
    result.result_count = 25000
    result.reduction_ratio = 0.75
    result.processing_time_ms = 350.0
    result.parameters = {"voxel_size": 0.1}
    return result


@pytest.fixture
def mock_anonymization_3d_result():
    """Create a mock Anonymization3DResult."""
    from backend.cv.anonymization_3d import Anonymization3DResult

    return Anonymization3DResult(
        output_path="/data/scan_anon.pcd",
        original_point_count=100000,
        anonymized_point_count=99800,
        face_regions_found=3,
        points_removed=200,
        points_noised=0,
        verification_passed=True,
        processing_time_ms=5200.0,
        views_processed=8,
        mode="remove",
    )


@pytest.fixture
def mock_bag_info():
    """Create a mock BagInfo."""
    from backend.data.ros_types import BagFormat, BagInfo, TopicInfo

    return BagInfo(
        path="/data/recording.bag",
        format=BagFormat.ROS1,
        duration_sec=120.5,
        start_time=1700000000.0,
        end_time=1700000120.5,
        topics=[
            TopicInfo(
                name="/velodyne_points",
                msg_type="sensor_msgs/msg/PointCloud2",
                message_count=1200,
                frequency_hz=10.0,
            ),
            TopicInfo(
                name="/camera/image_raw",
                msg_type="sensor_msgs/msg/Image",
                message_count=3600,
                frequency_hz=30.0,
            ),
            TopicInfo(
                name="/imu/data",
                msg_type="sensor_msgs/msg/Imu",
                message_count=12000,
                frequency_hz=100.0,
            ),
        ],
        message_count=16800,
    )


# =============================================================================
# PointCloudStatsTool Tests
# =============================================================================


class TestPointCloudStatsTool:
    """Tests for PointCloudStatsTool."""

    @pytest.mark.asyncio
    async def test_stats_valid_file(self, temp_pointcloud, mock_pointcloud_stats):
        """Test getting stats for a valid point cloud file."""
        with patch("backend.cv.pointcloud.PointCloudProcessor") as mock_cls:
            mock_processor = MagicMock()
            mock_processor.get_stats.return_value = mock_pointcloud_stats
            mock_cls.return_value = mock_processor

            tool = PointCloudStatsTool()
            result = await tool.run(path=str(temp_pointcloud))

            assert result.success
            assert result.data["point_count"] == 100000
            assert result.data["has_colors"] is True
            assert result.data["has_intensity"] is True
            assert "bounds" in result.data

    @pytest.mark.asyncio
    async def test_stats_nonexistent_file(self):
        """Test stats with non-existent file."""
        tool = PointCloudStatsTool()
        result = await tool.run(path="/nonexistent/scan.pcd")

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_stats_unsupported_format(self, tmp_path):
        """Test stats with unsupported format."""
        text_file = tmp_path / "data.txt"
        text_file.touch()

        tool = PointCloudStatsTool()
        result = await tool.run(path=str(text_file))

        assert not result.success
        assert "unsupported" in result.error.lower()

    @pytest.mark.asyncio
    async def test_stats_directory_path(self, tmp_path):
        """Test stats with directory path instead of file."""
        tool = PointCloudStatsTool()
        result = await tool.run(path=str(tmp_path))

        assert not result.success
        assert "file" in result.error.lower()


# =============================================================================
# ProcessPointCloudTool Tests
# =============================================================================


class TestProcessPointCloudTool:
    """Tests for ProcessPointCloudTool."""

    @pytest.mark.asyncio
    async def test_voxel_downsample(self, temp_pointcloud, tmp_path, mock_processing_result):
        """Test voxel downsampling operation."""
        output = tmp_path / "output.pcd"

        with patch("backend.cv.pointcloud.PointCloudProcessor") as mock_cls:
            mock_processor = MagicMock()
            mock_processor.process_file.return_value = mock_processing_result
            mock_cls.return_value = mock_processor

            tool = ProcessPointCloudTool()
            result = await tool.run(
                input_path=str(temp_pointcloud),
                output_path=str(output),
                operation="voxel",
                voxel_size=0.1,
            )

            assert result.success
            assert result.data["operation"] == "voxel"
            assert result.data["reduction_ratio"] == 0.75

    @pytest.mark.asyncio
    async def test_voxel_missing_size(self, temp_pointcloud, tmp_path):
        """Test voxel operation without voxel_size fails."""
        output = tmp_path / "output.pcd"

        tool = ProcessPointCloudTool()
        result = await tool.run(
            input_path=str(temp_pointcloud),
            output_path=str(output),
            operation="voxel",
        )

        assert not result.success
        assert "voxel_size" in result.error.lower()

    @pytest.mark.asyncio
    async def test_random_missing_count(self, temp_pointcloud, tmp_path):
        """Test random operation without target_count fails."""
        output = tmp_path / "output.pcd"

        tool = ProcessPointCloudTool()
        result = await tool.run(
            input_path=str(temp_pointcloud),
            output_path=str(output),
            operation="random",
        )

        assert not result.success
        assert "target_count" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unsupported_output_format(self, temp_pointcloud, tmp_path):
        """Test output with unsupported format."""
        output = tmp_path / "output.las"  # LAS is input-only

        tool = ProcessPointCloudTool()
        result = await tool.run(
            input_path=str(temp_pointcloud),
            output_path=str(output),
            operation="voxel",
            voxel_size=0.1,
        )

        assert not result.success
        assert "unsupported output" in result.error.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_input(self, tmp_path):
        """Test processing with non-existent input."""
        output = tmp_path / "output.pcd"

        tool = ProcessPointCloudTool()
        result = await tool.run(
            input_path="/nonexistent/scan.pcd",
            output_path=str(output),
            operation="voxel",
            voxel_size=0.1,
        )

        assert not result.success
        assert "not found" in result.error.lower()


# =============================================================================
# Project3DTo2DTool Tests
# =============================================================================


class TestProject3DTo2DTool:
    """Tests for Project3DTo2DTool."""

    @pytest.mark.asyncio
    async def test_missing_inputs(self, tmp_path):
        """Test projection without detections or pointcloud fails."""
        calib = tmp_path / "calib.txt"
        calib.touch()

        tool = Project3DTo2DTool()
        result = await tool.run(calibration_path=str(calib))

        assert not result.success
        assert "detections_path" in result.error.lower() or "pointcloud_path" in result.error.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_calibration(self):
        """Test projection with non-existent calibration file."""
        tool = Project3DTo2DTool()
        result = await tool.run(
            calibration_path="/nonexistent/calib.txt",
            detections_path="/some/dets.json",
        )

        assert not result.success
        assert "not found" in result.error.lower()


# =============================================================================
# AnonymizePointCloudTool Tests
# =============================================================================


class TestAnonymizePointCloudTool:
    """Tests for AnonymizePointCloudTool."""

    @pytest.mark.asyncio
    async def test_anonymize_remove_mode(
        self, temp_pointcloud, tmp_path, mock_anonymization_3d_result
    ):
        """Test anonymization with 'remove' mode."""
        output = tmp_path / "output.pcd"

        with patch("backend.cv.anonymization_3d.PointCloudAnonymizer") as mock_cls:
            mock_anonymizer = MagicMock()
            mock_anonymizer.anonymize.return_value = mock_anonymization_3d_result
            mock_cls.return_value = mock_anonymizer

            tool = AnonymizePointCloudTool()
            result = await tool.run(
                input_path=str(temp_pointcloud),
                output_path=str(output),
                mode="remove",
            )

            assert result.success
            assert result.data["mode"] == "remove"
            assert result.data["faces_found"] == 3
            assert result.data["points_removed"] == 200
            assert result.data["verified"] is True
            mock_anonymizer.load.assert_called_once_with("auto")
            mock_anonymizer.unload.assert_called_once()

    @pytest.mark.asyncio
    async def test_anonymize_noise_mode(
        self, temp_pointcloud, tmp_path, mock_anonymization_3d_result
    ):
        """Test anonymization with 'noise' mode."""
        output = tmp_path / "output.pcd"

        # Modify mock for noise mode
        mock_anonymization_3d_result.mode = "noise"
        mock_anonymization_3d_result.points_removed = 0
        mock_anonymization_3d_result.points_noised = 200

        with patch("backend.cv.anonymization_3d.PointCloudAnonymizer") as mock_cls:
            mock_anonymizer = MagicMock()
            mock_anonymizer.anonymize.return_value = mock_anonymization_3d_result
            mock_cls.return_value = mock_anonymizer

            tool = AnonymizePointCloudTool()
            result = await tool.run(
                input_path=str(temp_pointcloud),
                output_path=str(output),
                mode="noise",
            )

            assert result.success
            assert result.data["mode"] == "noise"
            assert result.data["points_noised"] == 200

    @pytest.mark.asyncio
    async def test_anonymize_nonexistent_input(self, tmp_path):
        """Test anonymization with non-existent input."""
        output = tmp_path / "output.pcd"

        tool = AnonymizePointCloudTool()
        result = await tool.run(
            input_path="/nonexistent/scan.pcd",
            output_path=str(output),
        )

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_anonymize_unsupported_input_format(self, tmp_path):
        """Test anonymization with unsupported input format."""
        invalid = tmp_path / "scan.obj"
        invalid.touch()
        output = tmp_path / "output.pcd"

        tool = AnonymizePointCloudTool()
        result = await tool.run(
            input_path=str(invalid),
            output_path=str(output),
        )

        assert not result.success
        assert "unsupported" in result.error.lower()

    @pytest.mark.asyncio
    async def test_anonymize_unsupported_output_format(self, temp_pointcloud, tmp_path):
        """Test anonymization with unsupported output format."""
        output = tmp_path / "output.las"

        tool = AnonymizePointCloudTool()
        result = await tool.run(
            input_path=str(temp_pointcloud),
            output_path=str(output),
        )

        assert not result.success
        assert "unsupported output" in result.error.lower()

    @pytest.mark.asyncio
    async def test_anonymize_invalid_confidence(self, temp_pointcloud, tmp_path):
        """Test anonymization with invalid face confidence."""
        output = tmp_path / "output.pcd"

        tool = AnonymizePointCloudTool()
        result = await tool.run(
            input_path=str(temp_pointcloud),
            output_path=str(output),
            face_confidence=1.5,
        )

        assert not result.success
        assert "face_confidence" in result.error.lower()

    @pytest.mark.asyncio
    async def test_anonymize_invalid_num_views(self, temp_pointcloud, tmp_path):
        """Test anonymization with invalid num_views."""
        output = tmp_path / "output.pcd"

        tool = AnonymizePointCloudTool()
        result = await tool.run(
            input_path=str(temp_pointcloud),
            output_path=str(output),
            num_views=0,
        )

        assert not result.success
        assert "num_views" in result.error.lower()

    @pytest.mark.asyncio
    async def test_anonymize_gpu_oom_fallback(self, temp_pointcloud, tmp_path):
        """Test GPU OOM error is caught and reported."""
        output = tmp_path / "output.pcd"

        with patch("backend.cv.anonymization_3d.PointCloudAnonymizer") as mock_cls:
            mock_anonymizer = MagicMock()
            mock_anonymizer.anonymize.side_effect = RuntimeError("CUDA out of memory")
            mock_cls.return_value = mock_anonymizer

            tool = AnonymizePointCloudTool()
            result = await tool.run(
                input_path=str(temp_pointcloud),
                output_path=str(output),
            )

            assert not result.success
            assert "out of memory" in result.error.lower()
            mock_anonymizer.unload.assert_called_once()


# =============================================================================
# ExtractRosbagTool Tests
# =============================================================================


class TestExtractRosbagTool:
    """Tests for ExtractRosbagTool."""

    @pytest.fixture
    def mock_bag_file(self, tmp_path):
        """Create a mock .bag file."""
        bag = tmp_path / "recording.bag"
        bag.write_bytes(b"#ROSBAG V2.0\n")
        return bag

    @pytest.mark.asyncio
    async def test_extract_synced(self, mock_bag_file, tmp_path, mock_bag_info):
        """Test synchronized extraction of LiDAR and camera data."""
        output_dir = tmp_path / "extracted"

        with patch("backend.data.rosbag_parser.RosbagParser") as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser.get_info.return_value = mock_bag_info
            mock_parser.get_pointcloud_topics.return_value = [mock_bag_info.topics[0]]
            mock_parser.get_image_topics.return_value = [mock_bag_info.topics[1]]
            mock_parser.sync_frames.return_value = []  # Empty for unit test
            mock_parser_cls.return_value = mock_parser

            tool = ExtractRosbagTool()
            result = await tool.run(
                bag_path=str(mock_bag_file),
                output_dir=str(output_dir),
            )

            assert result.success
            assert result.data["bag_format"] == "ros1"
            assert result.data["pointcloud_topic"] == "/velodyne_points"
            assert result.data["image_topic"] == "/camera/image_raw"
            assert result.data["sync_enabled"] is True

    @pytest.mark.asyncio
    async def test_extract_no_sync(self, mock_bag_file, tmp_path, mock_bag_info):
        """Test separate extraction without synchronization."""
        output_dir = tmp_path / "extracted"

        with patch("backend.data.rosbag_parser.RosbagParser") as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser.get_info.return_value = mock_bag_info
            mock_parser.get_pointcloud_topics.return_value = [mock_bag_info.topics[0]]
            mock_parser.get_image_topics.return_value = [mock_bag_info.topics[1]]
            mock_parser.iter_pointcloud2.return_value = iter([])
            mock_parser.iter_images.return_value = iter([])
            mock_parser_cls.return_value = mock_parser

            tool = ExtractRosbagTool()
            result = await tool.run(
                bag_path=str(mock_bag_file),
                output_dir=str(output_dir),
                sync_sensors=False,
            )

            assert result.success
            assert result.data["sync_enabled"] is False

    @pytest.mark.asyncio
    async def test_extract_explicit_topics(self, mock_bag_file, tmp_path, mock_bag_info):
        """Test extraction with explicit topic names."""
        output_dir = tmp_path / "extracted"

        with patch("backend.data.rosbag_parser.RosbagParser") as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser.get_info.return_value = mock_bag_info
            mock_parser.sync_frames.return_value = []
            mock_parser_cls.return_value = mock_parser

            tool = ExtractRosbagTool()
            result = await tool.run(
                bag_path=str(mock_bag_file),
                output_dir=str(output_dir),
                pointcloud_topic="/velodyne_points",
                image_topic="/camera/image_raw",
            )

            assert result.success
            assert result.data["pointcloud_topic"] == "/velodyne_points"
            assert result.data["image_topic"] == "/camera/image_raw"

    @pytest.mark.asyncio
    async def test_extract_nonexistent_bag(self, tmp_path):
        """Test extraction with non-existent bag file."""
        tool = ExtractRosbagTool()
        result = await tool.run(
            bag_path="/nonexistent/recording.bag",
            output_dir=str(tmp_path / "out"),
        )

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_extract_unsupported_format(self, tmp_path):
        """Test extraction with unsupported file format."""
        invalid = tmp_path / "data.csv"
        invalid.touch()

        tool = ExtractRosbagTool()
        result = await tool.run(
            bag_path=str(invalid),
            output_dir=str(tmp_path / "out"),
        )

        assert not result.success
        assert "unsupported" in result.error.lower()

    @pytest.mark.asyncio
    async def test_extract_invalid_max_frames(self, mock_bag_file, tmp_path):
        """Test extraction with invalid max_frames."""
        tool = ExtractRosbagTool()
        result = await tool.run(
            bag_path=str(mock_bag_file),
            output_dir=str(tmp_path / "out"),
            max_frames=0,
        )

        assert not result.success
        assert "max_frames" in result.error.lower()

    @pytest.mark.asyncio
    async def test_extract_no_sensor_topics(self, mock_bag_file, tmp_path):
        """Test extraction when no sensor topics are found."""
        from backend.data.ros_types import BagFormat, BagInfo, TopicInfo

        empty_info = BagInfo(
            path=str(mock_bag_file),
            format=BagFormat.ROS1,
            duration_sec=10.0,
            start_time=0.0,
            end_time=10.0,
            topics=[
                TopicInfo(
                    name="/imu/data",
                    msg_type="sensor_msgs/msg/Imu",
                    message_count=1000,
                ),
            ],
        )

        with patch("backend.data.rosbag_parser.RosbagParser") as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser.get_info.return_value = empty_info
            mock_parser.get_pointcloud_topics.return_value = []
            mock_parser.get_image_topics.return_value = []
            mock_parser_cls.return_value = mock_parser

            tool = ExtractRosbagTool()
            result = await tool.run(
                bag_path=str(mock_bag_file),
                output_dir=str(tmp_path / "out"),
            )

            assert not result.success
            assert "no pointcloud2 or image topics" in result.error.lower()


# =============================================================================
# Tool Registration and Docstring Tests
# =============================================================================


class TestPointCloudToolRegistration:
    """Test that all point cloud tools are properly registered."""

    def test_all_pointcloud_tools_registered(self):
        """Verify all point cloud tools are in the registry."""
        from backend.agent.tools import get_tool_registry, initialize_tools

        initialize_tools()
        registry = get_tool_registry()

        expected_tools = [
            "pointcloud_stats",
            "process_pointcloud",
            "detect_3d",
            "project_3d_to_2d",
            "anonymize_pointcloud",
            "extract_rosbag",
        ]

        for tool_name in expected_tools:
            assert registry.has(tool_name), f"Tool '{tool_name}' not registered"

    def test_pointcloud_tools_have_descriptions(self):
        """Verify all point cloud tools have meaningful descriptions."""
        tools = [
            PointCloudStatsTool,
            ProcessPointCloudTool,
            Detect3DTool,
            Project3DTo2DTool,
            AnonymizePointCloudTool,
            ExtractRosbagTool,
        ]

        for tool_cls in tools:
            assert len(tool_cls.description) > 50, (
                f"{tool_cls.name} description too short: {len(tool_cls.description)} chars"
            )

    def test_pointcloud_stats_docstring_content(self):
        """Tool description should mention key capabilities."""
        desc = PointCloudStatsTool.description.lower()
        assert "point count" in desc or "metadata" in desc or "statistics" in desc

    def test_detect_3d_docstring_content(self):
        """Tool description should mention supported classes."""
        desc = Detect3DTool.description.lower()
        assert "vehicle" in desc or "car" in desc
        assert "pedestrian" in desc

    def test_anonymize_3d_docstring_content(self):
        """Tool description should mention anonymization modes and GDPR."""
        desc = AnonymizePointCloudTool.description.lower()
        assert "remove" in desc
        assert "noise" in desc
        assert "gdpr" in desc or "privacy" in desc

    def test_extract_rosbag_docstring_content(self):
        """Tool description should mention ROS formats and sync."""
        desc = ExtractRosbagTool.description.lower()
        assert "ros1" in desc or ".bag" in desc
        assert "ros2" in desc or ".db3" in desc
        assert "synchroniz" in desc

    def test_tools_have_parameters(self):
        """Verify all tools define parameters."""
        tools = [
            PointCloudStatsTool,
            ProcessPointCloudTool,
            Detect3DTool,
            Project3DTo2DTool,
            AnonymizePointCloudTool,
            ExtractRosbagTool,
        ]

        for tool_cls in tools:
            assert len(tool_cls.parameters) > 0, (
                f"{tool_cls.name} has no parameters defined"
            )

    def test_tools_generate_valid_schemas(self):
        """Verify all tools can generate valid JSON schemas."""
        tools = [
            PointCloudStatsTool(),
            ProcessPointCloudTool(),
            Detect3DTool(),
            Project3DTo2DTool(),
            AnonymizePointCloudTool(),
            ExtractRosbagTool(),
        ]

        for tool in tools:
            schema = tool.get_schema()
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]
