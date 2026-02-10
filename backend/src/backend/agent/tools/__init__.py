"""
Cloumask Agent Tool System.

This package provides the tool abstraction layer for the agent,
allowing it to invoke CV operations with structured parameters,
validation, and result handling.

Implements specs: 06-tool-system, 07-tool-implementations, 08-cv-tools

Example:
    from backend.agent.tools import (
        BaseTool,
        ToolParameter,
        ToolResult,
        ToolCategory,
        get_tool_registry,
        register_tool,
        success_result,
        error_result,
    )

    @register_tool
    class MyTool(BaseTool):
        name = "my_tool"
        description = "Does something useful"
        category = ToolCategory.UTILITY
        parameters = [
            ToolParameter("input", str, "Input to process"),
        ]

        async def execute(self, input: str) -> ToolResult:
            return success_result({"output": input.upper()})

Available Tools:
    - ScanDirectoryTool: Scan directories to analyze dataset contents
    - DetectTool: Object detection (YOLO11, YOLO-World, or SAM3 quality mode)
    - SegmentTool: Segmentation using SAM3 (text), SAM2, or MobileSAM
    - AnonymizeTool: Anonymize faces and plates (blur, blackbox, pixelate, mask)
    - FaceDetectTool: Face detection (SCRFD, YuNet, or SAM3 quality mode)
    - Detect3DTool: 3D object detection (PV-RCNN++, CenterPoint)
    - Project3DTo2DTool: Project 3D detections to 2D image coordinates
    - ProcessPointCloudTool: Point cloud processing (downsample, filter, normals)
    - PointCloudStatsTool: Get point cloud metadata and statistics
    - AnonymizePointCloudTool: 3D point cloud face anonymization (remove/noise)
    - ExtractRosbagTool: Extract point clouds and images from ROS bags
    - ConvertFormatTool: Convert datasets between annotation formats
    - FindDuplicatesTool: Find duplicate and near-duplicate images in datasets
    - ExportTool: Export annotations to various formats (stub)
    - CustomScriptTool: Execute user-defined Python scripts
"""

# Import tool implementations to trigger registration via @register_tool decorator
from backend.agent.tools.anonymize import AnonymizeTool
from backend.agent.tools.anonymize_3d import AnonymizePointCloudTool
from backend.agent.tools.base import (
    BaseTool,
    ProgressCallback,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.convert import ConvertFormatTool
from backend.agent.tools.custom import CustomScriptTool
from backend.agent.tools.detect import DetectTool
from backend.agent.tools.detect_3d import Detect3DTool
from backend.agent.tools.discovery import (
    discover_tools,
    initialize_tools,
    list_available_tools,
    reload_tools,
)
from backend.agent.tools.duplicates import FindDuplicatesTool
from backend.agent.tools.export import ExportTool
from backend.agent.tools.faces import FaceDetectTool
from backend.agent.tools.fusion import Project3DTo2DTool
from backend.agent.tools.pointcloud_process import PointCloudStatsTool, ProcessPointCloudTool
from backend.agent.tools.registry import (
    ToolRegistry,
    get_tool_registry,
    register_tool,
)
from backend.agent.tools.rosbag import ExtractRosbagTool
from backend.agent.tools.scan import ScanDirectoryTool
from backend.agent.tools.segment import SegmentTool

__all__ = [
    # Base types
    "BaseTool",
    "ToolCategory",
    "ToolParameter",
    "ToolResult",
    "ProgressCallback",
    # Result helpers
    "success_result",
    "error_result",
    # Registry
    "ToolRegistry",
    "get_tool_registry",
    "register_tool",
    # Discovery
    "discover_tools",
    "initialize_tools",
    "reload_tools",
    "list_available_tools",
    # Tool implementations
    "ScanDirectoryTool",
    "DetectTool",
    "SegmentTool",
    "AnonymizeTool",
    "FaceDetectTool",
    "Detect3DTool",
    "Project3DTo2DTool",
    "ConvertFormatTool",
    "FindDuplicatesTool",
    "ExportTool",
    "CustomScriptTool",
    "ProcessPointCloudTool",
    "PointCloudStatsTool",
    "AnonymizePointCloudTool",
    "ExtractRosbagTool",
]
