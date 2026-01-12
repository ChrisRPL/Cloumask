"""
Cloumask Agent Tool System.

This package provides the tool abstraction layer for the agent,
allowing it to invoke CV operations with structured parameters,
validation, and result handling.

Implements specs: 06-tool-system, 07-tool-implementations

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
    - AnonymizeTool: Blur faces and license plates (stub)
    - DetectTool: Object detection using YOLO (stub)
    - ExportTool: Export annotations to various formats (stub)
"""

# Import tool implementations to trigger registration via @register_tool decorator
from backend.agent.tools.anonymize import AnonymizeTool
from backend.agent.tools.base import (
    BaseTool,
    ProgressCallback,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.detect import DetectTool
from backend.agent.tools.discovery import (
    discover_tools,
    initialize_tools,
    list_available_tools,
    reload_tools,
)
from backend.agent.tools.export import ExportTool
from backend.agent.tools.registry import (
    ToolRegistry,
    get_tool_registry,
    register_tool,
)
from backend.agent.tools.scan import ScanDirectoryTool

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
    "AnonymizeTool",
    "DetectTool",
    "ExportTool",
]
