"""
Cloumask Agent Tool System.

This package provides the tool abstraction layer for the agent,
allowing it to invoke CV operations with structured parameters,
validation, and result handling.

Implements spec: 06-tool-system

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
"""

from backend.agent.tools.base import (
    BaseTool,
    ProgressCallback,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.discovery import (
    discover_tools,
    initialize_tools,
    list_available_tools,
    reload_tools,
)
from backend.agent.tools.registry import (
    ToolRegistry,
    get_tool_registry,
    register_tool,
)

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
]
