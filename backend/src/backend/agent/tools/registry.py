"""
Tool registry for the Cloumask agent.

This module provides the singleton registry for discovering and accessing
tools, along with decorators for automatic registration.

Implements spec: 06-tool-system
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.agent.tools.base import BaseTool, ToolCategory

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for discovering and accessing tools.

    Implements singleton pattern to ensure one global registry.
    Tools can be registered by instance or by class, and retrieved
    by name or category.

    Example:
        registry = get_tool_registry()
        registry.register(ScanDirectoryTool())

        tool = registry.get("scan_directory")
        schemas = registry.get_schemas()  # For LLM tool calling
    """

    _instance: ToolRegistry | None = None
    _tools: dict[str, BaseTool]

    def __new__(cls) -> ToolRegistry:
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def register_class(self, tool_class: type[BaseTool]) -> None:
        """
        Register a tool by class (instantiates it).

        Args:
            tool_class: The tool class to instantiate and register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        tool = tool_class()
        self.register(tool)

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            name: The name of the tool to unregister.

        Returns:
            True if the tool was found and removed, False otherwise.
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug("Unregistered tool: %s", name)
            return True
        return False

    def get(self, name: str) -> BaseTool | None:
        """
        Get a tool by name.

        Args:
            name: The tool name to look up.

        Returns:
            The tool instance if found, None otherwise.
        """
        return self._tools.get(name)

    def get_all(self) -> list[BaseTool]:
        """
        Get all registered tools.

        Returns:
            List of all registered tool instances.
        """
        return list(self._tools.values())

    def get_by_category(self, category: ToolCategory) -> list[BaseTool]:
        """
        Get tools by category.

        Args:
            category: The category to filter by.

        Returns:
            List of tools matching the category.
        """
        return [t for t in self._tools.values() if t.category == category]

    def get_names(self) -> list[str]:
        """
        Get names of all registered tools.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """
        Get JSON schemas for all tools (for LLM tool calling).

        Returns:
            List of OpenAI-compatible function schemas.
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def has(self, name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            name: The tool name to check.

        Returns:
            True if a tool with that name is registered.
        """
        return name in self._tools

    def clear(self) -> None:
        """Clear all registered tools (for testing)."""
        self._tools.clear()
        logger.debug("Cleared all tools from registry")

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered using 'in' operator."""
        return name in self._tools


def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.

    The ToolRegistry class implements the singleton pattern via __new__,
    so this always returns the same instance.

    Returns:
        The singleton ToolRegistry instance.
    """
    return ToolRegistry()


def register_tool(tool_class: type[BaseTool]) -> type[BaseTool]:
    """
    Decorator to automatically register a tool class.

    Use this decorator on tool classes to have them automatically
    registered when the module is imported.

    Example:
        @register_tool
        class ScanDirectoryTool(BaseTool):
            name = "scan_directory"
            ...

    Args:
        tool_class: The tool class to register.

    Returns:
        The same tool class (unmodified).
    """
    get_tool_registry().register_class(tool_class)
    return tool_class
