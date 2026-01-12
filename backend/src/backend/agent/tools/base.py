"""
Base classes and types for the Cloumask agent tool system.

This module provides the abstraction layer for all tools that the agent
can invoke. Includes parameter definitions, result types, and the abstract
base class that all tools must inherit from.

Implements spec: 06-tool-system
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)


# Type for progress callback: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]


class ToolCategory(str, Enum):
    """Categories for organizing tools."""

    SCAN = "scan"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    ANONYMIZATION = "anonymization"
    EXPORT = "export"
    UTILITY = "utility"


@dataclass
class ToolParameter:
    """
    Definition of a tool parameter.

    Describes a single parameter that a tool accepts, including its type,
    description, constraints, and default value. Used for validation and
    JSON schema generation for LLM tool calling.
    """

    name: str
    type: type
    description: str
    required: bool = True
    default: Any = None
    enum_values: list[Any] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """
        Convert parameter definition to JSON Schema format.

        Returns a dict compatible with OpenAI function calling schema,
        mapping Python types to JSON Schema types.

        Returns:
            JSON Schema representation of this parameter.
        """
        schema: dict[str, Any] = {
            "description": self.description,
        }

        # Map Python types to JSON Schema types
        type_map: dict[type, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        schema["type"] = type_map.get(self.type, "string")

        if self.enum_values:
            schema["enum"] = self.enum_values

        if self.default is not None:
            schema["default"] = self.default

        return schema


@dataclass
class ToolResult:
    """
    Result of a tool execution.

    Encapsulates the outcome of running a tool, including success/failure
    status, result data, error information, and timing metrics.
    """

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize result for state storage and chat display.

        For successful results, flattens data into the result dict.
        For failures, includes error message.

        Returns:
            Dict representation suitable for state storage.
        """
        result: dict[str, Any] = {
            "success": self.success,
            "duration_seconds": self.duration_seconds,
        }

        if self.success:
            result["data"] = self.data or {}
            # Flatten data into result for easier access
            result.update(self.data or {})
        else:
            result["error"] = self.error

        result["_meta"] = self.metadata
        return result


# -----------------------------------------------------------------------------
# Result Helper Functions
# -----------------------------------------------------------------------------


def success_result(data: dict[str, Any], **metadata: Any) -> ToolResult:
    """
    Create a successful tool result.

    Args:
        data: Result data from the tool execution.
        **metadata: Additional metadata to include.

    Returns:
        ToolResult with success=True and the provided data.
    """
    return ToolResult(
        success=True,
        data=data,
        metadata=metadata,
    )


def error_result(error: str, **metadata: Any) -> ToolResult:
    """
    Create a failed tool result.

    Args:
        error: Error message describing the failure.
        **metadata: Additional metadata to include.

    Returns:
        ToolResult with success=False and the error message.
    """
    return ToolResult(
        success=False,
        error=error,
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Base Tool Class
# -----------------------------------------------------------------------------


T = TypeVar("T", bound="BaseTool")


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Provides the interface and common functionality for tools that the
    agent can invoke. Subclasses must implement the execute() method
    and define their name, description, and parameters.

    Attributes:
        name: Unique tool identifier used for registration and invocation.
        description: Human-readable description shown to the LLM.
        category: Category for organizing tools in the registry.
        parameters: List of ToolParameter definitions for this tool.

    Example:
        class ScanDirectoryTool(BaseTool):
            name = "scan_directory"
            description = "Scan a directory for image and video files"
            category = ToolCategory.SCAN
            parameters = [
                ToolParameter("path", str, "Directory path to scan"),
                ToolParameter("recursive", bool, "Scan subdirectories", default=True),
            ]

            async def execute(self, path: str, recursive: bool = True) -> ToolResult:
                # Implementation
                return success_result({"files": [...]})
    """

    # Class attributes to be overridden by subclasses
    name: str = "base_tool"
    description: str = "Base tool description"
    category: ToolCategory = ToolCategory.UTILITY
    parameters: list[ToolParameter] = []

    # Instance attributes for execution
    _progress_callback: ProgressCallback | None = None
    _start_time: datetime | None = None
    _last_progress: int = 0

    def __init__(self) -> None:
        """Initialize the tool instance."""
        self._progress_callback = None
        self._start_time = None
        self._last_progress = 0

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with given parameters.

        Must be implemented by subclasses to perform the actual work.
        Should return a ToolResult with success/failure and data.

        Args:
            **kwargs: Tool parameters as defined in the parameters list.

        Returns:
            ToolResult with execution outcome.
        """
        pass

    def validate_parameters(self, **kwargs: Any) -> str | None:
        """
        Validate parameters before execution.

        Checks that required parameters are present and have correct types.
        Also validates enum constraints if specified.

        Args:
            **kwargs: Parameters to validate.

        Returns:
            Error message if validation fails, None if valid.
        """
        for param in self.parameters:
            # Check required parameters
            if param.required and param.name not in kwargs and param.default is None:
                return f"Missing required parameter: {param.name}"

            # Validate provided parameters
            if param.name in kwargs:
                value = kwargs[param.name]

                # Type checking
                if not isinstance(value, param.type):
                    return (
                        f"Parameter {param.name} must be {param.type.__name__}, "
                        f"got {type(value).__name__}"
                    )

                # Enum validation
                if param.enum_values and value not in param.enum_values:
                    return f"Parameter {param.name} must be one of {param.enum_values}"

        return None

    def get_schema(self) -> dict[str, Any]:
        """
        Generate JSON Schema for LLM tool calling.

        Returns an OpenAI-compatible function schema that describes
        this tool's name, description, and parameters.

        Returns:
            Dict with "type": "function" and function definition.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required and param.default is None:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """
        Set callback for progress updates during execution.

        Args:
            callback: Function to call with (current, total, message).
        """
        self._progress_callback = callback

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """
        Report progress to callback if set.

        Args:
            current: Current progress count.
            total: Total items to process.
            message: Optional status message.
        """
        if self._progress_callback:
            self._progress_callback(current, total, message)
        self._last_progress = current

    def _start_execution(self) -> None:
        """Mark execution start for timing."""
        self._start_time = datetime.now()

    def _end_execution(self) -> float:
        """
        Mark execution end and return duration.

        Returns:
            Duration in seconds since _start_execution was called.
        """
        if self._start_time:
            duration = (datetime.now() - self._start_time).total_seconds()
            self._start_time = None
            return duration
        return 0.0

    async def run(self, **kwargs: Any) -> ToolResult:
        """
        Run the tool with validation and timing.

        This is the main entry point that handles:
        - Parameter validation
        - Default value application
        - Execution timing
        - Exception handling

        Args:
            **kwargs: Tool parameters.

        Returns:
            ToolResult with execution outcome.
        """
        # Validate parameters
        validation_error = self.validate_parameters(**kwargs)
        if validation_error:
            return ToolResult(
                success=False,
                error=validation_error,
            )

        # Apply defaults for missing optional parameters
        for param in self.parameters:
            if param.name not in kwargs and param.default is not None:
                kwargs[param.name] = param.default

        # Execute with timing
        self._start_execution()
        try:
            result = await self.execute(**kwargs)
            duration = self._end_execution()
            result.duration_seconds = duration
            return result
        except Exception as e:
            duration = self._end_execution()
            logger.exception("Tool %s execution failed: %s", self.name, e)
            return ToolResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
