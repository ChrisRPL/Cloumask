# Tool System

> **Status:** 🔴 Not Started
> **Priority:** P0 (Critical)
> **Dependencies:** 01-state-types
> **Estimated Complexity:** Medium

## Overview

Design and implement the tool abstraction layer that allows the agent to invoke CV operations. This includes a base tool class, tool registry, result types, and metadata generation for LLM tool calling.

## Goals

- [ ] Abstract base class for all tools
- [ ] Tool registry with dynamic registration
- [ ] Structured result types (success/error)
- [ ] Tool metadata for LLM (JSON schema, descriptions)
- [ ] Async execution support
- [ ] Progress callback mechanism

## Technical Design

### Tool Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Type, TypeVar
from enum import Enum
import json


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
    """Definition of a tool parameter."""
    name: str
    type: Type
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[list] = None

    def to_json_schema(self) -> dict:
        """Convert to JSON Schema for LLM."""
        schema = {
            "description": self.description,
        }

        # Map Python types to JSON Schema types
        type_map = {
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
    """Result of a tool execution."""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for state storage."""
        result = {
            "success": self.success,
            "duration_seconds": self.duration_seconds,
        }
        if self.success:
            result["data"] = self.data or {}
            result.update(self.data or {})  # Flatten data into result
        else:
            result["error"] = self.error
        result["_meta"] = self.metadata
        return result


ProgressCallback = Callable[[int, int, str], None]  # current, total, message


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Subclasses must implement:
    - name: Unique tool identifier
    - description: Human-readable description for LLM
    - parameters: List of ToolParameter definitions
    - execute(): Async execution method
    """

    # Class attributes to be overridden
    name: str = "base_tool"
    description: str = "Base tool description"
    category: ToolCategory = ToolCategory.UTILITY
    parameters: list[ToolParameter] = []

    # Optional callbacks
    _progress_callback: Optional[ProgressCallback] = None

    def __init__(self):
        """Initialize the tool."""
        self._start_time: Optional[datetime] = None
        self._last_progress: int = 0

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Must be overridden by subclasses.

        Returns:
            ToolResult with success/failure and data
        """
        pass

    def validate_parameters(self, **kwargs) -> Optional[str]:
        """
        Validate parameters before execution.

        Returns:
            Error message if validation fails, None otherwise
        """
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                if param.default is None:
                    return f"Missing required parameter: {param.name}"

            if param.name in kwargs:
                value = kwargs[param.name]
                # Type checking
                if not isinstance(value, param.type):
                    return f"Parameter {param.name} must be {param.type.__name__}, got {type(value).__name__}"
                # Enum validation
                if param.enum_values and value not in param.enum_values:
                    return f"Parameter {param.name} must be one of {param.enum_values}"

        return None

    def get_schema(self) -> dict:
        """
        Generate JSON Schema for LLM tool calling.

        Returns OpenAI-compatible function schema.
        """
        properties = {}
        required = []

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
                }
            }
        }

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(current, total, message)
        self._last_progress = current

    def _start_execution(self) -> None:
        """Mark execution start."""
        self._start_time = datetime.now()

    def _end_execution(self) -> float:
        """Mark execution end and return duration."""
        if self._start_time:
            duration = (datetime.now() - self._start_time).total_seconds()
            self._start_time = None
            return duration
        return 0.0

    async def run(self, **kwargs) -> ToolResult:
        """
        Run the tool with validation and timing.

        This is the main entry point that handles:
        - Parameter validation
        - Execution timing
        - Error handling
        """
        # Validate parameters
        validation_error = self.validate_parameters(**kwargs)
        if validation_error:
            return ToolResult(
                success=False,
                error=validation_error,
            )

        # Apply defaults
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
            return ToolResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
```

### Tool Registry

```python
from typing import Dict, List, Type


class ToolRegistry:
    """
    Registry for discovering and accessing tools.

    Singleton pattern ensures one global registry.
    """

    _instance: Optional['ToolRegistry'] = None
    _tools: Dict[str, BaseTool] = {}

    def __new__(cls) -> 'ToolRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_class(self, tool_class: Type[BaseTool]) -> None:
        """Register a tool class (instantiates it)."""
        tool = tool_class()
        self.register(tool)

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get tools by category."""
        return [t for t in self._tools.values() if t.category == category]

    def get_schemas(self) -> List[dict]:
        """Get JSON schemas for all tools (for LLM)."""
        return [tool.get_schema() for tool in self._tools.values()]

    def clear(self) -> None:
        """Clear all registered tools (for testing)."""
        self._tools.clear()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return ToolRegistry()


def register_tool(tool_class: Type[BaseTool]) -> Type[BaseTool]:
    """Decorator to register a tool class."""
    get_tool_registry().register_class(tool_class)
    return tool_class
```

### Tool Discovery

```python
import importlib
import pkgutil
from pathlib import Path


def discover_tools(package_path: str = "agent.tools") -> None:
    """
    Auto-discover and register tools from a package.

    Looks for classes decorated with @register_tool.
    """
    package = importlib.import_module(package_path)
    package_dir = Path(package.__file__).parent

    for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        if module_name.startswith("_"):
            continue
        full_name = f"{package_path}.{module_name}"
        importlib.import_module(full_name)


def initialize_tools() -> None:
    """Initialize the tool system on startup."""
    # Clear any existing registrations
    get_tool_registry().clear()

    # Discover and register all tools
    discover_tools("agent.tools")
```

### Result Helpers

```python
def success_result(
    data: dict,
    **metadata
) -> ToolResult:
    """Create a successful tool result."""
    return ToolResult(
        success=True,
        data=data,
        metadata=metadata,
    )


def error_result(
    error: str,
    **metadata
) -> ToolResult:
    """Create a failed tool result."""
    return ToolResult(
        success=False,
        error=error,
        metadata=metadata,
    )
```

## Implementation Tasks

- [ ] Create `backend/agent/tools/__init__.py`
- [ ] Create `backend/agent/tools/base.py`
  - [ ] Implement `ToolCategory` enum
  - [ ] Implement `ToolParameter` dataclass
  - [ ] Implement `ToolResult` dataclass
  - [ ] Implement `BaseTool` abstract class
  - [ ] Implement `validate_parameters()` method
  - [ ] Implement `get_schema()` method
  - [ ] Implement `run()` wrapper method
- [ ] Create `backend/agent/tools/registry.py`
  - [ ] Implement `ToolRegistry` singleton
  - [ ] Implement `register()` method
  - [ ] Implement `get()` method
  - [ ] Implement `get_schemas()` method
  - [ ] Implement `@register_tool` decorator
- [ ] Create `backend/agent/tools/discovery.py`
  - [ ] Implement `discover_tools()` function
  - [ ] Implement `initialize_tools()` function
- [ ] Add result helper functions

## Testing

### Unit Tests

```python
# tests/agent/tools/test_base.py

class MockTool(BaseTool):
    name = "mock_tool"
    description = "A mock tool for testing"
    category = ToolCategory.UTILITY
    parameters = [
        ToolParameter("path", str, "File path", required=True),
        ToolParameter("limit", int, "Max items", required=False, default=100),
    ]

    async def execute(self, path: str, limit: int = 100) -> ToolResult:
        return success_result({"path": path, "limit": limit})


def test_tool_parameter_json_schema():
    """Parameter should convert to JSON schema."""
    param = ToolParameter("name", str, "A name", required=True)
    schema = param.to_json_schema()
    assert schema["type"] == "string"
    assert schema["description"] == "A name"


def test_tool_parameter_enum_schema():
    """Enum parameter should include enum values."""
    param = ToolParameter("format", str, "Output format",
                          enum_values=["yolo", "coco", "pascal"])
    schema = param.to_json_schema()
    assert schema["enum"] == ["yolo", "coco", "pascal"]


def test_tool_validate_missing_required():
    """Should fail when required param missing."""
    tool = MockTool()
    error = tool.validate_parameters()  # Missing path
    assert "path" in error


def test_tool_validate_wrong_type():
    """Should fail when param has wrong type."""
    tool = MockTool()
    error = tool.validate_parameters(path=123)  # Should be string
    assert "must be str" in error


def test_tool_get_schema():
    """Tool should generate valid schema."""
    tool = MockTool()
    schema = tool.get_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "mock_tool"
    assert "path" in schema["function"]["parameters"]["properties"]
    assert "path" in schema["function"]["parameters"]["required"]
    assert "limit" not in schema["function"]["parameters"]["required"]


@pytest.mark.asyncio
async def test_tool_run_success():
    """Run should execute and return result."""
    tool = MockTool()
    result = await tool.run(path="/data")
    assert result.success == True
    assert result.data["path"] == "/data"
    assert result.data["limit"] == 100  # Default
    assert result.duration_seconds > 0


@pytest.mark.asyncio
async def test_tool_run_validation_failure():
    """Run should return error on validation failure."""
    tool = MockTool()
    result = await tool.run()  # Missing path
    assert result.success == False
    assert "path" in result.error
```

### Registry Tests

```python
# tests/agent/tools/test_registry.py

def test_registry_singleton():
    """Registry should be singleton."""
    r1 = ToolRegistry()
    r2 = ToolRegistry()
    assert r1 is r2


def test_registry_register_and_get():
    """Should register and retrieve tools."""
    registry = get_tool_registry()
    registry.clear()

    tool = MockTool()
    registry.register(tool)

    retrieved = registry.get("mock_tool")
    assert retrieved is tool


def test_registry_duplicate_registration():
    """Should raise on duplicate registration."""
    registry = get_tool_registry()
    registry.clear()

    registry.register(MockTool())
    with pytest.raises(ValueError):
        registry.register(MockTool())


def test_registry_get_schemas():
    """Should return schemas for all tools."""
    registry = get_tool_registry()
    registry.clear()
    registry.register(MockTool())

    schemas = registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "mock_tool"


def test_register_decorator():
    """Decorator should register tool."""
    registry = get_tool_registry()
    registry.clear()

    @register_tool
    class DecoratedTool(BaseTool):
        name = "decorated"
        description = "Test"
        async def execute(self, **kwargs):
            return success_result({})

    assert registry.get("decorated") is not None
```

### Edge Cases

- [ ] Tool with no parameters
- [ ] Tool with all optional parameters
- [ ] Parameter with complex type (list of dicts)
- [ ] Very long parameter values
- [ ] Tool execution timeout
- [ ] Tool that raises exception during execute

## Acceptance Criteria

- [ ] BaseTool can be subclassed to create new tools
- [ ] Parameters are validated before execution
- [ ] JSON schemas are generated correctly for LLM
- [ ] Registry stores and retrieves tools
- [ ] Decorator pattern works for registration
- [ ] Progress callbacks are invoked during execution
- [ ] Timing information is captured

## Files to Create/Modify

```
backend/
├── agent/
│   └── tools/
│       ├── __init__.py      # Exports
│       ├── base.py          # BaseTool, ToolParameter, ToolResult
│       ├── registry.py      # ToolRegistry, decorators
│       └── discovery.py     # Auto-discovery
└── tests/
    └── agent/
        └── tools/
            ├── test_base.py
            └── test_registry.py
```

## Dependencies

```
# No additional dependencies for base tool system
```

## Notes

- Consider adding async generators for streaming progress
- May need to add cancellation support for long-running tools
- Future: Add tool versioning for compatibility
- Consider adding tool dependencies (must run after X)
