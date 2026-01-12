"""
Tests for the tool base classes and types.

Tests ToolCategory, ToolParameter, ToolResult, and BaseTool functionality.

Implements spec: 06-tool-system (testing section)
"""

from __future__ import annotations

import pytest

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)

# -----------------------------------------------------------------------------
# Mock Tool for Testing
# -----------------------------------------------------------------------------


class MockTool(BaseTool):
    """A mock tool for testing the base class functionality."""

    name = "mock_tool"
    description = "A mock tool for testing"
    category = ToolCategory.UTILITY
    parameters = [
        ToolParameter("path", str, "File path", required=True),
        ToolParameter("limit", int, "Max items", required=False, default=100),
        ToolParameter("format", str, "Output format", required=False, default="json", enum_values=["json", "csv", "xml"]),
    ]

    async def execute(self, path: str, limit: int = 100, **kwargs: object) -> ToolResult:
        """Execute the mock tool."""
        return success_result({"path": path, "limit": limit})


class NoParamTool(BaseTool):
    """Tool with no parameters for edge case testing."""

    name = "no_param_tool"
    description = "Tool with no parameters"
    category = ToolCategory.UTILITY
    parameters = []

    async def execute(self, **kwargs: object) -> ToolResult:
        """Execute the no-param tool."""
        return success_result({"status": "ok"})


class AllOptionalTool(BaseTool):
    """Tool with all optional parameters."""

    name = "all_optional_tool"
    description = "Tool with all optional parameters"
    category = ToolCategory.UTILITY
    parameters = [
        ToolParameter("a", str, "Optional A", required=False, default="default_a"),
        ToolParameter("b", int, "Optional B", required=False, default=42),
    ]

    async def execute(self, a: str = "default_a", b: int = 42, **kwargs: object) -> ToolResult:
        """Execute with optional params."""
        return success_result({"a": a, "b": b})


class FailingTool(BaseTool):
    """Tool that raises exceptions for error handling tests."""

    name = "failing_tool"
    description = "Tool that fails"
    category = ToolCategory.UTILITY
    parameters = []

    async def execute(self, **kwargs: object) -> ToolResult:
        """Raise an exception."""
        raise ValueError("Intentional test failure")


# -----------------------------------------------------------------------------
# ToolParameter Tests
# -----------------------------------------------------------------------------


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_json_schema_string(self) -> None:
        """String parameter should convert to JSON schema."""
        param = ToolParameter("name", str, "A name", required=True)
        schema = param.to_json_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "A name"

    def test_json_schema_integer(self) -> None:
        """Integer parameter should convert to JSON schema."""
        param = ToolParameter("count", int, "Item count")
        schema = param.to_json_schema()
        assert schema["type"] == "integer"

    def test_json_schema_number(self) -> None:
        """Float parameter should convert to number in JSON schema."""
        param = ToolParameter("confidence", float, "Confidence score")
        schema = param.to_json_schema()
        assert schema["type"] == "number"

    def test_json_schema_boolean(self) -> None:
        """Boolean parameter should convert to JSON schema."""
        param = ToolParameter("enabled", bool, "Enable feature")
        schema = param.to_json_schema()
        assert schema["type"] == "boolean"

    def test_json_schema_array(self) -> None:
        """List parameter should convert to array in JSON schema."""
        param = ToolParameter("items", list, "List of items")
        schema = param.to_json_schema()
        assert schema["type"] == "array"

    def test_json_schema_object(self) -> None:
        """Dict parameter should convert to object in JSON schema."""
        param = ToolParameter("config", dict, "Configuration")
        schema = param.to_json_schema()
        assert schema["type"] == "object"

    def test_json_schema_with_enum(self) -> None:
        """Enum parameter should include enum values in schema."""
        param = ToolParameter(
            "format",
            str,
            "Output format",
            enum_values=["yolo", "coco", "pascal"],
        )
        schema = param.to_json_schema()
        assert schema["enum"] == ["yolo", "coco", "pascal"]

    def test_json_schema_with_default(self) -> None:
        """Default value should be included in schema."""
        param = ToolParameter("limit", int, "Max items", default=100)
        schema = param.to_json_schema()
        assert schema["default"] == 100

    def test_json_schema_unknown_type(self) -> None:
        """Unknown types should default to string."""
        param = ToolParameter("custom", object, "Custom type")
        schema = param.to_json_schema()
        assert schema["type"] == "string"


# -----------------------------------------------------------------------------
# ToolResult Tests
# -----------------------------------------------------------------------------


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result_to_dict(self) -> None:
        """Success result should serialize correctly."""
        result = ToolResult(
            success=True,
            data={"count": 10, "path": "/data"},
            duration_seconds=1.5,
            metadata={"tool_version": "1.0"},
        )
        d = result.to_dict()

        assert d["success"] is True
        assert d["count"] == 10
        assert d["path"] == "/data"
        assert d["data"] == {"count": 10, "path": "/data"}
        assert d["duration_seconds"] == 1.5
        assert d["_meta"]["tool_version"] == "1.0"

    def test_error_result_to_dict(self) -> None:
        """Error result should serialize with error message."""
        result = ToolResult(
            success=False,
            error="File not found",
            duration_seconds=0.1,
        )
        d = result.to_dict()

        assert d["success"] is False
        assert d["error"] == "File not found"
        assert "data" not in d

    def test_empty_data_success(self) -> None:
        """Success with no data should still work."""
        result = ToolResult(success=True)
        d = result.to_dict()

        assert d["success"] is True
        assert d["data"] == {}


# -----------------------------------------------------------------------------
# Result Helper Tests
# -----------------------------------------------------------------------------


class TestResultHelpers:
    """Tests for success_result and error_result helpers."""

    def test_success_result_helper(self) -> None:
        """success_result should create successful ToolResult."""
        result = success_result({"count": 5}, source="test")

        assert result.success is True
        assert result.data == {"count": 5}
        assert result.metadata["source"] == "test"
        assert result.error is None

    def test_error_result_helper(self) -> None:
        """error_result should create failed ToolResult."""
        result = error_result("Something went wrong", code=500)

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.metadata["code"] == 500
        assert result.data is None


# -----------------------------------------------------------------------------
# BaseTool Validation Tests
# -----------------------------------------------------------------------------


class TestToolValidation:
    """Tests for parameter validation."""

    def test_validate_missing_required(self) -> None:
        """Should fail when required param is missing."""
        tool = MockTool()
        error = tool.validate_parameters()
        assert error is not None
        assert "path" in error

    def test_validate_wrong_type(self) -> None:
        """Should fail when param has wrong type."""
        tool = MockTool()
        error = tool.validate_parameters(path=123)
        assert error is not None
        assert "must be str" in error

    def test_validate_enum_invalid(self) -> None:
        """Should fail when enum value is invalid."""
        tool = MockTool()
        error = tool.validate_parameters(path="/data", format="invalid")
        assert error is not None
        assert "must be one of" in error

    def test_validate_enum_valid(self) -> None:
        """Should pass with valid enum value."""
        tool = MockTool()
        error = tool.validate_parameters(path="/data", format="json")
        assert error is None

    def test_validate_with_defaults(self) -> None:
        """Should pass when only required params provided."""
        tool = MockTool()
        error = tool.validate_parameters(path="/data")
        assert error is None

    def test_validate_no_params(self) -> None:
        """Tool with no params should validate with no args."""
        tool = NoParamTool()
        error = tool.validate_parameters()
        assert error is None

    def test_validate_all_optional(self) -> None:
        """Tool with all optional params should validate with no args."""
        tool = AllOptionalTool()
        error = tool.validate_parameters()
        assert error is None


# -----------------------------------------------------------------------------
# BaseTool Schema Tests
# -----------------------------------------------------------------------------


class TestToolSchema:
    """Tests for JSON schema generation."""

    def test_get_schema_structure(self) -> None:
        """Schema should have correct structure."""
        tool = MockTool()
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mock_tool"
        assert schema["function"]["description"] == "A mock tool for testing"
        assert "parameters" in schema["function"]

    def test_get_schema_properties(self) -> None:
        """Schema should include all parameters."""
        tool = MockTool()
        schema = tool.get_schema()
        properties = schema["function"]["parameters"]["properties"]

        assert "path" in properties
        assert "limit" in properties
        assert "format" in properties

    def test_get_schema_required(self) -> None:
        """Schema should mark required params correctly."""
        tool = MockTool()
        schema = tool.get_schema()
        required = schema["function"]["parameters"]["required"]

        assert "path" in required
        # limit has default, so not required
        assert "limit" not in required
        # format has default, so not required
        assert "format" not in required

    def test_get_schema_no_params(self) -> None:
        """Tool with no params should have empty properties."""
        tool = NoParamTool()
        schema = tool.get_schema()

        assert schema["function"]["parameters"]["properties"] == {}
        assert schema["function"]["parameters"]["required"] == []


# -----------------------------------------------------------------------------
# BaseTool Execution Tests
# -----------------------------------------------------------------------------


class TestToolExecution:
    """Tests for tool execution via run()."""

    @pytest.mark.asyncio
    async def test_run_success(self) -> None:
        """Run should execute and return result."""
        tool = MockTool()
        result = await tool.run(path="/data", format="json")

        assert result.success is True
        assert result.data is not None
        assert result.data["path"] == "/data"
        assert result.data["limit"] == 100  # Default
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_run_with_defaults(self) -> None:
        """Run should apply default values."""
        tool = AllOptionalTool()
        result = await tool.run()

        assert result.success is True
        assert result.data is not None
        assert result.data["a"] == "default_a"
        assert result.data["b"] == 42

    @pytest.mark.asyncio
    async def test_run_override_defaults(self) -> None:
        """Run should allow overriding defaults."""
        tool = AllOptionalTool()
        result = await tool.run(a="custom", b=99)

        assert result.success is True
        assert result.data is not None
        assert result.data["a"] == "custom"
        assert result.data["b"] == 99

    @pytest.mark.asyncio
    async def test_run_validation_failure(self) -> None:
        """Run should return error on validation failure."""
        tool = MockTool()
        result = await tool.run()  # Missing path

        assert result.success is False
        assert result.error is not None
        assert "path" in result.error

    @pytest.mark.asyncio
    async def test_run_exception_handling(self) -> None:
        """Run should catch exceptions and return error result."""
        tool = FailingTool()
        result = await tool.run()

        assert result.success is False
        assert result.error is not None
        assert "Intentional test failure" in result.error
        assert result.duration_seconds >= 0


# -----------------------------------------------------------------------------
# Progress Callback Tests
# -----------------------------------------------------------------------------


class TestProgressCallback:
    """Tests for progress reporting."""

    def test_set_progress_callback(self) -> None:
        """Should set progress callback."""
        tool = MockTool()
        calls: list[tuple[int, int, str]] = []

        def callback(current: int, total: int, message: str) -> None:
            calls.append((current, total, message))

        tool.set_progress_callback(callback)
        tool.report_progress(5, 10, "Processing")

        assert len(calls) == 1
        assert calls[0] == (5, 10, "Processing")

    def test_report_progress_no_callback(self) -> None:
        """Should not raise when no callback set."""
        tool = MockTool()
        # Should not raise
        tool.report_progress(1, 10, "Test")
        assert tool._last_progress == 1


# -----------------------------------------------------------------------------
# ToolCategory Tests
# -----------------------------------------------------------------------------


class TestToolCategory:
    """Tests for ToolCategory enum."""

    def test_category_values(self) -> None:
        """All expected categories should exist."""
        assert ToolCategory.SCAN.value == "scan"
        assert ToolCategory.DETECTION.value == "detection"
        assert ToolCategory.SEGMENTATION.value == "segmentation"
        assert ToolCategory.ANONYMIZATION.value == "anonymization"
        assert ToolCategory.EXPORT.value == "export"
        assert ToolCategory.UTILITY.value == "utility"

    def test_category_is_string(self) -> None:
        """Category should be usable as string."""
        # ToolCategory inherits from str, so .value gives the string
        assert ToolCategory.SCAN.value == "scan"
        assert str(ToolCategory.SCAN.value) == "scan"
