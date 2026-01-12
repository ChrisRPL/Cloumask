"""Tests for LLM tool calling module."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agent.llm.tools import (
    execute_tool_call,
    extract_tool_calls,
    format_tool_result_for_display,
    run_tool_loop,
)
from backend.agent.tools.base import BaseTool, ToolParameter, success_result
from backend.agent.tools.registry import ToolRegistry


class TestExtractToolCalls:
    """Tests for extract_tool_calls function."""

    def test_extract_langchain_format(self, ai_message_with_tool_calls):
        """Should extract tool calls from LangChain format."""
        calls = extract_tool_calls(ai_message_with_tool_calls)

        assert len(calls) == 1
        assert calls[0]["id"] == "call_123"
        assert calls[0]["name"] == "scan_directory"
        assert calls[0]["arguments"] == {"path": "/data"}

    def test_extract_json_function_format(self, ai_message_with_json_tool_call):
        """Should extract from JSON function format in content."""
        calls = extract_tool_calls(ai_message_with_json_tool_call)

        assert len(calls) == 1
        assert calls[0]["name"] == "scan_directory"
        assert calls[0]["arguments"] == {"path": "/data"}

    def test_extract_json_tool_calls_array(self):
        """Should extract from tool_calls array in JSON content."""
        response = AIMessage(
            content='{"tool_calls": [{"name": "detect", "arguments": {"input": "/img"}}]}'
        )

        calls = extract_tool_calls(response)

        assert len(calls) == 1
        assert calls[0]["name"] == "detect"

    def test_extract_direct_call_format(self):
        """Should extract from direct call format in JSON content."""
        response = AIMessage(
            content='{"name": "export", "arguments": {"format": "yolo"}}'
        )

        calls = extract_tool_calls(response)

        assert len(calls) == 1
        assert calls[0]["name"] == "export"
        assert calls[0]["arguments"] == {"format": "yolo"}

    def test_no_tool_calls_plain_text(self, ai_message_plain):
        """Should return empty list for plain text response."""
        calls = extract_tool_calls(ai_message_plain)
        assert calls == []

    def test_no_tool_calls_empty_content(self):
        """Should return empty list for empty content."""
        response = AIMessage(content="")
        calls = extract_tool_calls(response)
        assert calls == []

    def test_invalid_json_returns_empty(self):
        """Should return empty list for invalid JSON."""
        response = AIMessage(content="{invalid json}")
        calls = extract_tool_calls(response)
        assert calls == []

    def test_generates_ids_if_missing(self):
        """Should generate IDs if not provided in JSON content format."""
        # Test the JSON content parsing path where IDs might be missing
        response = AIMessage(
            content='{"function": {"name": "test", "arguments": {}}}'
        )

        calls = extract_tool_calls(response)
        assert calls[0]["id"] == "call_0"
        assert calls[0]["name"] == "test"

    def test_multiple_tool_calls(self):
        """Should extract multiple tool calls."""
        response = AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "scan_directory", "args": {"path": "/a"}},
                {"id": "c2", "name": "detect", "args": {"input": "/b"}},
                {"id": "c3", "name": "export", "args": {"format": "coco"}},
            ],
        )

        calls = extract_tool_calls(response)

        assert len(calls) == 3
        assert [c["name"] for c in calls] == ["scan_directory", "detect", "export"]


class TestExecuteToolCall:
    """Tests for execute_tool_call function."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock tool registry."""
        registry = ToolRegistry()
        registry.clear()
        return registry

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""

        class MockTool(BaseTool):
            name = "mock_tool"
            description = "A mock tool for testing"
            parameters = [ToolParameter("arg1", str, "Test argument")]

            async def execute(self, arg1: str = "default"):
                return success_result({"arg1": arg1, "result": "success"})

        return MockTool()

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_registry, mock_tool):
        """Should execute tool and return ToolMessage."""
        mock_registry.register(mock_tool)

        result = await execute_tool_call(
            {"id": "call_1", "name": "mock_tool", "arguments": {"arg1": "test"}},
            registry=mock_registry,
        )

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "call_1"

        content = json.loads(result.content)
        assert content["success"] is True
        assert content["data"]["arg1"] == "test"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, mock_registry):
        """Should return error for unknown tool."""
        result = await execute_tool_call(
            {"id": "call_1", "name": "nonexistent", "arguments": {}},
            registry=mock_registry,
        )

        content = json.loads(result.content)
        assert "error" in content
        assert "nonexistent" in content["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_with_string_arguments(self, mock_registry, mock_tool):
        """Should parse JSON string arguments."""
        mock_registry.register(mock_tool)

        result = await execute_tool_call(
            {
                "id": "call_1",
                "name": "mock_tool",
                "arguments": '{"arg1": "from_json"}',
            },
            registry=mock_registry,
        )

        content = json.loads(result.content)
        assert content["success"] is True
        assert content["data"]["arg1"] == "from_json"

    @pytest.mark.asyncio
    async def test_execute_with_invalid_json_arguments(self, mock_registry):
        """Should return error for invalid JSON arguments."""
        result = await execute_tool_call(
            {"id": "call_1", "name": "mock_tool", "arguments": "not valid json"},
            registry=mock_registry,
        )

        content = json.loads(result.content)
        assert "error" in content

    @pytest.mark.asyncio
    async def test_execute_tool_raises_exception(self, mock_registry):
        """Should catch and return tool execution errors."""

        class FailingTool(BaseTool):
            name = "failing_tool"
            description = "Always fails"

            async def execute(self):
                raise ValueError("Tool error")

        mock_registry.register(FailingTool())

        result = await execute_tool_call(
            {"id": "call_1", "name": "failing_tool", "arguments": {}},
            registry=mock_registry,
        )

        content = json.loads(result.content)
        assert "error" in content
        assert "Tool error" in content["error"]

    @pytest.mark.asyncio
    async def test_uses_global_registry_by_default(self):
        """Should use global registry when none provided."""
        # This test just verifies no exception is raised
        result = await execute_tool_call(
            {"id": "call_1", "name": "unknown", "arguments": {}},
        )
        assert isinstance(result, ToolMessage)


class TestRunToolLoop:
    """Tests for run_tool_loop function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock OllamaProvider."""
        provider = MagicMock()
        provider.invoke = AsyncMock()
        return provider

    @pytest.fixture
    def mock_registry_with_tool(self):
        """Create a registry with a mock tool."""

        class EchoTool(BaseTool):
            name = "echo"
            description = "Echo back the message"
            parameters = [ToolParameter("message", str, "Message to echo")]

            async def execute(self, message: str = ""):
                return success_result({"echoed": message})

        registry = ToolRegistry()
        registry.clear()
        registry.register(EchoTool())
        return registry

    @pytest.mark.asyncio
    async def test_tool_loop_no_tool_calls(self, mock_provider):
        """Should exit immediately when no tool calls."""
        mock_provider.invoke.return_value = AIMessage(content="Done")

        messages = [HumanMessage(content="Hello")]
        result = await run_tool_loop(mock_provider, messages, registry=ToolRegistry())

        assert len(result) == 2  # Original + AI response
        assert mock_provider.invoke.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_loop_executes_tools(
        self, mock_provider, mock_registry_with_tool
    ):
        """Should execute tool calls and add results."""
        # First call returns tool call, second returns final response
        mock_provider.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[{"id": "c1", "name": "echo", "args": {"message": "test"}}],
            ),
            AIMessage(content="Done!"),
        ]

        messages = [HumanMessage(content="Echo test")]
        result = await run_tool_loop(
            mock_provider, messages, registry=mock_registry_with_tool
        )

        # Original + AI with tool + tool result + final AI
        assert len(result) == 4
        assert isinstance(result[2], ToolMessage)
        assert result[3].content == "Done!"

    @pytest.mark.asyncio
    async def test_tool_loop_max_iterations(
        self, mock_provider, mock_registry_with_tool
    ):
        """Should stop at max iterations."""
        # Always return tool calls
        mock_provider.invoke.return_value = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "echo", "args": {"message": "loop"}}],
        )

        messages = [HumanMessage(content="Loop forever")]
        await run_tool_loop(
            mock_provider, messages, max_iterations=3, registry=mock_registry_with_tool
        )

        assert mock_provider.invoke.call_count == 3

    @pytest.mark.asyncio
    async def test_tool_loop_multiple_tools_per_call(
        self, mock_provider, mock_registry_with_tool
    ):
        """Should execute all tool calls in a single response."""
        mock_provider.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "echo", "args": {"message": "first"}},
                    {"id": "c2", "name": "echo", "args": {"message": "second"}},
                ],
            ),
            AIMessage(content="Both done!"),
        ]

        messages = [HumanMessage(content="Two echoes")]
        result = await run_tool_loop(
            mock_provider, messages, registry=mock_registry_with_tool
        )

        # Original + AI with tools + 2 tool results + final AI
        assert len(result) == 5
        tool_results = [m for m in result if isinstance(m, ToolMessage)]
        assert len(tool_results) == 2


class TestFormatToolResultForDisplay:
    """Tests for format_tool_result_for_display function."""

    def test_format_success_result(self):
        """Should format successful result nicely."""
        message = ToolMessage(
            content='{"success": true, "data": {"count": 10, "format": "yolo"}}',
            tool_call_id="test",
        )

        result = format_tool_result_for_display(message)

        assert "count: 10" in result
        assert "format: yolo" in result

    def test_format_error_result(self):
        """Should format error message."""
        message = ToolMessage(
            content='{"error": "File not found"}',
            tool_call_id="test",
        )

        result = format_tool_result_for_display(message)

        assert "Error: File not found" in result

    def test_format_list_data(self):
        """Should show count for list data."""
        message = ToolMessage(
            content='{"success": true, "data": {"files": [1, 2, 3, 4, 5]}}',
            tool_call_id="test",
        )

        result = format_tool_result_for_display(message)

        assert "files: 5 items" in result

    def test_format_invalid_json(self):
        """Should return raw content for invalid JSON."""
        message = ToolMessage(
            content="Not valid JSON",
            tool_call_id="test",
        )

        result = format_tool_result_for_display(message)

        assert result == "Not valid JSON"

    def test_format_skips_meta_fields(self):
        """Should skip fields starting with underscore."""
        message = ToolMessage(
            content='{"success": true, "data": {"result": "ok", "_internal": "hidden"}}',
            tool_call_id="test",
        )

        result = format_tool_result_for_display(message)

        assert "result: ok" in result
        assert "_internal" not in result
