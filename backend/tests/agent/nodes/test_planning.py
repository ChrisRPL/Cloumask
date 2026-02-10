"""
Tests for the planning nodes: understand and generate_plan.

These tests verify:
- validate_plan() correctly validates plan structures
- format_plan_for_display() produces readable output
- understand() parses user requests correctly (mocked LLM)
- generate_plan() creates valid plans (mocked LLM)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agent.nodes.plan import (
    VALID_TOOLS,
    format_plan_for_display,
    generate_plan,
    validate_plan,
)
from backend.agent.nodes.understand import understand
from backend.agent.state import MessageRole, PipelineState, StepStatus

# -----------------------------------------------------------------------------
# validate_plan() tests
# -----------------------------------------------------------------------------


class TestValidatePlan:
    """Tests for the validate_plan function."""

    def test_empty_plan_fails(self) -> None:
        """Empty plan should fail validation."""
        result = validate_plan([])
        assert result == "Plan is empty"

    def test_unknown_tool_fails(self) -> None:
        """Unknown tool should fail validation."""
        plan = [{"tool_name": "unknown_tool", "parameters": {"foo": "bar"}}]
        result = validate_plan(plan)
        assert result is not None
        assert "unknown tool" in result.lower()

    def test_missing_tool_name_fails(self) -> None:
        """Step without tool_name should fail."""
        plan = [{"parameters": {"path": "/data"}}]
        result = validate_plan(plan)
        assert result is not None
        assert "no tool_name" in result.lower()

    def test_missing_parameters_fails(self) -> None:
        """Step without parameters should fail."""
        plan = [{"tool_name": "scan_directory"}]
        result = validate_plan(plan)
        assert result is not None
        assert "no parameters" in result.lower()

    def test_valid_plan_passes(self) -> None:
        """Valid plan should pass validation."""
        plan = [
            {"tool_name": "scan_directory", "parameters": {"path": "/data"}},
            {
                "tool_name": "detect",
                "parameters": {"input_path": "/data", "classes": ["car"]},
            },
        ]
        result = validate_plan(plan)
        assert result is None

    def test_scan_directory_missing_path_fails(self) -> None:
        """scan_directory without path should fail."""
        plan = [{"tool_name": "scan_directory", "parameters": {"recursive": True}}]
        result = validate_plan(plan)
        assert result is not None
        assert "path" in result.lower()

    def test_anonymize_missing_output_path_fails(self) -> None:
        """anonymize without output_path should fail."""
        plan = [
            {
                "tool_name": "anonymize",
                "parameters": {"input_path": "/data"},
            }
        ]
        result = validate_plan(plan)
        assert result is not None
        assert "output_path" in result.lower()

    def test_detect_missing_classes_fails(self) -> None:
        """detect without classes should fail."""
        plan = [
            {
                "tool_name": "detect",
                "parameters": {"input_path": "/data"},
            }
        ]
        result = validate_plan(plan)
        assert result is not None
        assert "classes" in result.lower()

    def test_segment_missing_prompt_fails(self) -> None:
        """segment without prompt should fail."""
        plan = [
            {
                "tool_name": "segment",
                "parameters": {"input_path": "/data"},
            }
        ]
        result = validate_plan(plan)
        assert result is not None
        assert "prompt" in result.lower()

    def test_export_missing_format_fails(self) -> None:
        """export without format should fail."""
        plan = [
            {
                "tool_name": "export",
                "parameters": {"input_path": "/data", "output_path": "/out"},
            }
        ]
        result = validate_plan(plan)
        assert result is not None
        assert "format" in result.lower()

    def test_convert_format_missing_target_format_fails(self) -> None:
        """convert_format without target_format should fail."""
        plan = [
            {
                "tool_name": "convert_format",
                "parameters": {"source_path": "/data", "output_path": "/out"},
            }
        ]
        result = validate_plan(plan)
        assert result is not None
        assert "target_format" in result.lower()

    def test_find_duplicates_missing_path_fails(self) -> None:
        """find_duplicates without path should fail."""
        plan = [
            {
                "tool_name": "find_duplicates",
                "parameters": {"threshold": 0.95},
            }
        ]
        result = validate_plan(plan)
        assert result is not None
        assert "path" in result.lower()

    def test_all_tools_can_pass(self) -> None:
        """All valid tools should be able to pass validation."""
        for tool in VALID_TOOLS:
            if tool == "scan_directory":
                params = {"path": "/data"}
            elif tool == "anonymize":
                params = {"input_path": "/data", "output_path": "/out"}
            elif tool == "detect":
                params = {"input_path": "/data", "classes": ["car"]}
            elif tool == "segment":
                params = {"input_path": "/data", "prompt": "cars"}
            elif tool == "export":
                params = {"input_path": "/data", "output_path": "/out", "format": "yolo"}
            elif tool == "convert_format":
                params = {
                    "source_path": "/data/source",
                    "output_path": "/data/converted",
                    "target_format": "yolo",
                }
            elif tool == "find_duplicates":
                params = {"path": "/data/images", "method": "phash", "threshold": 0.9}
            else:
                continue

            plan = [{"tool_name": tool, "parameters": params}]
            result = validate_plan(plan)
            assert result is None, f"Tool {tool} failed validation: {result}"


# -----------------------------------------------------------------------------
# format_plan_for_display() tests
# -----------------------------------------------------------------------------


class TestFormatPlanForDisplay:
    """Tests for the format_plan_for_display function."""

    def test_empty_plan(self) -> None:
        """Empty plan should return appropriate message."""
        result = format_plan_for_display([])
        assert "no steps" in result.lower()

    def test_single_step(self) -> None:
        """Single step should be formatted correctly."""
        plan = [
            {
                "tool_name": "scan_directory",
                "parameters": {"path": "/data"},
                "description": "Scan input directory",
                "status": StepStatus.PENDING.value,
            }
        ]
        result = format_plan_for_display(plan)
        assert "Step 1" in result
        assert "scan_directory" in result
        assert "Scan input directory" in result
        assert "/data" in result

    def test_multiple_steps(self) -> None:
        """Multiple steps should all appear."""
        plan = [
            {
                "tool_name": "scan_directory",
                "parameters": {"path": "/data"},
                "description": "Step one",
            },
            {
                "tool_name": "detect",
                "parameters": {"input_path": "/data", "classes": ["car"]},
                "description": "Step two",
            },
        ]
        result = format_plan_for_display(plan)
        assert "Step 1" in result
        assert "Step 2" in result
        assert "scan_directory" in result
        assert "detect" in result

    def test_status_icons(self) -> None:
        """Status icons should appear based on step status."""
        plan = [
            {"tool_name": "scan_directory", "parameters": {"path": "/data"}, "status": "pending"},
            {"tool_name": "detect", "parameters": {"input_path": "/data", "classes": []}, "status": "completed"},
        ]
        result = format_plan_for_display(plan)
        assert "[ ]" in result  # pending
        assert "[x]" in result  # completed

    def test_list_parameters(self) -> None:
        """List parameters should be formatted correctly."""
        plan = [
            {
                "tool_name": "detect",
                "parameters": {"input_path": "/data", "classes": ["car", "truck", "person"]},
                "description": "Detect objects",
            }
        ]
        result = format_plan_for_display(plan)
        assert "car" in result
        assert "truck" in result
        assert "person" in result


# -----------------------------------------------------------------------------
# understand() tests (mocked LLM)
# -----------------------------------------------------------------------------


class TestUnderstandNode:
    """Tests for the understand node with mocked LLM."""

    @pytest.fixture
    def base_state(self) -> PipelineState:
        """Create a base state with a user message."""
        return PipelineState(
            messages=[
                {
                    "role": MessageRole.USER.value,
                    "content": "scan /data/images",
                    "timestamp": "2024-01-01T00:00:00",
                    "tool_calls": [],
                    "tool_call_id": None,
                }
            ],
            metadata={},
            plan=[],
            plan_approved=False,
            current_step=0,
            execution_results={},
            checkpoints=[],
            awaiting_user=False,
            last_error=None,
            retry_count=0,
        )

    @pytest.mark.asyncio
    async def test_no_user_message(self) -> None:
        """Should return error when no user message exists."""
        state = PipelineState(
            messages=[],
            metadata={},
            plan=[],
            plan_approved=False,
            current_step=0,
            execution_results={},
            checkpoints=[],
            awaiting_user=False,
            last_error=None,
            retry_count=0,
        )
        result = await understand(state)
        assert "last_error" in result
        assert "no user message" in result["last_error"].lower()

    @pytest.mark.asyncio
    async def test_parse_simple_request(self, base_state: PipelineState) -> None:
        """Should parse a simple scan request."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "intent": "scan",
            "input_path": "/data/images",
            "input_type": "images",
            "operations": ["scan"],
            "parameters": {},
            "output_path": None,
            "clarification_needed": None,
        })

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await understand(base_state)

            assert "metadata" in result
            assert "understanding" in result["metadata"]
            assert result["metadata"]["understanding"]["intent"] == "scan"
            assert result["metadata"]["understanding"]["input_path"] == "/data/images"

    @pytest.mark.asyncio
    async def test_clarification_needed(self, base_state: PipelineState) -> None:
        """Should request clarification for unclear requests."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "intent": None,
            "input_path": None,
            "input_type": None,
            "operations": [],
            "parameters": {},
            "output_path": None,
            "clarification_needed": "What would you like to do with your data?",
        })

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await understand(base_state)

            assert result.get("awaiting_user") is True
            assert len(result.get("messages", [])) > 0
            last_msg = result["messages"][-1]
            assert "What would you like to do" in last_msg["content"]

    @pytest.mark.asyncio
    async def test_handles_json_in_code_block(self, base_state: PipelineState) -> None:
        """Should extract JSON from markdown code blocks."""
        mock_response = MagicMock()
        mock_response.content = """Here's my understanding:

```json
{
    "intent": "anonymize",
    "input_path": "/dashcam",
    "input_type": "video",
    "operations": ["anonymize"],
    "parameters": {"target": "faces"},
    "output_path": null,
    "clarification_needed": null
}
```"""

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await understand(base_state)

            assert "metadata" in result
            assert result["metadata"]["understanding"]["intent"] == "anonymize"


# -----------------------------------------------------------------------------
# generate_plan() tests (mocked LLM)
# -----------------------------------------------------------------------------


class TestGeneratePlanNode:
    """Tests for the generate_plan node with mocked LLM."""

    @pytest.fixture
    def state_with_understanding(self) -> PipelineState:
        """Create a state with understanding already extracted."""
        return PipelineState(
            messages=[
                {
                    "role": MessageRole.USER.value,
                    "content": "scan /data/images",
                    "timestamp": "2024-01-01T00:00:00",
                    "tool_calls": [],
                    "tool_call_id": None,
                },
                {
                    "role": MessageRole.ASSISTANT.value,
                    "content": "I understand...",
                    "timestamp": "2024-01-01T00:00:01",
                    "tool_calls": [],
                    "tool_call_id": None,
                },
            ],
            metadata={
                "understanding": {
                    "intent": "scan",
                    "input_path": "/data/images",
                    "input_type": "images",
                    "operations": ["scan"],
                    "parameters": {},
                    "output_path": None,
                }
            },
            plan=[],
            plan_approved=False,
            current_step=0,
            execution_results={},
            checkpoints=[],
            awaiting_user=False,
            last_error=None,
            retry_count=0,
        )

    @pytest.mark.asyncio
    async def test_no_understanding(self) -> None:
        """Should return error when no understanding exists."""
        state = PipelineState(
            messages=[],
            metadata={},
            plan=[],
            plan_approved=False,
            current_step=0,
            execution_results={},
            checkpoints=[],
            awaiting_user=False,
            last_error=None,
            retry_count=0,
        )
        result = await generate_plan(state)
        assert "last_error" in result
        assert "no understanding" in result["last_error"].lower()

    @pytest.mark.asyncio
    async def test_generates_valid_plan(
        self, state_with_understanding: PipelineState
    ) -> None:
        """Should generate a valid plan from understanding."""
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            {
                "tool_name": "scan_directory",
                "parameters": {"path": "/data/images", "recursive": True},
                "description": "Scan input directory",
            }
        ])

        with patch("backend.agent.nodes.plan.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await generate_plan(state_with_understanding)

            assert "plan" in result
            assert len(result["plan"]) == 1
            assert result["plan"][0]["tool_name"] == "scan_directory"
            assert result["current_step"] == 0
            assert result["plan_approved"] is False

    @pytest.mark.asyncio
    async def test_plan_includes_step_metadata(
        self, state_with_understanding: PipelineState
    ) -> None:
        """Generated plan steps should include required metadata."""
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            {
                "tool_name": "scan_directory",
                "parameters": {"path": "/data"},
                "description": "Test step",
            }
        ])

        with patch("backend.agent.nodes.plan.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await generate_plan(state_with_understanding)

            step = result["plan"][0]
            assert "id" in step
            assert step["id"].startswith("step-")
            assert step["status"] == StepStatus.PENDING.value
            assert step["result"] is None
            assert step["error"] is None

    @pytest.mark.asyncio
    async def test_handles_invalid_plan_response(
        self, state_with_understanding: PipelineState
    ) -> None:
        """Should handle LLM returning invalid plan gracefully."""
        mock_response = MagicMock()
        # Return plan with unknown tool
        mock_response.content = json.dumps([
            {
                "tool_name": "unknown_tool",
                "parameters": {"foo": "bar"},
                "description": "Invalid step",
            }
        ])

        with patch("backend.agent.nodes.plan.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await generate_plan(state_with_understanding)

            # Should set awaiting_user and have an error
            assert result.get("awaiting_user") is True or result.get("last_error") is not None

    @pytest.mark.asyncio
    async def test_formats_plan_in_message(
        self, state_with_understanding: PipelineState
    ) -> None:
        """Plan should be formatted in the assistant message."""
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            {
                "tool_name": "scan_directory",
                "parameters": {"path": "/data"},
                "description": "Scan the data folder",
            }
        ])

        with patch("backend.agent.nodes.plan.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await generate_plan(state_with_understanding)

            messages = result.get("messages", [])
            assert len(messages) > 0
            last_msg = messages[-1]
            assert "scan_directory" in last_msg["content"]
            assert "proceed" in last_msg["content"].lower()


# -----------------------------------------------------------------------------
# Edge cases and integration tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for planning nodes."""

    def test_validate_plan_with_none(self) -> None:
        """validate_plan should handle edge cases gracefully."""
        # Empty list
        assert validate_plan([]) == "Plan is empty"

    def test_format_plan_handles_missing_fields(self) -> None:
        """format_plan_for_display should handle missing fields."""
        plan = [
            {
                "tool_name": "scan_directory",
                "parameters": {"path": "/data"},
                # Missing description and status
            }
        ]
        result = format_plan_for_display(plan)
        assert "Step 1" in result
        assert "scan_directory" in result

    def test_valid_tools_constant(self) -> None:
        """VALID_TOOLS should contain expected tools."""
        expected = {
            "scan_directory",
            "anonymize",
            "detect",
            "segment",
            "export",
            "convert_format",
            "find_duplicates",
        }
        assert expected == VALID_TOOLS
