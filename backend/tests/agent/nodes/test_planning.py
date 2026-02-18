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
    build_rule_based_plan,
    format_plan_for_display,
    generate_plan,
    validate_plan,
)
from backend.agent.nodes.understand import understand
from backend.agent.state import MessageRole, PipelineState, StepStatus
from backend.agent.tools.base import BaseTool, ToolCategory, ToolParameter, success_result
from backend.agent.tools.registry import get_tool_registry


def _state_with_message(content: str) -> PipelineState:
    return PipelineState(
        messages=[
            {
                "role": MessageRole.USER.value,
                "content": content,
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
        assert "missing required 'path'" in result.lower()

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

    def test_detect_missing_classes_passes(self) -> None:
        """detect without classes should pass (planner must not invent classes)."""
        plan = [
            {
                "tool_name": "detect",
                "parameters": {"input_path": "/data"},
            }
        ]
        result = validate_plan(plan)
        assert result is None

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

    def test_export_missing_output_format_fails(self) -> None:
        """export without output_format should fail."""
        plan = [
            {
                "tool_name": "export",
                "parameters": {"source_path": "/data", "output_path": "/out"},
            }
        ]
        result = validate_plan(plan)
        assert result is not None
        assert "output_format" in result.lower()

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

    def test_label_qa_missing_path_fails(self) -> None:
        """label_qa without path should fail."""
        plan = [
            {
                "tool_name": "label_qa",
                "parameters": {"generate_report": True},
            }
        ]
        result = validate_plan(plan)
        assert result is not None
        assert "path" in result.lower()

    def test_split_dataset_missing_output_path_fails(self) -> None:
        """split_dataset without output_path should fail."""
        plan = [
            {
                "tool_name": "split_dataset",
                "parameters": {"path": "/data/source"},
            }
        ]
        result = validate_plan(plan)
        assert result is not None
        assert "output_path" in result.lower()

    def test_run_script_generated_code_without_script_path_passes(self) -> None:
        """run_script should accept generated_code-only planner output."""
        plan = [
            {
                "tool_name": "run_script",
                "parameters": {"input_path": "/data"},
                "generated_code": "print('hello')",
            }
        ]
        result = validate_plan(plan)
        assert result is None

    def test_registry_registered_tool_is_accepted(self) -> None:
        """Validation should accept dynamically registered custom tools."""

        class _AdhocTool(BaseTool):
            name = "adhoc_plan_test_tool"
            description = "Adhoc planner validation test tool"
            category = ToolCategory.UTILITY
            parameters = [ToolParameter("input_path", str, "Input path", required=True)]

            async def execute(self, **kwargs: object):  # type: ignore[override]
                return success_result({"ok": True})

        registry = get_tool_registry()
        tool = _AdhocTool()
        registry.register(tool)
        try:
            result = validate_plan(
                [{"tool_name": tool.name, "parameters": {"input_path": "/data"}}]
            )
            assert result is None
        finally:
            registry.unregister(tool.name)

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
                params = {
                    "source_path": "/data/source",
                    "output_path": "/data/exported",
                    "output_format": "yolo",
                }
            elif tool == "convert_format":
                params = {
                    "source_path": "/data/source",
                    "output_path": "/data/converted",
                    "target_format": "yolo",
                }
            elif tool == "find_duplicates":
                params = {"path": "/data/images", "method": "phash", "threshold": 0.9}
            elif tool == "label_qa":
                params = {"path": "/data/dataset", "generate_report": True}
            elif tool == "split_dataset":
                params = {"path": "/data/source", "output_path": "/data/split"}
            elif tool == "run_script":
                params = {"input_path": "/data"}
            elif tool == "review":
                params = {"source_path": "/data/detections", "image_dir": "/data/images"}
            else:
                # Point cloud tools and others — skip specific validation
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

    @pytest.mark.asyncio
    async def test_fast_path_multistep_task_skips_llm(self) -> None:
        """Clear multi-operation tasks should bypass LLM understanding."""
        state = _state_with_message(
            "detect cars and people in /tmp/images, anonymize faces, then export yolo"
        )

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            result = await understand(state)
            mock_get_llm.assert_not_called()

        understanding = result["metadata"]["understanding"]
        assert understanding["input_path"] == "/tmp/images"
        assert "detect" in understanding["operations"]
        assert "anonymize" in understanding["operations"]
        assert "export" in understanding["operations"]

    @pytest.mark.asyncio
    async def test_fast_path_detect_classes_excludes_anonymization_targets(self) -> None:
        """Detection classes should not be polluted by anonymization targets."""
        state = _state_with_message(
            "detect cars, people, traffic lights and road signs in /tmp/images, "
            "then anonymize faces and plates and export yolo"
        )

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            result = await understand(state)
            mock_get_llm.assert_not_called()

        params = result["metadata"]["understanding"]["parameters"]
        assert set(params["classes"]) == {"car", "person", "traffic light", "road sign"}
        assert params["target"] == "all"

    @pytest.mark.asyncio
    async def test_fast_path_respects_faces_only_negation(self) -> None:
        """Fast-path should map 'only faces / not plates' to faces target."""
        state = _state_with_message(
            "detect cars and people in /tmp/images, then anonymize only faces "
            "and do not anonymize plates, then export yolo"
        )

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            result = await understand(state)
            mock_get_llm.assert_not_called()

        params = result["metadata"]["understanding"]["parameters"]
        assert set(params["classes"]) == {"car", "person"}
        assert params["target"] == "faces"

    @pytest.mark.asyncio
    async def test_fast_path_strips_trailing_path_punctuation(self) -> None:
        """Fast-path path extraction should ignore sentence punctuation."""
        state = _state_with_message("detect cars and people in /tmp/images.")

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            result = await understand(state)
            mock_get_llm.assert_not_called()

        understanding = result["metadata"]["understanding"]
        assert understanding["input_path"] == "/tmp/images"

    @pytest.mark.asyncio
    async def test_fast_path_supports_unquoted_paths_with_spaces(self) -> None:
        """Fast-path should capture common unquoted absolute paths with spaces."""
        state = _state_with_message(
            "detect people in /Users/krzysztof/Documents/data images and export yolo"
        )

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            result = await understand(state)
            mock_get_llm.assert_not_called()

        understanding = result["metadata"]["understanding"]
        assert understanding["input_path"] == "/Users/krzysztof/Documents/data images"

    @pytest.mark.asyncio
    async def test_fast_path_stops_path_before_operation_keywords(self) -> None:
        """Path extraction should stop before operation words after spaced paths."""
        state = _state_with_message(
            "detect people in /Users/krzysztof/Documents/data images anonymize only faces and export yolo"
        )

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            result = await understand(state)
            mock_get_llm.assert_not_called()

        understanding = result["metadata"]["understanding"]
        assert understanding["input_path"] == "/Users/krzysztof/Documents/data images"

    @pytest.mark.asyncio
    async def test_fast_path_stops_after_sentence_punctuation(self) -> None:
        """Path extraction should not append words after punctuation-delimited paths."""
        state = _state_with_message(
            "Use /Users/krzysztof/Documents/data. Create a plan to detect cars."
        )

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            result = await understand(state)
            mock_get_llm.assert_not_called()

        understanding = result["metadata"]["understanding"]
        assert understanding["input_path"] == "/Users/krzysztof/Documents/data"

    @pytest.mark.asyncio
    async def test_fast_path_keeps_segment_target_and_custom_final_step(self) -> None:
        """Segment target + custom final step should stay specific in the generated plan."""
        state = _state_with_message(
            "Segment roads in /data/urban and add a final step for RF-DETR training from Roboflow."
        )

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            result = await understand(state)
            mock_get_llm.assert_not_called()

        understanding = result["metadata"]["understanding"]
        assert understanding["parameters"]["prompt"] == "roads"
        assert understanding["parameters"]["custom_step_description"] == "RF-DETR training from Roboflow"
        assert "segment" in understanding["operations"]
        assert "script" in understanding["operations"]

        plan = build_rule_based_plan(understanding)
        assert plan is not None
        tool_names = [step["tool_name"] for step in plan]
        assert "segment" in tool_names
        assert tool_names[-1] == "run_script"

        segment_step = next(step for step in plan if step["tool_name"] == "segment")
        script_step = next(step for step in plan if step["tool_name"] == "run_script")
        assert segment_step["parameters"]["prompt"] == "roads"
        assert script_step["description"] == "RF-DETR training from Roboflow"

    @pytest.mark.asyncio
    async def test_fast_path_segment_prompt_ignores_followup_custom_step(self) -> None:
        """Segmentation prompt extraction should stop before follow-up workflow clauses."""
        state = _state_with_message(
            "Segment roads and add a final step for RF-DETR training from Roboflow in /data/urban."
        )

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            result = await understand(state)
            mock_get_llm.assert_not_called()

        understanding = result["metadata"]["understanding"]
        assert understanding["parameters"]["prompt"] == "roads"
        assert understanding["parameters"]["custom_step_description"] == "RF-DETR training from Roboflow"

    @pytest.mark.asyncio
    async def test_fast_path_custom_step_ignores_trailing_instructions(self) -> None:
        """Custom final-step extraction should stop before trailing pipeline commands."""
        state = _state_with_message(
            "Detect objects in /data/site and add a final step for RF-DETR training from Roboflow and export yolo."
        )

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            result = await understand(state)
            mock_get_llm.assert_not_called()

        understanding = result["metadata"]["understanding"]
        assert understanding["parameters"]["custom_step_description"] == "RF-DETR training from Roboflow"

    @pytest.mark.asyncio
    async def test_fast_path_custom_step_detect_does_not_default_classes(self) -> None:
        """Custom-step detect flow should not inject person/car classes by default."""
        state = _state_with_message(
            "Detect objects in /data/site and add a final step for Roboflow RF-DETR training."
        )

        with patch("backend.agent.nodes.understand.get_llm") as mock_get_llm:
            result = await understand(state)
            mock_get_llm.assert_not_called()

        understanding = result["metadata"]["understanding"]
        plan = build_rule_based_plan(understanding)
        assert plan is not None

        detect_step = next(step for step in plan if step["tool_name"] == "detect")
        script_step = next(step for step in plan if step["tool_name"] == "run_script")
        assert "classes" not in detect_step["parameters"]
        assert script_step["description"] == "Roboflow RF-DETR training"


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

    @pytest.mark.asyncio
    async def test_rule_based_plan_skips_llm_for_multistep_task(self) -> None:
        """Deterministic planner should bypass LLM for common workflows."""
        state = PipelineState(
            messages=[
                {
                    "role": MessageRole.USER.value,
                    "content": "detect and export",
                    "timestamp": "2024-01-01T00:00:00",
                    "tool_calls": [],
                    "tool_call_id": None,
                }
            ],
            metadata={
                "understanding": {
                    "intent": "detect",
                    "input_path": "/tmp/images",
                    "input_type": "images",
                    "operations": ["detect", "export"],
                    "parameters": {"classes": ["car", "person"], "format": "yolo"},
                    "output_path": None,
                    "clarification_needed": None,
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

        with patch("backend.agent.nodes.plan.get_llm") as mock_get_llm:
            result = await generate_plan(state)
            mock_get_llm.assert_not_called()

        tool_names = [step["tool_name"] for step in result["plan"]]
        assert tool_names[0] == "scan_directory"
        assert "detect" in tool_names
        assert "export" in tool_names


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
            "label_qa",
            "split_dataset",
            "run_script",
            "review",
            "pointcloud_stats",
            "process_pointcloud",
            "detect_3d",
            "project_3d_to_2d",
            "anonymize_pointcloud",
            "extract_rosbag",
        }
        assert expected == VALID_TOOLS
