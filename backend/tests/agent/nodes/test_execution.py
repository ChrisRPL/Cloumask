"""
Tests for the execution nodes: execute_step and complete.

These tests verify:
- execute_step_node() executes tools and updates state correctly
- complete_node() generates summary and statistics
- Helper functions work as expected
- Error handling and retry logic
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.agent.nodes.complete import (
    calculate_final_stats,
    complete_node,
    generate_summary,
)
from backend.agent.nodes.execute import (
    StubTool,
    execute_step_node,
    format_step_result,
    is_retryable,
    register_stub_tools,
    update_progress,
)
from backend.agent.tools import ToolRegistry, get_tool_registry
from backend.agent.state import MessageRole, PipelineState, StepStatus

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Clear tool registry before each test."""
    get_tool_registry().clear()


@pytest.fixture
def base_state() -> PipelineState:
    """Create a base state for execution tests."""
    return PipelineState(
        messages=[
            {
                "role": MessageRole.USER.value,
                "content": "scan /data",
                "timestamp": "2026-01-01T00:00:00",
                "tool_calls": [],
                "tool_call_id": None,
            }
        ],
        plan=[
            {
                "id": "step-1",
                "tool_name": "scan_directory",
                "parameters": {"path": "/data"},
                "description": "Scan input directory",
                "status": StepStatus.PENDING.value,
                "result": None,
                "error": None,
                "started_at": None,
                "completed_at": None,
            }
        ],
        plan_approved=True,
        current_step=0,
        execution_results={},
        checkpoints=[],
        awaiting_user=False,
        metadata={},
        last_error=None,
        retry_count=0,
    )


@pytest.fixture
def multi_step_state() -> PipelineState:
    """Create a state with multiple steps for integration tests."""
    return PipelineState(
        messages=[],
        plan=[
            {
                "id": "step-1",
                "tool_name": "scan_directory",
                "parameters": {"path": "/data"},
                "description": "Scan input",
                "status": StepStatus.PENDING.value,
                "result": None,
                "error": None,
            },
            {
                "id": "step-2",
                "tool_name": "detect",
                "parameters": {"input_path": "/data", "classes": ["car", "person"]},
                "description": "Detect objects",
                "status": StepStatus.PENDING.value,
                "result": None,
                "error": None,
            },
            {
                "id": "step-3",
                "tool_name": "anonymize",
                "parameters": {"input_path": "/data", "output_path": "/out"},
                "description": "Anonymize faces",
                "status": StepStatus.PENDING.value,
                "result": None,
                "error": None,
            },
        ],
        plan_approved=True,
        current_step=0,
        execution_results={},
        checkpoints=[],
        awaiting_user=False,
        metadata={},
        last_error=None,
        retry_count=0,
    )


# -----------------------------------------------------------------------------
# format_step_result() tests
# -----------------------------------------------------------------------------


class TestFormatStepResult:
    """Tests for the format_step_result function."""

    def test_format_scan_result(self) -> None:
        """Scan result should show file count and formats."""
        step = {"tool_name": "scan_directory", "description": "Scan input"}
        result = {"total_files": 100, "formats": ["jpg", "png"]}

        output = format_step_result(step, result)
        assert "100 files" in output
        assert "jpg" in output
        assert "png" in output

    def test_format_detect_result(self) -> None:
        """Detect result should show object counts."""
        step = {"tool_name": "detect", "description": "Detect objects"}
        result = {"count": 50, "classes": {"car": 30, "person": 20}}

        output = format_step_result(step, result)
        assert "50 objects" in output
        assert "car: 30" in output
        assert "person: 20" in output

    def test_format_anonymize_result(self) -> None:
        """Anonymize result should show face/plate counts."""
        step = {"tool_name": "anonymize", "description": "Anonymize"}
        result = {"files_processed": 25, "faces_blurred": 45, "plates_blurred": 12}

        output = format_step_result(step, result)
        assert "25 files" in output
        assert "45 faces" in output
        assert "12 plates" in output

    def test_format_segment_result(self) -> None:
        """Segment result should show mask counts."""
        step = {"tool_name": "segment", "description": "Segment"}
        result = {"count": 30, "masks_generated": 30}

        output = format_step_result(step, result)
        assert "30 masks" in output
        assert "30 objects" in output

    def test_format_export_result(self) -> None:
        """Export result should show output path and format."""
        step = {"tool_name": "export", "description": "Export"}
        result = {"output_path": "/output/data", "format": "yolo"}

        output = format_step_result(step, result)
        assert "/output/data" in output
        assert "yolo" in output

    def test_format_with_timing(self) -> None:
        """Result with timing should show duration."""
        step = {
            "tool_name": "scan_directory",
            "description": "Scan",
            "started_at": "2026-01-01T10:00:00",
            "completed_at": "2026-01-01T10:00:30",
        }
        result = {"total_files": 10}

        output = format_step_result(step, result)
        assert "30.0s" in output

    def test_format_generic_result(self) -> None:
        """Unknown tool should show generic key-value output."""
        step = {"tool_name": "custom_tool", "description": "Custom"}
        result = {"custom_field": "value", "count": 5}

        output = format_step_result(step, result)
        assert "custom_field: value" in output
        assert "count: 5" in output


# -----------------------------------------------------------------------------
# is_retryable() tests
# -----------------------------------------------------------------------------


class TestIsRetryable:
    """Tests for the is_retryable function."""

    def test_timeout_is_retryable(self) -> None:
        """Timeout errors should be retryable."""
        assert is_retryable(TimeoutError("timed out")) is True

    def test_connection_error_is_retryable(self) -> None:
        """Connection errors should be retryable."""
        assert is_retryable(ConnectionError("failed")) is True

    def test_io_error_is_retryable(self) -> None:
        """IO errors should be retryable."""
        assert is_retryable(OSError("disk busy")) is True

    def test_value_error_not_retryable(self) -> None:
        """Value errors should not be retryable."""
        assert is_retryable(ValueError("invalid")) is False

    def test_key_error_not_retryable(self) -> None:
        """Key errors should not be retryable."""
        assert is_retryable(KeyError("missing")) is False

    def test_temporary_message_is_retryable(self) -> None:
        """Error with 'temporary' in message should be retryable."""
        assert is_retryable(RuntimeError("temporary failure")) is True

    def test_busy_message_is_retryable(self) -> None:
        """Error with 'busy' in message should be retryable."""
        assert is_retryable(RuntimeError("resource busy")) is True


# -----------------------------------------------------------------------------
# update_progress() tests
# -----------------------------------------------------------------------------


class TestUpdateProgress:
    """Tests for the update_progress function."""

    def test_updates_progress_percent(self) -> None:
        """Should calculate correct progress percentage."""
        state: PipelineState = {"metadata": {}}
        update_progress(state, 0, 4, {})
        assert state["metadata"]["progress_percent"] == 25.0

        update_progress(state, 1, 4, {})
        assert state["metadata"]["progress_percent"] == 50.0

    def test_tracks_files_processed(self) -> None:
        """Should accumulate files_processed."""
        state: PipelineState = {"metadata": {}}
        update_progress(state, 0, 2, {"files_processed": 10})
        assert state["metadata"]["processed_files"] == 10

        update_progress(state, 1, 2, {"files_processed": 15})
        assert state["metadata"]["processed_files"] == 25

    def test_tracks_items_detected(self) -> None:
        """Should accumulate count as total_items."""
        state: PipelineState = {"metadata": {}}
        update_progress(state, 0, 2, {"count": 50})
        assert state["metadata"]["total_items"] == 50

        update_progress(state, 1, 2, {"count": 30})
        assert state["metadata"]["total_items"] == 80

    def test_handles_empty_result(self) -> None:
        """Should handle result with no special fields."""
        state: PipelineState = {"metadata": {}}
        update_progress(state, 0, 1, {})
        assert "progress_percent" in state["metadata"]


# -----------------------------------------------------------------------------
# calculate_final_stats() tests
# -----------------------------------------------------------------------------


class TestCalculateFinalStats:
    """Tests for the calculate_final_stats function."""

    def test_counts_step_statuses(self) -> None:
        """Should correctly count step statuses."""
        plan = [
            {"status": "completed"},
            {"status": "completed"},
            {"status": "failed"},
            {"status": "skipped"},
        ]
        results: dict[str, Any] = {}

        stats = calculate_final_stats(plan, results)

        assert stats["total_steps"] == 4
        assert stats["completed_steps"] == 2
        assert stats["failed_steps"] == 1
        assert stats["skipped_steps"] == 1
        assert stats["success_rate"] == 0.5

    def test_calculates_duration(self) -> None:
        """Should sum durations from step timestamps."""
        plan = [
            {
                "status": "completed",
                "started_at": "2026-01-01T10:00:00",
                "completed_at": "2026-01-01T10:00:30",
            },
            {
                "status": "completed",
                "started_at": "2026-01-01T10:00:30",
                "completed_at": "2026-01-01T10:01:00",
            },
        ]
        results: dict[str, Any] = {}

        stats = calculate_final_stats(plan, results)
        assert stats["total_duration_seconds"] == 60.0

    def test_aggregates_file_counts(self) -> None:
        """Should aggregate files_processed from results."""
        plan = [{"status": "completed"}, {"status": "completed"}]
        results = {
            "step-1": {"files_processed": 50},
            "step-2": {"files_processed": 50, "count": 100},
        }

        stats = calculate_final_stats(plan, results)
        assert stats["files_processed"] == 100
        assert stats["items_detected"] == 100

    def test_aggregates_anonymization_counts(self) -> None:
        """Should sum faces and plates blurred."""
        plan = [{"status": "completed"}]
        results = {"step-1": {"faces_blurred": 30, "plates_blurred": 10}}

        stats = calculate_final_stats(plan, results)
        assert stats["items_anonymized"] == 40

    def test_calculates_average_confidence(self) -> None:
        """Should average confidence scores."""
        plan = [{"status": "completed"}, {"status": "completed"}]
        results = {
            "step-1": {"confidence": 0.8},
            "step-2": {"confidence": 0.9},
        }

        stats = calculate_final_stats(plan, results)
        assert stats["average_confidence"] == pytest.approx(0.85)

    def test_handles_empty_plan(self) -> None:
        """Should handle empty plan gracefully."""
        stats = calculate_final_stats([], {})
        assert stats["total_steps"] == 0
        assert stats["success_rate"] == 0

    def test_ignores_error_results(self) -> None:
        """Should not count results that contain errors."""
        plan = [{"status": "failed"}]
        results = {"step-1": {"error": "Something went wrong"}}

        stats = calculate_final_stats(plan, results)
        assert stats["files_processed"] == 0


# -----------------------------------------------------------------------------
# generate_summary() tests
# -----------------------------------------------------------------------------


class TestGenerateSummary:
    """Tests for the generate_summary function."""

    def test_success_summary(self) -> None:
        """Should indicate all steps completed."""
        plan = [{"status": "completed"}]
        results: dict[str, Any] = {}
        stats = {
            "total_steps": 1,
            "completed_steps": 1,
            "failed_steps": 0,
            "skipped_steps": 0,
            "success_rate": 1.0,
            "total_duration_seconds": 10.0,
            "files_processed": 100,
            "items_detected": 50,
            "items_anonymized": 0,
            "masks_generated": 0,
            "average_confidence": 0.9,
        }

        summary = generate_summary(plan, results, stats)

        assert "## Pipeline Complete" in summary
        assert "All steps completed successfully" in summary
        assert "1/1 completed" in summary
        assert "10.0s" in summary
        assert "100" in summary
        assert "90" in summary  # 90% confidence

    def test_failure_summary(self) -> None:
        """Should list failed steps."""
        plan = [
            {"status": "completed", "tool_name": "scan"},
            {"status": "failed", "tool_name": "detect", "description": "Detect objects", "error": "GPU OOM"},
        ]
        results: dict[str, Any] = {}
        stats = {
            "total_steps": 2,
            "completed_steps": 1,
            "failed_steps": 1,
            "skipped_steps": 0,
            "success_rate": 0.5,
            "total_duration_seconds": 5.0,
            "files_processed": 0,
            "items_detected": 0,
            "items_anonymized": 0,
            "masks_generated": 0,
            "average_confidence": None,
        }

        summary = generate_summary(plan, results, stats)

        assert "1 failed step" in summary
        assert "### Failed Steps" in summary
        assert "Detect objects" in summary
        assert "GPU OOM" in summary

    def test_export_output_path(self) -> None:
        """Should show output paths from export results."""
        plan = [{"status": "completed"}]
        results = {"step-1": {"output_path": "/export/data.zip"}}
        stats = {
            "total_steps": 1,
            "completed_steps": 1,
            "failed_steps": 0,
            "skipped_steps": 0,
            "success_rate": 1.0,
            "total_duration_seconds": 0,
            "files_processed": 0,
            "items_detected": 0,
            "items_anonymized": 0,
            "masks_generated": 0,
            "average_confidence": None,
        }

        summary = generate_summary(plan, results, stats)

        assert "### Output" in summary
        assert "/export/data.zip" in summary


# -----------------------------------------------------------------------------
# execute_step_node() tests
# -----------------------------------------------------------------------------


class TestExecuteStepNode:
    """Tests for the execute_step_node function."""

    @pytest.mark.asyncio
    async def test_execute_success(self, base_state: PipelineState) -> None:
        """Should execute step and update state correctly."""
        register_stub_tools()

        result = await execute_step_node(base_state)

        assert result["current_step"] == 1
        assert "step-1" in result["execution_results"]
        assert result["plan"][0]["status"] == StepStatus.COMPLETED.value
        assert result["plan"][0]["result"] is not None
        assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, base_state: PipelineState) -> None:
        """Should fail gracefully for unknown tool."""
        base_state["plan"][0]["tool_name"] = "unknown_tool"

        result = await execute_step_node(base_state)

        assert result["current_step"] == 1
        assert result["plan"][0]["status"] == StepStatus.FAILED.value
        assert "Unknown tool" in result["plan"][0]["error"]
        assert "error" in result["execution_results"]["step-1"]

    @pytest.mark.asyncio
    async def test_execute_past_end(self, base_state: PipelineState) -> None:
        """Should handle executing past end of plan."""
        base_state["current_step"] = 10

        result = await execute_step_node(base_state)

        assert "last_error" in result
        assert "No more steps" in result["last_error"]

    @pytest.mark.asyncio
    async def test_execute_with_custom_tool(self, base_state: PipelineState) -> None:
        """Should work with custom tool implementations."""
        custom_result = {"custom": "data", "count": 42}

        # Register a stub tool that returns custom data
        tool = StubTool("scan_directory", lambda **kw: custom_result)
        get_tool_registry().register(tool)

        result = await execute_step_node(base_state)

        assert result["execution_results"]["step-1"] == custom_result

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, base_state: PipelineState) -> None:
        """Should retry on transient errors."""
        from backend.agent.tools import BaseTool, ToolCategory, ToolResult, error_result

        call_count = 0

        class FailingTool(BaseTool):
            """Tool that fails with timeout on first call."""

            name = "scan_directory"
            description = "Failing scan tool"
            category = ToolCategory.SCAN
            parameters = []

            async def execute(self, **kwargs: Any) -> ToolResult:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    # Return error result with retryable message
                    raise TimeoutError("Connection timeout")
                return ToolResult(success=True, data={"success": True})

        get_tool_registry().register(FailingTool())

        # First call should trigger retry
        result = await execute_step_node(base_state)

        assert result["retry_count"] == 1
        assert result["plan"][0]["status"] == StepStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_no_retry_on_value_error(self, base_state: PipelineState) -> None:
        """Should not retry on non-transient errors."""
        from backend.agent.tools import BaseTool, ToolCategory, ToolResult

        class NonRetryableFailingTool(BaseTool):
            """Tool that fails with non-retryable error."""

            name = "scan_directory"
            description = "Non-retryable failing tool"
            category = ToolCategory.SCAN
            parameters = []

            async def execute(self, **kwargs: Any) -> ToolResult:
                raise ValueError("Invalid parameter")

        get_tool_registry().register(NonRetryableFailingTool())

        result = await execute_step_node(base_state)

        # Should move to next step, not retry
        assert result["current_step"] == 1
        assert result["retry_count"] == 0
        assert result["plan"][0]["status"] == StepStatus.FAILED.value


# -----------------------------------------------------------------------------
# complete_node() tests
# -----------------------------------------------------------------------------


class TestCompleteNode:
    """Tests for the complete_node function."""

    @pytest.mark.asyncio
    async def test_complete_generates_summary(self) -> None:
        """Should generate summary message."""
        state: PipelineState = {
            "plan": [
                {
                    "id": "step-1",
                    "tool_name": "scan_directory",
                    "status": "completed",
                    "started_at": "2026-01-01T10:00:00",
                    "completed_at": "2026-01-01T10:00:10",
                }
            ],
            "current_step": 1,
            "execution_results": {"step-1": {"total_files": 50}},
            "messages": [],
            "metadata": {},
            "checkpoints": [],
            "plan_approved": True,
            "awaiting_user": False,
            "last_error": None,
            "retry_count": 0,
        }

        result = await complete_node(state)

        assert "final_stats" in result["metadata"]
        assert "completed_at" in result["metadata"]
        assert len(result["messages"]) > 0
        assert "complete" in result["messages"][-1]["content"].lower()

    @pytest.mark.asyncio
    async def test_complete_sets_awaiting_user_false(self) -> None:
        """Should set awaiting_user to False."""
        state: PipelineState = {
            "plan": [],
            "current_step": 0,
            "execution_results": {},
            "messages": [],
            "metadata": {},
            "checkpoints": [],
            "plan_approved": True,
            "awaiting_user": True,
            "last_error": None,
            "retry_count": 0,
        }

        result = await complete_node(state)

        assert result["awaiting_user"] is False

    @pytest.mark.asyncio
    async def test_complete_includes_stats(self) -> None:
        """Should include final statistics in metadata."""
        state: PipelineState = {
            "plan": [
                {"status": "completed"},
                {"status": "failed"},
            ],
            "current_step": 2,
            "execution_results": {
                "step-1": {"files_processed": 100, "count": 50},
            },
            "messages": [],
            "metadata": {},
            "checkpoints": [],
            "plan_approved": True,
            "awaiting_user": False,
            "last_error": None,
            "retry_count": 0,
        }

        result = await complete_node(state)

        stats = result["metadata"]["final_stats"]
        assert stats["total_steps"] == 2
        assert stats["completed_steps"] == 1
        assert stats["failed_steps"] == 1
        assert stats["files_processed"] == 100


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------


class TestExecutionIntegration:
    """Integration tests for full execution flow."""

    @pytest.mark.asyncio
    async def test_full_execution_flow(self, multi_step_state: PipelineState) -> None:
        """Test executing multiple steps in sequence."""
        register_stub_tools()

        state = multi_step_state

        # Execute all steps
        for _i in range(3):
            result = await execute_step_node(state)
            # Update state with result
            state.update(result)  # type: ignore[typeddict-item]

        assert state["current_step"] == 3
        assert len(state["execution_results"]) == 3
        assert all(s["status"] == "completed" for s in state["plan"])

    @pytest.mark.asyncio
    async def test_execute_then_complete(self, base_state: PipelineState) -> None:
        """Test executing steps then completing."""
        register_stub_tools()

        # Execute
        exec_result = await execute_step_node(base_state)
        base_state.update(exec_result)  # type: ignore[typeddict-item]

        # Complete
        complete_result = await complete_node(base_state)
        base_state.update(complete_result)  # type: ignore[typeddict-item]

        assert "final_stats" in base_state["metadata"]
        assert base_state["metadata"]["final_stats"]["completed_steps"] == 1
        assert "## Pipeline Complete" in base_state["messages"][-1]["content"]


# -----------------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for execution nodes."""

    @pytest.mark.asyncio
    async def test_execute_empty_plan(self) -> None:
        """Should handle empty plan."""
        state: PipelineState = {
            "plan": [],
            "current_step": 0,
            "execution_results": {},
            "messages": [],
            "metadata": {},
            "checkpoints": [],
            "plan_approved": True,
            "awaiting_user": False,
            "last_error": None,
            "retry_count": 0,
        }

        result = await execute_step_node(state)

        assert "last_error" in result
        assert "No more steps" in result["last_error"]

    @pytest.mark.asyncio
    async def test_complete_all_failed(self) -> None:
        """Should generate appropriate summary for all failures."""
        state: PipelineState = {
            "plan": [
                {"status": "failed", "tool_name": "scan", "description": "Scan", "error": "Disk error"},
                {"status": "failed", "tool_name": "detect", "description": "Detect", "error": "GPU error"},
            ],
            "current_step": 2,
            "execution_results": {
                "step-1": {"error": "Disk error"},
                "step-2": {"error": "GPU error"},
            },
            "messages": [],
            "metadata": {},
            "checkpoints": [],
            "plan_approved": True,
            "awaiting_user": False,
            "last_error": None,
            "retry_count": 0,
        }

        result = await complete_node(state)

        assert result["metadata"]["final_stats"]["failed_steps"] == 2
        assert result["metadata"]["final_stats"]["success_rate"] == 0
        summary = result["messages"][-1]["content"]
        assert "2 failed" in summary
        assert "Disk error" in summary
        assert "GPU error" in summary

    def test_format_step_result_missing_fields(self) -> None:
        """Should handle step with missing optional fields."""
        step = {"tool_name": "scan_directory"}
        result = {"total_files": 10}

        output = format_step_result(step, result)

        assert "scan_directory" in output
        assert "10 files" in output

    def test_calculate_stats_invalid_timestamps(self) -> None:
        """Should handle invalid timestamps gracefully."""
        plan = [
            {
                "status": "completed",
                "started_at": "invalid",
                "completed_at": "also invalid",
            }
        ]

        stats = calculate_final_stats(plan, {})

        # Should not crash, duration should be 0
        assert stats["total_duration_seconds"] == 0.0
