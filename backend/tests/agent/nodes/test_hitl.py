"""
Tests for the Human-in-the-Loop nodes: await_approval and checkpoint.

These tests verify:
- await_approval_node() correctly pauses execution and sets state
- checkpoint_node() evaluates quality and creates checkpoints
- handle_user_response() processes user decisions correctly
- apply_plan_edits() modifies plans as expected
- determine_trigger() detects appropriate checkpoint conditions
- calculate_quality_metrics() computes accurate metrics
"""

from __future__ import annotations

import pytest

from backend.agent.nodes.approval import (
    apply_plan_edits,
    await_approval_node,
    handle_user_response,
)
from backend.agent.nodes.checkpoint import (
    CONFIDENCE_DROP_THRESHOLD,
    CRITICAL_TOOLS,
    ERROR_RATE_THRESHOLD,
    PERCENTAGE_THRESHOLDS,
    calculate_quality_metrics,
    checkpoint_node,
    determine_trigger,
    format_checkpoint_message,
)
from backend.agent.state import (
    CheckpointTrigger,
    MessageRole,
    PipelineState,
    QualityMetrics,
    StepStatus,
    UserDecision,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def base_state() -> PipelineState:
    """Create a base state for HITL tests."""
    return PipelineState(
        messages=[
            {
                "role": MessageRole.USER.value,
                "content": "process /data",
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
            },
            {
                "id": "step-2",
                "tool_name": "detect",
                "parameters": {"input_path": "/data", "classes": ["car"]},
                "description": "Detect cars",
                "status": StepStatus.PENDING.value,
                "result": None,
                "error": None,
            },
        ],
        plan_approved=False,
        current_step=0,
        execution_results={},
        checkpoints=[],
        awaiting_user=False,
        metadata={},
        last_error=None,
        retry_count=0,
    )


@pytest.fixture
def executing_state() -> PipelineState:
    """State during execution (plan approved, some steps done)."""
    return PipelineState(
        messages=[],
        plan=[
            {
                "id": "step-1",
                "tool_name": "scan_directory",
                "parameters": {"path": "/data"},
                "description": "Scan input",
                "status": StepStatus.COMPLETED.value,
                "result": {"total_files": 100},
                "started_at": "2026-01-01T10:00:00",
                "completed_at": "2026-01-01T10:00:10",
            },
            {
                "id": "step-2",
                "tool_name": "detect",
                "parameters": {},
                "description": "Detect objects",
                "status": StepStatus.COMPLETED.value,
                "result": {"count": 50, "confidence": 0.9},
                "started_at": "2026-01-01T10:00:10",
                "completed_at": "2026-01-01T10:00:30",
            },
            {
                "id": "step-3",
                "tool_name": "anonymize",
                "parameters": {},
                "description": "Anonymize faces",
                "status": StepStatus.PENDING.value,
            },
            {
                "id": "step-4",
                "tool_name": "export",
                "parameters": {},
                "description": "Export results",
                "status": StepStatus.PENDING.value,
            },
        ],
        plan_approved=True,
        current_step=2,
        execution_results={
            "step-1": {"total_files": 100, "files_processed": 100},
            "step-2": {"count": 50, "confidence": 0.9},
        },
        checkpoints=[],
        awaiting_user=False,
        metadata={},
        last_error=None,
        retry_count=0,
    )


# -----------------------------------------------------------------------------
# await_approval_node() tests
# -----------------------------------------------------------------------------


class TestAwaitApprovalNode:
    """Tests for the await_approval_node function."""

    def test_awaits_plan_approval(self, base_state: PipelineState) -> None:
        """Should set awaiting_user and add AWAIT_PLAN_APPROVAL message."""
        result = await_approval_node(base_state)

        assert result["awaiting_user"] is True
        assert len(result["messages"]) > 0
        assert "AWAIT_PLAN_APPROVAL" in result["messages"][-1]["content"]

    def test_awaits_checkpoint_approval(self, executing_state: PipelineState) -> None:
        """Should await checkpoint approval when plan is approved."""
        # Add an unresolved checkpoint
        executing_state["checkpoints"] = [
            {
                "id": "ckpt-123",
                "step_index": 2,
                "trigger_reason": CheckpointTrigger.PERCENTAGE.value,
                "created_at": "2026-01-01T10:00:30",
                "resolved_at": None,
            }
        ]

        result = await_approval_node(executing_state)

        assert result["awaiting_user"] is True
        assert "AWAIT_CHECKPOINT:ckpt-123" in result["messages"][-1]["content"]

    def test_no_message_for_resolved_checkpoint(
        self, executing_state: PipelineState
    ) -> None:
        """Should not add checkpoint message if already resolved."""
        executing_state["checkpoints"] = [
            {
                "id": "ckpt-123",
                "step_index": 2,
                "resolved_at": "2026-01-01T10:00:35",
                "user_decision": UserDecision.APPROVE.value,
            }
        ]

        result = await_approval_node(executing_state)

        # Should just set awaiting_user, no new checkpoint message
        assert result["awaiting_user"] is True


# -----------------------------------------------------------------------------
# handle_user_response() tests
# -----------------------------------------------------------------------------


class TestHandleUserResponse:
    """Tests for the handle_user_response function."""

    def test_approve_sets_plan_approved(self, base_state: PipelineState) -> None:
        """Approve should set plan_approved and clear awaiting_user."""
        base_state["awaiting_user"] = True

        result = handle_user_response(base_state, {"decision": "approve"})

        assert result["awaiting_user"] is False
        assert result["plan_approved"] is True

    def test_approve_resolves_checkpoint(self, executing_state: PipelineState) -> None:
        """Approve should resolve pending checkpoint."""
        executing_state["checkpoints"] = [
            {
                "id": "ckpt-123",
                "resolved_at": None,
                "user_decision": None,
                "user_feedback": None,
            }
        ]
        executing_state["awaiting_user"] = True

        result = handle_user_response(
            executing_state,
            {"decision": "approve", "message": "Looks good"},
        )

        checkpoint = result["checkpoints"][0]
        assert checkpoint["user_decision"] == UserDecision.APPROVE.value
        assert checkpoint["user_feedback"] == "Looks good"
        assert checkpoint["resolved_at"] is not None

    def test_cancel_skips_remaining_steps(self, executing_state: PipelineState) -> None:
        """Cancel should skip remaining steps."""
        executing_state["awaiting_user"] = True

        result = handle_user_response(executing_state, {"decision": "cancel"})

        assert result["awaiting_user"] is False
        # Steps 3 and 4 should be skipped (indices 2 and 3, current_step=2)
        assert result["plan"][2]["status"] == StepStatus.SKIPPED.value
        assert result["plan"][3]["status"] == StepStatus.SKIPPED.value
        # Earlier completed steps unchanged
        assert result["plan"][0]["status"] == StepStatus.COMPLETED.value

    def test_edit_with_edits_applies_changes(
        self, base_state: PipelineState
    ) -> None:
        """Edit with plan_edits should apply modifications."""
        base_state["awaiting_user"] = True

        edits = [
            {"action": "modify", "step_index": 0, "changes": {"parameters": {"path": "/new/path"}}}
        ]
        result = handle_user_response(
            base_state,
            {"decision": "edit", "plan_edits": edits},
        )

        assert result["awaiting_user"] is False
        assert result["plan"][0]["parameters"]["path"] == "/new/path"

    def test_edit_without_edits_awaits_more_input(
        self, base_state: PipelineState
    ) -> None:
        """Edit without plan_edits should wait for details."""
        base_state["awaiting_user"] = True

        result = handle_user_response(base_state, {"decision": "edit"})

        assert result["awaiting_user"] is True
        assert "What changes" in result["messages"][-1]["content"]

    def test_retry_resets_retry_count(self, executing_state: PipelineState) -> None:
        """Retry should reset retry_count."""
        executing_state["awaiting_user"] = True
        executing_state["retry_count"] = 3

        result = handle_user_response(executing_state, {"decision": "retry"})

        assert result["awaiting_user"] is False
        assert result["retry_count"] == 0

    def test_user_message_added_to_conversation(
        self, base_state: PipelineState
    ) -> None:
        """User's message should be added to conversation."""
        base_state["awaiting_user"] = True

        result = handle_user_response(
            base_state,
            {"decision": "approve", "message": "Confirmed, proceed"},
        )

        user_messages = [m for m in result["messages"] if m["role"] == "user"]
        assert any("Confirmed, proceed" in m["content"] for m in user_messages)

    def test_invalid_decision_defaults_to_approve(
        self, base_state: PipelineState
    ) -> None:
        """Invalid decision should default to approve."""
        base_state["awaiting_user"] = True

        result = handle_user_response(base_state, {"decision": "invalid_decision"})

        assert result["awaiting_user"] is False
        assert result["plan_approved"] is True


# -----------------------------------------------------------------------------
# apply_plan_edits() tests
# -----------------------------------------------------------------------------


class TestApplyPlanEdits:
    """Tests for the apply_plan_edits function."""

    def test_modify_step_parameters(self) -> None:
        """Modify edit should update step parameters."""
        plan = [
            {"tool_name": "detect", "parameters": {"confidence": 0.5}},
        ]
        edits = [
            {
                "action": "modify",
                "step_index": 0,
                "changes": {"parameters": {"confidence": 0.8}},
            }
        ]

        result = apply_plan_edits(plan, edits)

        assert result[0]["parameters"]["confidence"] == 0.8

    def test_modify_merges_parameters(self) -> None:
        """Modify should merge parameters, not replace entirely."""
        plan = [
            {
                "tool_name": "detect",
                "parameters": {"confidence": 0.5, "classes": ["car"]},
            },
        ]
        edits = [
            {
                "action": "modify",
                "step_index": 0,
                "changes": {"parameters": {"confidence": 0.8}},
            }
        ]

        result = apply_plan_edits(plan, edits)

        assert result[0]["parameters"]["confidence"] == 0.8
        assert result[0]["parameters"]["classes"] == ["car"]

    def test_remove_step(self) -> None:
        """Remove edit should delete step."""
        plan = [
            {"tool_name": "a"},
            {"tool_name": "b"},
            {"tool_name": "c"},
        ]
        edits = [{"action": "remove", "step_index": 1}]

        result = apply_plan_edits(plan, edits)

        assert len(result) == 2
        assert result[0]["tool_name"] == "a"
        assert result[1]["tool_name"] == "c"

    def test_add_step(self) -> None:
        """Add edit should insert new step."""
        plan = [{"tool_name": "a"}, {"tool_name": "c"}]
        edits = [
            {
                "action": "add",
                "after_index": 0,
                "step": {"tool_name": "b", "description": "New step"},
            }
        ]

        result = apply_plan_edits(plan, edits)

        assert len(result) == 3
        assert result[1]["tool_name"] == "b"
        assert result[1]["status"] == StepStatus.PENDING.value

    def test_add_step_generates_id(self) -> None:
        """Add should generate id if not provided."""
        plan = [{"tool_name": "a"}]
        edits = [{"action": "add", "after_index": 0, "step": {"tool_name": "b"}}]

        result = apply_plan_edits(plan, edits)

        assert "id" in result[1]
        assert result[1]["id"].startswith("step-")

    def test_reorder_step(self) -> None:
        """Reorder edit should move step."""
        plan = [
            {"tool_name": "a"},
            {"tool_name": "b"},
            {"tool_name": "c"},
        ]
        edits = [{"action": "reorder", "from_index": 2, "to_index": 0}]

        result = apply_plan_edits(plan, edits)

        assert result[0]["tool_name"] == "c"
        assert result[1]["tool_name"] == "a"
        assert result[2]["tool_name"] == "b"

    def test_invalid_indices_ignored(self) -> None:
        """Invalid indices should be ignored without error."""
        plan = [{"tool_name": "a"}]
        edits = [
            {"action": "modify", "step_index": 999, "changes": {}},
            {"action": "remove", "step_index": -1},
            {"action": "reorder", "from_index": 0, "to_index": 100},
        ]

        result = apply_plan_edits(plan, edits)

        # Plan should be unchanged
        assert len(result) == 1
        assert result[0]["tool_name"] == "a"

    def test_multiple_edits_applied_in_order(self) -> None:
        """Multiple edits should be applied sequentially."""
        plan = [
            {"tool_name": "a"},
            {"tool_name": "b"},
        ]
        edits = [
            {"action": "add", "after_index": 1, "step": {"tool_name": "c"}},
            {"action": "remove", "step_index": 0},
        ]

        result = apply_plan_edits(plan, edits)

        # After add: [a, b, c]
        # After remove index 0: [b, c]
        assert len(result) == 2
        assert result[0]["tool_name"] == "b"
        assert result[1]["tool_name"] == "c"


# -----------------------------------------------------------------------------
# determine_trigger() tests
# -----------------------------------------------------------------------------


class TestDetermineTrigger:
    """Tests for the determine_trigger function."""

    def test_no_trigger_on_first_step(self) -> None:
        """Should not trigger on first step."""
        state: PipelineState = {
            "plan": [{"tool_name": "step_1"}],
            "current_step": 0,
            "execution_results": {},
            "checkpoints": [],
        }

        trigger = determine_trigger(state)

        assert trigger is None

    def test_trigger_percentage_at_25_percent(self) -> None:
        """Should trigger at 25% progress."""
        state: PipelineState = {
            "plan": [{"tool_name": f"step_{i}"} for i in range(4)],
            "current_step": 1,  # 25%
            "execution_results": {},
            "checkpoints": [],
        }

        trigger = determine_trigger(state)

        assert trigger == CheckpointTrigger.PERCENTAGE

    def test_trigger_percentage_at_50_percent(self) -> None:
        """Should trigger at 50% progress."""
        state: PipelineState = {
            "plan": [{"tool_name": f"step_{i}"} for i in range(4)],
            "current_step": 2,  # 50%
            "execution_results": {},
            "checkpoints": [],
        }

        trigger = determine_trigger(state)

        assert trigger == CheckpointTrigger.PERCENTAGE

    def test_no_duplicate_checkpoint_at_same_step(self) -> None:
        """Should not trigger if already checkpointed at this step."""
        state: PipelineState = {
            "plan": [{"tool_name": f"step_{i}"} for i in range(4)],
            "current_step": 1,
            "execution_results": {},
            "checkpoints": [{"step_index": 1}],  # Already checkpointed
        }

        trigger = determine_trigger(state)

        assert trigger is None

    def test_trigger_critical_step_anonymize(self) -> None:
        """Should trigger after anonymize tool."""
        state: PipelineState = {
            "plan": [
                {"tool_name": "scan_directory"},
                {"tool_name": "anonymize"},  # Critical
            ],
            "current_step": 2,  # Just completed anonymize
            "execution_results": {},
            "checkpoints": [],
        }

        trigger = determine_trigger(state)

        assert trigger == CheckpointTrigger.CRITICAL_STEP

    def test_trigger_critical_step_segment(self) -> None:
        """Should trigger after segment tool."""
        state: PipelineState = {
            "plan": [
                {"tool_name": "scan_directory"},
                {"tool_name": "segment"},  # Critical
            ],
            "current_step": 2,
            "execution_results": {},
            "checkpoints": [],
        }

        trigger = determine_trigger(state)

        assert trigger == CheckpointTrigger.CRITICAL_STEP

    def test_trigger_critical_step_detect_3d(self) -> None:
        """Should trigger after detect_3d tool."""
        state: PipelineState = {
            "plan": [
                {"tool_name": "scan_directory"},
                {"tool_name": "detect_3d"},  # Critical
            ],
            "current_step": 2,
            "execution_results": {},
            "checkpoints": [],
        }

        trigger = determine_trigger(state)

        assert trigger == CheckpointTrigger.CRITICAL_STEP

    def test_trigger_explicitly_marked_critical(self) -> None:
        """Should trigger for steps marked as critical."""
        state: PipelineState = {
            "plan": [
                {"tool_name": "custom_tool", "critical": True},
            ],
            "current_step": 1,
            "execution_results": {},
            "checkpoints": [],
        }

        trigger = determine_trigger(state)

        assert trigger == CheckpointTrigger.CRITICAL_STEP

    def test_trigger_quality_drop(self) -> None:
        """Should trigger when confidence drops >15%."""
        # Use 20 steps and current_step=8 to avoid percentage thresholds
        # (10%=2, 25%=5, 50%=10 - 8 is between 25% and 50%)
        # Quality drop: overall_avg - recent_avg must be > 0.15
        # With these values:
        # overall = (0.95*3 + 0.50*5) / 8 = (2.85 + 2.50) / 8 = 0.669
        # recent = (0.50*5) / 5 = 0.50
        # drop = 0.669 - 0.50 = 0.169 > 0.15
        state: PipelineState = {
            "plan": [{"tool_name": f"step_{i}"} for i in range(20)],
            "current_step": 8,  # 40% - between 25% and 50%
            "execution_results": {
                "step-0": {"confidence": 0.95},
                "step-1": {"confidence": 0.95},
                "step-2": {"confidence": 0.95},
                "step-3": {"confidence": 0.50},  # Recent - drop
                "step-4": {"confidence": 0.50},  # Recent
                "step-5": {"confidence": 0.50},  # Recent
                "step-6": {"confidence": 0.50},  # Recent
                "step-7": {"confidence": 0.50},  # Recent
            },
            "checkpoints": [{"step_index": 5}],  # Already checkpointed at 25%
        }

        trigger = determine_trigger(state)

        assert trigger == CheckpointTrigger.QUALITY_DROP

    def test_trigger_error_rate(self) -> None:
        """Should trigger when error rate >5%."""
        state: PipelineState = {
            "plan": [{"tool_name": f"step_{i}"} for i in range(10)],
            "current_step": 5,
            "execution_results": {
                "step-0": {"success": True},
                "step-1": {"error": "Failed"},  # Error
                "step-2": {"success": True},
                "step-3": {"success": True},
                "step-4": {"success": True},
            },
            "checkpoints": [],
        }

        trigger = determine_trigger(state)

        # 1 error / 5 = 20% > 5%
        assert trigger == CheckpointTrigger.ERROR_RATE


# -----------------------------------------------------------------------------
# calculate_quality_metrics() tests
# -----------------------------------------------------------------------------


class TestCalculateQualityMetrics:
    """Tests for the calculate_quality_metrics function."""

    def test_calculates_average_confidence(self) -> None:
        """Should calculate average confidence score."""
        state: PipelineState = {
            "plan": [],
            "execution_results": {
                "step-1": {"confidence": 0.8},
                "step-2": {"confidence": 0.9},
            },
        }

        metrics = calculate_quality_metrics(state)

        assert metrics.average_confidence == pytest.approx(0.85)

    def test_counts_errors(self) -> None:
        """Should count errors in results."""
        state: PipelineState = {
            "plan": [],
            "execution_results": {
                "step-1": {"error": "Failed"},
                "step-2": {"success": True},
                "step-3": {"error": "Also failed"},
            },
        }

        metrics = calculate_quality_metrics(state)

        assert metrics.error_count == 2

    def test_sums_files_processed(self) -> None:
        """Should sum files_processed from all results."""
        state: PipelineState = {
            "plan": [],
            "execution_results": {
                "step-1": {"files_processed": 50},
                "step-2": {"files_processed": 30},
            },
        }

        metrics = calculate_quality_metrics(state)

        assert metrics.total_processed == 80

    def test_sums_items_processed(self) -> None:
        """Should also count items_processed."""
        state: PipelineState = {
            "plan": [],
            "execution_results": {
                "step-1": {"items_processed": 100},
            },
        }

        metrics = calculate_quality_metrics(state)

        assert metrics.total_processed == 100

    def test_calculates_processing_speed(self) -> None:
        """Should calculate items per second."""
        state: PipelineState = {
            "plan": [
                {
                    "started_at": "2026-01-01T10:00:00",
                    "completed_at": "2026-01-01T10:00:10",  # 10 seconds
                }
            ],
            "execution_results": {
                "step-1": {"files_processed": 100},
            },
        }

        metrics = calculate_quality_metrics(state)

        assert metrics.processing_speed == pytest.approx(10.0)  # 100 / 10 = 10

    def test_handles_empty_results(self) -> None:
        """Should handle empty execution results."""
        state: PipelineState = {
            "plan": [],
            "execution_results": {},
        }

        metrics = calculate_quality_metrics(state)

        assert metrics.average_confidence == 0.0
        assert metrics.error_count == 0
        assert metrics.total_processed == 0
        assert metrics.processing_speed == 0.0

    def test_error_rate_property(self) -> None:
        """Should calculate error rate correctly."""
        state: PipelineState = {
            "plan": [],
            "execution_results": {
                "step-1": {"error": "Failed", "files_processed": 1},
                "step-2": {"success": True, "files_processed": 1},
                "step-3": {"success": True, "files_processed": 1},
                "step-4": {"success": True, "files_processed": 1},
            },
        }

        metrics = calculate_quality_metrics(state)

        # 1 error, 4 total processed
        assert metrics.error_rate == pytest.approx(0.25)


# -----------------------------------------------------------------------------
# format_checkpoint_message() tests
# -----------------------------------------------------------------------------


class TestFormatCheckpointMessage:
    """Tests for the format_checkpoint_message function."""

    def test_includes_trigger_reason(self) -> None:
        """Message should include trigger reason."""
        metrics = QualityMetrics(
            average_confidence=0.9,
            error_count=0,
            total_processed=100,
            processing_speed=10.0,
        )
        state: PipelineState = {
            "plan": [{"tool_name": "step"}],
            "current_step": 1,
        }

        message = format_checkpoint_message(
            CheckpointTrigger.PERCENTAGE, metrics, state
        )

        assert "percentage" in message.lower()

    def test_includes_progress(self) -> None:
        """Message should include progress percentage."""
        metrics = QualityMetrics(
            average_confidence=0.9,
            error_count=0,
            total_processed=100,
            processing_speed=10.0,
        )
        state: PipelineState = {
            "plan": [{"tool_name": f"step_{i}"} for i in range(4)],
            "current_step": 2,
        }

        message = format_checkpoint_message(
            CheckpointTrigger.PERCENTAGE, metrics, state
        )

        assert "50%" in message
        assert "2/4" in message

    def test_includes_quality_metrics(self) -> None:
        """Message should include quality metrics."""
        metrics = QualityMetrics(
            average_confidence=0.85,
            error_count=2,
            total_processed=100,
            processing_speed=10.0,
        )
        state: PipelineState = {
            "plan": [{"tool_name": "step"}],
            "current_step": 1,
        }

        message = format_checkpoint_message(
            CheckpointTrigger.PERCENTAGE, metrics, state
        )

        assert "85" in message  # 85% confidence
        assert "100" in message  # total processed
        assert "10" in message  # processing speed

    def test_critical_step_shows_tool_name(self) -> None:
        """Critical step message should show the tool name."""
        metrics = QualityMetrics(
            average_confidence=0.9,
            error_count=0,
            total_processed=50,
            processing_speed=5.0,
        )
        state: PipelineState = {
            "plan": [
                {"tool_name": "scan_directory"},
                {"tool_name": "anonymize"},
            ],
            "current_step": 2,
        }

        message = format_checkpoint_message(
            CheckpointTrigger.CRITICAL_STEP, metrics, state
        )

        assert "anonymize" in message.lower()
        assert "critical" in message.lower()

    def test_quality_drop_shows_warning(self) -> None:
        """Quality drop message should show warning."""
        metrics = QualityMetrics(
            average_confidence=0.65,
            error_count=0,
            total_processed=100,
            processing_speed=10.0,
        )
        state: PipelineState = {
            "plan": [{"tool_name": "step"}],
            "current_step": 1,
        }

        message = format_checkpoint_message(
            CheckpointTrigger.QUALITY_DROP, metrics, state
        )

        assert "Warning" in message
        assert "confidence" in message.lower()

    def test_error_rate_shows_warning(self) -> None:
        """Error rate message should show warning."""
        metrics = QualityMetrics(
            average_confidence=0.9,
            error_count=10,
            total_processed=100,
            processing_speed=10.0,
        )
        state: PipelineState = {
            "plan": [{"tool_name": "step"}],
            "current_step": 1,
        }

        message = format_checkpoint_message(
            CheckpointTrigger.ERROR_RATE, metrics, state
        )

        assert "Warning" in message
        assert "5%" in message

    def test_includes_options(self) -> None:
        """Message should list available options."""
        metrics = QualityMetrics(
            average_confidence=0.9,
            error_count=0,
            total_processed=100,
            processing_speed=10.0,
        )
        state: PipelineState = {
            "plan": [{"tool_name": "step"}],
            "current_step": 1,
        }

        message = format_checkpoint_message(
            CheckpointTrigger.PERCENTAGE, metrics, state
        )

        assert "Continue" in message
        assert "Edit Plan" in message
        assert "Cancel" in message


# -----------------------------------------------------------------------------
# checkpoint_node() tests
# -----------------------------------------------------------------------------


class TestCheckpointNode:
    """Tests for the checkpoint_node function."""

    def test_creates_checkpoint_on_trigger(self, executing_state: PipelineState) -> None:
        """Should create checkpoint when trigger condition met."""
        # Set up for 50% trigger (2/4 = 50%)
        result = checkpoint_node(executing_state)

        assert len(result.get("checkpoints", [])) > 0
        checkpoint = result["checkpoints"][0]
        assert checkpoint["id"].startswith("ckpt-")
        assert checkpoint["trigger_reason"] == CheckpointTrigger.PERCENTAGE.value

    def test_no_checkpoint_when_not_triggered(self) -> None:
        """Should return empty dict when no trigger."""
        state: PipelineState = {
            "plan": [{"tool_name": f"step_{i}"} for i in range(100)],
            "current_step": 3,  # 3% - below 10% threshold
            "execution_results": {},
            "checkpoints": [],
            "messages": [],
        }

        result = checkpoint_node(state)

        assert result == {} or "checkpoints" not in result

    def test_checkpoint_includes_quality_metrics(
        self, executing_state: PipelineState
    ) -> None:
        """Checkpoint should include quality metrics."""
        result = checkpoint_node(executing_state)

        if result:
            checkpoint = result["checkpoints"][0]
            metrics = checkpoint["quality_metrics"]
            assert "average_confidence" in metrics
            assert "error_count" in metrics
            assert "total_processed" in metrics
            assert "processing_speed" in metrics

    def test_checkpoint_adds_message(self, executing_state: PipelineState) -> None:
        """Checkpoint should add notification message."""
        result = checkpoint_node(executing_state)

        if result:
            assert len(result.get("messages", [])) > 0
            last_message = result["messages"][-1]
            assert last_message["role"] == "assistant"
            assert "Checkpoint" in last_message["content"]

    def test_critical_checkpoint_sets_awaiting_user(self) -> None:
        """Critical checkpoints should set awaiting_user."""
        state: PipelineState = {
            "plan": [
                {"tool_name": "scan_directory"},
                {"tool_name": "anonymize"},
            ],
            "current_step": 2,
            "execution_results": {},
            "checkpoints": [],
            "messages": [],
        }

        result = checkpoint_node(state)

        if result:
            assert result.get("awaiting_user") is True
            assert result["checkpoints"][0]["trigger_reason"] == CheckpointTrigger.CRITICAL_STEP.value


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------


class TestHITLIntegration:
    """Integration tests for the full HITL flow."""

    def test_checkpoint_creation_and_resolution(
        self, executing_state: PipelineState
    ) -> None:
        """Test checkpoint creation followed by user approval."""
        # Create checkpoint
        checkpoint_result = checkpoint_node(executing_state)

        if checkpoint_result:
            executing_state.update(checkpoint_result)  # type: ignore[typeddict-item]

            # User approves
            approval_result = handle_user_response(
                executing_state,
                {"decision": "approve", "message": "Looks good"},
            )

            assert approval_result["checkpoints"][0]["user_decision"] == "approve"
            assert approval_result["checkpoints"][0]["user_feedback"] == "Looks good"
            assert approval_result["checkpoints"][0]["resolved_at"] is not None
            assert approval_result["awaiting_user"] is False

    def test_plan_edit_flow(self, base_state: PipelineState) -> None:
        """Test plan approval with edits."""
        # User requests edit
        edit_result = handle_user_response(
            base_state,
            {
                "decision": "edit",
                "plan_edits": [
                    {
                        "action": "modify",
                        "step_index": 1,
                        "changes": {"parameters": {"classes": ["car", "truck"]}},
                    },
                    {
                        "action": "add",
                        "after_index": 1,
                        "step": {"tool_name": "export", "description": "Export results"},
                    },
                ],
            },
        )

        # Verify edits applied
        assert edit_result["plan"][1]["parameters"]["classes"] == ["car", "truck"]
        assert len(edit_result["plan"]) == 3
        assert edit_result["plan"][2]["tool_name"] == "export"

    def test_cancel_flow(self, executing_state: PipelineState) -> None:
        """Test pipeline cancellation."""
        cancel_result = handle_user_response(
            executing_state,
            {"decision": "cancel", "message": "Not needed anymore"},
        )

        # Remaining steps should be skipped
        skipped_count = sum(
            1
            for s in cancel_result["plan"]
            if s["status"] == StepStatus.SKIPPED.value
        )
        assert skipped_count == 2  # Steps 3 and 4


# -----------------------------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for HITL functionality."""

    def test_handle_response_with_empty_plan(self) -> None:
        """Should handle empty plan on cancel."""
        state: PipelineState = {
            "plan": [],
            "current_step": 0,
            "messages": [],
            "checkpoints": [],
            "awaiting_user": True,
            "plan_approved": False,
        }

        result = handle_user_response(state, {"decision": "cancel"})

        assert result["awaiting_user"] is False

    def test_apply_edits_to_empty_plan(self) -> None:
        """Should handle edits on empty plan."""
        result = apply_plan_edits([], [{"action": "remove", "step_index": 0}])

        assert result == []

    def test_checkpoint_at_100_percent(self) -> None:
        """Should not trigger at 100% (all steps done)."""
        state: PipelineState = {
            "plan": [{"tool_name": "step_1"}, {"tool_name": "step_2"}],
            "current_step": 2,  # 100%
            "execution_results": {},
            "checkpoints": [],
        }

        trigger = determine_trigger(state)

        # 50% would have been triggered at step 1
        # At 100% we should complete, not checkpoint again
        assert trigger != CheckpointTrigger.PERCENTAGE

    def test_metrics_with_non_dict_results(self) -> None:
        """Should handle non-dict values in results."""
        state: PipelineState = {
            "plan": [],
            "execution_results": {
                "step-1": "string value",  # type: ignore[dict-item]
                "step-2": None,  # type: ignore[dict-item]
                "step-3": {"confidence": 0.9},
            },
        }

        metrics = calculate_quality_metrics(state)

        # Should only process valid dict
        assert metrics.average_confidence == pytest.approx(0.9)

    def test_format_message_empty_plan(self) -> None:
        """Should handle empty plan gracefully."""
        metrics = QualityMetrics(
            average_confidence=0.0,
            error_count=0,
            total_processed=0,
            processing_speed=0.0,
        )
        state: PipelineState = {"plan": [], "current_step": 0}

        message = format_checkpoint_message(
            CheckpointTrigger.PERCENTAGE, metrics, state
        )

        assert "Checkpoint" in message


# -----------------------------------------------------------------------------
# Constants validation tests
# -----------------------------------------------------------------------------


class TestConstants:
    """Tests to verify constant values match spec."""

    def test_critical_tools(self) -> None:
        """Critical tools should include anonymize, segment, detect_3d."""
        assert "anonymize" in CRITICAL_TOOLS
        assert "segment" in CRITICAL_TOOLS
        assert "detect_3d" in CRITICAL_TOOLS

    def test_percentage_thresholds(self) -> None:
        """Percentage thresholds should be 10%, 25%, 50%."""
        assert 0.10 in PERCENTAGE_THRESHOLDS
        assert 0.25 in PERCENTAGE_THRESHOLDS
        assert 0.50 in PERCENTAGE_THRESHOLDS

    def test_error_rate_threshold(self) -> None:
        """Error rate threshold should be 5%."""
        assert pytest.approx(0.05) == ERROR_RATE_THRESHOLD

    def test_confidence_drop_threshold(self) -> None:
        """Confidence drop threshold should be 15%."""
        assert pytest.approx(0.15) == CONFIDENCE_DROP_THRESHOLD
