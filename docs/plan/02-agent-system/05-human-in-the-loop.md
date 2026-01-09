# Human-in-the-Loop

> **Status:** 🔴 Not Started
> **Priority:** P0 (Critical)
> **Dependencies:** 01-state-types, 02-langgraph-core
> **Estimated Complexity:** Medium

## Overview

Implement the `await_approval` and `checkpoint` nodes that pause execution for human review. This is critical for ensuring users maintain control over CV pipelines, especially for operations that modify data or have quality implications.

## Goals

- [ ] `await_approval` node: Pause and wait for user decision
- [ ] `checkpoint` node: Evaluate quality and decide if pause needed
- [ ] Support approve, edit, cancel, and retry decisions
- [ ] Checkpoint trigger logic (percentage, quality, critical steps)
- [ ] User feedback handling and plan modification

## Technical Design

### Await Approval Node

```python
from datetime import datetime
from agent.state import PipelineState, UserDecision, CheckpointTrigger


async def await_approval_node(state: PipelineState) -> PipelineState:
    """
    Pause execution and wait for user decision.

    This node marks the state as awaiting user input.
    The actual user response is handled by the API layer,
    which updates the state and resumes the graph.

    Updates state with:
    - awaiting_user: True
    - Current checkpoint/plan status for display
    """

    # Determine what we're waiting for
    plan_approved = state.get("plan_approved", False)
    checkpoints = state.get("checkpoints", [])

    if not plan_approved:
        # Waiting for initial plan approval
        state["awaiting_user"] = True
        state["messages"].append({
            "role": "system",
            "content": "AWAIT_PLAN_APPROVAL",
            "timestamp": datetime.now().isoformat(),
        })
        return state

    # Waiting for checkpoint approval
    if checkpoints:
        latest_checkpoint = checkpoints[-1]

        state["awaiting_user"] = True
        state["messages"].append({
            "role": "system",
            "content": f"AWAIT_CHECKPOINT:{latest_checkpoint['id']}",
            "timestamp": datetime.now().isoformat(),
        })

    return state


def handle_user_response(state: PipelineState, response: dict) -> PipelineState:
    """
    Process user's decision and update state accordingly.

    Called by API layer when user responds to approval request.

    Args:
        state: Current pipeline state
        response: {
            "decision": "approve" | "edit" | "cancel" | "retry",
            "message": Optional feedback message,
            "plan_edits": Optional list of plan modifications
        }
    """

    decision = UserDecision(response.get("decision", "approve"))
    message = response.get("message")
    plan_edits = response.get("plan_edits")

    # Add user message to conversation
    if message:
        state["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })

    # Handle based on decision
    if decision == UserDecision.APPROVE:
        state["awaiting_user"] = False
        state["plan_approved"] = True

        # Resolve any pending checkpoint
        checkpoints = state.get("checkpoints", [])
        if checkpoints and not checkpoints[-1].get("resolved_at"):
            checkpoints[-1]["user_decision"] = decision.value
            checkpoints[-1]["resolved_at"] = datetime.now().isoformat()
            checkpoints[-1]["user_feedback"] = message

        state["messages"].append({
            "role": "assistant",
            "content": "Continuing execution...",
            "timestamp": datetime.now().isoformat(),
        })

    elif decision == UserDecision.EDIT:
        state["awaiting_user"] = False

        if plan_edits:
            apply_plan_edits(state, plan_edits)
            state["messages"].append({
                "role": "assistant",
                "content": "Plan updated. Here's the revised plan:",
                "timestamp": datetime.now().isoformat(),
            })
        else:
            state["messages"].append({
                "role": "assistant",
                "content": "What changes would you like to make to the plan?",
                "timestamp": datetime.now().isoformat(),
            })
            state["awaiting_user"] = True

    elif decision == UserDecision.CANCEL:
        state["awaiting_user"] = False
        state["messages"].append({
            "role": "assistant",
            "content": "Pipeline cancelled.",
            "timestamp": datetime.now().isoformat(),
        })
        # Mark remaining steps as skipped
        plan = state.get("plan", [])
        current = state.get("current_step", 0)
        for step in plan[current:]:
            step["status"] = "skipped"

    elif decision == UserDecision.RETRY:
        state["awaiting_user"] = False
        state["retry_count"] = 0
        # Don't increment current_step, will retry
        state["messages"].append({
            "role": "assistant",
            "content": "Retrying the current step...",
            "timestamp": datetime.now().isoformat(),
        })

    return state


def apply_plan_edits(state: PipelineState, edits: list[dict]) -> None:
    """
    Apply user's edits to the plan.

    Edits format:
    [
        {"action": "modify", "step_index": 0, "changes": {"parameters": {...}}},
        {"action": "remove", "step_index": 2},
        {"action": "add", "after_index": 1, "step": {...}},
        {"action": "reorder", "from_index": 3, "to_index": 1}
    ]
    """

    plan = state.get("plan", [])

    for edit in edits:
        action = edit.get("action")

        if action == "modify":
            idx = edit["step_index"]
            if 0 <= idx < len(plan):
                changes = edit.get("changes", {})
                plan[idx].update(changes)

        elif action == "remove":
            idx = edit["step_index"]
            if 0 <= idx < len(plan):
                plan.pop(idx)

        elif action == "add":
            after_idx = edit.get("after_index", len(plan) - 1)
            new_step = edit["step"]
            plan.insert(after_idx + 1, new_step)

        elif action == "reorder":
            from_idx = edit["from_index"]
            to_idx = edit["to_index"]
            if 0 <= from_idx < len(plan) and 0 <= to_idx < len(plan):
                step = plan.pop(from_idx)
                plan.insert(to_idx, step)

    state["plan"] = plan
```

### Checkpoint Node

```python
from uuid import uuid4

from agent.state import (
    PipelineState,
    Checkpoint,
    CheckpointTrigger,
    QualityMetrics
)


async def checkpoint_node(state: PipelineState) -> PipelineState:
    """
    Evaluate execution quality and create checkpoint if needed.

    Updates state with:
    - New checkpoint in checkpoints list
    - Quality metrics
    - Checkpoint notification message
    """

    # Determine checkpoint trigger reason
    trigger = determine_trigger(state)

    if not trigger:
        # No checkpoint needed, continue execution
        return state

    # Calculate quality metrics
    metrics = calculate_quality_metrics(state)

    # Create checkpoint
    checkpoint = {
        "id": f"ckpt-{uuid4().hex[:8]}",
        "step_index": state.get("current_step", 0),
        "trigger_reason": trigger.value,
        "quality_metrics": metrics.__dict__,
        "created_at": datetime.now().isoformat(),
        "user_decision": None,
        "user_feedback": None,
        "resolved_at": None,
    }

    checkpoints = state.get("checkpoints", [])
    checkpoints.append(checkpoint)
    state["checkpoints"] = checkpoints

    # Generate checkpoint message
    message = format_checkpoint_message(trigger, metrics, state)
    state["messages"].append({
        "role": "assistant",
        "content": message,
        "timestamp": datetime.now().isoformat(),
    })

    return state


def determine_trigger(state: PipelineState) -> Optional[CheckpointTrigger]:
    """Determine if and why a checkpoint should be created."""

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    results = state.get("execution_results", {})

    if not plan or current_step == 0:
        return None

    total_steps = len(plan)
    progress = current_step / total_steps

    # Check for percentage triggers
    prev_progress = (current_step - 1) / total_steps
    for threshold in [0.10, 0.25, 0.50]:
        if prev_progress < threshold <= progress:
            return CheckpointTrigger.PERCENTAGE

    # Check for critical step
    last_step = plan[current_step - 1] if current_step > 0 else None
    if last_step:
        critical_tools = ["anonymize", "segment", "detect_3d"]
        if last_step.get("tool_name") in critical_tools:
            return CheckpointTrigger.CRITICAL_STEP

    # Check for quality issues
    if results:
        recent = list(results.values())[-5:]

        # Error rate check
        errors = sum(1 for r in recent if isinstance(r, dict) and r.get("error"))
        if errors / max(len(recent), 1) > 0.05:
            return CheckpointTrigger.ERROR_RATE

        # Confidence drop check
        all_conf = [r.get("confidence") for r in results.values()
                    if isinstance(r, dict) and "confidence" in r]
        recent_conf = [r.get("confidence") for r in recent
                       if isinstance(r, dict) and "confidence" in r]

        if all_conf and recent_conf:
            overall_avg = sum(all_conf) / len(all_conf)
            recent_avg = sum(recent_conf) / len(recent_conf)
            if overall_avg - recent_avg > 0.15:
                return CheckpointTrigger.QUALITY_DROP

    return None


def calculate_quality_metrics(state: PipelineState) -> QualityMetrics:
    """Calculate current quality metrics from execution results."""

    results = state.get("execution_results", {})
    plan = state.get("plan", [])

    # Collect confidence scores
    confidences = []
    error_count = 0
    total_processed = 0

    for step_id, result in results.items():
        if not isinstance(result, dict):
            continue

        if "confidence" in result:
            confidences.append(result["confidence"])

        if result.get("error"):
            error_count += 1

        total_processed += result.get("files_processed", 1)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Calculate processing speed
    total_duration = 0.0
    for step in plan:
        if step.get("started_at") and step.get("completed_at"):
            start = datetime.fromisoformat(step["started_at"])
            end = datetime.fromisoformat(step["completed_at"])
            total_duration += (end - start).total_seconds()

    speed = total_processed / total_duration if total_duration > 0 else 0.0

    return QualityMetrics(
        average_confidence=avg_confidence,
        error_count=error_count,
        total_processed=total_processed,
        processing_speed=speed,
    )


def format_checkpoint_message(
    trigger: CheckpointTrigger,
    metrics: QualityMetrics,
    state: PipelineState
) -> str:
    """Format checkpoint notification for display."""

    plan = state.get("plan", [])
    current = state.get("current_step", 0)
    progress = current / len(plan) * 100 if plan else 0

    lines = [f"## Checkpoint ({trigger.value})"]
    lines.append("")
    lines.append(f"**Progress:** {progress:.0f}% ({current}/{len(plan)} steps)")
    lines.append("")
    lines.append("### Quality Metrics")
    lines.append(f"- Avg Confidence: {metrics.average_confidence:.1%}")
    lines.append(f"- Error Rate: {metrics.error_rate:.1%}")
    lines.append(f"- Processing Speed: {metrics.processing_speed:.1f} items/sec")
    lines.append(f"- Total Processed: {metrics.total_processed}")

    # Add trigger-specific message
    lines.append("")
    if trigger == CheckpointTrigger.PERCENTAGE:
        lines.append(f"This is a progress checkpoint at {progress:.0f}%.")
        lines.append("Review the results so far and decide if you want to continue.")

    elif trigger == CheckpointTrigger.CRITICAL_STEP:
        last_tool = plan[current - 1]["tool_name"] if current > 0 else "unknown"
        lines.append(f"Just completed critical operation: **{last_tool}**")
        lines.append("Please verify the results before continuing.")

    elif trigger == CheckpointTrigger.QUALITY_DROP:
        lines.append("**Warning:** Significant drop in confidence scores detected.")
        lines.append("Consider reviewing recent results or adjusting parameters.")

    elif trigger == CheckpointTrigger.ERROR_RATE:
        lines.append("**Warning:** Error rate is above 5%.")
        lines.append("Some items may not be processing correctly.")

    lines.append("")
    lines.append("**Options:** Continue | Edit Plan | Cancel")

    return "\n".join(lines)
```

## Implementation Tasks

- [ ] Create `backend/agent/nodes/approval.py`
- [ ] Implement `await_approval_node()` function
- [ ] Implement `handle_user_response()` function
- [ ] Implement `apply_plan_edits()` helper
- [ ] Create `backend/agent/nodes/checkpoint.py`
- [ ] Implement `checkpoint_node()` function
- [ ] Implement `determine_trigger()` helper
- [ ] Implement `calculate_quality_metrics()` helper
- [ ] Implement `format_checkpoint_message()` helper
- [ ] Add API integration for user response handling
- [ ] Add SSE events for checkpoint notifications

## Testing

### Unit Tests

```python
# tests/agent/nodes/test_hitl.py

def test_determine_trigger_percentage():
    """Should trigger at 25% progress."""
    state = {
        "plan": [{"tool_name": f"step_{i}"} for i in range(4)],
        "current_step": 1,  # 25%
        "execution_results": {},
    }
    trigger = determine_trigger(state)
    assert trigger == CheckpointTrigger.PERCENTAGE


def test_determine_trigger_critical_step():
    """Should trigger after anonymize."""
    state = {
        "plan": [{"tool_name": "anonymize"}],
        "current_step": 1,
        "execution_results": {},
    }
    trigger = determine_trigger(state)
    assert trigger == CheckpointTrigger.CRITICAL_STEP


def test_determine_trigger_quality_drop():
    """Should trigger when confidence drops >15%."""
    state = {
        "plan": [{"tool_name": f"step_{i}"} for i in range(10)],
        "current_step": 5,
        "execution_results": {
            "step-0": {"confidence": 0.95},
            "step-1": {"confidence": 0.93},
            "step-2": {"confidence": 0.75},  # Recent
            "step-3": {"confidence": 0.70},  # Recent
            "step-4": {"confidence": 0.68},  # Recent
        },
    }
    trigger = determine_trigger(state)
    assert trigger == CheckpointTrigger.QUALITY_DROP


def test_handle_user_response_approve():
    """Approve should clear awaiting flag."""
    state = {"awaiting_user": True, "plan_approved": False, "messages": [], "checkpoints": []}
    result = handle_user_response(state, {"decision": "approve"})
    assert result["awaiting_user"] == False
    assert result["plan_approved"] == True


def test_handle_user_response_cancel():
    """Cancel should skip remaining steps."""
    state = {
        "awaiting_user": True,
        "plan": [{"status": "pending"}, {"status": "pending"}],
        "current_step": 0,
        "messages": [],
        "checkpoints": [],
    }
    result = handle_user_response(state, {"decision": "cancel"})
    assert all(s["status"] == "skipped" for s in result["plan"])


def test_apply_plan_edits_modify():
    """Modify edit should update step."""
    state = {
        "plan": [{"tool_name": "detect", "parameters": {"confidence": 0.5}}]
    }
    edits = [{"action": "modify", "step_index": 0, "changes": {"parameters": {"confidence": 0.8}}}]
    apply_plan_edits(state, edits)
    assert state["plan"][0]["parameters"]["confidence"] == 0.8


def test_apply_plan_edits_remove():
    """Remove edit should delete step."""
    state = {"plan": [{"tool_name": "a"}, {"tool_name": "b"}, {"tool_name": "c"}]}
    edits = [{"action": "remove", "step_index": 1}]
    apply_plan_edits(state, edits)
    assert len(state["plan"]) == 2
    assert state["plan"][1]["tool_name"] == "c"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_checkpoint_flow():
    """Test checkpoint creation and resolution."""
    state = {
        "plan": [{"tool_name": "scan_directory"}],
        "current_step": 1,
        "execution_results": {"step-1": {"files_processed": 100}},
        "messages": [],
        "checkpoints": [],
        "plan_approved": True,
    }

    # Create checkpoint
    state = await checkpoint_node(state)
    assert len(state["checkpoints"]) == 1

    # Simulate user approval
    state = handle_user_response(state, {"decision": "approve", "message": "Looks good"})
    assert state["checkpoints"][0]["user_decision"] == "approve"
    assert state["checkpoints"][0]["user_feedback"] == "Looks good"
```

### Edge Cases

- [ ] User responds to stale checkpoint
- [ ] Multiple rapid user responses
- [ ] Edit plan with invalid step indices
- [ ] Checkpoint at 100% (just before complete)
- [ ] No execution results yet

## Acceptance Criteria

- [ ] Agent pauses for plan approval before execution
- [ ] User can approve, edit, or cancel plan
- [ ] Plan edits are applied correctly
- [ ] Checkpoints trigger at 10%, 25%, 50%
- [ ] Checkpoints trigger after critical tools
- [ ] Quality drop triggers checkpoint warning
- [ ] Checkpoint messages are clear and actionable
- [ ] Pipeline can resume after checkpoint approval

## Files to Create/Modify

```
backend/
├── agent/
│   └── nodes/
│       ├── __init__.py
│       ├── approval.py       # await_approval_node
│       └── checkpoint.py     # checkpoint_node
└── tests/
    └── agent/
        └── nodes/
            └── test_hitl.py
```

## Notes

- Consider adding "skip to checkpoint" option for long pipelines
- Checkpoints should include sample preview images when available
- Future: Add estimated time to completion at checkpoints
- Consider WebSocket for real-time checkpoint interaction
