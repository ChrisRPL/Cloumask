# Agent Nodes: Execution

> **Status:** 🟢 Complete (Implemented; checklist backfill pending)
> **Priority:** P0 (Critical)
> **Dependencies:** 01-state-types, 02-langgraph-core, 06-tool-system
> **Estimated Complexity:** Medium

## Overview

Implement the `execute_step` and `complete` nodes that run pipeline tools and finalize execution. These nodes handle the actual work of invoking CV tools, capturing results, and generating completion summaries.

## Goals

- [ ] `execute_step` node: Invoke current tool and capture result
- [ ] `complete` node: Summarize execution and cleanup
- [ ] Progress tracking during execution
- [ ] Error handling with retry capability
- [ ] Result aggregation for final report

## Technical Design

### Execute Step Node

```python
from datetime import datetime
from typing import Any

from agent.state import PipelineState, StepStatus
from agent.tools import get_tool_registry


async def execute_step_node(state: PipelineState) -> PipelineState:
    """
    Execute the current pipeline step.

    Updates state with:
    - Step status (running -> completed/failed)
    - Execution result or error
    - Incremented current_step
    - Progress in metadata
    """

    plan = state.get("plan", [])
    current_idx = state.get("current_step", 0)

    # Validate we have a step to execute
    if current_idx >= len(plan):
        state["last_error"] = "No more steps to execute"
        return state

    step = plan[current_idx]
    tool_name = step["tool_name"]
    parameters = step.get("parameters", {})

    # Get tool from registry
    registry = get_tool_registry()
    tool = registry.get(tool_name)

    if not tool:
        step["status"] = "failed"
        step["error"] = f"Unknown tool: {tool_name}"
        state["execution_results"][step["id"]] = {"error": step["error"]}
        state["current_step"] = current_idx + 1
        return state

    # Mark step as running
    step["status"] = "running"
    step["started_at"] = datetime.now().isoformat()

    try:
        # Execute the tool
        result = await tool.execute(**parameters)

        # Mark success
        step["status"] = "completed"
        step["completed_at"] = datetime.now().isoformat()
        step["result"] = result

        # Store in execution results
        state["execution_results"][step["id"]] = result

        # Update progress
        update_progress(state, current_idx, len(plan), result)

        # Add progress message
        state["messages"].append({
            "role": "assistant",
            "content": format_step_result(step, result),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        # Mark failure
        step["status"] = "failed"
        step["completed_at"] = datetime.now().isoformat()
        step["error"] = str(e)

        state["execution_results"][step["id"]] = {"error": str(e)}

        # Check retry logic
        retry_count = state.get("retry_count", 0)
        if retry_count < 3 and is_retryable(e):
            state["retry_count"] = retry_count + 1
            state["messages"].append({
                "role": "assistant",
                "content": f"Step failed: {e}. Retrying ({retry_count + 1}/3)...",
                "timestamp": datetime.now().isoformat(),
            })
            return state  # Don't increment, will retry

        state["messages"].append({
            "role": "assistant",
            "content": f"Step failed: {e}. Moving to next step.",
            "timestamp": datetime.now().isoformat(),
        })

    # Move to next step
    state["current_step"] = current_idx + 1
    state["retry_count"] = 0

    return state


def update_progress(
    state: PipelineState,
    current: int,
    total: int,
    result: dict
) -> None:
    """Update progress metrics in metadata."""

    metadata = state.get("metadata", {})

    # Calculate progress percentage
    progress = (current + 1) / total * 100
    metadata["progress_percent"] = progress

    # Track files processed if applicable
    if "files_processed" in result:
        metadata["processed_files"] = metadata.get("processed_files", 0) + result["files_processed"]

    # Track items detected/processed
    if "count" in result:
        metadata["total_items"] = metadata.get("total_items", 0) + result["count"]

    state["metadata"] = metadata


def format_step_result(step: dict, result: dict) -> str:
    """Format step result for chat display."""

    tool_name = step["tool_name"]
    description = step.get("description", tool_name)

    lines = [f"**{description}** completed"]

    # Format based on tool type
    if tool_name == "scan_directory":
        lines.append(f"Found {result.get('total_files', 0)} files")
        if result.get("formats"):
            lines.append(f"Formats: {', '.join(result['formats'])}")

    elif tool_name == "detect":
        lines.append(f"Detected {result.get('count', 0)} objects")
        if result.get("classes"):
            class_summary = ", ".join(f"{k}: {v}" for k, v in result["classes"].items())
            lines.append(f"Classes: {class_summary}")

    elif tool_name == "anonymize":
        lines.append(f"Processed {result.get('files_processed', 0)} files")
        lines.append(f"Anonymized {result.get('faces_blurred', 0)} faces, {result.get('plates_blurred', 0)} plates")

    elif tool_name == "export":
        lines.append(f"Exported to {result.get('output_path', 'output')}")
        lines.append(f"Format: {result.get('format', 'unknown')}")

    else:
        # Generic result display
        for key, value in result.items():
            if key not in ["_meta", "raw_data"]:
                lines.append(f"{key}: {value}")

    # Add timing if available
    if step.get("started_at") and step.get("completed_at"):
        start = datetime.fromisoformat(step["started_at"])
        end = datetime.fromisoformat(step["completed_at"])
        duration = (end - start).total_seconds()
        lines.append(f"Duration: {duration:.1f}s")

    return "\n".join(lines)


def is_retryable(error: Exception) -> bool:
    """Determine if an error is retryable."""

    # Retryable errors
    retryable_types = (
        TimeoutError,
        ConnectionError,
        IOError,
    )

    if isinstance(error, retryable_types):
        return True

    # Check for specific error messages
    error_msg = str(error).lower()
    retryable_messages = ["timeout", "connection", "temporary", "busy"]

    return any(msg in error_msg for msg in retryable_messages)
```

### Complete Node

```python
async def complete_node(state: PipelineState) -> PipelineState:
    """
    Finalize pipeline execution.

    Updates state with:
    - Final summary message
    - Aggregated statistics
    - Completion timestamp
    """

    plan = state.get("plan", [])
    results = state.get("execution_results", {})
    metadata = state.get("metadata", {})

    # Calculate final statistics
    stats = calculate_final_stats(plan, results)
    metadata["final_stats"] = stats
    metadata["completed_at"] = datetime.now().isoformat()
    state["metadata"] = metadata

    # Generate summary message
    summary = generate_summary(plan, results, stats)

    state["messages"].append({
        "role": "assistant",
        "content": summary,
        "timestamp": datetime.now().isoformat(),
    })

    return state


def calculate_final_stats(plan: list[dict], results: dict) -> dict:
    """Calculate aggregate statistics for the pipeline."""

    completed = sum(1 for s in plan if s.get("status") == "completed")
    failed = sum(1 for s in plan if s.get("status") == "failed")
    skipped = sum(1 for s in plan if s.get("status") == "skipped")

    # Calculate total duration
    total_duration = 0.0
    for step in plan:
        if step.get("started_at") and step.get("completed_at"):
            start = datetime.fromisoformat(step["started_at"])
            end = datetime.fromisoformat(step["completed_at"])
            total_duration += (end - start).total_seconds()

    # Aggregate counts from results
    files_processed = 0
    items_detected = 0
    items_anonymized = 0

    for result in results.values():
        if isinstance(result, dict):
            files_processed += result.get("files_processed", 0)
            items_detected += result.get("count", 0)
            items_anonymized += result.get("faces_blurred", 0) + result.get("plates_blurred", 0)

    # Calculate average confidence if available
    confidences = []
    for result in results.values():
        if isinstance(result, dict) and "confidence" in result:
            confidences.append(result["confidence"])

    avg_confidence = sum(confidences) / len(confidences) if confidences else None

    return {
        "total_steps": len(plan),
        "completed_steps": completed,
        "failed_steps": failed,
        "skipped_steps": skipped,
        "success_rate": completed / len(plan) if plan else 0,
        "total_duration_seconds": total_duration,
        "files_processed": files_processed,
        "items_detected": items_detected,
        "items_anonymized": items_anonymized,
        "average_confidence": avg_confidence,
    }


def generate_summary(plan: list[dict], results: dict, stats: dict) -> str:
    """Generate human-readable completion summary."""

    lines = ["## Pipeline Complete"]
    lines.append("")

    # Overall status
    if stats["failed_steps"] == 0:
        lines.append("All steps completed successfully.")
    else:
        lines.append(f"Completed with {stats['failed_steps']} failed step(s).")

    lines.append("")
    lines.append("### Summary")
    lines.append(f"- Steps: {stats['completed_steps']}/{stats['total_steps']} completed")
    lines.append(f"- Duration: {stats['total_duration_seconds']:.1f}s")

    if stats["files_processed"] > 0:
        lines.append(f"- Files processed: {stats['files_processed']}")

    if stats["items_detected"] > 0:
        lines.append(f"- Objects detected: {stats['items_detected']}")

    if stats["items_anonymized"] > 0:
        lines.append(f"- Items anonymized: {stats['items_anonymized']}")

    if stats["average_confidence"]:
        lines.append(f"- Avg confidence: {stats['average_confidence']:.1%}")

    # List any failures
    failed_steps = [s for s in plan if s.get("status") == "failed"]
    if failed_steps:
        lines.append("")
        lines.append("### Failed Steps")
        for step in failed_steps:
            lines.append(f"- {step['description']}: {step.get('error', 'Unknown error')}")

    # Output location if available
    export_results = [r for r in results.values() if isinstance(r, dict) and "output_path" in r]
    if export_results:
        lines.append("")
        lines.append("### Output")
        for r in export_results:
            lines.append(f"- {r['output_path']}")

    return "\n".join(lines)
```

## Implementation Tasks

- [ ] Create `backend/agent/nodes/execute.py`
- [ ] Implement `execute_step_node()` function
- [ ] Implement `update_progress()` helper
- [ ] Implement `format_step_result()` helper
- [ ] Implement `is_retryable()` helper
- [ ] Create `backend/agent/nodes/complete.py`
- [ ] Implement `complete_node()` function
- [ ] Implement `calculate_final_stats()` helper
- [ ] Implement `generate_summary()` helper
- [ ] Add retry logic with backoff
- [ ] Add timeout handling for long-running tools

## Testing

### Unit Tests

```python
# tests/agent/nodes/test_execution.py

def test_format_step_result_scan():
    """Scan result should show file count."""
    step = {"tool_name": "scan_directory", "description": "Scan input"}
    result = {"total_files": 100, "formats": ["jpg", "png"]}

    output = format_step_result(step, result)
    assert "100 files" in output
    assert "jpg" in output


def test_format_step_result_detect():
    """Detect result should show object counts."""
    step = {"tool_name": "detect", "description": "Detect objects"}
    result = {"count": 50, "classes": {"car": 30, "person": 20}}

    output = format_step_result(step, result)
    assert "50 objects" in output
    assert "car: 30" in output


def test_is_retryable_timeout():
    """Timeout errors should be retryable."""
    assert is_retryable(TimeoutError("timed out")) == True


def test_is_retryable_value_error():
    """Value errors should not be retryable."""
    assert is_retryable(ValueError("invalid")) == False


def test_calculate_final_stats():
    """Stats should aggregate correctly."""
    plan = [
        {"status": "completed", "started_at": "2026-01-01T10:00:00", "completed_at": "2026-01-01T10:00:30"},
        {"status": "completed", "started_at": "2026-01-01T10:00:30", "completed_at": "2026-01-01T10:01:00"},
        {"status": "failed"},
    ]
    results = {
        "step-1": {"files_processed": 50},
        "step-2": {"files_processed": 50, "count": 100},
    }

    stats = calculate_final_stats(plan, results)

    assert stats["total_steps"] == 3
    assert stats["completed_steps"] == 2
    assert stats["failed_steps"] == 1
    assert stats["files_processed"] == 100
    assert stats["total_duration_seconds"] == 60.0


@pytest.mark.asyncio
async def test_execute_step_success():
    """Execute step should update state correctly."""
    state = {
        "plan": [
            {"id": "step-1", "tool_name": "scan_directory", "parameters": {"path": "/data"}, "status": "pending"}
        ],
        "current_step": 0,
        "execution_results": {},
        "messages": [],
        "metadata": {},
    }

    with patch("agent.tools.get_tool_registry") as mock_registry:
        mock_tool = AsyncMock()
        mock_tool.execute.return_value = {"total_files": 10}
        mock_registry.return_value.get.return_value = mock_tool

        result = await execute_step_node(state)

        assert result["plan"][0]["status"] == "completed"
        assert result["current_step"] == 1
        assert "step-1" in result["execution_results"]
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_execution_flow():
    """Test executing multiple steps."""
    state = {
        "plan": [
            {"id": "step-1", "tool_name": "scan_directory", "parameters": {"path": "/data"}, "description": "Scan", "status": "pending"},
            {"id": "step-2", "tool_name": "detect", "parameters": {"input_path": "/data"}, "description": "Detect", "status": "pending"},
        ],
        "current_step": 0,
        "execution_results": {},
        "messages": [],
        "metadata": {},
    }

    # Execute both steps
    state = await execute_step_node(state)
    state = await execute_step_node(state)

    assert state["current_step"] == 2
    assert len(state["execution_results"]) == 2


@pytest.mark.asyncio
async def test_complete_after_execution():
    """Complete node should generate summary."""
    state = {
        "plan": [
            {"id": "step-1", "tool_name": "scan_directory", "status": "completed",
             "started_at": "2026-01-01T10:00:00", "completed_at": "2026-01-01T10:00:10"}
        ],
        "current_step": 1,
        "execution_results": {"step-1": {"total_files": 50}},
        "messages": [],
        "metadata": {},
    }

    result = await complete_node(state)

    assert "final_stats" in result["metadata"]
    assert len(result["messages"]) > 0
    assert "complete" in result["messages"][-1]["content"].lower()
```

### Edge Cases

- [ ] All steps fail
- [ ] Empty plan
- [ ] Tool not found in registry
- [ ] Tool returns None
- [ ] Tool throws unexpected exception
- [ ] Very long execution (timeout handling)

## Acceptance Criteria

- [ ] Steps execute in order
- [ ] Failed steps don't block subsequent steps
- [ ] Retry logic works for transient errors
- [ ] Progress updates are accurate
- [ ] Completion summary includes all statistics
- [ ] Failed steps are listed in summary
- [ ] Duration is calculated correctly

## Files to Create/Modify

```
backend/
├── agent/
│   └── nodes/
│       ├── __init__.py       # Export execute, complete
│       ├── execute.py        # execute_step_node
│       └── complete.py       # complete_node
└── tests/
    └── agent/
        └── nodes/
            └── test_execution.py
```

## Notes

- Consider adding step-level timeouts (e.g., 5 minutes max per step)
- Large file operations should stream progress updates
- Execution results should be cleaned up after completion to save memory
- Consider parallel execution for independent steps in future version
