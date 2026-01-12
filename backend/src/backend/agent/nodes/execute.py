"""
Execute step node for the Cloumask agent.

This module implements the execute_step_node that runs pipeline tools
and captures results, along with helpers for progress tracking,
result formatting, and retry logic.

Implements spec: 04-agent-nodes-execution
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from backend.agent.state import MessageRole, PipelineState, StepStatus
from backend.agent.tools import (
    BaseTool,
    ToolCategory,
    ToolResult,
    get_tool_registry,
    success_result,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Maximum retries for transient errors
MAX_RETRIES = 3


# -----------------------------------------------------------------------------
# Stub Tools (for testing until 07-tool-implementations)
# -----------------------------------------------------------------------------


class StubTool(BaseTool):
    """
    Base stub tool that returns simulated results.

    Used for development and testing before real tool implementations
    are available.
    """

    category = ToolCategory.UTILITY
    parameters = []

    def __init__(
        self,
        name: str,
        result_factory: Callable[..., dict[str, Any]],
        description: str = "Stub tool for testing",
    ) -> None:
        super().__init__()
        self.name = name
        self.description = description
        self._result_factory = result_factory

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the stub tool and return simulated results."""
        data = self._result_factory(**kwargs)
        return success_result(data)


def _scan_result(**kwargs: Any) -> dict[str, Any]:
    """Generate scan_directory result."""
    return {
        "total_files": 100,
        "formats": ["jpg", "png", "mp4"],
        "files_processed": 0,
        "path": kwargs.get("path", "/data"),
    }


def _detect_result(**kwargs: Any) -> dict[str, Any]:
    """Generate detect result."""
    classes = kwargs.get("classes", [])
    return {
        "count": 50,
        "classes": {cls: 10 for cls in classes} if classes else {"object": 50},
        "confidence": 0.85,
        "files_processed": 10,
    }


def _anonymize_result(**kwargs: Any) -> dict[str, Any]:
    """Generate anonymize result."""
    return {
        "files_processed": 25,
        "faces_blurred": 45,
        "plates_blurred": 12,
        "confidence": 0.92,
    }


def _segment_result(**kwargs: Any) -> dict[str, Any]:
    """Generate segment result."""
    return {
        "count": 30,
        "masks_generated": 30,
        "confidence": 0.88,
        "files_processed": 10,
    }


def _export_result(**kwargs: Any) -> dict[str, Any]:
    """Generate export result."""
    return {
        "output_path": kwargs.get("output_path", "/output"),
        "format": kwargs.get("format", "yolo"),
        "files_exported": 100,
        "annotations_exported": 500,
    }


def register_stub_tools() -> None:
    """
    Register stub tools for development and testing.

    These tools simulate real CV operations, returning realistic but
    fake results. Used before actual tool implementations (07-tool-implementations).
    """
    registry = get_tool_registry()

    # Only register if not already registered (idempotent)
    stub_tools = [
        ("scan_directory", _scan_result, "Scan directory for image/video files"),
        ("detect", _detect_result, "Detect objects in images"),
        ("anonymize", _anonymize_result, "Anonymize faces and license plates"),
        ("segment", _segment_result, "Generate segmentation masks"),
        ("export", _export_result, "Export dataset in various formats"),
    ]

    for name, factory, desc in stub_tools:
        if not registry.has(name):
            registry.register(StubTool(name, factory, desc))


# -----------------------------------------------------------------------------
# Progress and Formatting Helpers
# -----------------------------------------------------------------------------


def update_progress(
    state: PipelineState,
    current: int,
    total: int,
    result: dict[str, Any],
) -> None:
    """
    Update progress metrics in metadata.

    Args:
        state: Current pipeline state.
        current: Current step index (0-based).
        total: Total number of steps.
        result: Result from the executed tool.
    """
    metadata = state.get("metadata", {})

    # Calculate progress percentage
    progress = (current + 1) / total * 100 if total > 0 else 0
    metadata["progress_percent"] = progress

    # Track files processed if applicable
    if "files_processed" in result:
        metadata["processed_files"] = (
            metadata.get("processed_files", 0) + result["files_processed"]
        )

    # Track items detected/processed
    if "count" in result:
        metadata["total_items"] = metadata.get("total_items", 0) + result["count"]

    state["metadata"] = metadata


def format_step_result(step: dict[str, Any], result: dict[str, Any]) -> str:
    """
    Format step result for chat display.

    Args:
        step: The pipeline step that was executed.
        result: The result from tool execution.

    Returns:
        Formatted string for displaying in chat.
    """
    tool_name = step.get("tool_name", "unknown")
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
            class_summary = ", ".join(
                f"{k}: {v}" for k, v in result["classes"].items()
            )
            lines.append(f"Classes: {class_summary}")

    elif tool_name == "anonymize":
        lines.append(f"Processed {result.get('files_processed', 0)} files")
        lines.append(
            f"Anonymized {result.get('faces_blurred', 0)} faces, "
            f"{result.get('plates_blurred', 0)} plates"
        )

    elif tool_name == "segment":
        lines.append(f"Generated {result.get('masks_generated', 0)} masks")
        lines.append(f"Segmented {result.get('count', 0)} objects")

    elif tool_name == "export":
        lines.append(f"Exported to {result.get('output_path', 'output')}")
        lines.append(f"Format: {result.get('format', 'unknown')}")

    else:
        # Generic result display
        for key, value in result.items():
            if key not in ["_meta", "raw_data"]:
                lines.append(f"{key}: {value}")

    # Add timing if available
    started_at = step.get("started_at")
    completed_at = step.get("completed_at")
    if started_at and completed_at:
        try:
            start = datetime.fromisoformat(started_at)
            end = datetime.fromisoformat(completed_at)
            duration = (end - start).total_seconds()
            lines.append(f"Duration: {duration:.1f}s")
        except (ValueError, TypeError):
            pass

    return "\n".join(lines)


def is_retryable(error: Exception) -> bool:
    """
    Determine if an error is retryable.

    Args:
        error: The exception that was raised.

    Returns:
        True if the error is transient and should be retried.
    """
    # Retryable error types
    retryable_types = (
        TimeoutError,
        ConnectionError,
        IOError,
    )

    if isinstance(error, retryable_types):
        return True

    # Check for specific error messages
    error_msg = str(error).lower()
    retryable_messages = ["timeout", "connection", "temporary", "busy", "retry"]

    return any(msg in error_msg for msg in retryable_messages)


# -----------------------------------------------------------------------------
# Execute Step Node
# -----------------------------------------------------------------------------


async def execute_step_node(state: PipelineState) -> dict[str, Any]:
    """
    Execute the current pipeline step.

    Gets the current step from the plan, looks up the appropriate tool,
    executes it with the step parameters, and records the result.

    Updates state with:
    - Step status (running -> completed/failed)
    - Execution result or error
    - Incremented current_step
    - Progress in metadata

    Args:
        state: Current pipeline state.

    Returns:
        State update dict with execution results.
    """
    plan = state.get("plan", [])
    current_idx = state.get("current_step", 0)
    messages = list(state.get("messages", []))
    execution_results = dict(state.get("execution_results", {}))

    # Validate we have a step to execute
    if current_idx >= len(plan):
        logger.warning("No more steps to execute (current=%d, total=%d)", current_idx, len(plan))
        return {
            "last_error": "No more steps to execute",
            "awaiting_user": False,
        }

    # Get the step and make a mutable copy
    step = dict(plan[current_idx])
    plan = list(plan)  # Make plan mutable
    plan[current_idx] = step

    tool_name = step.get("tool_name", "")
    parameters = step.get("parameters", {})
    step_id = step.get("id", f"step-{current_idx}")

    logger.info("Executing step %d: %s", current_idx, tool_name)

    # Get tool from registry
    registry = get_tool_registry()
    tool = registry.get(tool_name)

    if not tool:
        logger.error("Unknown tool: %s", tool_name)
        step["status"] = StepStatus.FAILED.value
        step["error"] = f"Unknown tool: {tool_name}"
        step["completed_at"] = datetime.now().isoformat()
        execution_results[step_id] = {"error": step["error"]}

        messages.append({
            "role": MessageRole.ASSISTANT.value,
            "content": f"Step failed: Unknown tool '{tool_name}'",
            "timestamp": datetime.now().isoformat(),
            "tool_calls": [],
            "tool_call_id": None,
        })

        return {
            "plan": plan,
            "current_step": current_idx + 1,
            "execution_results": execution_results,
            "messages": messages,
            "retry_count": 0,
            "awaiting_user": False,
        }

    # Mark step as running
    step["status"] = StepStatus.RUNNING.value
    step["started_at"] = datetime.now().isoformat()

    # Execute the tool using the run() method for validation and timing
    tool_result = await tool.run(**parameters)

    if tool_result.success:
        # Mark success
        step["status"] = StepStatus.COMPLETED.value
        step["completed_at"] = datetime.now().isoformat()

        # Get result data (ToolResult.data or empty dict)
        result_data = tool_result.data or {}
        step["result"] = result_data

        # Store in execution results
        execution_results[step_id] = result_data

        # Update progress
        update_progress(state, current_idx, len(plan), result_data)

        # Add progress message
        messages.append({
            "role": MessageRole.ASSISTANT.value,
            "content": format_step_result(step, result_data),
            "timestamp": datetime.now().isoformat(),
            "tool_calls": [],
            "tool_call_id": None,
        })

        logger.info("Step %d completed successfully (%.2fs)", current_idx, tool_result.duration_seconds)
    else:
        # Tool execution failed
        error_msg = tool_result.error or "Unknown error"
        logger.error("Step %d failed: %s", current_idx, error_msg)

        # Mark failure
        step["status"] = StepStatus.FAILED.value
        step["completed_at"] = datetime.now().isoformat()
        step["error"] = error_msg

        execution_results[step_id] = {"error": error_msg}

        # Check retry logic for retryable errors
        retry_count = state.get("retry_count", 0)
        # Create a synthetic exception to check if retryable
        synthetic_error = RuntimeError(error_msg)
        if retry_count < MAX_RETRIES and is_retryable(synthetic_error):
            messages.append({
                "role": MessageRole.ASSISTANT.value,
                "content": f"Step failed: {error_msg}. Retrying ({retry_count + 1}/{MAX_RETRIES})...",
                "timestamp": datetime.now().isoformat(),
                "tool_calls": [],
                "tool_call_id": None,
            })

            # Reset step status for retry
            step["status"] = StepStatus.PENDING.value
            step["error"] = None
            step["started_at"] = None
            step["completed_at"] = None

            return {
                "plan": plan,
                "retry_count": retry_count + 1,
                "messages": messages,
                "awaiting_user": False,
            }

        # Non-retryable or max retries exceeded
        messages.append({
            "role": MessageRole.ASSISTANT.value,
            "content": f"Step failed: {error_msg}. Moving to next step.",
            "timestamp": datetime.now().isoformat(),
            "tool_calls": [],
            "tool_call_id": None,
        })

    # Move to next step
    return {
        "plan": plan,
        "current_step": current_idx + 1,
        "execution_results": execution_results,
        "messages": messages,
        "retry_count": 0,
        "awaiting_user": False,
    }


# Alias for backward compatibility with the spec
execute_step = execute_step_node
