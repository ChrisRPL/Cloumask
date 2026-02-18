"""
Execute step node for the Cloumask agent.

This module implements the execute_step_node that runs pipeline tools
and captures results, along with helpers for progress tracking,
result formatting, and retry logic.

Implements spec: 04-agent-nodes-execution
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.agent.state import MessageRole, PipelineState, StepStatus
from backend.agent.tools import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    get_tool_registry,
    success_result,
)
from backend.data.formats import detect_format

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Maximum retries for transient errors
MAX_RETRIES = 3
GENERATED_SCRIPT_ROOT = Path("/tmp/cloumask-generated-steps")


def _coerce_parameter_value(param: ToolParameter, value: Any) -> Any:
    """Coerce common UI payload shapes to the tool's expected parameter type."""
    if param.type is list and not isinstance(value, list):
        if isinstance(value, str):
            if not value.strip():
                return []
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, tuple | set):
            return list(value)
    return value


def _get_latest_dataset_artifact(execution_results: dict[str, Any]) -> tuple[str | None, str | None]:
    """Get the latest dataset-like path emitted by previous tool steps."""
    for result in reversed(list(execution_results.values())):
        if not isinstance(result, dict):
            continue

        annotations_path = result.get("annotations_path")
        if isinstance(annotations_path, str) and annotations_path.strip():
            annotation_format = result.get("annotation_format")
            return annotations_path, str(annotation_format) if annotation_format else None

        output_path = result.get("output_path")
        if isinstance(output_path, str) and output_path.strip():
            output_format = result.get("output_format") or result.get("format")
            return output_path, str(output_format) if output_format else None

    return None, None


def _resolve_export_parameters(
    parameters: dict[str, Any],
    execution_results: dict[str, Any],
) -> dict[str, Any]:
    """
    Resolve missing/invalid export source details from prior step outputs.

    This keeps execution resilient when planner output references raw image directories
    while a preceding detection step has already produced a labeled dataset.
    """
    resolved = dict(parameters)
    source_path_value = resolved.get("source_path")
    source_format_value = resolved.get("source_format")
    source_dir: Path | None = None

    if isinstance(source_path_value, str) and source_path_value.strip():
        source_dir = Path(source_path_value).expanduser()

    detected_source_format: str | None = None
    source_is_existing_dir = bool(source_dir and source_dir.exists() and source_dir.is_dir())
    if source_is_existing_dir and not source_format_value:
        detected_source_format = detect_format(source_dir)
        if detected_source_format:
            resolved["source_format"] = detected_source_format

    if source_is_existing_dir and (source_format_value or detected_source_format):
        return resolved

    artifact_path, artifact_format = _get_latest_dataset_artifact(execution_results)
    if not artifact_path:
        return resolved

    artifact_dir = Path(artifact_path).expanduser()
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        return resolved

    resolved["source_path"] = str(artifact_dir)
    if not source_format_value:
        if artifact_format:
            resolved["source_format"] = artifact_format
        else:
            detected_artifact_format = detect_format(artifact_dir)
            if detected_artifact_format:
                resolved["source_format"] = detected_artifact_format

    logger.info(
        "Resolved export source to previous artifact: %s (format=%s)",
        resolved.get("source_path"),
        resolved.get("source_format"),
    )
    return resolved


def _looks_like_inline_python_script(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    content = value.strip()
    if not content:
        return False

    if "\n" in content:
        return True

    if content.lower().startswith("#!/usr/bin/env python"):
        return True

    python_markers = ("def process(", "import ", "from ", "class ", "return ")
    return any(marker in content for marker in python_markers)


def _sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")
    return cleaned or "default"


def _default_run_script_output_path(input_path: str) -> str:
    input_p = Path(input_path).expanduser()
    if input_p.suffix:
        return str(input_p.with_name(f"{input_p.stem}_custom_output{input_p.suffix}"))
    return str(input_p.parent / f"{input_p.name}_custom_output")


def _prepare_run_script_parameters(
    step: dict[str, Any],
    parameters: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Normalize legacy run_script payloads into executable tool parameters."""
    resolved = dict(parameters)
    script_value = resolved.get("script")
    generated_code = step.get("generated_code")

    if (not isinstance(generated_code, str) or not generated_code.strip()) and _looks_like_inline_python_script(script_value):
        generated_code = str(script_value).strip()
        step_parameters = dict(step.get("parameters", {}))
        step_parameters.pop("script", None)
        step["parameters"] = step_parameters
        step["generated_code"] = generated_code

    if isinstance(generated_code, str) and generated_code.strip():
        pipeline_id = _sanitize_path_component(str(metadata.get("pipeline_id", "pipeline")))
        step_id = _sanitize_path_component(str(step.get("id", "step")))
        script_dir = GENERATED_SCRIPT_ROOT / pipeline_id
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / f"{step_id}.py"
        script_path.write_text(generated_code.strip(), encoding="utf-8")
        resolved["script_path"] = str(script_path)
        resolved.pop("script", None)
    elif "script_path" not in resolved and isinstance(script_value, str) and script_value.strip():
        # Backward compatibility for legacy planner shape (`script` as file path).
        resolved["script_path"] = script_value.strip()
        resolved.pop("script", None)

    if "output_path" not in resolved and isinstance(resolved.get("input_path"), str):
        input_path = resolved["input_path"].strip()
        if input_path:
            default_output = _default_run_script_output_path(input_path)
            resolved["output_path"] = default_output
            step_parameters = dict(step.get("parameters", {}))
            step_parameters.setdefault("output_path", default_output)
            step["parameters"] = step_parameters

    return resolved


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

    # Respect pre-marked skipped steps from edited plans.
    if step.get("status") == StepStatus.SKIPPED.value:
        now = datetime.now().isoformat()
        step["started_at"] = now
        step["completed_at"] = now
        execution_results[step.get("id", f"step-{current_idx}")] = {
            "status": StepStatus.SKIPPED.value
        }

        messages.append({
            "role": MessageRole.ASSISTANT.value,
            "content": f"Skipped step {current_idx + 1}: {step.get('description', 'Unnamed step')}",
            "timestamp": now,
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

    if tool_name == "run_script":
        parameters = _prepare_run_script_parameters(
            step,
            parameters,
            state.get("metadata", {}),
        )

    # Drop unsupported UI-only parameters and coerce common value shapes.
    parameter_defs = {param.name: param for param in getattr(tool, "parameters", [])}
    if parameter_defs:
        filtered_parameters: dict[str, Any] = {}
        dropped_parameters: list[str] = []
        for name, value in parameters.items():
            param_def = parameter_defs.get(name)
            if not param_def:
                dropped_parameters.append(name)
                continue
            coerced = _coerce_parameter_value(param_def, value)
            if coerced is not value:
                logger.debug(
                    "Coerced parameter %s for tool %s from %s to %s",
                    name,
                    tool_name,
                    type(value).__name__,
                    type(coerced).__name__,
                )
            filtered_parameters[name] = coerced

        if dropped_parameters:
            logger.debug(
                "Dropping unsupported parameters for tool %s: %s",
                tool_name,
                ", ".join(sorted(dropped_parameters)),
            )
        parameters = filtered_parameters

    if tool_name == "export":
        parameters = _resolve_export_parameters(parameters, execution_results)

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
