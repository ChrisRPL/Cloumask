"""
SSE event types and builder functions.

Defines a strict JSON event schema for all event types sent to the frontend
during agent execution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SSEEventType(str, Enum):
    """Types of SSE events sent to frontend."""

    # Agent state events
    MESSAGE = "message"  # Chat message (user or assistant)
    THINKING = "thinking"  # Agent is processing
    PLAN = "plan"  # Plan generated or updated
    PLAN_APPROVED = "plan_approved"  # Plan was approved by user

    # Tool events
    TOOL_START = "tool_start"  # Tool execution beginning
    TOOL_PROGRESS = "tool_progress"  # Progress update during tool
    TOOL_RESULT = "tool_result"  # Tool completed

    # Checkpoint events
    CHECKPOINT = "checkpoint"  # Checkpoint created
    AWAIT_INPUT = "await_input"  # Waiting for user input

    # Pipeline events
    STEP_START = "step_start"  # Pipeline step starting
    STEP_COMPLETE = "step_complete"  # Pipeline step done
    PIPELINE_COMPLETE = "complete"  # All done

    # Error events
    ERROR = "error"  # Error occurred
    WARNING = "warning"  # Non-fatal warning

    # Connection events
    CONNECTED = "connected"  # SSE connection established
    HEARTBEAT = "heartbeat"  # Keep-alive ping


@dataclass
class SSEEvent:
    """Base SSE event structure."""

    type: SSEEventType
    timestamp: str
    data: dict[str, Any]

    def to_sse(self) -> str:
        """Format as SSE message."""
        return f"event: {self.type.value}\ndata: {json.dumps(asdict(self))}\n\n"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data,
        }


# -----------------------------------------------------------------------------
# Specific Event Data Structures
# -----------------------------------------------------------------------------


@dataclass
class MessageEventData:
    """Data for MESSAGE events."""

    role: str  # "user" | "assistant" | "system"
    content: str
    message_id: str | None = None


@dataclass
class PlanEventData:
    """Data for PLAN events."""

    plan_id: str
    steps: list[dict[str, Any]]
    total_steps: int


@dataclass
class ToolStartEventData:
    """Data for TOOL_START events."""

    tool_name: str
    step_index: int
    parameters: dict[str, Any]


@dataclass
class ToolProgressEventData:
    """Data for TOOL_PROGRESS events."""

    tool_name: str
    step_index: int
    current: int
    total: int
    message: str
    percentage: float


@dataclass
class ToolResultEventData:
    """Data for TOOL_RESULT events."""

    tool_name: str
    step_index: int
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class CheckpointEventData:
    """Data for CHECKPOINT events."""

    checkpoint_id: str
    step_index: int
    trigger_reason: str
    progress_percent: float
    quality_metrics: dict[str, Any]
    message: str


@dataclass
class AwaitInputEventData:
    """Data for AWAIT_INPUT events."""

    input_type: str  # "plan_approval" | "checkpoint_approval" | "clarification"
    prompt: str
    options: list[str] | None = None


@dataclass
class StepEventData:
    """Data for STEP_START and STEP_COMPLETE events."""

    step_index: int
    step_id: str
    tool_name: str
    description: str
    status: str


@dataclass
class PipelineCompleteEventData:
    """Data for PIPELINE_COMPLETE events."""

    pipeline_id: str
    success: bool
    total_steps: int
    completed_steps: int
    failed_steps: int
    duration_seconds: float
    summary: str


@dataclass
class ErrorEventData:
    """Data for ERROR events."""

    error_code: str
    message: str
    details: dict[str, Any] | None = None
    recoverable: bool = True


@dataclass
class ConnectedEventData:
    """Data for CONNECTED events."""

    thread_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HeartbeatEventData:
    """Data for HEARTBEAT events."""

    sequence: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# -----------------------------------------------------------------------------
# Event Builder Functions
# -----------------------------------------------------------------------------


def create_event(
    event_type: SSEEventType,
    data: dict[str, Any] | Any,
) -> SSEEvent:
    """
    Create an SSE event with current timestamp.

    Args:
        event_type: The type of SSE event.
        data: Event data, either a dict or a dataclass instance.

    Returns:
        SSEEvent ready to send.
    """
    event_data: dict[str, Any]
    if hasattr(data, "__dataclass_fields__"):
        event_data = asdict(data)  # type: ignore[arg-type]
    elif isinstance(data, dict):
        event_data = data
    else:
        event_data = {"value": data}

    return SSEEvent(
        type=event_type,
        timestamp=datetime.now().isoformat(),
        data=event_data,
    )


def message_event(
    role: str,
    content: str,
    message_id: str | None = None,
) -> SSEEvent:
    """Create a MESSAGE event."""
    return create_event(
        SSEEventType.MESSAGE,
        MessageEventData(role=role, content=content, message_id=message_id),
    )


def thinking_event(message: str = "Processing...") -> SSEEvent:
    """Create a THINKING event."""
    return create_event(SSEEventType.THINKING, {"message": message})


def plan_event(plan_id: str, steps: list[dict[str, Any]]) -> SSEEvent:
    """Create a PLAN event."""
    return create_event(
        SSEEventType.PLAN,
        PlanEventData(plan_id=plan_id, steps=steps, total_steps=len(steps)),
    )


def tool_start_event(
    tool_name: str,
    step_index: int,
    parameters: dict[str, Any],
) -> SSEEvent:
    """Create a TOOL_START event."""
    return create_event(
        SSEEventType.TOOL_START,
        ToolStartEventData(
            tool_name=tool_name,
            step_index=step_index,
            parameters=parameters,
        ),
    )


def tool_progress_event(
    tool_name: str,
    step_index: int,
    current: int,
    total: int,
    message: str = "",
) -> SSEEvent:
    """Create a TOOL_PROGRESS event."""
    percentage = (current / total * 100) if total > 0 else 0
    return create_event(
        SSEEventType.TOOL_PROGRESS,
        ToolProgressEventData(
            tool_name=tool_name,
            step_index=step_index,
            current=current,
            total=total,
            message=message,
            percentage=percentage,
        ),
    )


def tool_result_event(
    tool_name: str,
    step_index: int,
    success: bool,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    duration_seconds: float = 0.0,
) -> SSEEvent:
    """Create a TOOL_RESULT event."""
    return create_event(
        SSEEventType.TOOL_RESULT,
        ToolResultEventData(
            tool_name=tool_name,
            step_index=step_index,
            success=success,
            result=result,
            error=error,
            duration_seconds=duration_seconds,
        ),
    )


def checkpoint_event(checkpoint: dict[str, Any], message: str) -> SSEEvent:
    """Create a CHECKPOINT event."""
    return create_event(
        SSEEventType.CHECKPOINT,
        CheckpointEventData(
            checkpoint_id=checkpoint.get("id", ""),
            step_index=checkpoint.get("step_index", 0),
            trigger_reason=checkpoint.get("trigger_reason", ""),
            progress_percent=checkpoint.get("progress_percent", 0),
            quality_metrics=checkpoint.get("quality_metrics", {}),
            message=message,
        ),
    )


def await_input_event(
    input_type: str,
    prompt: str,
    options: list[str] | None = None,
) -> SSEEvent:
    """Create an AWAIT_INPUT event."""
    return create_event(
        SSEEventType.AWAIT_INPUT,
        AwaitInputEventData(
            input_type=input_type,
            prompt=prompt,
            options=options,
        ),
    )


def step_start_event(
    step_index: int,
    step_id: str,
    tool_name: str,
    description: str,
) -> SSEEvent:
    """Create a STEP_START event."""
    return create_event(
        SSEEventType.STEP_START,
        StepEventData(
            step_index=step_index,
            step_id=step_id,
            tool_name=tool_name,
            description=description,
            status="running",
        ),
    )


def step_complete_event(
    step_index: int,
    step_id: str,
    tool_name: str,
    description: str,
    status: str,
) -> SSEEvent:
    """Create a STEP_COMPLETE event."""
    return create_event(
        SSEEventType.STEP_COMPLETE,
        StepEventData(
            step_index=step_index,
            step_id=step_id,
            tool_name=tool_name,
            description=description,
            status=status,
        ),
    )


def error_event(
    error_code: str,
    message: str,
    details: dict[str, Any] | None = None,
    recoverable: bool = True,
) -> SSEEvent:
    """Create an ERROR event."""
    return create_event(
        SSEEventType.ERROR,
        ErrorEventData(
            error_code=error_code,
            message=message,
            details=details,
            recoverable=recoverable,
        ),
    )


def warning_event(
    message: str,
    details: dict[str, Any] | None = None,
) -> SSEEvent:
    """Create a WARNING event."""
    return create_event(
        SSEEventType.WARNING,
        {"message": message, "details": details},
    )


def complete_event(stats: dict[str, Any]) -> SSEEvent:
    """Create a PIPELINE_COMPLETE event."""
    return create_event(
        SSEEventType.PIPELINE_COMPLETE,
        PipelineCompleteEventData(
            pipeline_id=stats.get("pipeline_id", ""),
            success=stats.get("failed_steps", 0) == 0,
            total_steps=stats.get("total_steps", 0),
            completed_steps=stats.get("completed_steps", 0),
            failed_steps=stats.get("failed_steps", 0),
            duration_seconds=stats.get("total_duration_seconds", 0),
            summary=stats.get("summary", ""),
        ),
    )


def connected_event(thread_id: str) -> SSEEvent:
    """Create a CONNECTED event."""
    return create_event(
        SSEEventType.CONNECTED,
        ConnectedEventData(thread_id=thread_id),
    )


def heartbeat_event(sequence: int) -> SSEEvent:
    """Create a HEARTBEAT event."""
    return create_event(
        SSEEventType.HEARTBEAT,
        HeartbeatEventData(sequence=sequence),
    )
