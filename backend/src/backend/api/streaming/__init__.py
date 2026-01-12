"""
SSE streaming module for real-time agent updates.

This module provides Server-Sent Events (SSE) streaming for the frontend
to receive real-time updates during agent execution, including:
- Agent state changes (thinking, planning, executing)
- Tool execution progress
- Checkpoint notifications
- Error and completion events
"""

from backend.api.streaming.batching import EventBatcher
from backend.api.streaming.events import (
    # Event data classes
    AwaitInputEventData,
    CheckpointEventData,
    ErrorEventData,
    MessageEventData,
    PipelineCompleteEventData,
    PlanEventData,
    SSEEvent,
    SSEEventType,
    StepEventData,
    ToolProgressEventData,
    ToolResultEventData,
    ToolStartEventData,
    # Builder functions
    checkpoint_event,
    complete_event,
    create_event,
    error_event,
    message_event,
    plan_event,
    thinking_event,
    tool_progress_event,
    tool_result_event,
    tool_start_event,
)

__all__ = [
    # Core types
    "SSEEventType",
    "SSEEvent",
    # Event data classes
    "MessageEventData",
    "PlanEventData",
    "ToolStartEventData",
    "ToolProgressEventData",
    "ToolResultEventData",
    "CheckpointEventData",
    "AwaitInputEventData",
    "StepEventData",
    "PipelineCompleteEventData",
    "ErrorEventData",
    # Builder functions
    "create_event",
    "message_event",
    "thinking_event",
    "plan_event",
    "tool_start_event",
    "tool_progress_event",
    "tool_result_event",
    "checkpoint_event",
    "error_event",
    "complete_event",
    # Batching
    "EventBatcher",
]
