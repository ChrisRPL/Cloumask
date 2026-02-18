"""
LangGraph agent state types for Cloumask pipeline execution.

This module defines the core type system for the agent state machine,
including message types, pipeline steps, checkpoints, and the central
PipelineState TypedDict.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict, cast

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class StepStatus(str, Enum):
    """Execution status of a pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckpointTrigger(str, Enum):
    """Reason a checkpoint was triggered."""

    PERCENTAGE = "percentage"
    QUALITY_DROP = "quality_drop"
    ERROR_RATE = "error_rate"
    CRITICAL_STEP = "critical_step"


class UserDecision(str, Enum):
    """User's decision at a checkpoint or plan review."""

    APPROVE = "approve"
    EDIT = "edit"
    CANCEL = "cancel"
    RETRY = "retry"


# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------


class ToolCall(BaseModel):
    """Represents an LLM's request to call a tool."""

    id: str = Field(description="Unique identifier for this tool call")
    name: str = Field(description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class Message(BaseModel):
    """A single message in the conversation history."""

    role: MessageRole = Field(description="Role of the message sender")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="When message was created")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls requested")
    tool_call_id: str | None = Field(default=None, description="ID of tool call this responds to")

    def to_dict(self) -> dict[str, Any]:
        """Serialize for LangGraph state storage."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tool_calls": [tc.model_dump() for tc in self.tool_calls],
            "tool_call_id": self.tool_call_id,
        }


class PipelineStep(BaseModel):
    """A single step in the execution plan."""

    id: str = Field(description="Unique step identifier")
    tool_name: str = Field(description="Name of the tool to execute")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    generated_code: str | None = Field(
        default=None,
        description="Inline generated Python code for custom script steps",
    )
    description: str = Field(description="Human-readable step description")
    status: StepStatus = Field(default=StepStatus.PENDING, description="Current status")
    result: dict[str, Any] | None = Field(default=None, description="Execution result")
    error: str | None = Field(default=None, description="Error message if failed")
    started_at: datetime | None = Field(default=None, description="When execution started")
    completed_at: datetime | None = Field(default=None, description="When execution completed")

    @property
    def duration_seconds(self) -> float | None:
        """Calculate step duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class QualityMetrics(BaseModel):
    """Quality metrics captured at a checkpoint."""

    average_confidence: float = Field(ge=0.0, le=1.0, description="Average confidence score")
    error_count: int = Field(ge=0, description="Number of errors encountered")
    total_processed: int = Field(ge=0, description="Total items processed")
    processing_speed: float = Field(ge=0.0, description="Items per second")

    @property
    def error_rate(self) -> float:
        """Calculate error rate as a fraction."""
        if self.total_processed == 0:
            return 0.0
        return self.error_count / self.total_processed


class Checkpoint(BaseModel):
    """A human-in-the-loop checkpoint."""

    id: str = Field(description="Unique checkpoint identifier")
    step_index: int = Field(ge=0, description="Index of the step that triggered checkpoint")
    trigger_reason: CheckpointTrigger = Field(description="Why checkpoint was triggered")
    quality_metrics: QualityMetrics = Field(description="Metrics at checkpoint time")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When checkpoint was created"
    )
    user_decision: UserDecision | None = Field(default=None, description="User's decision")
    user_feedback: str | None = Field(default=None, description="User's feedback message")
    resolved_at: datetime | None = Field(default=None, description="When checkpoint was resolved")


class UserFeedback(BaseModel):
    """User's response to a checkpoint or plan review."""

    decision: UserDecision = Field(description="The user's decision")
    message: str | None = Field(default=None, description="Optional feedback message")
    plan_edits: list[dict[str, Any]] | None = Field(default=None, description="Plan modifications")


class PipelineMetadata(BaseModel):
    """Metadata about the pipeline execution."""

    pipeline_id: str = Field(description="Unique pipeline identifier")
    created_at: datetime = Field(description="When pipeline was created")
    input_path: str | None = Field(default=None, description="Input data path")
    output_path: str | None = Field(default=None, description="Output data path")
    total_files: int = Field(default=0, ge=0, description="Total files to process")
    processed_files: int = Field(default=0, ge=0, description="Files processed so far")


# -----------------------------------------------------------------------------
# TypedDict for LangGraph State
# -----------------------------------------------------------------------------


class PipelineState(TypedDict, total=False):
    """
    The central state for the LangGraph agent.

    This TypedDict is passed between all nodes and persisted
    to SQLite for resume capability. Uses dict serialization
    of Pydantic models for compatibility.
    """

    # Conversation
    messages: list[dict[str, Any]]  # Serialized Message objects

    # Planning
    plan: list[dict[str, Any]]  # Serialized PipelineStep objects
    plan_approved: bool

    # Execution
    current_step: int
    execution_results: dict[str, dict[str, Any]]

    # Checkpoints
    checkpoints: list[dict[str, Any]]  # Serialized Checkpoint objects
    awaiting_user: bool

    # Metadata
    metadata: dict[str, Any]  # Serialized PipelineMetadata

    # Error handling
    last_error: str | None
    retry_count: int


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def serialize_state(state: PipelineState) -> str:
    """
    Convert state to JSON string for persistence.

    Args:
        state: The pipeline state to serialize.

    Returns:
        JSON string representation of the state.
    """
    return json.dumps(state, default=str)


def deserialize_state(json_str: str) -> PipelineState:
    """
    Restore state from JSON string.

    Args:
        json_str: JSON string to deserialize.

    Returns:
        Restored PipelineState.
    """
    return cast(PipelineState, json.loads(json_str))


def create_initial_state(user_message: str, pipeline_id: str) -> PipelineState:
    """
    Create a fresh pipeline state from a user request.

    Args:
        user_message: The user's initial request.
        pipeline_id: Unique identifier for this pipeline.

    Returns:
        A new PipelineState ready for processing.
    """
    initial_message = Message(
        role=MessageRole.USER,
        content=user_message,
    )

    metadata = PipelineMetadata(
        pipeline_id=pipeline_id,
        created_at=datetime.now(),
    )

    return PipelineState(
        messages=[initial_message.to_dict()],
        plan=[],
        plan_approved=False,
        current_step=0,
        execution_results={},
        checkpoints=[],
        awaiting_user=False,
        metadata=metadata.model_dump(mode="json"),
        last_error=None,
        retry_count=0,
    )
