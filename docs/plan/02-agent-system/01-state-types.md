# State Types

> **Status:** 🟢 Complete (Implemented; checklist backfill pending)
> **Priority:** P0 (Critical)
> **Dependencies:** 01-foundation (Python sidecar running)
> **Estimated Complexity:** Low

## Overview

Define the core type system for the LangGraph agent. These TypedDict and dataclass definitions form the contract between all agent nodes, tools, and the persistence layer.

## Goals

- [ ] Define `PipelineState` as the central LangGraph state
- [ ] Create message types for conversation history
- [ ] Define pipeline step representation
- [ ] Create checkpoint and quality metrics types
- [ ] Ensure all types are JSON-serializable for persistence

## Technical Design

### Type Hierarchy

```
PipelineState (TypedDict)
├── messages: list[Message]
├── plan: list[PipelineStep]
├── current_step: int
├── execution_results: dict[str, StepResult]
├── checkpoints: list[Checkpoint]
├── user_feedback: Optional[UserFeedback]
└── metadata: PipelineMetadata
```

### Core Types

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TypedDict


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents an LLM's request to call a tool."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A single message in the conversation history."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None  # For tool response messages

    def to_dict(self) -> dict:
        """Serialize for LangGraph state."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ],
            "tool_call_id": self.tool_call_id,
        }


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """A single step in the execution plan."""
    id: str
    tool_name: str
    parameters: dict[str, Any]
    description: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class CheckpointTrigger(str, Enum):
    PERCENTAGE = "percentage"      # 10%, 25%, 50% progress
    QUALITY_DROP = "quality_drop"  # Confidence drop >15%
    ERROR_RATE = "error_rate"      # Error rate >5%
    CRITICAL_STEP = "critical_step"  # After anonymize, segment, detect_3d


class UserDecision(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    CANCEL = "cancel"
    RETRY = "retry"


@dataclass
class QualityMetrics:
    """Quality metrics captured at a checkpoint."""
    average_confidence: float
    error_count: int
    total_processed: int
    processing_speed: float  # items per second

    @property
    def error_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.error_count / self.total_processed


@dataclass
class Checkpoint:
    """A human-in-the-loop checkpoint."""
    id: str
    step_index: int
    trigger_reason: CheckpointTrigger
    quality_metrics: QualityMetrics
    created_at: datetime = field(default_factory=datetime.now)
    user_decision: Optional[UserDecision] = None
    user_feedback: Optional[str] = None
    resolved_at: Optional[datetime] = None


@dataclass
class UserFeedback:
    """User's response to a checkpoint or plan review."""
    decision: UserDecision
    message: Optional[str] = None
    plan_edits: Optional[list[dict]] = None  # For plan modifications


@dataclass
class PipelineMetadata:
    """Metadata about the pipeline execution."""
    pipeline_id: str
    created_at: datetime
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    total_files: int = 0
    processed_files: int = 0


class PipelineState(TypedDict, total=False):
    """
    The central state for the LangGraph agent.

    This TypedDict is passed between all nodes and persisted
    to SQLite for resume capability.
    """
    # Conversation
    messages: list[dict]  # Serialized Message objects

    # Planning
    plan: list[dict]  # Serialized PipelineStep objects
    plan_approved: bool

    # Execution
    current_step: int
    execution_results: dict[str, dict]

    # Checkpoints
    checkpoints: list[dict]  # Serialized Checkpoint objects
    awaiting_user: bool

    # Metadata
    metadata: dict

    # Error handling
    last_error: Optional[str]
    retry_count: int
```

### Serialization Helpers

```python
def serialize_state(state: PipelineState) -> str:
    """Convert state to JSON string for persistence."""
    import json
    return json.dumps(state, default=str)


def deserialize_state(json_str: str) -> PipelineState:
    """Restore state from JSON string."""
    import json
    return json.loads(json_str)


def create_initial_state(user_message: str, pipeline_id: str) -> PipelineState:
    """Create a fresh pipeline state from a user request."""
    from uuid import uuid4

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
        metadata=metadata.__dict__,
        last_error=None,
        retry_count=0,
    )
```

## Implementation Tasks

- [ ] Create `backend/agent/state.py` module
- [ ] Implement all Enum classes
- [ ] Implement Message dataclass with serialization
- [ ] Implement PipelineStep dataclass
- [ ] Implement QualityMetrics dataclass
- [ ] Implement Checkpoint dataclass
- [ ] Implement UserFeedback dataclass
- [ ] Implement PipelineMetadata dataclass
- [ ] Implement PipelineState TypedDict
- [ ] Add serialization/deserialization helpers
- [ ] Add `create_initial_state` factory function

## Testing

### Unit Tests

```python
# tests/agent/test_state.py

def test_message_serialization():
    """Message should serialize to dict and back."""
    msg = Message(role=MessageRole.USER, content="Hello")
    data = msg.to_dict()
    assert data["role"] == "user"
    assert data["content"] == "Hello"
    assert "timestamp" in data


def test_pipeline_step_duration():
    """Duration should calculate correctly."""
    step = PipelineStep(
        id="step-1",
        tool_name="scan_directory",
        parameters={"path": "/data"},
        description="Scan input directory",
    )
    assert step.duration_seconds is None

    step.started_at = datetime(2026, 1, 1, 10, 0, 0)
    step.completed_at = datetime(2026, 1, 1, 10, 0, 30)
    assert step.duration_seconds == 30.0


def test_quality_metrics_error_rate():
    """Error rate should calculate correctly."""
    metrics = QualityMetrics(
        average_confidence=0.85,
        error_count=5,
        total_processed=100,
        processing_speed=10.0,
    )
    assert metrics.error_rate == 0.05


def test_create_initial_state():
    """Initial state should have correct structure."""
    state = create_initial_state("Scan /data", "pipe-123")
    assert len(state["messages"]) == 1
    assert state["messages"][0]["role"] == "user"
    assert state["plan"] == []
    assert state["current_step"] == 0
    assert state["awaiting_user"] == False


def test_state_json_serializable():
    """Entire state should be JSON-serializable."""
    state = create_initial_state("Test message", "test-id")
    json_str = serialize_state(state)
    restored = deserialize_state(json_str)
    assert restored["messages"] == state["messages"]
```

### Edge Cases

- [ ] Empty messages list
- [ ] Unicode content in messages
- [ ] Very long content strings
- [ ] Missing optional fields
- [ ] Invalid enum values (should raise)

## Acceptance Criteria

- [ ] All types can be instantiated with valid data
- [ ] All types serialize to JSON without errors
- [ ] All types can be deserialized from JSON
- [ ] Enums have correct string values
- [ ] Duration calculations work correctly
- [ ] Error rate calculations handle division by zero
- [ ] Unit tests pass with >90% coverage

## Files to Create/Modify

```
backend/
├── agent/
│   ├── __init__.py          # Export state types
│   └── state.py              # All type definitions
└── tests/
    └── agent/
        └── test_state.py     # Unit tests
```

## Notes

- Using `TypedDict` for `PipelineState` instead of dataclass because LangGraph requires dict-like state
- All datetime objects use ISO format when serialized
- Consider adding Pydantic models later for validation, but start with dataclasses for simplicity
