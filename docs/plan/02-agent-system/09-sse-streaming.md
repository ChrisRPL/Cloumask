# SSE Streaming

> **Status:** 🟢 Complete (Implemented; checklist backfill pending)
> **Priority:** P0 (Critical)
> **Dependencies:** 02-langgraph-core, 04-agent-nodes-execution
> **Estimated Complexity:** Medium

## Overview

Implement Server-Sent Events (SSE) for real-time streaming of agent updates to the frontend. This includes defining a strict event schema, FastAPI endpoints, and handling for various event types.

## Goals

- [ ] Define strict JSON event schema for all event types
- [ ] FastAPI SSE endpoint for agent streaming
- [ ] Progress streaming during tool execution
- [ ] Checkpoint notification events
- [ ] Error and completion events
- [ ] Frontend connection handling guidance

## Technical Design

### Event Schema

```python
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union
import json


class SSEEventType(str, Enum):
    """Types of SSE events sent to frontend."""

    # Agent state events
    MESSAGE = "message"           # Chat message (user or assistant)
    THINKING = "thinking"         # Agent is processing
    PLAN = "plan"                 # Plan generated or updated
    PLAN_APPROVED = "plan_approved"

    # Tool events
    TOOL_START = "tool_start"     # Tool execution beginning
    TOOL_PROGRESS = "tool_progress"  # Progress update during tool
    TOOL_RESULT = "tool_result"   # Tool completed

    # Checkpoint events
    CHECKPOINT = "checkpoint"     # Checkpoint created
    AWAIT_INPUT = "await_input"   # Waiting for user input

    # Pipeline events
    STEP_START = "step_start"     # Pipeline step starting
    STEP_COMPLETE = "step_complete"  # Pipeline step done
    PIPELINE_COMPLETE = "complete"  # All done

    # Error events
    ERROR = "error"               # Error occurred
    WARNING = "warning"           # Non-fatal warning

    # Connection events
    CONNECTED = "connected"       # SSE connection established
    HEARTBEAT = "heartbeat"       # Keep-alive ping


@dataclass
class SSEEvent:
    """Base SSE event structure."""
    type: SSEEventType
    timestamp: str
    data: dict

    def to_sse(self) -> str:
        """Format as SSE message."""
        return f"event: {self.type.value}\ndata: {json.dumps(asdict(self))}\n\n"


# Specific event data structures

@dataclass
class MessageEventData:
    """Data for MESSAGE events."""
    role: str  # "user" | "assistant" | "system"
    content: str
    message_id: Optional[str] = None


@dataclass
class PlanEventData:
    """Data for PLAN events."""
    plan_id: str
    steps: list[dict]  # List of PipelineStep dicts
    total_steps: int


@dataclass
class ToolStartEventData:
    """Data for TOOL_START events."""
    tool_name: str
    step_index: int
    parameters: dict


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
    result: Optional[dict] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class CheckpointEventData:
    """Data for CHECKPOINT events."""
    checkpoint_id: str
    step_index: int
    trigger_reason: str
    progress_percent: float
    quality_metrics: dict
    message: str


@dataclass
class AwaitInputEventData:
    """Data for AWAIT_INPUT events."""
    input_type: str  # "plan_approval" | "checkpoint_approval" | "clarification"
    prompt: str
    options: Optional[list[str]] = None


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
    details: Optional[dict] = None
    recoverable: bool = True
```

### Event Builder

```python
from typing import Union


def create_event(
    event_type: SSEEventType,
    data: Union[dict, Any],
) -> SSEEvent:
    """Create an SSE event with current timestamp."""
    if hasattr(data, "__dict__"):
        data = asdict(data)
    elif not isinstance(data, dict):
        data = {"value": data}

    return SSEEvent(
        type=event_type,
        timestamp=datetime.now().isoformat(),
        data=data,
    )


# Convenience functions for common events

def message_event(role: str, content: str, message_id: str = None) -> SSEEvent:
    """Create a MESSAGE event."""
    return create_event(
        SSEEventType.MESSAGE,
        MessageEventData(role=role, content=content, message_id=message_id),
    )


def thinking_event(message: str = "Processing...") -> SSEEvent:
    """Create a THINKING event."""
    return create_event(SSEEventType.THINKING, {"message": message})


def plan_event(plan_id: str, steps: list[dict]) -> SSEEvent:
    """Create a PLAN event."""
    return create_event(
        SSEEventType.PLAN,
        PlanEventData(plan_id=plan_id, steps=steps, total_steps=len(steps)),
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


def checkpoint_event(checkpoint: dict, message: str) -> SSEEvent:
    """Create a CHECKPOINT event."""
    return create_event(
        SSEEventType.CHECKPOINT,
        CheckpointEventData(
            checkpoint_id=checkpoint["id"],
            step_index=checkpoint["step_index"],
            trigger_reason=checkpoint["trigger_reason"],
            progress_percent=checkpoint.get("progress_percent", 0),
            quality_metrics=checkpoint.get("quality_metrics", {}),
            message=message,
        ),
    )


def error_event(
    error_code: str,
    message: str,
    details: dict = None,
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


def complete_event(stats: dict) -> SSEEvent:
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
```

### FastAPI SSE Endpoint

```python
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from agent.graph import run_agent, compile_agent, create_agent_graph
from agent.state import PipelineState


router = APIRouter(prefix="/api", tags=["streaming"])

# Store active connections for cleanup
active_connections: dict[str, asyncio.Event] = {}


@router.get("/chat/stream/{thread_id}")
async def stream_chat(
    thread_id: str,
    request: Request,
) -> EventSourceResponse:
    """
    SSE endpoint for streaming agent responses.

    Frontend connects to this endpoint and receives real-time updates
    as the agent processes requests.
    """

    async def event_generator() -> AsyncGenerator[dict, None]:
        # Create cancel event for this connection
        cancel_event = asyncio.Event()
        active_connections[thread_id] = cancel_event

        try:
            # Send connected event
            yield {
                "event": SSEEventType.CONNECTED.value,
                "data": json.dumps({
                    "thread_id": thread_id,
                    "timestamp": datetime.now().isoformat(),
                }),
            }

            # Start heartbeat task
            heartbeat_task = asyncio.create_task(heartbeat_generator())

            # Wait for initial message or resume
            # This will be triggered by a POST to /chat/send
            while not cancel_event.is_set():
                if await request.is_disconnected():
                    break

                # Check for pending agent events
                event = await get_next_event(thread_id, timeout=1.0)
                if event:
                    yield {
                        "event": event.type.value,
                        "data": json.dumps(asdict(event)),
                    }

            heartbeat_task.cancel()

        finally:
            # Cleanup
            active_connections.pop(thread_id, None)

    return EventSourceResponse(event_generator())


async def heartbeat_generator() -> None:
    """Send heartbeat every 30 seconds."""
    while True:
        await asyncio.sleep(30)
        # Heartbeat is handled by the main generator


@router.post("/chat/send/{thread_id}")
async def send_message(
    thread_id: str,
    message: dict,  # {"content": str, "decision": Optional[str]}
) -> dict:
    """
    Send a message to the agent.

    This triggers agent processing, which streams events
    back through the SSE endpoint.
    """
    content = message.get("content", "")
    decision = message.get("decision")  # For checkpoint responses

    # Queue the message for processing
    await queue_agent_request(thread_id, content, decision)

    return {"status": "queued", "thread_id": thread_id}


# Agent event queue management

_event_queues: dict[str, asyncio.Queue] = {}


async def queue_agent_request(
    thread_id: str,
    content: str,
    decision: str = None,
) -> None:
    """Queue a request for agent processing."""

    if thread_id not in _event_queues:
        _event_queues[thread_id] = asyncio.Queue()

    # Start agent processing in background
    asyncio.create_task(
        process_agent_request(thread_id, content, decision)
    )


async def process_agent_request(
    thread_id: str,
    content: str,
    decision: str = None,
) -> None:
    """Process agent request and emit events."""

    queue = _event_queues.get(thread_id)
    if not queue:
        return

    try:
        # Emit thinking event
        await queue.put(thinking_event("Processing your request..."))

        # Run agent
        graph = create_agent_graph()
        compiled = compile_agent(graph)

        async for state_update in run_agent(content, thread_id, compiled):
            # Convert state updates to SSE events
            events = state_to_events(state_update)
            for event in events:
                await queue.put(event)

    except Exception as e:
        await queue.put(error_event(
            "AGENT_ERROR",
            str(e),
            recoverable=True,
        ))


async def get_next_event(thread_id: str, timeout: float = 1.0) -> Optional[SSEEvent]:
    """Get next event from queue with timeout."""
    queue = _event_queues.get(thread_id)
    if not queue:
        return None

    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


def state_to_events(state_update: dict) -> list[SSEEvent]:
    """Convert LangGraph state update to SSE events."""
    events = []

    # Check for new messages
    if "messages" in state_update:
        for msg in state_update.get("messages", [])[-1:]:  # Only last message
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            if content and not content.startswith("AWAIT_"):
                events.append(message_event(role, content))

    # Check for plan updates
    if "plan" in state_update and state_update.get("plan"):
        plan = state_update["plan"]
        if plan:
            events.append(plan_event(
                plan_id=state_update.get("metadata", {}).get("pipeline_id", ""),
                steps=plan,
            ))

    # Check for checkpoint
    if "checkpoints" in state_update:
        checkpoints = state_update.get("checkpoints", [])
        if checkpoints:
            latest = checkpoints[-1]
            if not latest.get("resolved_at"):
                events.append(checkpoint_event(
                    latest,
                    f"Checkpoint at step {latest['step_index']}",
                ))

    # Check for awaiting user
    if state_update.get("awaiting_user"):
        plan_approved = state_update.get("plan_approved", False)
        if not plan_approved:
            events.append(create_event(
                SSEEventType.AWAIT_INPUT,
                AwaitInputEventData(
                    input_type="plan_approval",
                    prompt="Review the plan and approve, edit, or cancel.",
                    options=["Approve", "Edit", "Cancel"],
                ),
            ))
        else:
            events.append(create_event(
                SSEEventType.AWAIT_INPUT,
                AwaitInputEventData(
                    input_type="checkpoint_approval",
                    prompt="Review the checkpoint and decide how to proceed.",
                    options=["Continue", "Edit Plan", "Cancel"],
                ),
            ))

    return events
```

### Batching and Rate Limiting

```python
import time
from collections import deque


class EventBatcher:
    """
    Batch rapid events to avoid overwhelming the frontend.

    Combines multiple TOOL_PROGRESS events within a window.
    """

    def __init__(self, batch_window_ms: int = 100):
        self.batch_window = batch_window_ms / 1000
        self._pending: deque = deque()
        self._last_emit: float = 0

    def add(self, event: SSEEvent) -> Optional[SSEEvent]:
        """
        Add event, return it if should emit immediately.

        Progress events are batched; other events emit immediately.
        """
        now = time.time()

        # Non-progress events emit immediately
        if event.type != SSEEventType.TOOL_PROGRESS:
            return event

        # Check if we should emit batched progress
        if now - self._last_emit >= self.batch_window:
            self._last_emit = now
            # Return latest progress (discard intermediate)
            self._pending.clear()
            return event

        # Queue for later
        self._pending.append(event)
        return None

    def flush(self) -> list[SSEEvent]:
        """Get all pending events."""
        events = list(self._pending)
        self._pending.clear()
        return events
```

## Implementation Tasks

- [ ] Create `backend/api/streaming/__init__.py`
- [ ] Create `backend/api/streaming/events.py`
  - [ ] Define `SSEEventType` enum
  - [ ] Define all event data classes
  - [ ] Implement `SSEEvent` class with `to_sse()` method
  - [ ] Implement event builder functions
- [ ] Create `backend/api/streaming/endpoints.py`
  - [ ] Implement `/chat/stream/{thread_id}` SSE endpoint
  - [ ] Implement `/chat/send/{thread_id}` POST endpoint
  - [ ] Implement event queue management
  - [ ] Implement state-to-events conversion
- [ ] Create `backend/api/streaming/batching.py`
  - [ ] Implement `EventBatcher` class
  - [ ] Add rate limiting logic
- [ ] Add `sse-starlette` to requirements

## Testing

### Unit Tests

```python
# tests/api/streaming/test_events.py

def test_sse_event_format():
    """SSE event should format correctly."""
    event = message_event("assistant", "Hello!")
    sse_str = event.to_sse()

    assert "event: message" in sse_str
    assert "data:" in sse_str
    assert "Hello!" in sse_str
    assert sse_str.endswith("\n\n")


def test_tool_progress_event():
    """Progress event should calculate percentage."""
    event = tool_progress_event("scan", 0, 50, 100, "Scanning...")

    assert event.data["percentage"] == 50.0
    assert event.data["message"] == "Scanning..."


def test_checkpoint_event():
    """Checkpoint event should include all fields."""
    checkpoint = {
        "id": "ckpt-123",
        "step_index": 2,
        "trigger_reason": "percentage",
        "quality_metrics": {"confidence": 0.9},
    }
    event = checkpoint_event(checkpoint, "Checkpoint reached")

    assert event.type == SSEEventType.CHECKPOINT
    assert event.data["checkpoint_id"] == "ckpt-123"


def test_error_event():
    """Error event should include error details."""
    event = error_event("TOOL_FAILED", "Scan failed", recoverable=True)

    assert event.type == SSEEventType.ERROR
    assert event.data["recoverable"] == True
```

### Batching Tests

```python
# tests/api/streaming/test_batching.py

def test_batcher_immediate_non_progress():
    """Non-progress events should emit immediately."""
    batcher = EventBatcher()
    event = message_event("assistant", "Hello")

    result = batcher.add(event)
    assert result is event


def test_batcher_batches_progress():
    """Rapid progress events should be batched."""
    batcher = EventBatcher(batch_window_ms=100)

    # First progress emits
    e1 = tool_progress_event("scan", 0, 1, 100)
    result1 = batcher.add(e1)
    assert result1 is e1

    # Rapid follow-up is batched
    e2 = tool_progress_event("scan", 0, 2, 100)
    result2 = batcher.add(e2)
    assert result2 is None

    # Flush returns pending
    pending = batcher.flush()
    assert len(pending) == 1
```

### Integration Tests

```python
# tests/api/streaming/test_endpoints.py

@pytest.mark.asyncio
async def test_sse_connection(client):
    """SSE endpoint should accept connection."""
    async with client.stream("GET", "/api/chat/stream/test-123") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Should receive connected event
        line = await response.aiter_lines().__anext__()
        assert "connected" in line


@pytest.mark.asyncio
async def test_send_message(client):
    """Send endpoint should queue message."""
    response = await client.post(
        "/api/chat/send/test-123",
        json={"content": "Scan /data"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_full_stream_flow(client):
    """Test full SSE flow with message."""
    events = []

    # Connect to SSE
    async with client.stream("GET", "/api/chat/stream/test-123") as stream:
        # Send message
        await client.post(
            "/api/chat/send/test-123",
            json={"content": "Hello"},
        )

        # Collect events
        async for line in stream.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
                if len(events) >= 3:
                    break

    # Should have connected, thinking, and message
    event_types = [e["type"] for e in events]
    assert "connected" in event_types
```

## Acceptance Criteria

- [ ] SSE connection established and maintained
- [ ] All event types properly formatted
- [ ] Message events show user and assistant messages
- [ ] Tool progress events update in real-time
- [ ] Checkpoint events pause for user input
- [ ] Error events include recovery information
- [ ] Connection handles disconnection gracefully
- [ ] Event batching prevents flooding
- [ ] Heartbeat keeps connection alive

## Files to Create/Modify

```
backend/
├── api/
│   └── streaming/
│       ├── __init__.py
│       ├── events.py      # Event types and builders
│       ├── endpoints.py   # FastAPI routes
│       └── batching.py    # Rate limiting
└── tests/
    └── api/
        └── streaming/
            ├── test_events.py
            ├── test_batching.py
            └── test_endpoints.py
```

## Dependencies

```
# requirements.txt additions
sse-starlette>=1.6.0
```

## Frontend Integration Notes

```typescript
// Example frontend SSE handling

const eventSource = new EventSource(`/api/chat/stream/${threadId}`);

eventSource.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  // Handle message event
  addMessage(data.data);
});

eventSource.addEventListener('tool_progress', (event) => {
  const data = JSON.parse(event.data);
  // Update progress bar
  updateProgress(data.data.percentage);
});

eventSource.addEventListener('checkpoint', (event) => {
  const data = JSON.parse(event.data);
  // Show checkpoint modal
  showCheckpointModal(data.data);
});

eventSource.addEventListener('error', (event) => {
  const data = JSON.parse(event.data);
  // Show error notification
  showError(data.data.message);
});

// Cleanup on unmount
eventSource.close();
```

## Notes

- SSE is simpler than WebSocket for this use case (one-way streaming)
- Consider adding compression for large payloads
- Frontend should handle reconnection on disconnect
- Event IDs could be added for replay capability
- Consider adding JWT auth to SSE endpoint
