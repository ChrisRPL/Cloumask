"""
FastAPI SSE endpoints for agent streaming.

Provides endpoints for:
- SSE connection for receiving real-time agent updates
- Sending messages to the agent
- Resuming from checkpoints
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.agent.graph import compile_agent, run_agent
from backend.agent.state import (
    PipelineState,
    StepStatus,
    UserDecision,
    create_initial_state,
)
from backend.api.streaming.events import (
    SSEEvent,
    await_input_event,
    checkpoint_event,
    complete_event,
    connected_event,
    error_event,
    heartbeat_event,
    message_event,
    plan_event,
    step_complete_event,
    step_start_event,
    thinking_event,
    tool_result_event,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["streaming"])

# Store active connections for cleanup
_active_connections: dict[str, asyncio.Event] = {}

# Event queues for each thread
_event_queues: dict[str, asyncio.Queue[SSEEvent]] = {}

# Track thread state for resumption
_thread_states: dict[str, dict[str, Any]] = {}


class SendMessageRequest(BaseModel):
    """Request body for sending a message."""

    content: str = Field(..., description="The message content")
    decision: UserDecision | None = Field(
        default=None,
        description="User decision for checkpoint/approval responses",
    )
    plan_edits: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional plan modifications when decision is EDIT",
    )


class SendMessageResponse(BaseModel):
    """Response for send message endpoint."""

    status: str = Field(..., description="Request status (queued, error)")
    thread_id: str = Field(..., description="Thread identifier")
    message_id: str | None = Field(default=None, description="Assigned message ID")


class ThreadInfo(BaseModel):
    """Information about a chat thread."""

    thread_id: str
    created: bool
    awaiting_user: bool = False
    current_step: int = 0
    total_steps: int = 0


@router.get("/stream/{thread_id}")
async def stream_chat(
    thread_id: str,
    request: Request,
) -> EventSourceResponse:
    """
    SSE endpoint for streaming agent responses.

    Frontend connects to this endpoint and receives real-time updates
    as the agent processes requests.

    Args:
        thread_id: Unique identifier for this chat thread.
        request: FastAPI request object for disconnect detection.

    Returns:
        EventSourceResponse for SSE streaming.
    """

    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        # Create cancel event for this connection
        cancel_event = asyncio.Event()
        _active_connections[thread_id] = cancel_event

        # Ensure we have a queue for this thread
        if thread_id not in _event_queues:
            _event_queues[thread_id] = asyncio.Queue()

        queue = _event_queues[thread_id]
        heartbeat_sequence = 0

        try:
            # Send connected event
            connected = connected_event(thread_id)
            yield {
                "event": connected.type.value,
                "data": json.dumps(connected.to_dict()),
            }

            # Main event loop
            while not cancel_event.is_set():
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from thread {thread_id}")
                    break

                try:
                    # Wait for events with timeout for heartbeat
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {
                        "event": event.type.value,
                        "data": json.dumps(event.to_dict()),
                    }
                except TimeoutError:
                    # Send heartbeat
                    heartbeat_sequence += 1
                    heartbeat = heartbeat_event(heartbeat_sequence)
                    yield {
                        "event": heartbeat.type.value,
                        "data": json.dumps(heartbeat.to_dict()),
                    }

        except asyncio.CancelledError:
            logger.info(f"SSE connection cancelled for thread {thread_id}")
        finally:
            # Cleanup
            _active_connections.pop(thread_id, None)
            logger.info(f"SSE cleanup complete for thread {thread_id}")

    return EventSourceResponse(event_generator())


@router.post("/send/{thread_id}")
async def send_message(
    thread_id: str,
    message: SendMessageRequest,
) -> SendMessageResponse:
    """
    Send a message to the agent.

    This triggers agent processing, which streams events
    back through the SSE endpoint.

    Args:
        thread_id: Thread to send message to.
        message: Message content and optional decision.

    Returns:
        Status response.
    """
    message_id = str(uuid.uuid4())

    # Queue the message for processing
    await queue_agent_request(
        thread_id=thread_id,
        content=message.content,
        decision=message.decision,
        plan_edits=message.plan_edits,
    )

    return SendMessageResponse(
        status="queued",
        thread_id=thread_id,
        message_id=message_id,
    )


@router.post("/thread/new")
async def create_thread() -> ThreadInfo:
    """
    Create a new chat thread.

    Returns:
        Thread information with new thread_id.
    """
    thread_id = str(uuid.uuid4())
    _event_queues[thread_id] = asyncio.Queue()

    return ThreadInfo(
        thread_id=thread_id,
        created=True,
    )


@router.get("/thread/{thread_id}")
async def get_thread_info(thread_id: str) -> ThreadInfo:
    """
    Get information about a chat thread.

    Args:
        thread_id: Thread identifier.

    Returns:
        Thread information.

    Raises:
        HTTPException: If thread not found.
    """
    if thread_id not in _event_queues:
        raise HTTPException(status_code=404, detail="Thread not found")

    state = _thread_states.get(thread_id)
    plan = state.get("plan", []) if state else []

    return ThreadInfo(
        thread_id=thread_id,
        created=False,
        awaiting_user=state.get("awaiting_user", False) if state else False,
        current_step=state.get("current_step", 0) if state else 0,
        total_steps=len(plan),
    )


@router.delete("/thread/{thread_id}")
async def close_thread(thread_id: str) -> dict[str, str]:
    """
    Close a chat thread and cleanup resources.

    Args:
        thread_id: Thread to close.

    Returns:
        Confirmation message.
    """
    # Signal cancellation if connected
    if thread_id in _active_connections:
        _active_connections[thread_id].set()
        _active_connections.pop(thread_id, None)

    # Cleanup queues and state
    _event_queues.pop(thread_id, None)
    _thread_states.pop(thread_id, None)

    return {"status": "closed", "thread_id": thread_id}


# -----------------------------------------------------------------------------
# Agent Event Queue Management
# -----------------------------------------------------------------------------


async def queue_agent_request(
    thread_id: str,
    content: str,
    decision: UserDecision | None = None,
    plan_edits: list[dict[str, Any]] | None = None,
) -> None:
    """
    Queue a request for agent processing.

    Args:
        thread_id: Thread identifier.
        content: Message content.
        decision: Optional user decision for checkpoints.
        plan_edits: Optional plan modifications.
    """
    if thread_id not in _event_queues:
        _event_queues[thread_id] = asyncio.Queue()

    # Start agent processing in background
    asyncio.create_task(
        process_agent_request(thread_id, content, decision, plan_edits)
    )


async def process_agent_request(
    thread_id: str,
    content: str,
    decision: UserDecision | None = None,
    plan_edits: list[dict[str, Any]] | None = None,
) -> None:
    """
    Process agent request and emit events.

    Args:
        thread_id: Thread identifier.
        content: Message content.
        decision: Optional user decision.
        plan_edits: Optional plan modifications.
    """
    queue = _event_queues.get(thread_id)
    if not queue:
        return

    try:
        # Emit thinking event
        await queue.put(thinking_event("Processing your request..."))

        # Get or create initial state
        initial_state: PipelineState
        if thread_id in _thread_states and decision is not None:
            # Resume from existing state with decision
            initial_state = _thread_states[thread_id]  # type: ignore[assignment]
            # Update state with user decision
            await _apply_user_decision(initial_state, decision, content, plan_edits)
        else:
            # New request - create initial state
            pipeline_id = str(uuid.uuid4())
            initial_state = create_initial_state(content, pipeline_id)

        # Run agent and stream events
        async with compile_agent(":memory:") as compiled:
            async for state_update in run_agent(compiled, initial_state, thread_id):
                # Store latest state
                _thread_states[thread_id] = state_update

                # Convert state updates to SSE events
                events = state_to_events(state_update, thread_id)
                for event in events:
                    await queue.put(event)

                # Check if we should pause for user input
                if state_update.get("awaiting_user", False):
                    break

    except Exception as e:
        logger.exception(f"Error processing agent request for {thread_id}")
        await queue.put(
            error_event(
                "AGENT_ERROR",
                str(e),
                recoverable=True,
            )
        )


async def _apply_user_decision(
    state: PipelineState,
    decision: UserDecision,
    message: str,
    plan_edits: list[dict[str, Any]] | None,
) -> None:
    """
    Apply user decision to state.

    Args:
        state: Current pipeline state.
        decision: User's decision.
        message: User's message.
        plan_edits: Optional plan modifications.
    """
    # Update the last checkpoint with decision
    checkpoints = state.get("checkpoints", [])
    if checkpoints:
        checkpoints[-1]["user_decision"] = decision.value
        checkpoints[-1]["user_feedback"] = message

    # Clear awaiting flag
    state["awaiting_user"] = False

    # Apply plan edits if provided
    if plan_edits and decision == UserDecision.EDIT:
        state["plan"] = plan_edits

    # If approving plan for first time
    if not state.get("plan_approved", False) and decision == UserDecision.APPROVE:
        state["plan_approved"] = True


def state_to_events(state_update: dict[str, Any], thread_id: str) -> list[SSEEvent]:
    """
    Convert LangGraph state update to SSE events.

    Analyzes the state to determine which events should be emitted.

    Args:
        state_update: Current pipeline state dictionary.
        thread_id: Thread identifier for event context.

    Returns:
        List of SSE events to emit.
    """
    events: list[SSEEvent] = []

    # Check for new messages
    messages = state_update.get("messages", [])
    if messages:
        last_msg = messages[-1]
        role = last_msg.get("role", "assistant")
        content = last_msg.get("content", "")
        # Only emit if content exists and isn't an internal marker
        if content and not content.startswith("AWAIT_"):
            events.append(message_event(role, content))

    # Check for plan updates
    plan = state_update.get("plan", [])
    if plan and not state_update.get("plan_approved", False):
        pipeline_id = (
            state_update.get("metadata", {}).get("pipeline_id", "") or thread_id
        )
        events.append(plan_event(pipeline_id, plan))

    # Check for current step changes (execution progress)
    current_step = state_update.get("current_step", 0)
    if current_step > 0 and plan:
        # Emit step completion for the just-completed step
        if current_step <= len(plan):
            completed_step_data = plan[current_step - 1]
            step_status = completed_step_data.get("status", "completed")

            # Emit step complete event
            events.append(
                step_complete_event(
                    step_index=current_step - 1,
                    step_id=completed_step_data.get("id", f"step-{current_step - 1}"),
                    tool_name=completed_step_data.get("tool_name", ""),
                    description=completed_step_data.get("description", ""),
                    status=step_status,
                )
            )

            # Emit tool result
            result = completed_step_data.get("result")
            error = completed_step_data.get("error")
            success = step_status == StepStatus.COMPLETED.value

            events.append(
                tool_result_event(
                    tool_name=completed_step_data.get("tool_name", ""),
                    step_index=current_step - 1,
                    success=success,
                    result=result,
                    error=error,
                )
            )

        # Emit step start for the next step if not complete
        if current_step < len(plan):
            next_step_data = plan[current_step]
            events.append(
                step_start_event(
                    step_index=current_step,
                    step_id=next_step_data.get("id", f"step-{current_step}"),
                    tool_name=next_step_data.get("tool_name", ""),
                    description=next_step_data.get("description", ""),
                )
            )

    # Check for checkpoint
    checkpoints = state_update.get("checkpoints", [])
    if checkpoints:
        latest = checkpoints[-1]
        if not latest.get("resolved_at"):
            events.append(
                checkpoint_event(
                    latest,
                    f"Checkpoint at step {latest.get('step_index', 0)}",
                )
            )

    # Check for awaiting user
    if state_update.get("awaiting_user"):
        plan_approved = state_update.get("plan_approved", False)
        if not plan_approved:
            events.append(
                await_input_event(
                    input_type="plan_approval",
                    prompt="Review the plan and approve, edit, or cancel.",
                    options=["Approve", "Edit", "Cancel"],
                )
            )
        else:
            events.append(
                await_input_event(
                    input_type="checkpoint_approval",
                    prompt="Review the checkpoint and decide how to proceed.",
                    options=["Continue", "Edit Plan", "Cancel"],
                )
            )

    # Check for completion (all steps done)
    if plan and current_step >= len(plan):
        execution_results = state_update.get("execution_results", {})
        metadata = state_update.get("metadata", {})

        # Calculate stats
        total_steps = len(plan)
        completed_steps = sum(
            1
            for r in execution_results.values()
            if r.get("status") == StepStatus.COMPLETED.value
        )
        failed_steps = sum(
            1
            for r in execution_results.values()
            if r.get("status") == StepStatus.FAILED.value
        )

        events.append(
            complete_event(
                {
                    "pipeline_id": metadata.get("pipeline_id", thread_id),
                    "total_steps": total_steps,
                    "completed_steps": completed_steps,
                    "failed_steps": failed_steps,
                    "total_duration_seconds": 0,  # TODO: Track actual duration
                    "summary": f"Completed {completed_steps}/{total_steps} steps",
                }
            )
        )

    return events


async def get_next_event(thread_id: str, timeout: float = 1.0) -> SSEEvent | None:
    """
    Get next event from queue with timeout.

    Args:
        thread_id: Thread identifier.
        timeout: Maximum wait time in seconds.

    Returns:
        Next SSE event or None if timeout.
    """
    queue = _event_queues.get(thread_id)
    if not queue:
        return None

    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except TimeoutError:
        return None
