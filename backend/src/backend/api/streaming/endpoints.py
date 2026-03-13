"""
FastAPI SSE endpoints for agent streaming.

Provides endpoints for:
- SSE connection for receiving real-time agent updates
- Sending messages to the agent
- Resuming from checkpoints
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from backend.agent.checkpoints import CheckpointManager
from backend.agent.graph import compile_agent, run_agent
from backend.agent.state import (
    PipelineState,
    StepStatus,
    UserDecision,
    create_initial_state,
)
from backend.api.streaming.batching import EventBatcher
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

# Configuration
THREAD_TIMEOUT_SECONDS = 3600  # 1 hour TTL for idle threads
CLEANUP_INTERVAL_SECONDS = 300  # Run cleanup every 5 minutes
MAX_CONNECTIONS_PER_THREAD = 1  # Only allow one SSE connection per thread
CHECKPOINT_DB_PATH = os.getenv("CLOUMASK_CHECKPOINT_DB", str(Path("data") / "checkpoints.db"))


class ThreadState:
    """Thread state with activity tracking."""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self.queue: asyncio.Queue[SSEEvent] = asyncio.Queue()
        self.cancel_event = asyncio.Event()
        self.created_at = time.time()
        self.last_activity = time.time()
        self.pipeline_state: dict[str, Any] = {}
        self.active_task: asyncio.Task[None] | None = None
        self.connection_count = 0
        # Track emitted state to prevent duplicates
        self.last_emitted_message_count = 0
        self.last_emitted_plan_hash: str | None = None
        self.last_emitted_checkpoint_id: str | None = None
        self.plan_approved_emitted = False

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def is_expired(self) -> bool:
        """Check if thread has exceeded TTL."""
        return time.time() - self.last_activity > THREAD_TIMEOUT_SECONDS


# Thread storage with proper encapsulation
_threads: dict[str, ThreadState] = {}
_cleanup_task: asyncio.Task[None] | None = None
_checkpoint_manager: CheckpointManager | None = None


def _get_checkpoint_db_path() -> str:
    """Return the configured checkpoint database path."""
    if CHECKPOINT_DB_PATH == ":memory:":
        return CHECKPOINT_DB_PATH

    db_path = Path(CHECKPOINT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path)


def _get_checkpoint_manager() -> CheckpointManager:
    """Return a lazily constructed checkpoint manager."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager(_get_checkpoint_db_path())
    return _checkpoint_manager


def _extract_persisted_state(thread_id: str) -> dict[str, Any]:
    """Load the latest persisted pipeline state for a thread."""
    snapshot = _get_checkpoint_manager().get_snapshot(thread_id)
    if not snapshot:
        return {}

    checkpoint_data = snapshot.get("checkpoint_data")
    if not isinstance(checkpoint_data, dict):
        return {}

    channel_values = checkpoint_data.get("channel_values")
    if isinstance(channel_values, dict):
        return dict(channel_values)

    return dict(checkpoint_data)


def _rehydrate_thread(thread_id: str) -> ThreadState | None:
    """Rebuild in-memory thread state from persisted metadata/checkpoints."""
    manager = _get_checkpoint_manager()
    if not manager.saver.get_thread(thread_id):
        return None

    thread = ThreadState(thread_id)
    thread.pipeline_state = _extract_persisted_state(thread_id)
    _threads[thread_id] = thread
    _sync_legacy_exports()
    return thread


def _persist_thread_state(
    thread_id: str,
    state: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save the latest thread state for restart-safe resume."""
    manager = _get_checkpoint_manager()
    manager.create_thread(thread_id)
    manager.save_snapshot(
        thread_id,
        f"state-{uuid.uuid4()}",
        {"channel_values": state},
        metadata=metadata,
    )


def _get_thread(thread_id: str) -> ThreadState | None:
    """Get thread state, returning None if not found."""
    thread = _threads.get(thread_id)
    if thread is None:
        thread = _rehydrate_thread(thread_id)
    if thread:
        thread.touch()
    return thread


def _get_or_create_thread(thread_id: str) -> ThreadState:
    """Get or create thread state."""
    if thread_id not in _threads:
        _threads[thread_id] = ThreadState(thread_id)
    thread = _threads[thread_id]
    thread.touch()
    return thread


def _delete_thread(thread_id: str, *, delete_persisted: bool = False) -> None:
    """Delete in-memory thread state and optionally purge persisted data."""
    thread = _threads.pop(thread_id, None)
    if thread:
        thread.cancel_event.set()
        if thread.active_task and not thread.active_task.done():
            thread.active_task.cancel()
    if delete_persisted:
        _get_checkpoint_manager().delete_thread(thread_id)
    _sync_legacy_exports()


async def _cleanup_expired_threads() -> None:
    """Periodic task to clean up expired threads."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        expired = [tid for tid, thread in _threads.items() if thread.is_expired()]
        for tid in expired:
            logger.info(f"Cleaning up expired thread {tid}")
            _delete_thread(tid, delete_persisted=False)


def _ensure_cleanup_task() -> None:
    """Ensure the cleanup task is running."""
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_expired_threads())


# Legacy exports for test compatibility
_event_queues: dict[str, asyncio.Queue[SSEEvent]] = {}
_thread_states: dict[str, dict[str, Any]] = {}


def _sync_legacy_exports() -> None:
    """Sync legacy exports for backward compatibility with tests."""
    _event_queues.clear()
    _thread_states.clear()
    for tid, thread in _threads.items():
        _event_queues[tid] = thread.queue
        _thread_states[tid] = thread.pipeline_state


class SendMessageRequest(BaseModel):
    """Request body for sending a message."""

    content: str = Field(..., description="The message content", min_length=1)
    decision: UserDecision | None = Field(
        default=None,
        description="User decision for checkpoint/approval responses",
    )
    plan_edits: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional plan modifications when decision is EDIT",
    )

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate content is not just whitespace."""
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace")
        return v


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


class ThreadSummary(BaseModel):
    """Summary of a resumable chat thread."""

    thread_id: str
    title: str | None = None
    status: str
    awaiting_user: bool = False
    current_step: int = 0
    total_steps: int = 0
    last_message: str = ""
    updated_at: str | None = None
    created_at: str | None = None


class ThreadListResponse(BaseModel):
    """List of resumable chat threads."""

    threads: list[ThreadSummary]


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
    # Ensure cleanup task is running
    _ensure_cleanup_task()

    # Get or create thread
    thread = _get_or_create_thread(thread_id)

    # Check for existing connection (prevent race condition)
    if thread.connection_count >= MAX_CONNECTIONS_PER_THREAD:
        raise HTTPException(status_code=409, detail="Thread already has an active SSE connection")

    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        thread.connection_count += 1
        heartbeat_sequence = 0

        try:
            # Send connected event
            connected = connected_event(thread_id)
            yield {
                "event": connected.type.value,
                "data": json.dumps(connected.to_dict()),
            }

            # Main event loop
            while not thread.cancel_event.is_set():
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from thread {thread_id}")
                    break

                try:
                    # Wait for events with timeout for heartbeat
                    event = await asyncio.wait_for(thread.queue.get(), timeout=30.0)
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
            thread.connection_count -= 1
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

    Raises:
        HTTPException: If thread not found.
    """
    # Validate thread exists
    thread = _get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    message_id = str(uuid.uuid4())

    # Queue the message for processing
    await queue_agent_request(
        thread_id=thread_id,
        content=message.content,
        decision=message.decision,
        plan_edits=message.plan_edits,
    )

    # Sync legacy exports for tests
    _sync_legacy_exports()

    return SendMessageResponse(
        status="queued",
        thread_id=thread_id,
        message_id=message_id,
    )


@router.post("/threads")
async def create_thread() -> ThreadInfo:
    """
    Create a new chat thread.

    Returns:
        Thread information with new thread_id.
    """
    # Ensure cleanup task is running
    _ensure_cleanup_task()

    thread_id = str(uuid.uuid4())
    _get_checkpoint_manager().create_thread(thread_id)
    _get_or_create_thread(thread_id)

    # Sync legacy exports for tests
    _sync_legacy_exports()

    return ThreadInfo(
        thread_id=thread_id,
        created=True,
    )


@router.get("/threads")
async def list_threads(limit: int = 20) -> ThreadListResponse:
    """
    List active resumable chat threads.

    Args:
        limit: Maximum number of threads to return.

    Returns:
        Ordered list of active resumable thread summaries.
    """
    manager = _get_checkpoint_manager()
    threads = manager.list_active_threads()
    if limit > 0:
        threads = threads[:limit]

    return ThreadListResponse(
        threads=[ThreadSummary.model_validate(thread) for thread in threads],
    )


@router.get("/threads/{thread_id}")
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
    thread = _get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    state = thread.pipeline_state
    plan = state.get("plan", []) if state else []

    return ThreadInfo(
        thread_id=thread_id,
        created=False,
        awaiting_user=state.get("awaiting_user", False) if state else False,
        current_step=state.get("current_step", 0) if state else 0,
        total_steps=len(plan),
    )


@router.delete("/threads/{thread_id}")
async def close_thread(thread_id: str) -> dict[str, str]:
    """
    Close a chat thread and cleanup resources.

    Args:
        thread_id: Thread to close.

    Returns:
        Confirmation message.
    """
    _delete_thread(thread_id, delete_persisted=True)

    # Sync legacy exports for tests
    _sync_legacy_exports()

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
    thread = _get_thread(thread_id)
    if not thread:
        return

    # Cancel any existing task for this thread
    if thread.active_task and not thread.active_task.done():
        thread.active_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await thread.active_task

    # Start agent processing in background with tracking
    thread.active_task = asyncio.create_task(
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
    thread = _get_thread(thread_id)
    if not thread:
        return

    queue = thread.queue
    batcher = EventBatcher(batch_window_ms=100)
    manager = _get_checkpoint_manager()

    try:
        # Emit thinking event
        await queue.put(thinking_event("Processing your request..."))

        # Handle CANCEL decision
        if decision == UserDecision.CANCEL:
            await queue.put(message_event("system", "Operation cancelled by user."))
            await queue.put(
                complete_event(
                    {
                        "pipeline_id": thread.pipeline_state.get("metadata", {}).get(
                            "pipeline_id", thread_id
                        ),
                        "total_steps": len(thread.pipeline_state.get("plan", [])),
                        "completed_steps": thread.pipeline_state.get("current_step", 0),
                        "failed_steps": 0,
                        "summary": "Cancelled by user",
                    }
                )
            )
            if thread.pipeline_state:
                _persist_thread_state(
                    thread_id,
                    {
                        **thread.pipeline_state,
                        "awaiting_user": False,
                    },
                    metadata={"status": "cancelled"},
                )
            manager.mark_cancelled(thread_id)
            # Reset thread state for new requests
            thread.pipeline_state = {}
            thread.last_emitted_message_count = 0
            thread.last_emitted_plan_hash = None
            thread.last_emitted_checkpoint_id = None
            thread.plan_approved_emitted = False
            return

        # Get or create initial state
        initial_state: PipelineState
        if thread.pipeline_state and decision is not None:
            # Resume from existing state with decision
            initial_state = dict(thread.pipeline_state)  # type: ignore[assignment]
            # Update state with user decision
            _apply_user_decision(initial_state, decision, content, plan_edits)
            thread.pipeline_state = dict(initial_state)
        else:
            # New request - create initial state
            pipeline_id = str(uuid.uuid4())
            initial_state = create_initial_state(content, pipeline_id)
            # Reset thread state to a full baseline for this run.
            thread.pipeline_state = dict(initial_state)
            # Reset tracking for new conversation
            thread.last_emitted_message_count = 0
            thread.last_emitted_plan_hash = None
            thread.last_emitted_checkpoint_id = None
            thread.plan_approved_emitted = False

        manager.create_thread(thread_id)
        _persist_thread_state(thread_id, thread.pipeline_state)

        # Run agent and stream events
        async with compile_agent(_get_checkpoint_db_path()) as compiled:
            async for state_update in run_agent(compiled, initial_state, thread_id):
                # Merge partial updates into full thread state so resume/approval
                # decisions retain plan + metadata across node outputs.
                thread.pipeline_state = {
                    **thread.pipeline_state,
                    **state_update,
                }
                _persist_thread_state(thread_id, thread.pipeline_state)

                # Convert state updates to SSE events with deduplication
                events = state_to_events(thread.pipeline_state, thread_id, thread)
                for event in events:
                    # Apply batching for progress events
                    batched = batcher.add(event)
                    if batched:
                        await queue.put(batched)

                # Flush any remaining batched events periodically
                for pending in batcher.flush():
                    await queue.put(pending)

                # Check if we should pause for user input
                if thread.pipeline_state.get("awaiting_user", False):
                    manager.mark_active(thread_id)
                    break

            else:
                manager.mark_completed(thread_id)

    except asyncio.CancelledError:
        logger.info(f"Agent request cancelled for thread {thread_id}")
        raise
    except Exception as e:
        logger.exception(f"Error processing agent request for {thread_id}")
        # Sanitize error message - don't expose internal details
        safe_message = _sanitize_error_message(str(e))
        await queue.put(
            error_event(
                "AGENT_ERROR",
                safe_message,
                recoverable=True,
            )
        )


def _sanitize_error_message(message: str) -> str:
    """
    Sanitize error message to avoid exposing internal details.

    Args:
        message: Raw error message.

    Returns:
        Sanitized message safe for frontend.
    """
    # List of patterns that might expose internal details
    sensitive_patterns = [
        "password",
        "secret",
        "token",
        "key",
        "credential",
        "/home/",
        "/root/",
        "traceback",
        'File "',
    ]

    message_lower = message.lower()
    for pattern in sensitive_patterns:
        if pattern.lower() in message_lower:
            return "An internal error occurred. Please try again."

    # Truncate very long messages
    if len(message) > 200:
        return message[:200] + "..."

    return message


def _apply_user_decision(
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

    # Apply plan edits whenever the frontend sends them.
    # This allows "approve + edited plan" flows from the plan editor.
    if plan_edits:
        state["plan"] = plan_edits

    # If approving plan for first time
    if not state.get("plan_approved", False) and decision == UserDecision.APPROVE:
        state["plan_approved"] = True


def _compute_plan_hash(plan: list[dict[str, Any]]) -> str:
    """Compute a simple hash of plan for change detection."""
    return str(len(plan)) + ":" + ",".join(step.get("id", str(i)) for i, step in enumerate(plan))


def state_to_events(
    state_update: dict[str, Any],
    thread_id: str,
    thread: ThreadState | None = None,
) -> list[SSEEvent]:
    """
    Convert LangGraph state update to SSE events.

    Analyzes the state to determine which events should be emitted.
    Uses thread state to prevent duplicate events.

    Args:
        state_update: Current pipeline state dictionary.
        thread_id: Thread identifier for event context.
        thread: Thread state for deduplication tracking.

    Returns:
        List of SSE events to emit.
    """
    events: list[SSEEvent] = []

    # Check for new messages (with deduplication)
    messages = state_update.get("messages", [])
    start_idx = thread.last_emitted_message_count if thread else 0
    if len(messages) > start_idx:
        for msg in messages[start_idx:]:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            # Only emit if content exists and isn't an internal marker
            if content and not content.startswith("AWAIT_"):
                events.append(message_event(role, content))
        if thread:
            thread.last_emitted_message_count = len(messages)

    # Check for plan updates (with deduplication)
    plan = state_update.get("plan", [])
    plan_approved = state_update.get("plan_approved", False)

    if plan:
        plan_hash = _compute_plan_hash(plan)
        should_emit_plan = thread is None or thread.last_emitted_plan_hash != plan_hash

        if should_emit_plan and not plan_approved:
            pipeline_id = state_update.get("metadata", {}).get("pipeline_id", "") or thread_id
            events.append(plan_event(pipeline_id, plan))
            if thread:
                thread.last_emitted_plan_hash = plan_hash

    # Emit PLAN_APPROVED event when plan is first approved
    if plan_approved and thread and not thread.plan_approved_emitted:
        events.append(message_event("system", "Plan approved. Starting execution..."))
        thread.plan_approved_emitted = True

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

    # Check for checkpoint (with deduplication)
    checkpoints = state_update.get("checkpoints", [])
    if checkpoints:
        latest = checkpoints[-1]
        checkpoint_id = latest.get("id", "")
        is_new_checkpoint = thread is None or thread.last_emitted_checkpoint_id != checkpoint_id

        if not latest.get("resolved_at") and is_new_checkpoint:
            events.append(
                checkpoint_event(
                    latest,
                    f"Checkpoint at step {latest.get('step_index', 0)}",
                )
            )
            if thread:
                thread.last_emitted_checkpoint_id = checkpoint_id

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
            1 for step in plan if step.get("status") == StepStatus.COMPLETED.value
        )
        failed_steps = sum(1 for step in plan if step.get("status") == StepStatus.FAILED.value)
        # Fallback for legacy states where plan statuses are missing.
        if completed_steps == 0 and failed_steps == 0 and execution_results:
            completed_steps = sum(
                1
                for r in execution_results.values()
                if isinstance(r, dict) and r.get("status") == StepStatus.COMPLETED.value
            )
            failed_steps = sum(
                1
                for r in execution_results.values()
                if isinstance(r, dict) and r.get("status") == StepStatus.FAILED.value
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
    thread = _get_thread(thread_id)
    if not thread:
        return None

    try:
        return await asyncio.wait_for(thread.queue.get(), timeout=timeout)
    except TimeoutError:
        return None
