"""
LangGraph state machine definition for the Cloumask agent.

This module defines the core graph structure with nodes, edges, and routing
logic that controls the flow between understanding, planning, execution,
and checkpoints.

Graph Structure:
    START -> intent_router -> (chat_reply | understand -> plan -> await_approval)
                                       |
               +-----------------------+-----------------------+
               |                       |                       |
               v                       v                       v
          execute_step               plan                  complete
               |                                               |
               v                                               v
       route_after_execution                                  END
               |
       +-------+-------+
       |       |       |
       v       v       v
   checkpoint  |   complete
       |       |
       v       v
   await_approval (loop back)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from backend.agent.nodes import (
    await_approval,
    chat_reply,
    complete,
    create_checkpoint,
    execute_step,
    generate_plan,
    route_intent,
    understand,
)
from backend.agent.state import (
    PipelineState,
    StepStatus,
    UserDecision,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


# -----------------------------------------------------------------------------
# Routing Functions
# -----------------------------------------------------------------------------


def route_after_approval(
    state: PipelineState,
) -> Literal["execute_step", "generate_plan", "complete"]:
    """
    Determine next step after user reviews plan/checkpoint.

    Routing logic:
    - If user cancelled -> complete (end gracefully)
    - If user requested edit -> generate_plan (re-plan with edits)
    - Otherwise -> execute_step (proceed with execution)

    Args:
        state: Current pipeline state.

    Returns:
        Node name to route to.
    """
    # Check the latest checkpoint for user decision
    checkpoints = state.get("checkpoints", [])
    if checkpoints:
        last_checkpoint = checkpoints[-1]
        decision = last_checkpoint.get("user_decision")

        if decision == UserDecision.CANCEL.value:
            return "complete"
        if decision == UserDecision.EDIT.value:
            return "generate_plan"

    # Default: proceed with execution
    return "execute_step"


def route_from_start(
    state: PipelineState,
) -> Literal["intent_router", "await_approval", "execute_step", "complete"]:
    """
    Determine the correct entry node when (re)starting graph execution.

    This allows resuming from a persisted in-memory state without always
    re-running understand -> generate_plan. In particular, after a user approves
    a plan, execution should continue at execute_step instead of planning again.
    """
    # If state is terminal, route directly to completion.
    if state.get("last_error") and not state.get("plan"):
        return "complete"

    plan = state.get("plan", [])
    if not plan:
        return "intent_router"

    if state.get("awaiting_user", False):
        return "await_approval"

    if state.get("plan_approved", False):
        return "execute_step"

    # Plan exists but is not approved yet; wait for user approval.
    return "await_approval"


def route_after_intent_router(
    state: PipelineState,
) -> Literal["chat_reply", "understand"]:
    """
    Route requests after deterministic intent classification.

    - chat: Send a lightweight conversational reply and end turn.
    - task: Continue to LLM understanding + planning flow.
    """
    metadata = state.get("metadata", {})
    route = str(metadata.get("intent_route", "task")).lower()
    if route == "chat":
        return "chat_reply"
    return "understand"


def route_after_execution(
    state: PipelineState,
) -> Literal["create_checkpoint", "execute_step", "complete"]:
    """
    Determine next step after executing a pipeline step.

    Routing logic:
    - If checkpoint conditions met -> create_checkpoint
    - If all steps done -> complete
    - Otherwise -> execute_step (next step)

    Args:
        state: Current pipeline state.

    Returns:
        Node name to route to.
    """
    # Check if checkpoint is needed first
    if should_checkpoint(state):
        return "create_checkpoint"

    # Check if all steps are complete
    plan_steps = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if current_step >= len(plan_steps):
        return "complete"

    return "execute_step"


def route_after_checkpoint(
    state: PipelineState,
) -> Literal["await_approval", "execute_step"]:
    """
    Determine if checkpoint needs human review.

    Routing logic:
    - If awaiting_user flag is set -> await_approval
    - Otherwise -> execute_step (continue automatically)

    Args:
        state: Current pipeline state.

    Returns:
        Node name to route to.
    """
    if state.get("awaiting_user", False):
        return "await_approval"
    return "execute_step"


def should_checkpoint(state: PipelineState) -> bool:
    """
    Determine if we should create a checkpoint.

    Checkpoint triggers:
    - Percentage-based: 10%, 25%, 50% progress
    - Quality-based: Error rate >5%
    - Critical steps: Marked as critical in plan

    Args:
        state: Current pipeline state.

    Returns:
        True if checkpoint should trigger.
    """
    plan_steps = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if not plan_steps:
        return False

    total_steps = len(plan_steps)

    # Get existing checkpoint step indices to avoid duplicates
    checkpoints = state.get("checkpoints", [])
    checkpoint_steps = {cp.get("step_index", -1) for cp in checkpoints}

    # Percentage-based checkpoints (10%, 25%, 50%)
    percentage_thresholds = [0.10, 0.25, 0.50]
    for threshold in percentage_thresholds:
        threshold_step = int(threshold * total_steps)
        # Trigger if we've reached the threshold and haven't checkpointed there
        if current_step >= threshold_step > 0 and threshold_step not in checkpoint_steps:
            return True

    # Quality-based checkpoints: Error rate >5%
    execution_results = state.get("execution_results", {})
    if execution_results:
        error_count = sum(
            1
            for result in execution_results.values()
            if result.get("status") == StepStatus.FAILED.value
        )
        total_executed = len(execution_results)
        if total_executed > 0 and (error_count / total_executed) > 0.05:
            return True

    # Critical step checkpoint
    if 0 < current_step <= len(plan_steps):
        # Check the just-completed step (current_step - 1) for critical flag
        completed_step = plan_steps[current_step - 1]
        if completed_step.get("critical", False):
            return True

    return False


# -----------------------------------------------------------------------------
# Graph Creation
# -----------------------------------------------------------------------------


def create_agent_graph() -> StateGraph:
    """
    Create the agent StateGraph with all nodes and edges.

    The graph follows this flow:
    1. intent_router: Deterministic chat/task routing
    2. chat_reply: Fast-path response for smalltalk
    3. understand: Parse task requests
    4. plan: Generate execution plan
    5. await_approval: Wait for user approval
    6. execute_step: Execute plan steps (loops until done)
    7. checkpoint: Pause for human review at milestones
    8. complete: Finalize execution

    Returns:
        Configured StateGraph ready for compilation.
    """
    graph = StateGraph(PipelineState)

    # Add all nodes
    graph.add_node("intent_router", route_intent)
    graph.add_node("chat_reply", chat_reply)
    graph.add_node("understand", understand)
    graph.add_node("generate_plan", generate_plan)
    graph.add_node("await_approval", await_approval)
    graph.add_node("execute_step", execute_step)
    graph.add_node("create_checkpoint", create_checkpoint)
    graph.add_node("complete", complete)

    # Entry routing: support both fresh requests and resume paths.
    graph.add_conditional_edges(
        START,
        route_from_start,
        {
            "intent_router": "intent_router",
            "await_approval": "await_approval",
            "execute_step": "execute_step",
            "complete": "complete",
        },
    )

    # Fast-path chat routing.
    graph.add_conditional_edges(
        "intent_router",
        route_after_intent_router,
        {
            "chat_reply": "chat_reply",
            "understand": "understand",
        },
    )

    # Linear flow: understand -> generate_plan -> await_approval
    graph.add_edge("understand", "generate_plan")
    graph.add_edge("generate_plan", "await_approval")
    graph.add_edge("chat_reply", END)

    # Conditional edge after approval
    graph.add_conditional_edges(
        "await_approval",
        route_after_approval,
        {
            "execute_step": "execute_step",
            "generate_plan": "generate_plan",
            "complete": "complete",
        },
    )

    # Conditional edge after execution
    graph.add_conditional_edges(
        "execute_step",
        route_after_execution,
        {
            "create_checkpoint": "create_checkpoint",
            "execute_step": "execute_step",
            "complete": "complete",
        },
    )

    # Conditional edge after checkpoint
    graph.add_conditional_edges(
        "create_checkpoint",
        route_after_checkpoint,
        {
            "await_approval": "await_approval",
            "execute_step": "execute_step",
        },
    )

    # Terminal edge: complete -> END
    graph.add_edge("complete", END)

    return graph


# -----------------------------------------------------------------------------
# Graph Compilation and Execution
# -----------------------------------------------------------------------------


@asynccontextmanager
async def compile_agent(
    db_path: str = "checkpoints.db",
) -> AsyncGenerator[CompiledStateGraph, None]:
    """
    Compile the agent graph with AsyncSqliteSaver persistence.

    This is an async context manager that handles checkpointer lifecycle.

    Args:
        db_path: Path to SQLite database for checkpoints.
            Use ":memory:" for in-memory testing.

    Yields:
        Compiled graph with checkpointing enabled.

    Example:
        async with compile_agent() as graph:
            async for state in run_agent(graph, initial_state, "thread-1"):
                print(state)
    """
    graph = create_agent_graph()

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
        yield compiled


async def run_agent(
    compiled_graph: CompiledStateGraph,
    initial_state: PipelineState,
    thread_id: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run the agent and yield state updates.

    Args:
        compiled_graph: Compiled LangGraph instance.
        initial_state: Starting state for the pipeline.
        thread_id: Unique identifier for this execution thread.
            Used for checkpoint persistence and resume capability.

    Yields:
        State dict after each node execution.

    Example:
        async for state in run_agent(graph, initial_state, "thread-123"):
            print(f"Current step: {state.get('current_step')}")
            if state.get('awaiting_user'):
                # Handle human-in-the-loop
                break
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    async for event in compiled_graph.astream(
        initial_state,
        config,  # type: ignore[arg-type]
        stream_mode="values",
    ):
        yield event


__all__ = [
    # Graph creation
    "create_agent_graph",
    "compile_agent",
    "run_agent",
    # Routing functions
    "route_from_start",
    "route_after_intent_router",
    "route_after_approval",
    "route_after_execution",
    "route_after_checkpoint",
    "should_checkpoint",
]
