# LangGraph Core

> **Status:** 🟢 Complete (Implemented; checklist backfill pending)
> **Priority:** P0 (Critical)
> **Dependencies:** 01-state-types
> **Estimated Complexity:** Medium

## Overview

Set up the LangGraph state machine that orchestrates the agent's behavior. This defines the graph structure with nodes, edges, and conditional routing logic that controls the flow between understanding, planning, execution, and checkpoints.

## Goals

- [ ] Create the StateGraph with PipelineState
- [ ] Define all node entry points (actual logic in other specs)
- [ ] Implement conditional edge routing
- [ ] Configure memory/persistence backend
- [ ] Compile graph to executable runnable

## Technical Design

### State Machine Flow

```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    ▼                                     │
START ──> understand ──> plan ──> await_approval ──>──────┤
                              │         │                 │
                         [cancel]    [approve]            │
                              │         │                 │
                              ▼         ▼                 │
                            END    execute_step ──>───────┤
                                        │                 │
                                        ▼                 │
                              ┌──> checkpoint ──>─────────┘
                              │         │
                              │    [needs_review]
                              │         │
                              │         ▼
                              │   await_approval
                              │         │
                              │    [continue]
                              │         │
                              └─────────┘
                                        │
                                   [complete]
                                        │
                                        ▼
                                    complete ──> END
```

### Graph Definition

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.state import PipelineState


def create_agent_graph() -> StateGraph:
    """Create the main agent state graph."""

    # Initialize graph with state type
    graph = StateGraph(PipelineState)

    # Add nodes (implementations imported from node modules)
    graph.add_node("understand", understand_node)
    graph.add_node("plan", plan_node)
    graph.add_node("await_approval", await_approval_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("checkpoint", checkpoint_node)
    graph.add_node("complete", complete_node)

    # Set entry point
    graph.set_entry_point("understand")

    # Add edges
    graph.add_edge("understand", "plan")

    # Conditional: after plan, wait for approval
    graph.add_edge("plan", "await_approval")

    # Conditional: after approval
    graph.add_conditional_edges(
        "await_approval",
        route_after_approval,
        {
            "execute": "execute_step",
            "cancel": END,
            "edit": "plan",  # Go back to planning with edits
        }
    )

    # Conditional: after execution
    graph.add_conditional_edges(
        "execute_step",
        route_after_execution,
        {
            "checkpoint": "checkpoint",
            "continue": "execute_step",  # Next step
            "complete": "complete",
        }
    )

    # Conditional: after checkpoint
    graph.add_conditional_edges(
        "checkpoint",
        route_after_checkpoint,
        {
            "await": "await_approval",  # Needs human review
            "continue": "execute_step",
        }
    )

    # Terminal node
    graph.add_edge("complete", END)

    return graph


def compile_agent(
    graph: StateGraph,
    checkpoint_path: str = "checkpoints.db"
) -> CompiledGraph:
    """Compile graph with persistence."""

    # SQLite-based checkpointing for resume capability
    memory = SqliteSaver.from_conn_string(f"sqlite:///{checkpoint_path}")

    return graph.compile(checkpointer=memory)
```

### Routing Functions

```python
def route_after_approval(state: PipelineState) -> str:
    """Determine next step after user reviews plan/checkpoint."""

    if not state.get("awaiting_user"):
        return "execute"

    # Check the latest user feedback
    messages = state.get("messages", [])
    if not messages:
        return "execute"

    last_message = messages[-1]
    if last_message.get("role") != "user":
        return "execute"

    content = last_message.get("content", "").lower()

    if "cancel" in content or "stop" in content:
        return "cancel"
    elif "edit" in content or "change" in content:
        return "edit"
    else:
        return "execute"


def route_after_execution(state: PipelineState) -> str:
    """Determine next step after executing a pipeline step."""

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    # Check if all steps complete
    if current_step >= len(plan):
        return "complete"

    # Check if checkpoint is needed
    if should_checkpoint(state):
        return "checkpoint"

    return "continue"


def route_after_checkpoint(state: PipelineState) -> str:
    """Determine if checkpoint needs human review."""

    checkpoints = state.get("checkpoints", [])
    if not checkpoints:
        return "continue"

    latest = checkpoints[-1]
    trigger = latest.get("trigger_reason")

    # Critical checkpoints always need review
    if trigger in ["quality_drop", "error_rate", "critical_step"]:
        return "await"

    # Percentage checkpoints are informational
    return "continue"


def should_checkpoint(state: PipelineState) -> bool:
    """Determine if we should create a checkpoint."""

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    results = state.get("execution_results", {})

    if not plan:
        return False

    total_steps = len(plan)
    progress = current_step / total_steps

    # Percentage-based checkpoints
    percentage_triggers = [0.10, 0.25, 0.50]
    prev_progress = (current_step - 1) / total_steps if current_step > 0 else 0

    for trigger in percentage_triggers:
        if prev_progress < trigger <= progress:
            return True

    # Critical step checkpoints
    critical_tools = ["anonymize", "segment", "detect_3d"]
    current_tool = plan[current_step - 1].get("tool_name", "") if current_step > 0 else ""
    if current_tool in critical_tools:
        return True

    # Quality-based checkpoints (check recent results)
    if results:
        recent_results = list(results.values())[-5:]
        confidences = [r.get("confidence", 1.0) for r in recent_results if "confidence" in r]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            # Check for confidence drop
            all_confidences = [r.get("confidence", 1.0) for r in results.values() if "confidence" in r]
            if all_confidences:
                overall_avg = sum(all_confidences) / len(all_confidences)
                if overall_avg - avg_confidence > 0.15:
                    return True

        # Error rate check
        error_count = sum(1 for r in recent_results if r.get("error"))
        if error_count / max(len(recent_results), 1) > 0.05:
            return True

    return False
```

### Graph Execution

```python
async def run_agent(
    user_message: str,
    thread_id: str,
    compiled_graph: CompiledGraph,
) -> AsyncGenerator[dict, None]:
    """
    Run the agent and yield state updates.

    Args:
        user_message: The user's natural language request
        thread_id: Unique identifier for this conversation thread
        compiled_graph: The compiled LangGraph

    Yields:
        State update dicts for SSE streaming
    """
    from agent.state import create_initial_state
    from uuid import uuid4

    # Create or restore state
    config = {"configurable": {"thread_id": thread_id}}

    # Check for existing state (resume capability)
    existing_state = compiled_graph.get_state(config)

    if existing_state and existing_state.values:
        # Resume from checkpoint
        state = existing_state.values
        # Add the new user message
        state["messages"].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat(),
        })
    else:
        # Fresh start
        state = create_initial_state(user_message, str(uuid4()))

    # Stream execution
    async for event in compiled_graph.astream(state, config):
        yield event
```

## Implementation Tasks

- [ ] Create `backend/agent/graph.py` module
- [ ] Implement `create_agent_graph()` function
- [ ] Import node functions (stubs initially)
- [ ] Implement `route_after_approval()` routing
- [ ] Implement `route_after_execution()` routing
- [ ] Implement `route_after_checkpoint()` routing
- [ ] Implement `should_checkpoint()` helper
- [ ] Implement `compile_agent()` with SQLite persistence
- [ ] Implement `run_agent()` async generator
- [ ] Add node stub functions for initial testing

## Testing

### Unit Tests

```python
# tests/agent/test_graph.py

def test_create_graph_has_all_nodes():
    """Graph should have all required nodes."""
    graph = create_agent_graph()
    nodes = graph.nodes
    expected = ["understand", "plan", "await_approval",
                "execute_step", "checkpoint", "complete"]
    for node in expected:
        assert node in nodes


def test_route_after_approval_execute():
    """Should route to execute when approved."""
    state = {
        "messages": [{"role": "user", "content": "yes, proceed"}],
        "awaiting_user": False,
    }
    assert route_after_approval(state) == "execute"


def test_route_after_approval_cancel():
    """Should route to cancel on user cancel."""
    state = {
        "messages": [{"role": "user", "content": "cancel this"}],
        "awaiting_user": True,
    }
    assert route_after_approval(state) == "cancel"


def test_route_after_execution_complete():
    """Should complete when all steps done."""
    state = {
        "plan": [{"tool_name": "scan"}],
        "current_step": 1,
        "execution_results": {},
    }
    assert route_after_execution(state) == "complete"


def test_should_checkpoint_at_25_percent():
    """Should checkpoint at 25% progress."""
    state = {
        "plan": [{"tool_name": f"step_{i}"} for i in range(4)],
        "current_step": 1,  # 25%
        "execution_results": {},
    }
    assert should_checkpoint(state) == True


def test_should_checkpoint_after_critical_tool():
    """Should checkpoint after anonymize."""
    state = {
        "plan": [{"tool_name": "anonymize"}],
        "current_step": 1,
        "execution_results": {},
    }
    assert should_checkpoint(state) == True
```

### Integration Tests

```python
# tests/agent/test_graph_integration.py

@pytest.mark.asyncio
async def test_full_graph_execution():
    """Test complete graph flow with mock nodes."""
    graph = create_agent_graph()
    compiled = compile_agent(graph, ":memory:")

    events = []
    async for event in run_agent("scan /data", "test-thread", compiled):
        events.append(event)

    # Should have gone through understand, plan, await_approval at minimum
    node_names = [e.get("node") for e in events if "node" in e]
    assert "understand" in node_names
    assert "plan" in node_names
```

### Edge Cases

- [ ] Empty plan (no steps to execute)
- [ ] Single step plan
- [ ] Plan with all critical steps
- [ ] Resume from mid-execution
- [ ] Multiple rapid user messages

## Acceptance Criteria

- [ ] Graph compiles without errors
- [ ] All routing functions return valid destinations
- [ ] Checkpoint logic triggers at correct thresholds
- [ ] Graph can be persisted to SQLite
- [ ] Graph can resume from saved state
- [ ] Async streaming works correctly

## Files to Create/Modify

```
backend/
├── agent/
│   ├── __init__.py          # Export graph functions
│   ├── graph.py              # Graph definition
│   └── nodes/
│       └── __init__.py       # Node stubs for testing
└── tests/
    └── agent/
        ├── test_graph.py     # Unit tests
        └── test_graph_integration.py
```

## Dependencies

```
# requirements.txt additions
langgraph>=0.2.0
langgraph-checkpoint>=0.2.0
```

## Notes

- Using async generators for streaming to support SSE
- SQLite checkpointing is simple but sufficient for desktop app
- Consider adding Redis backend later for multi-instance scenarios
- The `thread_id` maps to conversation/session ID from frontend
