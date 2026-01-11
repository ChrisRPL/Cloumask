"""Tests for LangGraph agent graph implementation."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.graph import StateGraph

from backend.agent.graph import (
    create_agent_graph,
    route_after_approval,
    route_after_checkpoint,
    route_after_execution,
    should_checkpoint,
)
from backend.agent.state import (
    PipelineState,
    StepStatus,
    UserDecision,
)


def _create_minimal_state(**overrides: Any) -> PipelineState:
    """Create a minimal PipelineState for testing."""
    base: PipelineState = {
        "messages": [],
        "plan": [],
        "plan_approved": False,
        "current_step": 0,
        "execution_results": {},
        "checkpoints": [],
        "awaiting_user": False,
        "metadata": {},
        "last_error": None,
        "retry_count": 0,
    }
    return {**base, **overrides}  # type: ignore[typeddict-item]


class TestCreateAgentGraph:
    """Tests for create_agent_graph function."""

    def test_returns_state_graph(self) -> None:
        """create_agent_graph should return a StateGraph instance."""
        graph = create_agent_graph()
        assert isinstance(graph, StateGraph)

    def test_graph_compiles_without_error(self) -> None:
        """Graph should compile without errors."""
        graph = create_agent_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_graph_has_all_nodes(self) -> None:
        """Graph should have all required nodes."""
        graph = create_agent_graph()
        # Access the builder's nodes
        expected_nodes = {
            "understand",
            "generate_plan",
            "await_approval",
            "execute_step",
            "create_checkpoint",
            "complete",
        }
        # StateGraph stores nodes in nodes property
        actual_nodes = set(graph.nodes.keys())
        assert expected_nodes == actual_nodes


class TestRouteAfterApproval:
    """Tests for route_after_approval routing function."""

    def test_returns_execute_step_by_default(self) -> None:
        """Should return 'execute_step' when no special conditions."""
        state = _create_minimal_state(plan_approved=True)
        assert route_after_approval(state) == "execute_step"

    def test_returns_execute_step_when_no_checkpoints(self) -> None:
        """Should return 'execute_step' when no checkpoints exist."""
        state = _create_minimal_state(checkpoints=[])
        assert route_after_approval(state) == "execute_step"

    def test_returns_complete_when_cancelled(self) -> None:
        """Should return 'complete' when user cancels."""
        state = _create_minimal_state(
            checkpoints=[{"user_decision": UserDecision.CANCEL.value}]
        )
        assert route_after_approval(state) == "complete"

    def test_returns_generate_plan_when_edit_requested(self) -> None:
        """Should return 'generate_plan' when user requests edit."""
        state = _create_minimal_state(
            checkpoints=[{"user_decision": UserDecision.EDIT.value}]
        )
        assert route_after_approval(state) == "generate_plan"

    def test_uses_latest_checkpoint(self) -> None:
        """Should use the most recent checkpoint's decision."""
        state = _create_minimal_state(
            checkpoints=[
                {"user_decision": UserDecision.CANCEL.value},
                {"user_decision": UserDecision.APPROVE.value},
            ]
        )
        # Latest is approve, so should execute
        assert route_after_approval(state) == "execute_step"


class TestRouteAfterExecution:
    """Tests for route_after_execution routing function."""

    def test_returns_complete_when_all_steps_done(self) -> None:
        """Should return 'complete' when all steps executed."""
        state = _create_minimal_state(
            plan=[{"id": "step-1", "tool_name": "test"}],
            current_step=1,
        )
        assert route_after_execution(state) == "complete"

    def test_returns_execute_step_when_more_steps(self) -> None:
        """Should return 'execute_step' when steps remain."""
        state = _create_minimal_state(
            plan=[
                {"id": "step-1", "tool_name": "test"},
                {"id": "step-2", "tool_name": "test"},
            ],
            current_step=0,
        )
        assert route_after_execution(state) == "execute_step"

    def test_returns_create_checkpoint_when_threshold_reached(self) -> None:
        """Should return 'create_checkpoint' at progress thresholds."""
        # Create state at 10% progress (step 1 of 10)
        plan = [{"id": f"step-{i}", "tool_name": "test"} for i in range(10)]
        state = _create_minimal_state(plan=plan, current_step=1)
        assert route_after_execution(state) == "create_checkpoint"

    def test_returns_complete_when_empty_plan(self) -> None:
        """Should return 'complete' when plan is empty."""
        state = _create_minimal_state(plan=[], current_step=0)
        assert route_after_execution(state) == "complete"


class TestRouteAfterCheckpoint:
    """Tests for route_after_checkpoint routing function."""

    def test_returns_await_approval_when_awaiting(self) -> None:
        """Should return 'await_approval' when awaiting user."""
        state = _create_minimal_state(awaiting_user=True)
        assert route_after_checkpoint(state) == "await_approval"

    def test_returns_execute_step_when_not_awaiting(self) -> None:
        """Should return 'execute_step' when not awaiting."""
        state = _create_minimal_state(awaiting_user=False)
        assert route_after_checkpoint(state) == "execute_step"


class TestShouldCheckpoint:
    """Tests for should_checkpoint trigger function."""

    def test_false_when_no_plan(self) -> None:
        """Should return False when plan is empty."""
        state = _create_minimal_state(plan=[])
        assert should_checkpoint(state) is False

    def test_true_at_10_percent(self) -> None:
        """Should trigger at 10% progress."""
        plan = [{"id": f"step-{i}", "tool_name": "test"} for i in range(10)]
        state = _create_minimal_state(plan=plan, current_step=1)
        assert should_checkpoint(state) is True

    def test_true_at_25_percent(self) -> None:
        """Should trigger at 25% progress."""
        plan = [{"id": f"step-{i}", "tool_name": "test"} for i in range(4)]
        state = _create_minimal_state(plan=plan, current_step=1)
        assert should_checkpoint(state) is True

    def test_true_at_50_percent(self) -> None:
        """Should trigger at 50% progress."""
        plan = [{"id": f"step-{i}", "tool_name": "test"} for i in range(4)]
        state = _create_minimal_state(plan=plan, current_step=2)
        assert should_checkpoint(state) is True

    def test_false_when_checkpoint_already_exists(self) -> None:
        """Should not trigger duplicate checkpoint at same step."""
        plan = [{"id": f"step-{i}", "tool_name": "test"} for i in range(10)]
        state = _create_minimal_state(
            plan=plan,
            current_step=1,
            checkpoints=[{"step_index": 1}],
        )
        assert should_checkpoint(state) is False

    def test_true_on_high_error_rate(self) -> None:
        """Should trigger when error rate exceeds 5%."""
        state = _create_minimal_state(
            plan=[{"id": "step-1", "tool_name": "test"}],
            current_step=0,
            execution_results={
                "step-0": {"status": StepStatus.FAILED.value},
                "step-1": {"status": StepStatus.COMPLETED.value},
            },
        )
        # 50% error rate > 5%
        assert should_checkpoint(state) is True

    def test_false_on_low_error_rate(self) -> None:
        """Should not trigger when error rate is below 5%."""
        results = {f"step-{i}": {"status": StepStatus.COMPLETED.value} for i in range(100)}
        state = _create_minimal_state(
            plan=[{"id": "step-1", "tool_name": "test"}],
            current_step=0,
            checkpoints=[{"step_index": 0}],  # Prevent percentage checkpoint
            execution_results=results,
        )
        assert should_checkpoint(state) is False

    def test_true_on_critical_step(self) -> None:
        """Should trigger after critical step."""
        plan = [{"id": "step-1", "tool_name": "anonymize", "critical": True}]
        state = _create_minimal_state(plan=plan, current_step=1)
        assert should_checkpoint(state) is True


class TestCompileAgent:
    """Tests for compile_agent async context manager."""

    @pytest.mark.asyncio
    async def test_yields_compiled_graph(self) -> None:
        """compile_agent should yield a compiled graph."""
        from contextlib import asynccontextmanager

        with patch("backend.agent.graph.AsyncSqliteSaver") as mock_saver:
            mock_checkpointer = AsyncMock()

            @asynccontextmanager
            async def mock_context_manager(db_path: str) -> Any:
                yield mock_checkpointer

            mock_saver.from_conn_string = mock_context_manager

            from backend.agent.graph import compile_agent

            async with compile_agent(":memory:") as graph:
                assert graph is not None

    @pytest.mark.asyncio
    async def test_uses_provided_db_path(self) -> None:
        """compile_agent should use provided database path."""
        from contextlib import asynccontextmanager

        captured_path: list[str] = []

        with patch("backend.agent.graph.AsyncSqliteSaver") as mock_saver:
            mock_checkpointer = AsyncMock()

            @asynccontextmanager
            async def mock_context_manager(db_path: str) -> Any:
                captured_path.append(db_path)
                yield mock_checkpointer

            mock_saver.from_conn_string = mock_context_manager

            from backend.agent.graph import compile_agent

            async with compile_agent("/custom/path.db") as _graph:
                pass

            assert captured_path[0] == "/custom/path.db"


class TestRunAgent:
    """Tests for run_agent async generator."""

    @pytest.mark.asyncio
    async def test_yields_state_updates(self) -> None:
        """run_agent should yield state updates."""
        from unittest.mock import MagicMock

        from backend.agent.graph import run_agent
        from backend.agent.state import create_initial_state

        # Create a mock graph with async stream that returns an async generator
        mock_graph = MagicMock()

        async def mock_astream(*args: Any, **kwargs: Any) -> Any:
            yield {"current_step": 0}
            yield {"current_step": 1}

        mock_graph.astream = mock_astream

        initial_state = create_initial_state("Test", "test-id")

        states = []
        async for state in run_agent(mock_graph, initial_state, "thread-1"):
            states.append(state)

        assert len(states) == 2
        assert states[0]["current_step"] == 0
        assert states[1]["current_step"] == 1

    @pytest.mark.asyncio
    async def test_passes_thread_id_in_config(self) -> None:
        """run_agent should pass thread_id in config."""
        from unittest.mock import MagicMock

        from backend.agent.graph import run_agent
        from backend.agent.state import create_initial_state

        captured_config: dict[str, Any] = {}

        async def mock_astream(
            state: Any, config: dict[str, Any], **kwargs: Any
        ) -> Any:
            captured_config.update(config)
            return
            yield  # Make it a generator

        mock_graph = MagicMock()
        mock_graph.astream = mock_astream

        initial_state = create_initial_state("Test", "test-id")

        async for _ in run_agent(mock_graph, initial_state, "my-thread"):
            pass

        assert captured_config["configurable"]["thread_id"] == "my-thread"


class TestNodeStubs:
    """Tests for node stub implementations."""

    def test_understand_returns_empty_dict(self) -> None:
        """understand node should return empty dict (pass-through)."""
        from backend.agent.nodes import understand

        state = _create_minimal_state()
        result = understand(state)
        assert result == {}

    def test_generate_plan_returns_empty_dict(self) -> None:
        """generate_plan node should return empty dict (pass-through)."""
        from backend.agent.nodes import generate_plan

        state = _create_minimal_state()
        result = generate_plan(state)
        assert result == {}

    def test_await_approval_sets_awaiting_user(self) -> None:
        """await_approval node should set awaiting_user flag."""
        from backend.agent.nodes import await_approval

        state = _create_minimal_state()
        result = await_approval(state)
        assert result["awaiting_user"] is True

    def test_execute_step_increments_current_step(self) -> None:
        """execute_step node should increment current_step."""
        from backend.agent.nodes import execute_step

        state = _create_minimal_state(current_step=0)
        result = execute_step(state)
        assert result["current_step"] == 1

    def test_create_checkpoint_creates_checkpoint_record(self) -> None:
        """create_checkpoint node should create checkpoint record."""
        from backend.agent.nodes import create_checkpoint

        state = _create_minimal_state(current_step=5)
        result = create_checkpoint(state)

        assert "checkpoints" in result
        assert len(result["checkpoints"]) == 1
        cp = result["checkpoints"][0]
        assert cp["step_index"] == 5
        assert cp["trigger_reason"] == "percentage"
        assert result["awaiting_user"] is True

    def test_complete_returns_awaiting_user_false(self) -> None:
        """complete node should set awaiting_user to False."""
        from backend.agent.nodes import complete

        state = _create_minimal_state(awaiting_user=True)
        result = complete(state)
        assert result["awaiting_user"] is False
