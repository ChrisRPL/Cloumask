"""Integration tests for checkpoint persistence with LangGraph."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.agent.checkpoints import (
    STATUS_ACTIVE,
    STATUS_COMPLETED,
    CheckpointManager,
    resume_pipeline,
)
from backend.agent.state import create_initial_state


@pytest.fixture
def manager(tmp_path: Path) -> CheckpointManager:
    """Create test manager with temp database."""
    db_path = tmp_path / "integration_test.db"
    return CheckpointManager(str(db_path))


class TestGraphIntegration:
    """Tests for integration with LangGraph."""

    @pytest.mark.asyncio
    async def test_get_checkpointer_returns_compatible_saver(
        self, manager: CheckpointManager
    ) -> None:
        """Checkpointer should be compatible with graph compilation."""
        async with manager.get_checkpointer() as checkpointer:
            # Should have required LangGraph checkpoint methods
            assert hasattr(checkpointer, "aget")
            assert hasattr(checkpointer, "aput")
            assert hasattr(checkpointer, "alist")

    @pytest.mark.asyncio
    async def test_checkpointer_with_graph_compilation(
        self, manager: CheckpointManager
    ) -> None:
        """Checkpointer should work with graph compilation."""
        from backend.agent.graph import create_agent_graph

        graph = create_agent_graph()

        async with manager.get_checkpointer() as checkpointer:
            compiled = graph.compile(checkpointer=checkpointer)
            assert compiled is not None


class TestResumePipeline:
    """Tests for resume_pipeline function."""

    @pytest.mark.asyncio
    async def test_resume_creates_new_thread_when_none_exists(
        self, manager: CheckpointManager
    ) -> None:
        """Should create new thread when no checkpoint exists."""
        mock_graph = MagicMock()
        events_yielded: list[dict] = []

        async def mock_astream(state: Any, config: Any, **kwargs: Any) -> Any:
            events_yielded.append({"state": state})
            return
            yield  # Make it a generator

        mock_graph.astream = mock_astream

        async for _ in resume_pipeline(
            "new-thread",
            "Test message",
            mock_graph,
            manager,
        ):
            pass

        # Should have created the thread
        thread = manager.saver.get_thread("new-thread")
        assert thread is not None
        assert thread["status"] == STATUS_ACTIVE

    @pytest.mark.asyncio
    async def test_resume_adds_message_to_existing_state(
        self, manager: CheckpointManager
    ) -> None:
        """Should add new message when resuming from checkpoint."""
        # Create existing checkpoint
        existing_state = {
            "channel_values": {
                "messages": [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "Response"},
                ],
                "plan": [],
                "current_step": 0,
                "awaiting_user": True,
            }
        }
        manager.save_snapshot("existing-thread", "ckpt-1", existing_state)

        mock_graph = MagicMock()
        captured_state: dict = {}

        async def mock_astream(state: Any, config: Any, **kwargs: Any) -> Any:
            captured_state.update(state)
            return
            yield  # Make it a generator

        mock_graph.astream = mock_astream

        async for _ in resume_pipeline(
            "existing-thread",
            "Continue processing",
            mock_graph,
            manager,
        ):
            pass

        # Should have added the new message
        assert len(captured_state.get("messages", [])) == 3
        assert captured_state["messages"][-1]["content"] == "Continue processing"
        assert captured_state["awaiting_user"] is False

    @pytest.mark.asyncio
    async def test_resume_yields_graph_events(
        self, manager: CheckpointManager
    ) -> None:
        """Should yield events from graph execution."""
        mock_graph = MagicMock()

        async def mock_astream(state: Any, config: Any, **kwargs: Any) -> Any:
            yield {"step": 1}
            yield {"step": 2}
            yield {"step": 3}

        mock_graph.astream = mock_astream

        events = []
        async for event in resume_pipeline(
            "new-thread",
            "Test message",
            mock_graph,
            manager,
        ):
            events.append(event)

        assert len(events) == 3
        assert events[0]["step"] == 1
        assert events[2]["step"] == 3


class TestFullWorkflow:
    """Tests for full checkpoint/resume workflow."""

    def test_save_load_cycle(self, manager: CheckpointManager) -> None:
        """Should persist state across save/load cycle."""
        # Create initial state
        initial_state = create_initial_state("Process images", "pipeline-123")

        # Simulate some progress
        state_with_progress = {
            "channel_values": {
                "messages": initial_state["messages"],
                "plan": [
                    {"id": "step-1", "tool_name": "scan", "status": "completed"},
                    {"id": "step-2", "tool_name": "detect", "status": "running"},
                    {"id": "step-3", "tool_name": "export", "status": "pending"},
                ],
                "plan_approved": True,
                "current_step": 1,
                "execution_results": {
                    "step-1": {"success": True, "files_found": 100},
                },
                "checkpoints": [],
                "awaiting_user": True,
                "metadata": initial_state["metadata"],
                "last_error": None,
                "retry_count": 0,
            }
        }

        # Save checkpoint
        manager.save_snapshot(
            "pipeline-123",
            "ckpt-1",
            state_with_progress,
            metadata={"trigger": "percentage", "progress": 33},
        )

        # Simulate app restart by creating new manager
        new_manager = CheckpointManager(manager.db_path)

        # Verify can resume
        assert new_manager.can_resume("pipeline-123")

        # Load state
        result = new_manager.get_resume_state("pipeline-123")
        assert result is not None

        loaded_state = result["checkpoint_data"]["channel_values"]
        assert loaded_state["current_step"] == 1
        assert loaded_state["plan_approved"] is True
        assert len(loaded_state["plan"]) == 3
        assert loaded_state["execution_results"]["step-1"]["files_found"] == 100

    def test_thread_lifecycle_workflow(
        self, manager: CheckpointManager
    ) -> None:
        """Should track thread through full lifecycle."""
        # Create thread
        manager.create_thread(
            "workflow-1",
            title="Test Workflow",
            input_path="/data/test",
        )

        # Save progress checkpoints
        for i in range(3):
            manager.save_snapshot(
                "workflow-1",
                f"ckpt-{i}",
                {"step": i, "channel_values": {"messages": [], "plan": []}},
            )

        # Verify active
        summary = manager.get_thread_summary("workflow-1")
        assert summary is not None
        assert summary["status"] == STATUS_ACTIVE

        # Mark completed
        manager.mark_completed("workflow-1")

        # Verify completed
        summary = manager.get_thread_summary("workflow-1")
        assert summary is not None
        assert summary["status"] == STATUS_COMPLETED

        # Should still be able to list snapshots
        snapshots = manager.list_snapshots("workflow-1")
        assert len(snapshots) == 3

    def test_multiple_threads_isolation(
        self, manager: CheckpointManager
    ) -> None:
        """Should isolate data between threads."""
        # Create two threads
        manager.save_snapshot("thread-a", "ckpt-1", {"data": "A"})
        manager.save_snapshot("thread-b", "ckpt-1", {"data": "B"})

        # Each should have its own data
        result_a = manager.get_snapshot("thread-a")
        result_b = manager.get_snapshot("thread-b")

        assert result_a is not None
        assert result_b is not None
        assert result_a["checkpoint_data"]["data"] == "A"
        assert result_b["checkpoint_data"]["data"] == "B"


class TestDatabasePersistence:
    """Tests for database persistence across instances."""

    def test_data_persists_across_instances(self, tmp_path: Path) -> None:
        """Data should persist when creating new manager instances."""
        db_path = tmp_path / "persistent.db"

        # First instance - create data
        manager1 = CheckpointManager(str(db_path))
        manager1.create_thread("persistent-thread", title="Persistent")
        manager1.save_snapshot("persistent-thread", "ckpt-1", {"value": 42})

        # Second instance - should see data
        manager2 = CheckpointManager(str(db_path))
        thread = manager2.saver.get_thread("persistent-thread")
        snapshot = manager2.get_snapshot("persistent-thread")

        assert thread is not None
        assert thread["title"] == "Persistent"
        assert snapshot is not None
        assert snapshot["checkpoint_data"]["value"] == 42

    def test_cleanup_persists(self, tmp_path: Path) -> None:
        """Cleanup results should persist."""
        db_path = tmp_path / "cleanup.db"

        # First instance - create and mark completed
        manager1 = CheckpointManager(str(db_path))
        manager1.create_thread("old-thread")
        manager1.save_snapshot("old-thread", "ckpt-1", {})
        manager1.mark_completed("old-thread")

        # Backdate
        with manager1.saver._get_conn() as conn:
            conn.execute(
                """
                UPDATE threads SET updated_at = datetime('now', '-10 days')
                WHERE thread_id = 'old-thread'
                """
            )

        # Second instance - cleanup
        manager2 = CheckpointManager(str(db_path))
        deleted = manager2.cleanup_old_checkpoints(days=7)
        assert deleted == 1

        # Third instance - verify deleted
        manager3 = CheckpointManager(str(db_path))
        assert manager3.saver.get_thread("old-thread") is None


class TestConcurrencyHandling:
    """Tests for concurrent access handling."""

    def test_multiple_snapshots_same_thread(
        self, manager: CheckpointManager
    ) -> None:
        """Should handle multiple snapshots for same thread."""
        # Save many snapshots rapidly
        for i in range(100):
            manager.save_snapshot("thread-1", f"ckpt-{i}", {"step": i})

        # All should be saved
        snapshots = manager.list_snapshots("thread-1")
        assert len(snapshots) == 100

        # Latest should be retrievable
        latest = manager.get_snapshot("thread-1")
        assert latest is not None
        assert latest["checkpoint_data"]["step"] == 99


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_handles_corrupted_metadata(
        self, manager: CheckpointManager
    ) -> None:
        """Should handle threads with invalid JSON metadata gracefully."""
        # Directly insert invalid metadata
        with manager.saver._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO threads (thread_id, metadata)
                VALUES ('invalid-meta', 'not valid json')
                """
            )

        # Should not crash when listing
        threads = manager.list_all_threads()
        # Thread might not appear or might appear with None metadata
        # depending on implementation - just ensure no crash
        assert isinstance(threads, list)

    def test_graceful_handling_of_missing_fields(
        self, manager: CheckpointManager
    ) -> None:
        """Should handle state with missing expected fields."""
        # Save minimal state
        manager.save_snapshot("minimal-thread", "ckpt-1", {})

        # Should not crash when getting summary
        summary = manager.get_thread_summary("minimal-thread")
        assert summary is not None
        assert summary["total_steps"] == 0
        assert summary["progress_percent"] == 0
