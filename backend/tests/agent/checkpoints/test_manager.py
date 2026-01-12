"""Tests for CheckpointManager implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agent.checkpoints.manager import CheckpointManager
from backend.agent.checkpoints.saver import (
    STATUS_ACTIVE,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
)


@pytest.fixture
def manager(tmp_path: Path) -> CheckpointManager:
    """Create test manager with temp database."""
    db_path = tmp_path / "test_checkpoints.db"
    return CheckpointManager(str(db_path))


def _create_checkpoint_data(
    messages: list[dict] | None = None,
    plan: list[dict] | None = None,
    current_step: int = 0,
    awaiting_user: bool = False,
) -> dict[str, Any]:
    """Create test checkpoint data."""
    return {
        "channel_values": {
            "messages": messages or [],
            "plan": plan or [],
            "current_step": current_step,
            "awaiting_user": awaiting_user,
        }
    }


class TestCanResume:
    """Tests for can_resume method."""

    def test_returns_false_for_new_thread(self, manager: CheckpointManager) -> None:
        """Should return False for non-existent thread."""
        assert manager.can_resume("new-thread") is False

    def test_returns_false_for_thread_without_snapshot(
        self, manager: CheckpointManager
    ) -> None:
        """Should return False for thread without checkpoint."""
        manager.create_thread("thread-1")
        assert manager.can_resume("thread-1") is False

    def test_returns_true_for_thread_with_snapshot(
        self, manager: CheckpointManager
    ) -> None:
        """Should return True after checkpoint stored."""
        manager.save_snapshot(
            "thread-1",
            "ckpt-1",
            _create_checkpoint_data(),
        )
        assert manager.can_resume("thread-1") is True


class TestGetResumeState:
    """Tests for get_resume_state method."""

    def test_returns_none_for_missing_thread(
        self, manager: CheckpointManager
    ) -> None:
        """Should return None for non-existent thread."""
        assert manager.get_resume_state("missing-thread") is None

    def test_returns_state_for_existing_checkpoint(
        self, manager: CheckpointManager
    ) -> None:
        """Should return state dict for existing checkpoint."""
        checkpoint_data = _create_checkpoint_data(
            messages=[{"role": "user", "content": "Hello"}]
        )
        manager.save_snapshot("thread-1", "ckpt-1", checkpoint_data)

        result = manager.get_resume_state("thread-1")

        assert result is not None
        assert "checkpoint_data" in result
        assert "thread_id" in result
        assert result["thread_id"] == "thread-1"


class TestGetThreadSummary:
    """Tests for get_thread_summary method."""

    def test_returns_none_for_missing_thread(
        self, manager: CheckpointManager
    ) -> None:
        """Should return None for non-existent thread."""
        assert manager.get_thread_summary("missing") is None

    def test_returns_summary_with_progress(
        self, manager: CheckpointManager
    ) -> None:
        """Should calculate progress from plan."""
        plan = [
            {"id": "step-1", "status": "completed"},
            {"id": "step-2", "status": "completed"},
            {"id": "step-3", "status": "pending"},
            {"id": "step-4", "status": "pending"},
        ]
        checkpoint_data = _create_checkpoint_data(plan=plan, current_step=2)
        manager.save_snapshot("thread-1", "ckpt-1", checkpoint_data)

        summary = manager.get_thread_summary("thread-1")

        assert summary is not None
        assert summary["total_steps"] == 4
        assert summary["completed_steps"] == 2
        assert summary["current_step"] == 2
        assert summary["progress_percent"] == 50.0

    def test_returns_summary_with_last_message(
        self, manager: CheckpointManager
    ) -> None:
        """Should include last message content."""
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Last message"},
        ]
        checkpoint_data = _create_checkpoint_data(messages=messages)
        manager.save_snapshot("thread-1", "ckpt-1", checkpoint_data)

        summary = manager.get_thread_summary("thread-1")

        assert summary is not None
        assert summary["last_message"] == "Last message"

    def test_returns_summary_with_awaiting_user(
        self, manager: CheckpointManager
    ) -> None:
        """Should include awaiting_user flag."""
        checkpoint_data = _create_checkpoint_data(awaiting_user=True)
        manager.save_snapshot("thread-1", "ckpt-1", checkpoint_data)

        summary = manager.get_thread_summary("thread-1")

        assert summary is not None
        assert summary["awaiting_user"] is True

    def test_handles_empty_plan(self, manager: CheckpointManager) -> None:
        """Should handle empty plan gracefully."""
        checkpoint_data = _create_checkpoint_data(plan=[])
        manager.save_snapshot("thread-1", "ckpt-1", checkpoint_data)

        summary = manager.get_thread_summary("thread-1")

        assert summary is not None
        assert summary["total_steps"] == 0
        assert summary["progress_percent"] == 0


class TestListActiveThreads:
    """Tests for list_active_threads method."""

    def test_returns_empty_for_no_threads(
        self, manager: CheckpointManager
    ) -> None:
        """Should return empty list when no threads exist."""
        assert manager.list_active_threads() == []

    def test_returns_only_active_threads(
        self, manager: CheckpointManager
    ) -> None:
        """Should only return active threads with checkpoints."""
        # Create threads with checkpoints
        for i in range(3):
            manager.save_snapshot(
                f"thread-{i}",
                "ckpt-1",
                _create_checkpoint_data(),
            )

        # Mark one as completed
        manager.mark_completed("thread-1")

        threads = manager.list_active_threads()
        thread_ids = [t["thread_id"] for t in threads]

        assert "thread-0" in thread_ids
        assert "thread-1" not in thread_ids
        assert "thread-2" in thread_ids

    def test_returns_summaries(self, manager: CheckpointManager) -> None:
        """Should return thread summaries, not just metadata."""
        plan = [{"id": "step-1", "status": "completed"}]
        manager.save_snapshot(
            "thread-1",
            "ckpt-1",
            _create_checkpoint_data(plan=plan),
        )

        threads = manager.list_active_threads()

        assert len(threads) == 1
        assert threads[0]["total_steps"] == 1
        assert threads[0]["completed_steps"] == 1


class TestThreadLifecycle:
    """Tests for thread lifecycle methods."""

    def test_create_thread(self, manager: CheckpointManager) -> None:
        """Should create thread with metadata."""
        result = manager.create_thread(
            "thread-1",
            title="Test Pipeline",
            input_path="/data/input",
            metadata={"key": "value"},
        )

        assert result["thread_id"] == "thread-1"
        assert result["title"] == "Test Pipeline"
        assert result["input_path"] == "/data/input"
        assert result["status"] == STATUS_ACTIVE

    def test_mark_completed(self, manager: CheckpointManager) -> None:
        """Should mark thread as completed."""
        manager.create_thread("thread-1")
        manager.mark_completed("thread-1")

        thread = manager.saver.get_thread("thread-1")
        assert thread is not None
        assert thread["status"] == STATUS_COMPLETED

    def test_mark_cancelled(self, manager: CheckpointManager) -> None:
        """Should mark thread as cancelled."""
        manager.create_thread("thread-1")
        manager.mark_cancelled("thread-1")

        thread = manager.saver.get_thread("thread-1")
        assert thread is not None
        assert thread["status"] == STATUS_CANCELLED

    def test_mark_active(self, manager: CheckpointManager) -> None:
        """Should mark thread as active."""
        manager.create_thread("thread-1")
        manager.mark_completed("thread-1")
        manager.mark_active("thread-1")

        thread = manager.saver.get_thread("thread-1")
        assert thread is not None
        assert thread["status"] == STATUS_ACTIVE

    def test_delete_thread(self, manager: CheckpointManager) -> None:
        """Should delete thread."""
        manager.create_thread("thread-1")
        manager.save_snapshot("thread-1", "ckpt-1", {})

        assert manager.delete_thread("thread-1") is True
        assert manager.can_resume("thread-1") is False

    def test_delete_missing_thread(self, manager: CheckpointManager) -> None:
        """Should return False for missing thread."""
        assert manager.delete_thread("missing") is False


class TestCheckpointOperations:
    """Tests for checkpoint save/get operations."""

    def test_save_and_get_snapshot(self, manager: CheckpointManager) -> None:
        """Should save and retrieve snapshot."""
        state = {"messages": [], "plan": [], "current_step": 5}
        manager.save_snapshot("thread-1", "ckpt-1", state)

        result = manager.get_snapshot("thread-1", "ckpt-1")

        assert result is not None
        assert result["checkpoint_data"]["current_step"] == 5

    def test_save_snapshot_with_metadata(
        self, manager: CheckpointManager
    ) -> None:
        """Should save snapshot with metadata."""
        manager.save_snapshot(
            "thread-1",
            "ckpt-1",
            {"data": "value"},
            metadata={"trigger": "percentage"},
        )

        result = manager.get_snapshot("thread-1")
        assert result is not None
        assert result["metadata"]["trigger"] == "percentage"

    def test_list_snapshots(self, manager: CheckpointManager) -> None:
        """Should list snapshots for thread."""
        for i in range(5):
            manager.save_snapshot("thread-1", f"ckpt-{i}", {"step": i})

        snapshots = manager.list_snapshots("thread-1")

        assert len(snapshots) == 5

    def test_list_snapshots_with_limit(
        self, manager: CheckpointManager
    ) -> None:
        """Should respect limit parameter."""
        for i in range(10):
            manager.save_snapshot("thread-1", f"ckpt-{i}", {"step": i})

        snapshots = manager.list_snapshots("thread-1", limit=3)

        assert len(snapshots) == 3


class TestCleanupAndStats:
    """Tests for cleanup and statistics methods."""

    def test_cleanup_old_checkpoints(self, manager: CheckpointManager) -> None:
        """Should cleanup old completed/cancelled threads."""
        manager.create_thread("old-thread")
        manager.save_snapshot("old-thread", "ckpt-1", {})
        manager.mark_completed("old-thread")

        # Manually backdate
        with manager.saver._get_conn() as conn:
            conn.execute(
                """
                UPDATE threads SET updated_at = datetime('now', '-10 days')
                WHERE thread_id = 'old-thread'
                """
            )

        deleted = manager.cleanup_old_checkpoints(days=7)

        assert deleted == 1
        assert not manager.can_resume("old-thread")

    def test_get_storage_stats(self, manager: CheckpointManager) -> None:
        """Should return storage statistics."""
        manager.create_thread("thread-1")
        manager.save_snapshot("thread-1", "ckpt-1", {})

        stats = manager.get_storage_stats()

        assert stats["total_threads"] == 1
        assert stats["active_threads"] == 1
        assert stats["total_snapshots"] == 1


class TestGetCheckpointer:
    """Tests for get_checkpointer async context manager."""

    @pytest.mark.asyncio
    async def test_yields_async_sqlite_saver(
        self, manager: CheckpointManager
    ) -> None:
        """Should yield an AsyncSqliteSaver instance."""
        async with manager.get_checkpointer() as checkpointer:
            # AsyncSqliteSaver should have async methods
            assert hasattr(checkpointer, "aget")
            assert hasattr(checkpointer, "aput")


class TestListAllThreads:
    """Tests for list_all_threads method."""

    def test_list_all_threads(self, manager: CheckpointManager) -> None:
        """Should list all threads regardless of status."""
        manager.create_thread("thread-1")
        manager.create_thread("thread-2")
        manager.create_thread("thread-3")
        manager.mark_completed("thread-2")

        threads = manager.list_all_threads()

        assert len(threads) == 3

    def test_list_all_threads_filtered(
        self, manager: CheckpointManager
    ) -> None:
        """Should filter by status."""
        manager.create_thread("thread-1")
        manager.create_thread("thread-2")
        manager.mark_completed("thread-2")

        active = manager.list_all_threads(status=STATUS_ACTIVE)
        completed = manager.list_all_threads(status=STATUS_COMPLETED)

        assert len(active) == 1
        assert len(completed) == 1
        assert active[0]["thread_id"] == "thread-1"
        assert completed[0]["thread_id"] == "thread-2"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_non_dict_checkpoint_data(
        self, manager: CheckpointManager
    ) -> None:
        """Should handle checkpoint data without channel_values."""
        # Some checkpoints might have flat structure
        manager.save_snapshot("thread-1", "ckpt-1", {
            "messages": [{"content": "test"}],
            "plan": [],
            "current_step": 0,
            "awaiting_user": False,
        })

        summary = manager.get_thread_summary("thread-1")

        assert summary is not None
        assert summary["last_message"] == "test"

    def test_handles_empty_messages(self, manager: CheckpointManager) -> None:
        """Should handle empty messages list."""
        manager.save_snapshot("thread-1", "ckpt-1", {
            "channel_values": {
                "messages": [],
                "plan": [],
            }
        })

        summary = manager.get_thread_summary("thread-1")

        assert summary is not None
        assert summary["last_message"] == ""
