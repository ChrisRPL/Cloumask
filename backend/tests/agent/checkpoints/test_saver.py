"""Tests for SQLiteCheckpointSaver implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.agent.checkpoints.saver import (
    SQLiteCheckpointSaver,
    STATUS_ACTIVE,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
)


@pytest.fixture
def saver(tmp_path: Path) -> SQLiteCheckpointSaver:
    """Create test saver with temp database."""
    db_path = tmp_path / "test_checkpoints.db"
    return SQLiteCheckpointSaver(str(db_path))


@pytest.fixture
def memory_saver() -> SQLiteCheckpointSaver:
    """Create test saver with in-memory database."""
    return SQLiteCheckpointSaver(":memory:")


class TestSQLiteCheckpointSaverInit:
    """Tests for SQLiteCheckpointSaver initialization."""

    def test_creates_database_file(self, tmp_path: Path) -> None:
        """Should create database file on initialization."""
        db_path = tmp_path / "new_checkpoint.db"
        SQLiteCheckpointSaver(str(db_path))
        assert db_path.exists()

    def test_works_with_memory_database(self) -> None:
        """Should work with in-memory database."""
        saver = SQLiteCheckpointSaver(":memory:")
        assert saver.db_path == ":memory:"

    def test_creates_tables(self, saver: SQLiteCheckpointSaver) -> None:
        """Should create required tables on initialization."""
        with saver._get_conn() as conn:
            # Check threads table exists
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='threads'"
            ).fetchone()
            assert result is not None

            # Check checkpoint_snapshots table exists
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoint_snapshots'"
            ).fetchone()
            assert result is not None

            # Check writes table exists
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='writes'"
            ).fetchone()
            assert result is not None


class TestThreadManagement:
    """Tests for thread management methods."""

    def test_ensure_thread_creates_new_thread(self, saver: SQLiteCheckpointSaver) -> None:
        """Should create thread if it doesn't exist."""
        saver.ensure_thread("thread-1", title="Test Thread")
        thread = saver.get_thread("thread-1")

        assert thread is not None
        assert thread["thread_id"] == "thread-1"
        assert thread["title"] == "Test Thread"
        assert thread["status"] == STATUS_ACTIVE

    def test_ensure_thread_updates_existing(self, saver: SQLiteCheckpointSaver) -> None:
        """Should update existing thread."""
        saver.ensure_thread("thread-1", title="Original")
        saver.ensure_thread("thread-1", title="Updated")

        thread = saver.get_thread("thread-1")
        assert thread is not None
        assert thread["title"] == "Updated"

    def test_ensure_thread_with_metadata(self, saver: SQLiteCheckpointSaver) -> None:
        """Should store thread metadata as JSON."""
        metadata = {"key": "value", "nested": {"data": 123}}
        saver.ensure_thread("thread-1", metadata=metadata)

        thread = saver.get_thread("thread-1")
        assert thread is not None
        assert thread["metadata"] == metadata

    def test_get_thread_returns_none_for_missing(
        self, saver: SQLiteCheckpointSaver
    ) -> None:
        """Should return None for non-existent thread."""
        assert saver.get_thread("missing-thread") is None

    def test_update_thread_status(self, saver: SQLiteCheckpointSaver) -> None:
        """Should update thread status."""
        saver.ensure_thread("thread-1")

        saver.update_thread_status("thread-1", STATUS_COMPLETED)
        thread = saver.get_thread("thread-1")
        assert thread is not None
        assert thread["status"] == STATUS_COMPLETED

        saver.update_thread_status("thread-1", STATUS_CANCELLED)
        thread = saver.get_thread("thread-1")
        assert thread is not None
        assert thread["status"] == STATUS_CANCELLED

    def test_update_thread_status_invalid(self, saver: SQLiteCheckpointSaver) -> None:
        """Should raise error for invalid status."""
        saver.ensure_thread("thread-1")

        with pytest.raises(ValueError, match="Invalid status"):
            saver.update_thread_status("thread-1", "invalid")

    def test_list_threads_all(self, saver: SQLiteCheckpointSaver) -> None:
        """Should list all threads."""
        saver.ensure_thread("thread-1")
        saver.ensure_thread("thread-2")
        saver.ensure_thread("thread-3")

        threads = saver.list_threads()
        assert len(threads) == 3

    def test_list_threads_by_status(self, saver: SQLiteCheckpointSaver) -> None:
        """Should filter threads by status."""
        saver.ensure_thread("thread-1")
        saver.ensure_thread("thread-2")
        saver.ensure_thread("thread-3")

        saver.update_thread_status("thread-2", STATUS_COMPLETED)

        active = saver.list_threads(status=STATUS_ACTIVE)
        assert len(active) == 2

        completed = saver.list_threads(status=STATUS_COMPLETED)
        assert len(completed) == 1
        assert completed[0]["thread_id"] == "thread-2"

    def test_list_threads_with_limit(self, saver: SQLiteCheckpointSaver) -> None:
        """Should respect limit parameter."""
        for i in range(10):
            saver.ensure_thread(f"thread-{i}")

        threads = saver.list_threads(limit=5)
        assert len(threads) == 5

    def test_delete_thread(self, saver: SQLiteCheckpointSaver) -> None:
        """Should delete thread and return True."""
        saver.ensure_thread("thread-1")
        assert saver.delete_thread("thread-1") is True
        assert saver.get_thread("thread-1") is None

    def test_delete_thread_missing(self, saver: SQLiteCheckpointSaver) -> None:
        """Should return False for missing thread."""
        assert saver.delete_thread("missing") is False


class TestCheckpointSnapshots:
    """Tests for checkpoint snapshot operations."""

    def test_put_and_get_snapshot(self, saver: SQLiteCheckpointSaver) -> None:
        """Should store and retrieve checkpoint snapshot."""
        checkpoint_data = {
            "id": "ckpt-1",
            "channel_values": {"messages": [{"content": "Hello"}]},
        }

        saver.put_snapshot("thread-1", "ckpt-1", checkpoint_data)
        result = saver.get_snapshot("thread-1", "ckpt-1")

        assert result is not None
        assert result["checkpoint_data"] == checkpoint_data

    def test_get_snapshot_latest(self, saver: SQLiteCheckpointSaver) -> None:
        """Should get latest snapshot when no ID specified."""
        for i in range(3):
            saver.put_snapshot(
                "thread-1",
                f"ckpt-{i}",
                {"id": f"ckpt-{i}", "step": i},
            )

        result = saver.get_snapshot("thread-1")
        assert result is not None
        assert result["checkpoint_data"]["id"] == "ckpt-2"

    def test_get_snapshot_specific(self, saver: SQLiteCheckpointSaver) -> None:
        """Should get specific snapshot by ID."""
        for i in range(3):
            saver.put_snapshot(
                "thread-1",
                f"ckpt-{i}",
                {"id": f"ckpt-{i}", "step": i},
            )

        result = saver.get_snapshot("thread-1", "ckpt-1")
        assert result is not None
        assert result["checkpoint_data"]["id"] == "ckpt-1"

    def test_get_snapshot_missing(self, saver: SQLiteCheckpointSaver) -> None:
        """Should return None for missing snapshot."""
        assert saver.get_snapshot("missing-thread") is None
        saver.ensure_thread("thread-1")
        assert saver.get_snapshot("thread-1", "missing-ckpt") is None

    def test_put_snapshot_with_metadata(self, saver: SQLiteCheckpointSaver) -> None:
        """Should store snapshot with metadata."""
        saver.put_snapshot(
            "thread-1",
            "ckpt-1",
            {"data": "value"},
            metadata={"step": 5, "trigger": "percentage"},
        )

        result = saver.get_snapshot("thread-1", "ckpt-1")
        assert result is not None
        assert result["metadata"]["step"] == 5
        assert result["metadata"]["trigger"] == "percentage"

    def test_put_snapshot_replaces_existing(self, saver: SQLiteCheckpointSaver) -> None:
        """Should replace existing snapshot with same ID."""
        saver.put_snapshot("thread-1", "ckpt-1", {"version": 1})
        saver.put_snapshot("thread-1", "ckpt-1", {"version": 2})

        result = saver.get_snapshot("thread-1", "ckpt-1")
        assert result is not None
        assert result["checkpoint_data"]["version"] == 2

    def test_list_snapshots(self, saver: SQLiteCheckpointSaver) -> None:
        """Should list snapshots for a thread."""
        for i in range(5):
            saver.put_snapshot("thread-1", f"ckpt-{i}", {"step": i})

        snapshots = saver.list_snapshots("thread-1")
        assert len(snapshots) == 5
        # Should be in reverse chronological order
        assert snapshots[0]["checkpoint_id"] == "ckpt-4"

    def test_list_snapshots_with_limit(self, saver: SQLiteCheckpointSaver) -> None:
        """Should respect limit parameter."""
        for i in range(10):
            saver.put_snapshot("thread-1", f"ckpt-{i}", {"step": i})

        snapshots = saver.list_snapshots("thread-1", limit=3)
        assert len(snapshots) == 3

    def test_delete_thread_cascades_to_snapshots(
        self, saver: SQLiteCheckpointSaver
    ) -> None:
        """Should delete snapshots when thread is deleted."""
        saver.put_snapshot("thread-1", "ckpt-1", {"data": "value"})
        saver.put_snapshot("thread-1", "ckpt-2", {"data": "value"})

        saver.delete_thread("thread-1")

        assert saver.get_snapshot("thread-1") is None
        assert saver.list_snapshots("thread-1") == []


class TestPendingWrites:
    """Tests for pending writes operations."""

    def test_put_and_get_writes(self, saver: SQLiteCheckpointSaver) -> None:
        """Should store and retrieve pending writes."""
        saver.ensure_thread("thread-1")
        writes = [("channel1", {"value": 1}), ("channel2", {"value": 2})]

        saver.put_writes("thread-1", "ckpt-1", writes)
        result = saver.get_writes("thread-1", "ckpt-1")

        assert len(result) == 2
        assert result[0] == ("channel1", {"value": 1})
        assert result[1] == ("channel2", {"value": 2})

    def test_get_writes_empty(self, saver: SQLiteCheckpointSaver) -> None:
        """Should return empty list for no writes."""
        saver.ensure_thread("thread-1")
        result = saver.get_writes("thread-1", "ckpt-1")
        assert result == []

    def test_clear_writes(self, saver: SQLiteCheckpointSaver) -> None:
        """Should clear pending writes."""
        saver.ensure_thread("thread-1")
        saver.put_writes("thread-1", "ckpt-1", [("ch", "val")])

        saver.clear_writes("thread-1", "ckpt-1")

        result = saver.get_writes("thread-1", "ckpt-1")
        assert result == []


class TestCleanupAndStats:
    """Tests for cleanup and statistics methods."""

    def test_cleanup_old_threads(self, saver: SQLiteCheckpointSaver) -> None:
        """Should delete old completed/cancelled threads."""
        # Create threads
        saver.ensure_thread("old-completed")
        saver.ensure_thread("old-cancelled")
        saver.ensure_thread("active-thread")

        # Mark statuses
        saver.update_thread_status("old-completed", STATUS_COMPLETED)
        saver.update_thread_status("old-cancelled", STATUS_CANCELLED)

        # Manually backdate old threads
        with saver._get_conn() as conn:
            conn.execute(
                """
                UPDATE threads SET updated_at = datetime('now', '-10 days')
                WHERE thread_id IN ('old-completed', 'old-cancelled')
                """
            )

        deleted = saver.cleanup_old_threads(days=7)
        assert deleted == 2

        # Active thread should remain
        assert saver.get_thread("active-thread") is not None
        assert saver.get_thread("old-completed") is None
        assert saver.get_thread("old-cancelled") is None

    def test_cleanup_does_not_delete_active(
        self, saver: SQLiteCheckpointSaver
    ) -> None:
        """Should not delete active threads regardless of age."""
        saver.ensure_thread("old-active")

        # Manually backdate
        with saver._get_conn() as conn:
            conn.execute(
                """
                UPDATE threads SET updated_at = datetime('now', '-30 days')
                WHERE thread_id = 'old-active'
                """
            )

        deleted = saver.cleanup_old_threads(days=7)
        assert deleted == 0
        assert saver.get_thread("old-active") is not None

    def test_get_storage_stats(self, saver: SQLiteCheckpointSaver) -> None:
        """Should return storage statistics."""
        # Create some data
        saver.ensure_thread("thread-1")
        saver.ensure_thread("thread-2")
        saver.ensure_thread("thread-3")
        saver.update_thread_status("thread-2", STATUS_COMPLETED)
        saver.update_thread_status("thread-3", STATUS_CANCELLED)

        saver.put_snapshot("thread-1", "ckpt-1", {"data": "value"})
        saver.put_writes("thread-1", "ckpt-1", [("ch", "val")])

        stats = saver.get_storage_stats()

        assert stats["total_threads"] == 3
        assert stats["active_threads"] == 1
        assert stats["completed_threads"] == 1
        assert stats["cancelled_threads"] == 1
        assert stats["total_snapshots"] == 1
        assert stats["total_writes"] == 1
        assert "db_size_bytes" in stats


class TestComplexDataTypes:
    """Tests for handling complex data types."""

    def test_snapshot_with_nested_objects(self, saver: SQLiteCheckpointSaver) -> None:
        """Should handle deeply nested objects."""
        complex_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "list": [1, 2, {"nested": True}],
                        "tuple_like": [1, 2, 3],
                    }
                }
            },
            "bytes_data": b"binary content",
        }

        saver.put_snapshot("thread-1", "ckpt-1", complex_data)
        result = saver.get_snapshot("thread-1", "ckpt-1")

        assert result is not None
        assert result["checkpoint_data"] == complex_data

    def test_snapshot_with_datetime(self, saver: SQLiteCheckpointSaver) -> None:
        """Should handle datetime objects via pickle."""
        from datetime import datetime

        data = {
            "created": datetime.now(),
            "messages": [{"timestamp": datetime.now()}],
        }

        saver.put_snapshot("thread-1", "ckpt-1", data)
        result = saver.get_snapshot("thread-1", "ckpt-1")

        assert result is not None
        assert isinstance(result["checkpoint_data"]["created"], datetime)
