# Checkpoint Persistence

> **Status:** 🔴 Not Started
> **Priority:** P0 (Critical)
> **Dependencies:** 01-state-types, 02-langgraph-core
> **Estimated Complexity:** Medium

## Overview

Implement SQLite-based persistence for LangGraph checkpoints, enabling pipeline resume capability after app restarts. This includes the database schema, checkpoint saver, and resume logic.

## Goals

- [ ] SQLite schema for checkpoint storage
- [ ] LangGraph checkpoint saver implementation
- [ ] Resume from checkpoint logic
- [ ] Checkpoint cleanup and expiration
- [ ] Thread/conversation management

## Technical Design

### Database Schema

```sql
-- checkpoints.sql

-- Main checkpoints table
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_id TEXT,
    checkpoint_data BLOB NOT NULL,  -- Serialized state
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(thread_id, checkpoint_id)
);

CREATE INDEX idx_checkpoints_thread ON checkpoints(thread_id);
CREATE INDEX idx_checkpoints_created ON checkpoints(created_at);

-- Thread metadata
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active',  -- active, completed, cancelled
    title TEXT,
    input_path TEXT,
    metadata TEXT  -- JSON
);

CREATE INDEX idx_threads_status ON threads(status);
CREATE INDEX idx_threads_updated ON threads(updated_at);

-- Writes table for LangGraph (stores pending writes)
CREATE TABLE IF NOT EXISTS writes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    value BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
);

CREATE INDEX idx_writes_thread_checkpoint ON writes(thread_id, checkpoint_id);
```

### Checkpoint Saver

```python
import sqlite3
import pickle
from typing import Any, Dict, Iterator, Optional, Tuple
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata


class SQLiteCheckpointSaver(BaseCheckpointSaver):
    """
    SQLite-based checkpoint saver for LangGraph.

    Stores serialized state in a local SQLite database,
    enabling resume capability across app restarts.
    """

    def __init__(self, db_path: str = "checkpoints.db"):
        super().__init__()
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    parent_id TEXT,
                    checkpoint_data BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(thread_id, checkpoint_id)
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
                    ON checkpoints(thread_id);

                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    title TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS writes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    value BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_writes_thread_checkpoint
                    ON writes(thread_id, checkpoint_id);
            """)

    @contextmanager
    def _get_conn(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> Dict[str, Any]:
        """Store a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        parent_id = config.get("configurable", {}).get("checkpoint_id")

        # Serialize checkpoint data
        checkpoint_data = pickle.dumps(checkpoint)
        metadata_json = pickle.dumps(metadata)

        with self._get_conn() as conn:
            # Ensure thread exists
            conn.execute("""
                INSERT OR IGNORE INTO threads (thread_id)
                VALUES (?)
            """, (thread_id,))

            # Update thread timestamp
            conn.execute("""
                UPDATE threads SET updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
            """, (thread_id,))

            # Store checkpoint
            conn.execute("""
                INSERT OR REPLACE INTO checkpoints
                (id, thread_id, checkpoint_id, parent_id, checkpoint_data, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                f"{thread_id}:{checkpoint_id}",
                thread_id,
                checkpoint_id,
                parent_id,
                checkpoint_data,
                metadata_json,
            ))

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def get(
        self,
        config: Dict[str, Any],
    ) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a thread."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        with self._get_conn() as conn:
            if checkpoint_id:
                # Get specific checkpoint
                row = conn.execute("""
                    SELECT checkpoint_data FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_id = ?
                """, (thread_id, checkpoint_id)).fetchone()
            else:
                # Get latest checkpoint
                row = conn.execute("""
                    SELECT checkpoint_data FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (thread_id,)).fetchone()

            if row:
                return pickle.loads(row["checkpoint_data"])
            return None

    def get_tuple(
        self,
        config: Dict[str, Any],
    ) -> Optional[Tuple[Checkpoint, CheckpointMetadata]]:
        """Get checkpoint with metadata."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        with self._get_conn() as conn:
            if checkpoint_id:
                row = conn.execute("""
                    SELECT checkpoint_data, metadata, parent_id FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_id = ?
                """, (thread_id, checkpoint_id)).fetchone()
            else:
                row = conn.execute("""
                    SELECT checkpoint_data, metadata, parent_id FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (thread_id,)).fetchone()

            if row:
                checkpoint = pickle.loads(row["checkpoint_data"])
                metadata = pickle.loads(row["metadata"]) if row["metadata"] else {}
                return (checkpoint, metadata)
            return None

    def list(
        self,
        config: Dict[str, Any],
        *,
        before: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Tuple[Checkpoint, CheckpointMetadata]]:
        """List checkpoints for a thread."""
        thread_id = config["configurable"]["thread_id"]

        query = """
            SELECT checkpoint_data, metadata FROM checkpoints
            WHERE thread_id = ?
        """
        params = [thread_id]

        if before:
            query += " AND created_at < (SELECT created_at FROM checkpoints WHERE checkpoint_id = ?)"
            params.append(before)

        query += " ORDER BY created_at DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_conn() as conn:
            for row in conn.execute(query, params):
                checkpoint = pickle.loads(row["checkpoint_data"])
                metadata = pickle.loads(row["metadata"]) if row["metadata"] else {}
                yield (checkpoint, metadata)

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: list[Tuple[str, Any]],
    ) -> None:
        """Store pending writes."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id", "")

        with self._get_conn() as conn:
            for channel, value in writes:
                conn.execute("""
                    INSERT INTO writes (thread_id, checkpoint_id, channel, value)
                    VALUES (?, ?, ?, ?)
                """, (thread_id, checkpoint_id, channel, pickle.dumps(value)))

    def get_writes(
        self,
        config: Dict[str, Any],
    ) -> list[Tuple[str, Any]]:
        """Get pending writes."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id", "")

        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT channel, value FROM writes
                WHERE thread_id = ? AND checkpoint_id = ?
                ORDER BY id
            """, (thread_id, checkpoint_id)).fetchall()

            return [(row["channel"], pickle.loads(row["value"])) for row in rows]
```

### Resume Logic

```python
from datetime import datetime, timedelta
from typing import Optional, List

from agent.state import PipelineState


class CheckpointManager:
    """
    High-level checkpoint management for the agent.

    Handles resume logic, thread management, and cleanup.
    """

    def __init__(self, saver: SQLiteCheckpointSaver):
        self.saver = saver

    def can_resume(self, thread_id: str) -> bool:
        """Check if a thread has resumable state."""
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = self.saver.get(config)
        return checkpoint is not None

    def get_resume_state(self, thread_id: str) -> Optional[dict]:
        """Get the state to resume from."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.saver.get_tuple(config)

        if not result:
            return None

        checkpoint, metadata = result
        return {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "thread_id": thread_id,
        }

    def get_thread_summary(self, thread_id: str) -> Optional[dict]:
        """Get summary of a thread's current state."""
        state = self.get_resume_state(thread_id)
        if not state:
            return None

        checkpoint = state["checkpoint"]
        values = checkpoint.get("channel_values", {})

        plan = values.get("plan", [])
        current_step = values.get("current_step", 0)
        completed = sum(1 for s in plan if s.get("status") == "completed")

        return {
            "thread_id": thread_id,
            "total_steps": len(plan),
            "completed_steps": completed,
            "current_step": current_step,
            "progress_percent": (completed / len(plan) * 100) if plan else 0,
            "awaiting_user": values.get("awaiting_user", False),
            "last_message": values.get("messages", [{}])[-1].get("content", ""),
        }

    def list_active_threads(self) -> List[dict]:
        """List all threads with resumable state."""
        with self.saver._get_conn() as conn:
            rows = conn.execute("""
                SELECT thread_id, status, title, updated_at, metadata
                FROM threads
                WHERE status = 'active'
                ORDER BY updated_at DESC
            """).fetchall()

            threads = []
            for row in rows:
                summary = self.get_thread_summary(row["thread_id"])
                if summary:
                    summary["title"] = row["title"]
                    summary["updated_at"] = row["updated_at"]
                    threads.append(summary)

            return threads

    def mark_completed(self, thread_id: str) -> None:
        """Mark a thread as completed."""
        with self.saver._get_conn() as conn:
            conn.execute("""
                UPDATE threads
                SET status = 'completed', updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
            """, (thread_id,))

    def mark_cancelled(self, thread_id: str) -> None:
        """Mark a thread as cancelled."""
        with self.saver._get_conn() as conn:
            conn.execute("""
                UPDATE threads
                SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
            """, (thread_id,))

    def delete_thread(self, thread_id: str) -> None:
        """Delete a thread and all its checkpoints."""
        with self.saver._get_conn() as conn:
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))

    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """Delete checkpoints older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)

        with self.saver._get_conn() as conn:
            # Get threads to delete
            rows = conn.execute("""
                SELECT thread_id FROM threads
                WHERE updated_at < ? AND status IN ('completed', 'cancelled')
            """, (cutoff,)).fetchall()

            thread_ids = [row["thread_id"] for row in rows]

            if not thread_ids:
                return 0

            placeholders = ",".join("?" * len(thread_ids))

            # Delete writes
            conn.execute(f"""
                DELETE FROM writes WHERE thread_id IN ({placeholders})
            """, thread_ids)

            # Delete checkpoints
            conn.execute(f"""
                DELETE FROM checkpoints WHERE thread_id IN ({placeholders})
            """, thread_ids)

            # Delete threads
            conn.execute(f"""
                DELETE FROM threads WHERE thread_id IN ({placeholders})
            """, thread_ids)

            return len(thread_ids)

    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        with self.saver._get_conn() as conn:
            stats = {}

            row = conn.execute("SELECT COUNT(*) as count FROM threads").fetchone()
            stats["total_threads"] = row["count"]

            row = conn.execute("""
                SELECT COUNT(*) as count FROM threads WHERE status = 'active'
            """).fetchone()
            stats["active_threads"] = row["count"]

            row = conn.execute("SELECT COUNT(*) as count FROM checkpoints").fetchone()
            stats["total_checkpoints"] = row["count"]

            # Database size
            stats["db_size_bytes"] = self.saver.db_path.stat().st_size

            return stats
```

### Integration with Graph

```python
from agent.checkpoints import SQLiteCheckpointSaver, CheckpointManager


def create_compiled_graph(
    checkpoint_path: str = "data/checkpoints.db"
) -> tuple:
    """
    Create compiled graph with checkpoint persistence.

    Returns:
        (compiled_graph, checkpoint_manager)
    """
    from agent.graph import create_agent_graph

    # Create checkpoint saver
    saver = SQLiteCheckpointSaver(checkpoint_path)
    manager = CheckpointManager(saver)

    # Create and compile graph
    graph = create_agent_graph()
    compiled = graph.compile(checkpointer=saver)

    return compiled, manager


async def resume_pipeline(
    thread_id: str,
    user_message: str,
    compiled_graph,
    manager: CheckpointManager,
) -> AsyncGenerator[dict, None]:
    """
    Resume a pipeline from checkpoint.

    If no checkpoint exists, starts fresh.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Check for existing state
    if manager.can_resume(thread_id):
        # Get existing state
        result = manager.get_resume_state(thread_id)
        checkpoint = result["checkpoint"]

        # Add new user message to state
        state = checkpoint.get("channel_values", {})
        state["messages"].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat(),
        })
        state["awaiting_user"] = False

        # Resume from checkpoint
        async for event in compiled_graph.astream(
            state,
            config,
        ):
            yield event
    else:
        # Fresh start
        from agent.state import create_initial_state
        state = create_initial_state(user_message, thread_id)

        async for event in compiled_graph.astream(
            state,
            config,
        ):
            yield event
```

## Implementation Tasks

- [ ] Create `backend/agent/checkpoints/__init__.py`
- [ ] Create `backend/agent/checkpoints/saver.py`
  - [ ] Implement `SQLiteCheckpointSaver` class
  - [ ] Implement `put()` method
  - [ ] Implement `get()` and `get_tuple()` methods
  - [ ] Implement `list()` method
  - [ ] Implement `put_writes()` and `get_writes()` methods
- [ ] Create `backend/agent/checkpoints/manager.py`
  - [ ] Implement `CheckpointManager` class
  - [ ] Implement `can_resume()` method
  - [ ] Implement `get_resume_state()` method
  - [ ] Implement `list_active_threads()` method
  - [ ] Implement cleanup methods
- [ ] Create `backend/agent/checkpoints/schema.sql`
- [ ] Add resume endpoints to API
- [ ] Add background cleanup task

## Testing

### Unit Tests

```python
# tests/agent/checkpoints/test_saver.py

@pytest.fixture
def saver(tmp_path):
    """Create test saver with temp database."""
    db_path = tmp_path / "test_checkpoints.db"
    return SQLiteCheckpointSaver(str(db_path))


def test_saver_put_and_get(saver):
    """Should store and retrieve checkpoint."""
    config = {"configurable": {"thread_id": "test-thread"}}
    checkpoint = {
        "id": "ckpt-1",
        "channel_values": {"messages": [{"content": "Hello"}]},
    }
    metadata = {"step": 1}

    saver.put(config, checkpoint, metadata)
    result = saver.get(config)

    assert result is not None
    assert result["id"] == "ckpt-1"
    assert result["channel_values"]["messages"][0]["content"] == "Hello"


def test_saver_get_latest(saver):
    """Should get latest checkpoint when no ID specified."""
    config = {"configurable": {"thread_id": "test-thread"}}

    # Store multiple checkpoints
    for i in range(3):
        checkpoint = {"id": f"ckpt-{i}", "channel_values": {"step": i}}
        saver.put(config, checkpoint, {})

    result = saver.get(config)
    assert result["id"] == "ckpt-2"


def test_saver_list_checkpoints(saver):
    """Should list checkpoints in order."""
    config = {"configurable": {"thread_id": "test-thread"}}

    for i in range(5):
        checkpoint = {"id": f"ckpt-{i}", "channel_values": {}}
        saver.put(config, checkpoint, {})

    checkpoints = list(saver.list(config, limit=3))
    assert len(checkpoints) == 3
    assert checkpoints[0][0]["id"] == "ckpt-4"  # Latest first
```

### Manager Tests

```python
# tests/agent/checkpoints/test_manager.py

@pytest.fixture
def manager(tmp_path):
    """Create test manager."""
    saver = SQLiteCheckpointSaver(str(tmp_path / "test.db"))
    return CheckpointManager(saver)


def test_can_resume_false_initially(manager):
    """Should return False for new thread."""
    assert manager.can_resume("new-thread") == False


def test_can_resume_true_after_checkpoint(manager):
    """Should return True after checkpoint stored."""
    config = {"configurable": {"thread_id": "test-thread"}}
    checkpoint = {"id": "ckpt-1", "channel_values": {}}
    manager.saver.put(config, checkpoint, {})

    assert manager.can_resume("test-thread") == True


def test_list_active_threads(manager):
    """Should list only active threads."""
    # Create some threads
    for i in range(3):
        config = {"configurable": {"thread_id": f"thread-{i}"}}
        checkpoint = {"id": "ckpt-1", "channel_values": {"plan": [], "messages": []}}
        manager.saver.put(config, checkpoint, {})

    # Mark one as completed
    manager.mark_completed("thread-1")

    threads = manager.list_active_threads()
    thread_ids = [t["thread_id"] for t in threads]

    assert "thread-0" in thread_ids
    assert "thread-1" not in thread_ids
    assert "thread-2" in thread_ids


def test_cleanup_old_checkpoints(manager):
    """Should delete old completed threads."""
    # Create old thread
    config = {"configurable": {"thread_id": "old-thread"}}
    checkpoint = {"id": "ckpt-1", "channel_values": {}}
    manager.saver.put(config, checkpoint, {})
    manager.mark_completed("old-thread")

    # Manually backdate
    with manager.saver._get_conn() as conn:
        conn.execute("""
            UPDATE threads SET updated_at = datetime('now', '-10 days')
            WHERE thread_id = 'old-thread'
        """)

    deleted = manager.cleanup_old_checkpoints(days=7)
    assert deleted == 1
    assert not manager.can_resume("old-thread")
```

### Integration Tests

```python
# tests/agent/checkpoints/test_integration.py

@pytest.mark.asyncio
async def test_resume_pipeline_flow(tmp_path):
    """Test full resume flow."""
    compiled, manager = create_compiled_graph(
        str(tmp_path / "test.db")
    )

    # First run - should create checkpoint
    events1 = []
    async for event in resume_pipeline(
        "test-thread",
        "Scan /data",
        compiled,
        manager,
    ):
        events1.append(event)

    assert manager.can_resume("test-thread")

    # Resume - should continue from checkpoint
    events2 = []
    async for event in resume_pipeline(
        "test-thread",
        "Yes, proceed",
        compiled,
        manager,
    ):
        events2.append(event)

    # Should have continued, not restarted
    summary = manager.get_thread_summary("test-thread")
    assert summary is not None
```

## Acceptance Criteria

- [ ] Checkpoints saved to SQLite after each node
- [ ] Pipeline can resume after app restart
- [ ] Thread list shows all active pipelines
- [ ] Completed/cancelled threads marked correctly
- [ ] Old checkpoints cleaned up automatically
- [ ] Storage stats available for monitoring
- [ ] Resume adds new message to existing state

## Files to Create/Modify

```
backend/
├── agent/
│   └── checkpoints/
│       ├── __init__.py      # Exports
│       ├── saver.py         # SQLiteCheckpointSaver
│       ├── manager.py       # CheckpointManager
│       └── schema.sql       # Database schema
├── data/
│   └── .gitkeep             # For checkpoints.db
└── tests/
    └── agent/
        └── checkpoints/
            ├── test_saver.py
            ├── test_manager.py
            └── test_integration.py
```

## API Endpoints

```python
# backend/api/routes/threads.py

@router.get("/threads")
async def list_threads():
    """List all resumable threads."""
    return manager.list_active_threads()


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get thread summary."""
    return manager.get_thread_summary(thread_id)


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread and its checkpoints."""
    manager.delete_thread(thread_id)
    return {"status": "deleted"}


@router.post("/threads/{thread_id}/cancel")
async def cancel_thread(thread_id: str):
    """Cancel an active thread."""
    manager.mark_cancelled(thread_id)
    return {"status": "cancelled"}
```

## Notes

- SQLite is sufficient for single-user desktop app
- Consider WAL mode for better concurrency
- Pickle is used for simplicity; consider msgpack for production
- Add database migrations if schema changes
- Background cleanup should run on app startup
