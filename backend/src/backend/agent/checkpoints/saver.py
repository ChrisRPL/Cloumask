"""
SQLite-based checkpoint saver with extended thread management.

This module wraps LangGraph's AsyncSqliteSaver with additional
functionality for thread tracking, metadata, and management.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

# Thread status constants
STATUS_ACTIVE = "active"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"


class SQLiteCheckpointSaver:
    """
    SQLite-based checkpoint saver for LangGraph.

    Stores serialized state in a local SQLite database,
    enabling resume capability across app restarts.

    This class manages the thread metadata table and provides
    methods for thread lifecycle management. It works alongside
    LangGraph's AsyncSqliteSaver which handles the actual
    checkpoint storage.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path = "checkpoints.db") -> None:
        """
        Initialize the checkpoint saver.

        Args:
            db_path: Path to SQLite database file.
                Use ":memory:" for in-memory testing.
        """
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema for thread management."""
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            schema_sql = schema_path.read_text()
        else:
            # Fallback inline schema for thread management
            schema_sql = self._get_inline_schema()

        with self._get_conn() as conn:
            conn.executescript(schema_sql)

    def _get_inline_schema(self) -> str:
        """Get inline schema SQL as fallback."""
        return """
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'cancelled')),
                title TEXT,
                input_path TEXT,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
            CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at);

            CREATE TABLE IF NOT EXISTS checkpoint_snapshots (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                parent_id TEXT,
                checkpoint_data BLOB NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(thread_id, checkpoint_id),
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_thread ON checkpoint_snapshots(thread_id);
            CREATE INDEX IF NOT EXISTS idx_snapshots_created ON checkpoint_snapshots(created_at);

            CREATE TABLE IF NOT EXISTS writes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                value BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_writes_thread_checkpoint ON writes(thread_id, checkpoint_id);
        """

    @contextmanager
    def _get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get database connection with context manager.

        Yields:
            SQLite connection with row factory.
        """
        conn = sqlite3.connect(
            str(self.db_path) if isinstance(self.db_path, Path) else self.db_path,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        # Enable foreign key support
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Thread Management
    # -------------------------------------------------------------------------

    def ensure_thread(
        self,
        thread_id: str,
        title: str | None = None,
        input_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Ensure a thread exists in the database.

        Creates the thread if it doesn't exist, or updates
        the timestamp if it does.

        Args:
            thread_id: Unique thread identifier.
            title: Optional human-readable title.
            input_path: Optional input data path.
            metadata: Optional JSON metadata.
        """
        with self._get_conn() as conn:
            # Try to insert, ignore if exists
            conn.execute(
                """
                INSERT OR IGNORE INTO threads (thread_id, title, input_path, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (
                    thread_id,
                    title,
                    input_path,
                    json.dumps(metadata) if metadata else None,
                ),
            )

            # Update timestamp and any provided fields
            updates = ["updated_at = CURRENT_TIMESTAMP"]
            params: list[Any] = []

            if title is not None:
                updates.append("title = ?")
                params.append(title)
            if input_path is not None:
                updates.append("input_path = ?")
                params.append(input_path)
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))

            params.append(thread_id)

            conn.execute(
                f"UPDATE threads SET {', '.join(updates)} WHERE thread_id = ?",
                params,
            )

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """
        Get thread information.

        Args:
            thread_id: Thread identifier.

        Returns:
            Thread data dict or None if not found.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT thread_id, created_at, updated_at, status, title, input_path, metadata
                FROM threads
                WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchone()

            if row:
                # Handle potentially corrupted metadata JSON
                metadata = None
                if row["metadata"]:
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        metadata = {"_raw": row["metadata"], "_error": "invalid_json"}

                return {
                    "thread_id": row["thread_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "status": row["status"],
                    "title": row["title"],
                    "input_path": row["input_path"],
                    "metadata": metadata,
                }
            return None

    def update_thread_status(self, thread_id: str, status: str) -> bool:
        """
        Update thread status.

        Args:
            thread_id: Thread identifier.
            status: New status (active, completed, cancelled).

        Returns:
            True if thread was updated, False if thread not found.

        Raises:
            ValueError: If status is not valid.
        """
        if status not in (STATUS_ACTIVE, STATUS_COMPLETED, STATUS_CANCELLED):
            raise ValueError(f"Invalid status: {status}")

        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE threads
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
                """,
                (status, thread_id),
            )
            return cursor.rowcount > 0

    def list_threads(
        self,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        List threads, optionally filtered by status.

        Args:
            status: Filter by status (optional).
            limit: Maximum number of threads to return (optional).

        Returns:
            List of thread data dicts.
        """
        query = """
            SELECT thread_id, created_at, updated_at, status, title, input_path, metadata
            FROM threads
        """
        params: list[Any] = []

        if status:
            query += " WHERE status = ?"
            params.append(status)

        query += " ORDER BY updated_at DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

            results = []
            for row in rows:
                # Handle potentially corrupted metadata JSON
                metadata = None
                if row["metadata"]:
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        metadata = {"_raw": row["metadata"], "_error": "invalid_json"}

                results.append({
                    "thread_id": row["thread_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "status": row["status"],
                    "title": row["title"],
                    "input_path": row["input_path"],
                    "metadata": metadata,
                })

            return results

    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread and all its data.

        Args:
            thread_id: Thread identifier.

        Returns:
            True if thread was deleted, False if not found.
        """
        with self._get_conn() as conn:
            # Delete writes first (foreign key)
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))

            # Delete snapshots
            conn.execute("DELETE FROM checkpoint_snapshots WHERE thread_id = ?", (thread_id,))

            # Delete thread
            cursor = conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))

            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Checkpoint Snapshot Operations
    # -------------------------------------------------------------------------

    def put_snapshot(
        self,
        thread_id: str,
        checkpoint_id: str,
        checkpoint_data: dict[str, Any],
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a checkpoint snapshot.

        Args:
            thread_id: Thread identifier.
            checkpoint_id: Checkpoint identifier.
            checkpoint_data: The checkpoint data to store.
            parent_id: Parent checkpoint ID (optional).
            metadata: Additional metadata (optional).

        Note:
            Checkpoint data is serialized using pickle. This is suitable
            for single-user desktop apps but should not be used with
            untrusted data. Consider msgpack for production deployments.
        """
        self.ensure_thread(thread_id)

        # Use high-precision timestamp for reliable ordering
        timestamp = datetime.now().isoformat()

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoint_snapshots
                (id, thread_id, checkpoint_id, parent_id, checkpoint_data, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{thread_id}:{checkpoint_id}",
                    thread_id,
                    checkpoint_id,
                    parent_id,
                    pickle.dumps(checkpoint_data),
                    json.dumps(metadata) if metadata else None,
                    timestamp,
                ),
            )

    def get_snapshot(
        self,
        thread_id: str,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Get a checkpoint snapshot.

        Args:
            thread_id: Thread identifier.
            checkpoint_id: Specific checkpoint ID, or None for latest.

        Returns:
            Checkpoint data dict or None if not found.
        """
        with self._get_conn() as conn:
            if checkpoint_id:
                row = conn.execute(
                    """
                    SELECT checkpoint_data, metadata, parent_id, created_at
                    FROM checkpoint_snapshots
                    WHERE thread_id = ? AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT checkpoint_data, metadata, parent_id, created_at
                    FROM checkpoint_snapshots
                    WHERE thread_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (thread_id,),
                ).fetchone()

            if row:
                return {
                    "checkpoint_data": pickle.loads(row["checkpoint_data"]),
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    "parent_id": row["parent_id"],
                    "created_at": row["created_at"],
                }
            return None

    def list_snapshots(
        self,
        thread_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        List checkpoint snapshots for a thread.

        Args:
            thread_id: Thread identifier.
            limit: Maximum number of snapshots to return (optional).

        Returns:
            List of snapshot data dicts (without checkpoint_data for efficiency).
        """
        query = """
            SELECT checkpoint_id, parent_id, metadata, created_at
            FROM checkpoint_snapshots
            WHERE thread_id = ?
            ORDER BY created_at DESC
        """
        params: list[Any] = [thread_id]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

            return [
                {
                    "checkpoint_id": row["checkpoint_id"],
                    "parent_id": row["parent_id"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    # -------------------------------------------------------------------------
    # Pending Writes Operations
    # -------------------------------------------------------------------------

    def put_writes(
        self,
        thread_id: str,
        checkpoint_id: str,
        writes: list[tuple[str, Any]],
    ) -> None:
        """
        Store pending writes for a checkpoint.

        Args:
            thread_id: Thread identifier.
            checkpoint_id: Checkpoint identifier.
            writes: List of (channel, value) tuples.

        Note:
            Values are serialized using pickle. Only use with trusted data.
        """
        # Ensure thread exists to satisfy foreign key constraint
        self.ensure_thread(thread_id)

        with self._get_conn() as conn:
            for channel, value in writes:
                conn.execute(
                    """
                    INSERT INTO writes (thread_id, checkpoint_id, channel, value)
                    VALUES (?, ?, ?, ?)
                    """,
                    (thread_id, checkpoint_id, channel, pickle.dumps(value)),
                )

    def get_writes(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> list[tuple[str, Any]]:
        """
        Get pending writes for a checkpoint.

        Args:
            thread_id: Thread identifier.
            checkpoint_id: Checkpoint identifier.

        Returns:
            List of (channel, value) tuples.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT channel, value FROM writes
                WHERE thread_id = ? AND checkpoint_id = ?
                ORDER BY id
                """,
                (thread_id, checkpoint_id),
            ).fetchall()

            return [(row["channel"], pickle.loads(row["value"])) for row in rows]

    def clear_writes(self, thread_id: str, checkpoint_id: str) -> None:
        """
        Clear pending writes for a checkpoint.

        Args:
            thread_id: Thread identifier.
            checkpoint_id: Checkpoint identifier.
        """
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM writes WHERE thread_id = ? AND checkpoint_id = ?",
                (thread_id, checkpoint_id),
            )

    # -------------------------------------------------------------------------
    # Cleanup and Statistics
    # -------------------------------------------------------------------------

    def cleanup_old_threads(self, days: int = 7) -> int:
        """
        Delete threads older than specified days.

        Only deletes completed or cancelled threads.

        Args:
            days: Age threshold in days.

        Returns:
            Number of threads deleted.
        """
        with self._get_conn() as conn:
            # Get threads to delete
            rows = conn.execute(
                """
                SELECT thread_id FROM threads
                WHERE updated_at < datetime('now', ?)
                AND status IN (?, ?)
                """,
                (f"-{days} days", STATUS_COMPLETED, STATUS_CANCELLED),
            ).fetchall()

            thread_ids = [row["thread_id"] for row in rows]

            if not thread_ids:
                return 0

            placeholders = ",".join("?" * len(thread_ids))

            # Delete writes
            conn.execute(
                f"DELETE FROM writes WHERE thread_id IN ({placeholders})",
                thread_ids,
            )

            # Delete snapshots
            conn.execute(
                f"DELETE FROM checkpoint_snapshots WHERE thread_id IN ({placeholders})",
                thread_ids,
            )

            # Delete threads
            conn.execute(
                f"DELETE FROM threads WHERE thread_id IN ({placeholders})",
                thread_ids,
            )

            return len(thread_ids)

    def get_storage_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dict with storage statistics.
        """
        with self._get_conn() as conn:
            stats: dict[str, Any] = {}

            # Thread counts
            row = conn.execute("SELECT COUNT(*) as count FROM threads").fetchone()
            stats["total_threads"] = row["count"]

            row = conn.execute(
                "SELECT COUNT(*) as count FROM threads WHERE status = ?",
                (STATUS_ACTIVE,),
            ).fetchone()
            stats["active_threads"] = row["count"]

            row = conn.execute(
                "SELECT COUNT(*) as count FROM threads WHERE status = ?",
                (STATUS_COMPLETED,),
            ).fetchone()
            stats["completed_threads"] = row["count"]

            row = conn.execute(
                "SELECT COUNT(*) as count FROM threads WHERE status = ?",
                (STATUS_CANCELLED,),
            ).fetchone()
            stats["cancelled_threads"] = row["count"]

            # Checkpoint counts
            row = conn.execute(
                "SELECT COUNT(*) as count FROM checkpoint_snapshots"
            ).fetchone()
            stats["total_snapshots"] = row["count"]

            # Writes count
            row = conn.execute("SELECT COUNT(*) as count FROM writes").fetchone()
            stats["total_writes"] = row["count"]

            # Database size
            if isinstance(self.db_path, Path) and self.db_path.exists():
                stats["db_size_bytes"] = self.db_path.stat().st_size
            else:
                stats["db_size_bytes"] = 0

            return stats


__all__ = [
    "SQLiteCheckpointSaver",
    "STATUS_ACTIVE",
    "STATUS_COMPLETED",
    "STATUS_CANCELLED",
]
