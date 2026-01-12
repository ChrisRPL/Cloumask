"""
High-level checkpoint management for the Cloumask agent.

This module provides the CheckpointManager class that handles
resume logic, thread management, and cleanup operations.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.agent.checkpoints.saver import (
    SQLiteCheckpointSaver,
    STATUS_ACTIVE,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class CheckpointManager:
    """
    High-level checkpoint management for the agent.

    Handles resume logic, thread management, and cleanup.
    Works in conjunction with LangGraph's AsyncSqliteSaver
    and our custom SQLiteCheckpointSaver for thread metadata.

    Attributes:
        saver: The SQLite checkpoint saver instance.
        db_path: Path to the SQLite database.
    """

    def __init__(self, db_path: str = "checkpoints.db") -> None:
        """
        Initialize the checkpoint manager.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self.saver = SQLiteCheckpointSaver(db_path)

    # -------------------------------------------------------------------------
    # Resume Capability
    # -------------------------------------------------------------------------

    def can_resume(self, thread_id: str) -> bool:
        """
        Check if a thread has resumable state.

        Args:
            thread_id: Thread identifier.

        Returns:
            True if thread exists and has checkpoint data.
        """
        thread = self.saver.get_thread(thread_id)
        if not thread:
            return False

        # Check if there's a checkpoint snapshot
        snapshot = self.saver.get_snapshot(thread_id)
        return snapshot is not None

    def get_resume_state(self, thread_id: str) -> dict[str, Any] | None:
        """
        Get the state to resume from.

        Args:
            thread_id: Thread identifier.

        Returns:
            Dict with checkpoint data and metadata, or None if not found.
        """
        snapshot = self.saver.get_snapshot(thread_id)
        if not snapshot:
            return None

        thread = self.saver.get_thread(thread_id)

        return {
            "checkpoint_data": snapshot["checkpoint_data"],
            "metadata": snapshot["metadata"],
            "thread_id": thread_id,
            "thread_info": thread,
            "created_at": snapshot["created_at"],
        }

    def get_thread_summary(self, thread_id: str) -> dict[str, Any] | None:
        """
        Get a summary of a thread's current state.

        Provides progress information and current status
        for display in the UI.

        Args:
            thread_id: Thread identifier.

        Returns:
            Summary dict or None if not found.
        """
        state = self.get_resume_state(thread_id)
        if not state:
            return None

        checkpoint_data = state["checkpoint_data"]

        # Extract state from checkpoint if it has channel_values
        # LangGraph stores state in channel_values
        values = checkpoint_data
        if isinstance(checkpoint_data, dict) and "channel_values" in checkpoint_data:
            values = checkpoint_data["channel_values"]

        plan = values.get("plan", [])
        current_step = values.get("current_step", 0)
        completed = sum(1 for s in plan if s.get("status") == "completed")

        # Get messages for last message content
        messages = values.get("messages", [])
        last_message = ""
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                last_message = last_msg.get("content", "")

        thread_info = state.get("thread_info", {}) or {}

        return {
            "thread_id": thread_id,
            "title": thread_info.get("title"),
            "status": thread_info.get("status", STATUS_ACTIVE),
            "total_steps": len(plan),
            "completed_steps": completed,
            "current_step": current_step,
            "progress_percent": (completed / len(plan) * 100) if plan else 0,
            "awaiting_user": values.get("awaiting_user", False),
            "last_message": last_message,
            "updated_at": thread_info.get("updated_at"),
            "created_at": thread_info.get("created_at"),
        }

    # -------------------------------------------------------------------------
    # Thread Listing
    # -------------------------------------------------------------------------

    def list_active_threads(self) -> list[dict[str, Any]]:
        """
        List all threads with resumable state.

        Returns:
            List of thread summaries.
        """
        threads = self.saver.list_threads(status=STATUS_ACTIVE)

        summaries = []
        for thread in threads:
            summary = self.get_thread_summary(thread["thread_id"])
            if summary:
                summaries.append(summary)

        return summaries

    def list_all_threads(
        self,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        List all threads with optional filtering.

        Args:
            status: Filter by status (active, completed, cancelled).
            limit: Maximum number of threads to return.

        Returns:
            List of thread data dicts.
        """
        return self.saver.list_threads(status=status, limit=limit)

    # -------------------------------------------------------------------------
    # Thread Lifecycle
    # -------------------------------------------------------------------------

    def create_thread(
        self,
        thread_id: str,
        title: str | None = None,
        input_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new thread.

        Args:
            thread_id: Unique thread identifier.
            title: Human-readable title.
            input_path: Input data path.
            metadata: Additional metadata.

        Returns:
            Thread data dict.
        """
        self.saver.ensure_thread(
            thread_id,
            title=title,
            input_path=input_path,
            metadata=metadata,
        )
        return self.saver.get_thread(thread_id) or {}

    def mark_completed(self, thread_id: str) -> None:
        """
        Mark a thread as completed.

        Args:
            thread_id: Thread identifier.
        """
        self.saver.update_thread_status(thread_id, STATUS_COMPLETED)

    def mark_cancelled(self, thread_id: str) -> None:
        """
        Mark a thread as cancelled.

        Args:
            thread_id: Thread identifier.
        """
        self.saver.update_thread_status(thread_id, STATUS_CANCELLED)

    def mark_active(self, thread_id: str) -> None:
        """
        Mark a thread as active.

        Args:
            thread_id: Thread identifier.
        """
        self.saver.update_thread_status(thread_id, STATUS_ACTIVE)

    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread and all its checkpoints.

        Args:
            thread_id: Thread identifier.

        Returns:
            True if deleted, False if not found.
        """
        return self.saver.delete_thread(thread_id)

    # -------------------------------------------------------------------------
    # Checkpoint Operations
    # -------------------------------------------------------------------------

    def save_snapshot(
        self,
        thread_id: str,
        checkpoint_id: str,
        state: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Save a checkpoint snapshot.

        This is typically called by the graph after each node
        execution to persist the state.

        Args:
            thread_id: Thread identifier.
            checkpoint_id: Checkpoint identifier.
            state: Pipeline state to save.
            metadata: Additional metadata.
        """
        self.saver.put_snapshot(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            checkpoint_data=state,
            metadata=metadata,
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
            checkpoint_id: Specific checkpoint, or None for latest.

        Returns:
            Snapshot data or None if not found.
        """
        return self.saver.get_snapshot(thread_id, checkpoint_id)

    def list_snapshots(
        self,
        thread_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        List checkpoint snapshots for a thread.

        Args:
            thread_id: Thread identifier.
            limit: Maximum number to return.

        Returns:
            List of snapshot metadata dicts.
        """
        return self.saver.list_snapshots(thread_id, limit)

    # -------------------------------------------------------------------------
    # Cleanup and Maintenance
    # -------------------------------------------------------------------------

    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """
        Delete checkpoints older than specified days.

        Only deletes completed or cancelled threads.

        Args:
            days: Age threshold in days.

        Returns:
            Number of threads deleted.
        """
        return self.saver.cleanup_old_threads(days)

    def get_storage_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dict with storage statistics including:
            - total_threads, active_threads, completed_threads, cancelled_threads
            - total_snapshots, total_writes
            - db_size_bytes
        """
        return self.saver.get_storage_stats()

    # -------------------------------------------------------------------------
    # Graph Integration
    # -------------------------------------------------------------------------

    @asynccontextmanager
    async def get_checkpointer(self) -> AsyncGenerator[AsyncSqliteSaver, None]:
        """
        Get an AsyncSqliteSaver for graph compilation.

        This creates a LangGraph-compatible checkpointer
        using the same database as the manager.

        Yields:
            AsyncSqliteSaver instance.

        Example:
            async with manager.get_checkpointer() as checkpointer:
                compiled = graph.compile(checkpointer=checkpointer)
        """
        async with AsyncSqliteSaver.from_conn_string(self.db_path) as checkpointer:
            yield checkpointer


async def create_compiled_graph_with_manager(
    db_path: str = "data/checkpoints.db",
) -> tuple[CompiledStateGraph, CheckpointManager]:
    """
    Create a compiled graph with checkpoint manager.

    This is a convenience function that creates both
    the compiled graph and manager for use together.

    Note: This function creates but does not yield the graph.
    For async context manager usage, use CheckpointManager.get_checkpointer()
    directly.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        Tuple of (compiled_graph, checkpoint_manager).
    """
    from backend.agent.graph import create_agent_graph

    manager = CheckpointManager(db_path)
    graph = create_agent_graph()

    # Create the checkpointer context
    async with manager.get_checkpointer() as checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
        # Note: This returns inside the context, so caller must
        # manage the lifecycle appropriately
        return compiled, manager


async def resume_pipeline(
    thread_id: str,
    user_message: str,
    compiled_graph: CompiledStateGraph,
    manager: CheckpointManager,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Resume a pipeline from checkpoint.

    If a checkpoint exists, resumes from it with the new message.
    If no checkpoint exists, starts a fresh pipeline.

    Args:
        thread_id: Thread identifier.
        user_message: New user message.
        compiled_graph: Compiled LangGraph instance.
        manager: Checkpoint manager instance.

    Yields:
        State updates from the graph.
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    if manager.can_resume(thread_id):
        # Get existing state
        result = manager.get_resume_state(thread_id)
        if result:
            checkpoint_data = result["checkpoint_data"]

            # Get channel_values from checkpoint if present
            if isinstance(checkpoint_data, dict) and "channel_values" in checkpoint_data:
                state = checkpoint_data["channel_values"]
            else:
                state = checkpoint_data

            # Add new user message to state
            messages = state.get("messages", [])
            messages.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat(),
            })
            state["messages"] = messages
            state["awaiting_user"] = False

            # Resume from checkpoint
            async for event in compiled_graph.astream(
                state,
                config,
                stream_mode="values",
            ):
                yield event
            return

    # Fresh start
    from backend.agent.state import create_initial_state

    state = create_initial_state(user_message, thread_id)

    # Ensure thread exists
    manager.create_thread(thread_id)

    async for event in compiled_graph.astream(
        state,
        config,
        stream_mode="values",
    ):
        yield event


__all__ = [
    "CheckpointManager",
    "create_compiled_graph_with_manager",
    "resume_pipeline",
]
