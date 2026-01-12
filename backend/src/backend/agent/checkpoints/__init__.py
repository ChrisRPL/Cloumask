"""
Checkpoint persistence module for Cloumask agent.

This module provides SQLite-based checkpoint persistence
for the LangGraph agent, enabling resume capability after
app restarts.

Components:
    - SQLiteCheckpointSaver: Low-level checkpoint storage
    - CheckpointManager: High-level checkpoint operations
    - resume_pipeline: Helper for resuming pipelines

Example:
    from backend.agent.checkpoints import CheckpointManager, resume_pipeline

    manager = CheckpointManager("checkpoints.db")

    # Check if a thread can be resumed
    if manager.can_resume("thread-123"):
        summary = manager.get_thread_summary("thread-123")
        print(f"Progress: {summary['progress_percent']:.1f}%")

    # Resume or start pipeline
    async for event in resume_pipeline(
        "thread-123",
        "Continue processing",
        compiled_graph,
        manager,
    ):
        print(event)
"""

from backend.agent.checkpoints.manager import (
    CheckpointManager,
    create_compiled_graph_with_manager,
    resume_pipeline,
)
from backend.agent.checkpoints.saver import (
    SQLiteCheckpointSaver,
    STATUS_ACTIVE,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
)

__all__ = [
    # Classes
    "SQLiteCheckpointSaver",
    "CheckpointManager",
    # Functions
    "create_compiled_graph_with_manager",
    "resume_pipeline",
    # Constants
    "STATUS_ACTIVE",
    "STATUS_COMPLETED",
    "STATUS_CANCELLED",
]
