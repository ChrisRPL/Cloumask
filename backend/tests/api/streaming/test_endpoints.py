"""Tests for SSE streaming endpoints."""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.agent.checkpoints import CheckpointManager
from backend.agent.state import UserDecision
from backend.api.main import app
from backend.api.streaming import endpoints as streaming_endpoints
from backend.api.streaming.endpoints import (
    ThreadState,
    process_agent_request,
    _event_queues,
    _get_or_create_thread,
    _sanitize_error_message,
    _thread_states,
    _threads,
    state_to_events,
)
from backend.api.streaming.events import SSEEventType


@pytest.fixture
def client() -> TestClient:
    """Create test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_state() -> None:
    """Clean up global state after each test."""
    yield
    _event_queues.clear()
    _thread_states.clear()
    _threads.clear()


@pytest.fixture
def checkpoint_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CheckpointManager:
    """Provide an isolated checkpoint manager for endpoint persistence tests."""
    db_path = tmp_path / "streaming-endpoints.db"
    manager = CheckpointManager(str(db_path))
    monkeypatch.setattr(streaming_endpoints, "CHECKPOINT_DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(streaming_endpoints, "_checkpoint_manager", manager, raising=False)
    return manager


class TestCreateThread:
    """Tests for thread creation endpoint."""

    def test_create_thread_returns_200(self, client: TestClient) -> None:
        """Create thread endpoint should return 200."""
        response = client.post("/api/chat/threads")
        assert response.status_code == 200

    def test_create_thread_returns_thread_id(self, client: TestClient) -> None:
        """Create thread should return a thread_id."""
        response = client.post("/api/chat/threads")
        data = response.json()

        assert "thread_id" in data
        assert len(data["thread_id"]) > 0

    def test_create_thread_marks_created(self, client: TestClient) -> None:
        """Create thread should mark created=True."""
        response = client.post("/api/chat/threads")
        data = response.json()

        assert data["created"] is True


class TestGetThreadInfo:
    """Tests for thread info endpoint."""

    def test_get_thread_not_found(self, client: TestClient) -> None:
        """Get thread info should return 404 for unknown thread."""
        response = client.get("/api/chat/threads/unknown-id")
        assert response.status_code == 404

    def test_get_thread_info(self, client: TestClient) -> None:
        """Get thread info should return thread state."""
        # First create a thread
        create_response = client.post("/api/chat/threads")
        thread_id = create_response.json()["thread_id"]

        # Then get info
        response = client.get(f"/api/chat/threads/{thread_id}")
        data = response.json()

        assert response.status_code == 200
        assert data["thread_id"] == thread_id
        assert data["created"] is False  # Not a new thread

    def test_get_thread_info_restores_persisted_state_after_memory_reset(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """Thread info should rehydrate from persisted checkpoint state after restart."""
        create_response = client.post("/api/chat/threads")
        thread_id = create_response.json()["thread_id"]

        checkpoint_manager.create_thread(thread_id, title="Persisted thread")
        checkpoint_manager.save_snapshot(
            thread_id,
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "pending"},
                    ],
                    "current_step": 0,
                    "awaiting_user": True,
                    "metadata": {"pipeline_id": "pipe-persisted"},
                }
            },
        )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get(f"/api/chat/threads/{thread_id}")
        data = response.json()

        assert response.status_code == 200
        assert data["thread_id"] == thread_id
        assert data["awaiting_user"] is True
        assert data["current_step"] == 0
        assert data["total_steps"] == 1


class TestListThreads:
    """Tests for thread listing endpoint."""

    def test_list_threads_returns_resumable_threads(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should return active resumable threads ordered by recency."""
        checkpoint_manager.create_thread("thread-older", title="Older")
        checkpoint_manager.save_snapshot(
            "thread-older",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [{"id": "step-1", "tool_name": "scan_directory", "status": "completed"}],
                    "current_step": 1,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Older thread"}],
                }
            },
        )

        checkpoint_manager.create_thread("thread-latest", title="Latest")
        checkpoint_manager.save_snapshot(
            "thread-latest",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [{"id": "step-2", "tool_name": "detect", "status": "pending"}],
                    "current_step": 0,
                    "awaiting_user": True,
                    "messages": [{"role": "assistant", "content": "Latest thread"}],
                }
            },
        )

        checkpoint_manager.create_thread("thread-empty", title="No snapshot")
        checkpoint_manager.mark_completed("thread-older")

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-latest"]
        assert data["threads"][0]["awaiting_user"] is True
        assert data["threads"][0]["total_steps"] == 1
        assert data["threads"][0]["summary"] == "awaiting review. Progress: 0/1 steps."

    def test_list_threads_keeps_recency_order_and_summary_fields(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should expose truthful summary fields while staying recency-ordered."""
        checkpoint_manager.create_thread("thread-review", title="Needs review")
        checkpoint_manager.save_snapshot(
            "thread-review",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "detect", "status": "pending"},
                    ],
                    "current_step": 1,
                    "awaiting_user": True,
                    "messages": [{"role": "assistant", "content": "Review this run"}],
                }
            },
        )

        checkpoint_manager.create_thread("thread-progress", title="Still running")
        checkpoint_manager.save_snapshot(
            "thread-progress",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "detect", "status": "completed"},
                        {"id": "step-3", "tool_name": "export", "status": "pending"},
                    ],
                    "current_step": 2,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Continuing export"}],
                }
            },
        )

        with checkpoint_manager.saver._get_conn() as conn:
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-14 10:00:00", "thread-review"),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-14 11:00:00", "thread-progress"),
            )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == [
            "thread-progress",
            "thread-review",
        ]

        latest = data["threads"][0]
        assert latest["title"] == "Still running"
        assert latest["status"] == "active"
        assert latest["resume_status"] == "in progress"
        assert latest["awaiting_user"] is False
        assert latest["current_step"] == 2
        assert latest["total_steps"] == 3
        assert latest["last_message"] == "Continuing export"
        assert latest["summary"] == "in progress. Progress: 2/3 steps."

        review = data["threads"][1]
        assert review["title"] == "Needs review"
        assert review["status"] == "active"
        assert review["resume_status"] == "awaiting review"
        assert review["awaiting_user"] is True
        assert review["current_step"] == 1
        assert review["total_steps"] == 2
        assert review["last_message"] == "Review this run"
        assert review["summary"] == "awaiting review. Progress: 1/2 steps."

    def test_list_threads_formats_ready_and_completed_summaries(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should format no-plan and clamped completed summaries truthfully."""
        checkpoint_manager.create_thread("thread-ready", title="Ready")
        checkpoint_manager.save_snapshot(
            "thread-ready",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [],
                    "current_step": 0,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Ready to start"}],
                }
            },
        )

        checkpoint_manager.create_thread("thread-complete", title="Complete")
        checkpoint_manager.save_snapshot(
            "thread-complete",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "export", "status": "completed"},
                    ],
                    "current_step": 5,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Finished run"}],
                }
            },
        )

        with checkpoint_manager.saver._get_conn() as conn:
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 09:00:00", "thread-ready"),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 10:00:00", "thread-complete"),
            )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == [
            "thread-complete",
            "thread-ready",
        ]
        assert data["threads"][0]["resume_status"] == "completed"
        assert data["threads"][0]["summary"] == "completed. Progress: 2/2 steps."
        assert data["threads"][1]["resume_status"] == "ready"
        assert data["threads"][1]["summary"] == "ready."

    def test_list_threads_keeps_recency_order_for_mixed_resume_states(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should stay recency-ordered even when summaries span different states."""
        checkpoint_manager.create_thread("thread-ready-newest", title="Ready newest")
        checkpoint_manager.save_snapshot(
            "thread-ready-newest",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [],
                    "current_step": 0,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Ready thread"}],
                }
            },
        )

        checkpoint_manager.create_thread("thread-progress-middle", title="Progress middle")
        checkpoint_manager.save_snapshot(
            "thread-progress-middle",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "detect", "status": "pending"},
                    ],
                    "current_step": 1,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Progress thread"}],
                }
            },
        )

        checkpoint_manager.create_thread("thread-review-oldest", title="Review oldest")
        checkpoint_manager.save_snapshot(
            "thread-review-oldest",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "review", "status": "pending"},
                    ],
                    "current_step": 1,
                    "awaiting_user": True,
                    "messages": [{"role": "assistant", "content": "Review thread"}],
                }
            },
        )

        with checkpoint_manager.saver._get_conn() as conn:
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:02:00", "thread-ready-newest"),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:01:00", "thread-progress-middle"),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:00:00", "thread-review-oldest"),
            )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == [
            "thread-ready-newest",
            "thread-progress-middle",
            "thread-review-oldest",
        ]
        assert data["threads"][0]["resume_status"] == "ready"
        assert data["threads"][0]["summary"] == "ready."
        assert data["threads"][1]["resume_status"] == "in progress"
        assert data["threads"][1]["summary"] == "in progress. Progress: 1/2 steps."
        assert data["threads"][2]["resume_status"] == "awaiting review"
        assert data["threads"][2]["summary"] == "awaiting review. Progress: 1/2 steps."

    def test_list_threads_keeps_failed_summary_in_mixed_recency_order(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should preserve recency ordering while exposing failed summary text."""
        checkpoint_manager.create_thread("thread-failed-newest", title="Failed newest")
        checkpoint_manager.save_snapshot(
            "thread-failed-newest",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "export", "status": "failed"},
                    ],
                    "current_step": 9,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Newest failed thread"}],
                }
            },
        )

        checkpoint_manager.create_thread("thread-progress-middle", title="Progress middle")
        checkpoint_manager.save_snapshot(
            "thread-progress-middle",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "detect", "status": "pending"},
                    ],
                    "current_step": 1,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Progress thread"}],
                }
            },
        )

        checkpoint_manager.create_thread("thread-review-oldest", title="Review oldest")
        checkpoint_manager.save_snapshot(
            "thread-review-oldest",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "review", "status": "pending"},
                    ],
                    "current_step": 1,
                    "awaiting_user": True,
                    "messages": [{"role": "assistant", "content": "Review thread"}],
                }
            },
        )

        with checkpoint_manager.saver._get_conn() as conn:
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:02:00", "thread-failed-newest"),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:01:00", "thread-progress-middle"),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:00:00", "thread-review-oldest"),
            )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == [
            "thread-failed-newest",
            "thread-progress-middle",
            "thread-review-oldest",
        ]
        assert data["threads"][0]["resume_status"] == "failed"
        assert data["threads"][0]["summary"] == "failed. Progress: 1/2 steps."
        assert data["threads"][1]["resume_status"] == "in progress"
        assert data["threads"][1]["summary"] == "in progress. Progress: 1/2 steps."
        assert data["threads"][2]["resume_status"] == "awaiting review"
        assert data["threads"][2]["summary"] == "awaiting review. Progress: 1/2 steps."

    def test_list_threads_clamps_missing_and_negative_current_step_in_summary(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should keep summary text stable for malformed step counters."""
        checkpoint_manager.create_thread("thread-missing-step", title="Missing step")
        checkpoint_manager.save_snapshot(
            "thread-missing-step",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "detect", "status": "pending"},
                    ],
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Missing current step"}],
                }
            },
        )

        checkpoint_manager.create_thread("thread-negative-step", title="Negative step")
        checkpoint_manager.save_snapshot(
            "thread-negative-step",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "review", "status": "pending"},
                        {"id": "step-3", "tool_name": "export", "status": "pending"},
                    ],
                    "current_step": -4,
                    "awaiting_user": True,
                    "messages": [{"role": "assistant", "content": "Negative current step"}],
                }
            },
        )

        with checkpoint_manager.saver._get_conn() as conn:
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:11:00", "thread-missing-step"),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:10:00", "thread-negative-step"),
            )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == [
            "thread-missing-step",
            "thread-negative-step",
        ]
        assert data["threads"][0]["summary"] == "in progress. Progress: 0/2 steps."
        assert data["threads"][1]["summary"] == "awaiting review. Progress: 0/3 steps."

    def test_list_threads_formats_empty_and_missing_plan_summaries(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should keep empty-plan summaries stable for ready and review states."""
        checkpoint_manager.create_thread("thread-review-empty", title="Review empty")
        checkpoint_manager.save_snapshot(
            "thread-review-empty",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [],
                    "current_step": 0,
                    "awaiting_user": True,
                    "messages": [{"role": "assistant", "content": "Awaiting review without steps"}],
                }
            },
        )

        checkpoint_manager.create_thread("thread-ready-missing-plan", title="Ready missing plan")
        checkpoint_manager.save_snapshot(
            "thread-ready-missing-plan",
            "ckpt-1",
            {
                "channel_values": {
                    "current_step": 4,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Ready without plan"}],
                }
            },
        )

        with checkpoint_manager.saver._get_conn() as conn:
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:21:00", "thread-review-empty"),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:20:00", "thread-ready-missing-plan"),
            )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == [
            "thread-review-empty",
            "thread-ready-missing-plan",
        ]
        assert data["threads"][0]["resume_status"] == "awaiting review"
        assert data["threads"][0]["summary"] == "awaiting review."
        assert data["threads"][1]["resume_status"] == "ready"
        assert data["threads"][1]["summary"] == "ready."

    def test_list_threads_skips_corrupted_non_list_plan_payloads(self, client: TestClient, checkpoint_manager: CheckpointManager) -> None:
        """List threads should not crash when persisted plan payloads are malformed."""
        checkpoint_manager.create_thread("thread-corrupted-plan", title="Corrupted")
        checkpoint_manager.save_snapshot(
            "thread-corrupted-plan",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": {"unexpected": "shape"},
                    "current_step": 3,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Corrupted plan payload"}],
                }
            },
        )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-corrupted-plan"]
        assert data["threads"][0]["total_steps"] == 0
        assert data["threads"][0]["resume_status"] == "ready"
        assert data["threads"][0]["summary"] == "ready."

    def test_list_threads_ignores_non_string_last_message_content(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should not surface malformed last-message payload content."""
        checkpoint_manager.create_thread("thread-bad-message", title="Bad message")
        checkpoint_manager.save_snapshot(
            "thread-bad-message",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [],
                    "current_step": 0,
                    "awaiting_user": False,
                    "messages": [
                        {"role": "assistant", "content": {"unexpected": "object"}},
                    ],
                }
            },
        )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-bad-message"]
        assert data["threads"][0]["last_message"] == ""
        assert data["threads"][0]["resume_status"] == "ready"
        assert data["threads"][0]["summary"] == "ready."

    def test_list_threads_ignores_malformed_timestamp_metadata(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """List threads should coerce malformed timestamp metadata to null."""
        checkpoint_manager.create_thread("thread-bad-timestamps", title="Bad timestamps")
        checkpoint_manager.save_snapshot(
            "thread-bad-timestamps",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [],
                    "current_step": 0,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Bad timestamps restored"}],
                }
            },
        )

        original_get_thread = checkpoint_manager.saver.get_thread

        def get_thread_with_bad_timestamps(thread_id: str) -> dict[str, object] | None:
            thread = original_get_thread(thread_id)
            if not thread or thread_id != "thread-bad-timestamps":
                return thread
            return {
                **thread,
                "updated_at": {"unexpected": "object"},
                "created_at": 123,
            }

        monkeypatch.setattr(checkpoint_manager.saver, "get_thread", get_thread_with_bad_timestamps)

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-bad-timestamps"]
        assert data["threads"][0]["updated_at"] is None
        assert data["threads"][0]["created_at"] is None
        assert data["threads"][0]["summary"] == "ready."

    def test_list_threads_ignores_malformed_title_and_status_metadata(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """List threads should coerce malformed title/status metadata to safe defaults."""
        checkpoint_manager.create_thread("thread-bad-metadata", title="Bad metadata")
        checkpoint_manager.save_snapshot(
            "thread-bad-metadata",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [],
                    "current_step": 0,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Bad metadata restored"}],
                }
            },
        )

        original_get_thread = checkpoint_manager.saver.get_thread

        def get_thread_with_bad_metadata(thread_id: str) -> dict[str, object] | None:
            thread = original_get_thread(thread_id)
            if not thread or thread_id != "thread-bad-metadata":
                return thread
            return {
                **thread,
                "title": {"unexpected": "object"},
                "status": ["active"],
            }

        monkeypatch.setattr(checkpoint_manager.saver, "get_thread", get_thread_with_bad_metadata)

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-bad-metadata"]
        assert data["threads"][0]["title"] is None
        assert data["threads"][0]["status"] == "active"
        assert data["threads"][0]["summary"] == "ready."

    def test_list_threads_keeps_resume_status_when_thread_status_metadata_is_unknown(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unknown thread metadata status should not change user-facing resume status."""
        checkpoint_manager.create_thread("thread-unknown-status", title="Unknown status")
        checkpoint_manager.save_snapshot(
            "thread-unknown-status",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [],
                    "current_step": 0,
                    "awaiting_user": True,
                    "messages": [{"role": "assistant", "content": "Unknown status restored"}],
                }
            },
        )

        original_get_thread = checkpoint_manager.saver.get_thread

        def get_thread_with_unknown_status(thread_id: str) -> dict[str, object] | None:
            thread = original_get_thread(thread_id)
            if not thread or thread_id != "thread-unknown-status":
                return thread
            return {
                **thread,
                "status": "mystery",
            }

        monkeypatch.setattr(checkpoint_manager.saver, "get_thread", get_thread_with_unknown_status)

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-unknown-status"]
        assert data["threads"][0]["status"] == "active"
        assert data["threads"][0]["resume_status"] == "awaiting review"
        assert data["threads"][0]["summary"] == "awaiting review."

    def test_list_threads_preserves_known_lifecycle_status_metadata(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Known lifecycle statuses should survive normalization unchanged."""
        checkpoint_manager.create_thread("thread-known-completed", title="Known completed")
        checkpoint_manager.save_snapshot(
            "thread-known-completed",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [],
                    "current_step": 0,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Known completed restored"}],
                }
            },
        )

        checkpoint_manager.create_thread("thread-known-cancelled", title="Known cancelled")
        checkpoint_manager.save_snapshot(
            "thread-known-cancelled",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [],
                    "current_step": 0,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Known cancelled restored"}],
                }
            },
        )

        original_get_thread = checkpoint_manager.saver.get_thread

        def get_thread_with_known_statuses(thread_id: str) -> dict[str, object] | None:
            thread = original_get_thread(thread_id)
            if not thread:
                return thread
            if thread_id == "thread-known-completed":
                return {
                    **thread,
                    "status": "completed",
                }
            if thread_id == "thread-known-cancelled":
                return {
                    **thread,
                    "status": "cancelled",
                }
            return thread

        monkeypatch.setattr(checkpoint_manager.saver, "get_thread", get_thread_with_known_statuses)

        with checkpoint_manager.saver._get_conn() as conn:
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:31:00", "thread-known-cancelled"),
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                ("2026-03-15 12:30:00", "thread-known-completed"),
            )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == [
            "thread-known-cancelled",
            "thread-known-completed",
        ]
        assert data["threads"][0]["status"] == "cancelled"
        assert data["threads"][0]["resume_status"] == "ready"
        assert data["threads"][0]["summary"] == "ready."
        assert data["threads"][1]["status"] == "completed"
        assert data["threads"][1]["resume_status"] == "ready"
        assert data["threads"][1]["summary"] == "ready."

    def test_list_threads_skips_corrupted_rows_without_string_thread_ids(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """List threads should ignore corrupted rows that cannot provide a valid thread id."""
        checkpoint_manager.create_thread("thread-valid-row", title="Valid row")
        checkpoint_manager.save_snapshot(
            "thread-valid-row",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [],
                    "current_step": 0,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Valid row restored"}],
                }
            },
        )

        original_list_threads = checkpoint_manager.saver.list_threads

        def list_threads_with_corruption(
            status: str | None = None,
            limit: int | None = None,
        ) -> list[object]:
            valid_rows = original_list_threads(status=status, limit=limit)
            return [
                *valid_rows,
                {"thread_id": None, "status": "active"},
                {"title": "Missing id", "status": "active"},
                "corrupted-row",
            ]

        monkeypatch.setattr(checkpoint_manager.saver, "list_threads", list_threads_with_corruption)

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-valid-row"]
        assert data["threads"][0]["summary"] == "ready."

    def test_list_threads_clamps_mixed_corrupted_step_and_message_payloads(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should stay stable when current_step and trailing messages are malformed."""
        checkpoint_manager.create_thread("thread-mixed-corruption", title="Mixed corruption")
        checkpoint_manager.save_snapshot(
            "thread-mixed-corruption",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "review", "status": "pending"},
                    ],
                    "current_step": {"unexpected": "object"},
                    "awaiting_user": False,
                    "messages": [
                        {"role": "assistant", "content": "Earlier valid message"},
                        "corrupted-tail",
                        {"role": "assistant", "content": {"unexpected": "object"}},
                    ],
                }
            },
        )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-mixed-corruption"]
        assert data["threads"][0]["current_step"] == 0
        assert data["threads"][0]["last_message"] == "Earlier valid message"
        assert data["threads"][0]["summary"] == "in progress. Progress: 0/2 steps."

    def test_list_threads_ignores_malformed_awaiting_user_values(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should not treat malformed awaiting_user payloads as review-required."""
        checkpoint_manager.create_thread("thread-bad-awaiting", title="Bad awaiting")
        checkpoint_manager.save_snapshot(
            "thread-bad-awaiting",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "review", "status": "pending"},
                    ],
                    "current_step": 1,
                    "awaiting_user": {"unexpected": "object"},
                    "messages": [{"role": "assistant", "content": "Bad awaiting restored"}],
                }
            },
        )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-bad-awaiting"]
        assert data["threads"][0]["awaiting_user"] is False
        assert data["threads"][0]["summary"] == "in progress. Progress: 1/2 steps."

    def test_list_threads_uses_completed_step_count_when_current_step_is_stale(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should not underreport persisted completion when current_step lags behind."""
        checkpoint_manager.create_thread("thread-stale-current-step", title="Stale progress")
        checkpoint_manager.save_snapshot(
            "thread-stale-current-step",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "export", "status": "completed"},
                    ],
                    "current_step": 0,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Finished run"}],
                }
            },
        )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-stale-current-step"]
        assert data["threads"][0]["current_step"] == 2
        assert data["threads"][0]["summary"] == "completed. Progress: 2/2 steps."

    def test_list_threads_does_not_mark_incomplete_plan_as_completed_from_stale_step(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should not overreport completion when current_step runs ahead of plan status."""
        checkpoint_manager.create_thread("thread-stale-complete", title="Stale complete")
        checkpoint_manager.save_snapshot(
            "thread-stale-complete",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "export", "status": "pending"},
                    ],
                    "current_step": 2,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Export still pending"}],
                }
            },
        )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-stale-complete"]
        assert data["threads"][0]["current_step"] == 1
        assert data["threads"][0]["summary"] == "in progress. Progress: 1/2 steps."

    def test_list_threads_does_not_mark_failed_plan_as_completed_from_stale_step(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """List threads should surface failed work as failed even when current_step overshoots."""
        checkpoint_manager.create_thread("thread-stale-failed", title="Stale failed")
        checkpoint_manager.save_snapshot(
            "thread-stale-failed",
            "ckpt-1",
            {
                "channel_values": {
                    "plan": [
                        {"id": "step-1", "tool_name": "scan_directory", "status": "completed"},
                        {"id": "step-2", "tool_name": "export", "status": "failed"},
                    ],
                    "current_step": 2,
                    "awaiting_user": False,
                    "messages": [{"role": "assistant", "content": "Export failed"}],
                }
            },
        )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get("/api/chat/threads")
        data = response.json()

        assert response.status_code == 200
        assert [thread["thread_id"] for thread in data["threads"]] == ["thread-stale-failed"]
        assert data["threads"][0]["current_step"] == 1
        assert data["threads"][0]["summary"] == "failed. Progress: 1/2 steps."


class TestGetThreadState:
    """Tests for thread state hydration endpoint."""

    def test_get_thread_state_returns_persisted_snapshot(
        self,
        client: TestClient,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """Thread state endpoint should expose the persisted channel values payload."""
        thread_id = "thread-hydrate-state"
        checkpoint_manager.create_thread(thread_id, title="Hydrate me")
        checkpoint_manager.save_snapshot(
            thread_id,
            "ckpt-1",
            {
                "channel_values": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "process this folder",
                            "timestamp": "2026-03-14T10:00:00.000Z",
                        },
                        {
                            "role": "assistant",
                            "content": "Here's a plan",
                            "timestamp": "2026-03-14T10:00:01.000Z",
                        },
                    ],
                    "plan": [
                        {
                            "id": "step-1",
                            "tool_name": "scan_directory",
                            "description": "Scan input",
                            "parameters": {"path": "/data/input"},
                            "status": "completed",
                        },
                        {
                            "id": "step-2",
                            "tool_name": "detect",
                            "description": "Detect people",
                            "parameters": {"classes": ["person"]},
                            "status": "pending",
                        },
                    ],
                    "plan_approved": False,
                    "current_step": 1,
                    "awaiting_user": True,
                    "metadata": {"pipeline_id": "pipe-hydrate"},
                    "checkpoints": [],
                    "execution_results": {},
                }
            },
        )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        response = client.get(f"/api/chat/threads/{thread_id}/state")
        data = response.json()

        assert response.status_code == 200
        assert data["thread_id"] == thread_id
        assert data["state"]["metadata"]["pipeline_id"] == "pipe-hydrate"
        assert len(data["state"]["messages"]) == 2
        assert len(data["state"]["plan"]) == 2
        assert data["state"]["awaiting_user"] is True
        assert data["state"]["current_step"] == 1

class TestCloseThread:
    """Tests for thread close endpoint."""

    def test_close_thread(self, client: TestClient) -> None:
        """Close thread should cleanup resources."""
        # Create thread
        create_response = client.post("/api/chat/threads")
        thread_id = create_response.json()["thread_id"]

        # Close it
        response = client.delete(f"/api/chat/threads/{thread_id}")
        data = response.json()

        assert response.status_code == 200
        assert data["status"] == "closed"
        assert data["thread_id"] == thread_id

    def test_close_thread_removes_from_queues(self, client: TestClient) -> None:
        """Closing thread should remove it from event queues."""
        create_response = client.post("/api/chat/threads")
        thread_id = create_response.json()["thread_id"]

        assert thread_id in _event_queues

        client.delete(f"/api/chat/threads/{thread_id}")

        assert thread_id not in _event_queues

    def test_close_nonexistent_thread(self, client: TestClient) -> None:
        """Closing nonexistent thread should succeed silently."""
        response = client.delete("/api/chat/threads/nonexistent-id")
        assert response.status_code == 200


class TestSendMessage:
    """Tests for send message endpoint."""

    def test_send_message_returns_queued(self, client: TestClient) -> None:
        """Send message should return queued status."""
        # Create thread first
        create_response = client.post("/api/chat/threads")
        thread_id = create_response.json()["thread_id"]

        # Send message
        response = client.post(
            f"/api/chat/send/{thread_id}",
            json={"content": "Hello"},
        )
        data = response.json()

        assert response.status_code == 200
        assert data["status"] == "queued"
        assert data["thread_id"] == thread_id

    def test_send_message_returns_message_id(self, client: TestClient) -> None:
        """Send message should return a message_id."""
        create_response = client.post("/api/chat/threads")
        thread_id = create_response.json()["thread_id"]

        response = client.post(
            f"/api/chat/send/{thread_id}",
            json={"content": "Test"},
        )
        data = response.json()

        assert "message_id" in data
        assert len(data["message_id"]) > 0

    def test_send_message_with_decision(self, client: TestClient) -> None:
        """Send message should accept optional decision."""
        create_response = client.post("/api/chat/threads")
        thread_id = create_response.json()["thread_id"]

        response = client.post(
            f"/api/chat/send/{thread_id}",
            json={"content": "Approved", "decision": "approve"},
        )

        assert response.status_code == 200

    def test_send_message_thread_not_found(self, client: TestClient) -> None:
        """Send message should return 404 for unknown thread."""
        response = client.post(
            "/api/chat/send/unknown-thread-id",
            json={"content": "Hello"},
        )
        assert response.status_code == 404

    def test_send_message_empty_content_rejected(self, client: TestClient) -> None:
        """Send message should reject empty content."""
        create_response = client.post("/api/chat/threads")
        thread_id = create_response.json()["thread_id"]

        response = client.post(
            f"/api/chat/send/{thread_id}",
            json={"content": ""},
        )
        assert response.status_code == 422  # Validation error

    def test_send_message_whitespace_content_rejected(self, client: TestClient) -> None:
        """Send message should reject whitespace-only content."""
        create_response = client.post("/api/chat/threads")
        thread_id = create_response.json()["thread_id"]

        response = client.post(
            f"/api/chat/send/{thread_id}",
            json={"content": "   "},
        )
        assert response.status_code == 422  # Validation error


class TestSSEStream:
    """Tests for SSE streaming endpoint."""

    def test_sse_endpoint_exists(self, client: TestClient) -> None:
        """SSE endpoint should be defined and accessible."""
        # Create thread first
        create_response = client.post("/api/chat/threads")
        thread_id = create_response.json()["thread_id"]

        # Just verify the endpoint is accessible
        # Full SSE testing requires async test client with proper handling
        # This test verifies the route exists and accepts GET requests
        from backend.api.main import app

        # Verify the route is registered
        route_paths = [route.path for route in app.routes]
        assert "/api/chat/stream/{thread_id}" in route_paths

        # Cleanup
        client.delete(f"/api/chat/threads/{thread_id}")


class TestSanitizeErrorMessage:
    """Tests for error message sanitization."""

    def test_sanitize_normal_message(self) -> None:
        """Normal error messages should pass through."""
        msg = "File not found"
        assert _sanitize_error_message(msg) == msg

    def test_sanitize_password_in_message(self) -> None:
        """Messages with password should be sanitized."""
        msg = "Invalid password for user admin"
        result = _sanitize_error_message(msg)
        assert "password" not in result.lower()
        assert "internal error" in result.lower()

    def test_sanitize_path_in_message(self) -> None:
        """Messages with file paths should be sanitized."""
        msg = "Error reading /home/user/secrets.txt"
        result = _sanitize_error_message(msg)
        assert "/home/" not in result
        assert "internal error" in result.lower()

    def test_sanitize_long_message(self) -> None:
        """Very long messages should be truncated."""
        msg = "A" * 300
        result = _sanitize_error_message(msg)
        assert len(result) <= 203  # 200 + "..."
        assert result.endswith("...")

    def test_sanitize_traceback_in_message(self) -> None:
        """Messages with traceback should be sanitized."""
        msg = 'Traceback (most recent call last): File "test.py"'
        result = _sanitize_error_message(msg)
        assert "internal error" in result.lower()


class TestStateToEvents:
    """Tests for state_to_events conversion function."""

    def test_state_to_events_message(self) -> None:
        """State with messages should produce message events."""
        state = {
            "messages": [
                {"role": "assistant", "content": "Hello there!"},
            ],
            "plan": [],
            "current_step": 0,
            "checkpoints": [],
            "awaiting_user": False,
        }

        events = state_to_events(state, "test-thread")

        assert len(events) >= 1
        message_events = [e for e in events if e.type == SSEEventType.MESSAGE]
        assert len(message_events) == 1
        assert message_events[0].data["content"] == "Hello there!"

    def test_state_to_events_plan(self) -> None:
        """State with plan should produce plan event."""
        state = {
            "messages": [],
            "plan": [
                {"id": "step-1", "tool_name": "scan", "description": "Scan files"},
            ],
            "plan_approved": False,
            "current_step": 0,
            "checkpoints": [],
            "awaiting_user": False,
            "metadata": {"pipeline_id": "pipe-123"},
        }

        events = state_to_events(state, "test-thread")

        plan_events = [e for e in events if e.type == SSEEventType.PLAN]
        assert len(plan_events) == 1
        assert plan_events[0].data["total_steps"] == 1

    def test_state_to_events_await_plan_approval(self) -> None:
        """State awaiting plan approval should produce await_input event."""
        state = {
            "messages": [],
            "plan": [{"id": "step-1", "tool_name": "scan"}],
            "plan_approved": False,
            "current_step": 0,
            "checkpoints": [],
            "awaiting_user": True,
        }

        events = state_to_events(state, "test-thread")

        await_events = [e for e in events if e.type == SSEEventType.AWAIT_INPUT]
        assert len(await_events) == 1
        assert await_events[0].data["input_type"] == "plan_approval"

    def test_state_to_events_await_checkpoint(self) -> None:
        """State awaiting checkpoint approval should produce await_input event."""
        state = {
            "messages": [],
            "plan": [{"id": "step-1", "tool_name": "scan"}],
            "plan_approved": True,
            "current_step": 1,
            "checkpoints": [],
            "awaiting_user": True,
        }

        events = state_to_events(state, "test-thread")

        await_events = [e for e in events if e.type == SSEEventType.AWAIT_INPUT]
        assert len(await_events) == 1
        assert await_events[0].data["input_type"] == "checkpoint_approval"

    def test_state_to_events_step_complete(self) -> None:
        """State with completed step should produce step events."""
        state = {
            "messages": [],
            "plan": [
                {
                    "id": "step-0",
                    "tool_name": "scan",
                    "description": "Scan files",
                    "status": "completed",
                    "result": {"files": 10},
                },
                {
                    "id": "step-1",
                    "tool_name": "detect",
                    "description": "Detect objects",
                    "status": "pending",
                },
            ],
            "plan_approved": True,
            "current_step": 1,
            "checkpoints": [],
            "awaiting_user": False,
            "execution_results": {},
        }

        events = state_to_events(state, "test-thread")

        # Should have step_complete for step 0 and step_start for step 1
        step_complete = [e for e in events if e.type == SSEEventType.STEP_COMPLETE]
        step_start = [e for e in events if e.type == SSEEventType.STEP_START]

        assert len(step_complete) == 1
        assert step_complete[0].data["step_index"] == 0

        assert len(step_start) == 1
        assert step_start[0].data["step_index"] == 1

    def test_state_to_events_checkpoint(self) -> None:
        """State with unresolved checkpoint should produce checkpoint event."""
        state = {
            "messages": [],
            "plan": [],
            "current_step": 0,
            "checkpoints": [
                {
                    "id": "ckpt-1",
                    "step_index": 2,
                    "trigger_reason": "percentage",
                    "quality_metrics": {},
                    "resolved_at": None,
                },
            ],
            "awaiting_user": False,
        }

        events = state_to_events(state, "test-thread")

        checkpoint_events = [e for e in events if e.type == SSEEventType.CHECKPOINT]
        assert len(checkpoint_events) == 1
        assert checkpoint_events[0].data["checkpoint_id"] == "ckpt-1"

    def test_state_to_events_pipeline_complete(self) -> None:
        """State with all steps done should produce complete event."""
        state = {
            "messages": [],
            "plan": [
                {"id": "step-0", "tool_name": "scan", "status": "completed"},
            ],
            "plan_approved": True,
            "current_step": 1,  # Past last step
            "checkpoints": [],
            "awaiting_user": False,
            "execution_results": {
                "step-0": {"status": "completed"},
            },
            "metadata": {"pipeline_id": "pipe-123"},
        }

        events = state_to_events(state, "test-thread")

        complete_events = [e for e in events if e.type == SSEEventType.PIPELINE_COMPLETE]
        assert len(complete_events) == 1
        assert complete_events[0].data["success"] is True

    def test_state_to_events_pipeline_complete_uses_plan_statuses(self) -> None:
        """Completion stats should derive from plan step statuses when available."""
        state = {
            "messages": [],
            "plan": [
                {"id": "step-0", "tool_name": "scan", "status": "completed"},
                {"id": "step-1", "tool_name": "detect", "status": "failed"},
                {"id": "step-2", "tool_name": "export", "status": "completed"},
            ],
            "plan_approved": True,
            "current_step": 3,
            "checkpoints": [],
            "awaiting_user": False,
            "execution_results": {
                # Legacy payloads may omit step status here.
                "step-0": {"files_processed": 10},
                "step-1": {"error": "boom"},
                "step-2": {"output_path": "/tmp/out"},
            },
            "metadata": {"pipeline_id": "pipe-xyz"},
        }

        events = state_to_events(state, "test-thread")
        complete_events = [e for e in events if e.type == SSEEventType.PIPELINE_COMPLETE]
        assert len(complete_events) == 1
        assert complete_events[0].data["completed_steps"] == 2
        assert complete_events[0].data["failed_steps"] == 1
        assert complete_events[0].data["success"] is False

    def test_state_to_events_ignores_internal_markers(self) -> None:
        """Messages starting with AWAIT_ should be ignored."""
        state = {
            "messages": [
                {"role": "system", "content": "AWAIT_APPROVAL"},
            ],
            "plan": [],
            "current_step": 0,
            "checkpoints": [],
            "awaiting_user": False,
        }

        events = state_to_events(state, "test-thread")

        message_events = [e for e in events if e.type == SSEEventType.MESSAGE]
        assert len(message_events) == 0


class TestStateToEventsDeduplication:
    """Tests for event deduplication in state_to_events."""

    def test_message_deduplication(self) -> None:
        """Duplicate messages should not be emitted."""
        thread = ThreadState("test-thread")
        state = {
            "messages": [
                {"role": "assistant", "content": "First message"},
            ],
            "plan": [],
            "current_step": 0,
            "checkpoints": [],
            "awaiting_user": False,
        }

        # First call should emit message
        events1 = state_to_events(state, "test-thread", thread)
        message_events1 = [e for e in events1 if e.type == SSEEventType.MESSAGE]
        assert len(message_events1) == 1

        # Second call with same state should not emit message
        events2 = state_to_events(state, "test-thread", thread)
        message_events2 = [e for e in events2 if e.type == SSEEventType.MESSAGE]
        assert len(message_events2) == 0

        # Adding new message should emit only the new one
        state["messages"].append({"role": "assistant", "content": "Second message"})
        events3 = state_to_events(state, "test-thread", thread)
        message_events3 = [e for e in events3 if e.type == SSEEventType.MESSAGE]
        assert len(message_events3) == 1
        assert message_events3[0].data["content"] == "Second message"

    def test_plan_deduplication(self) -> None:
        """Duplicate plan events should not be emitted."""
        thread = ThreadState("test-thread")
        state = {
            "messages": [],
            "plan": [
                {"id": "step-1", "tool_name": "scan"},
            ],
            "plan_approved": False,
            "current_step": 0,
            "checkpoints": [],
            "awaiting_user": False,
        }

        # First call should emit plan
        events1 = state_to_events(state, "test-thread", thread)
        plan_events1 = [e for e in events1 if e.type == SSEEventType.PLAN]
        assert len(plan_events1) == 1

        # Second call with same plan should not emit
        events2 = state_to_events(state, "test-thread", thread)
        plan_events2 = [e for e in events2 if e.type == SSEEventType.PLAN]
        assert len(plan_events2) == 0

    def test_checkpoint_deduplication(self) -> None:
        """Duplicate checkpoint events should not be emitted."""
        thread = ThreadState("test-thread")
        state = {
            "messages": [],
            "plan": [],
            "current_step": 0,
            "checkpoints": [
                {
                    "id": "ckpt-1",
                    "step_index": 2,
                    "trigger_reason": "percentage",
                    "quality_metrics": {},
                    "resolved_at": None,
                },
            ],
            "awaiting_user": False,
        }

        # First call should emit checkpoint
        events1 = state_to_events(state, "test-thread", thread)
        ckpt_events1 = [e for e in events1 if e.type == SSEEventType.CHECKPOINT]
        assert len(ckpt_events1) == 1

        # Second call with same checkpoint should not emit
        events2 = state_to_events(state, "test-thread", thread)
        ckpt_events2 = [e for e in events2 if e.type == SSEEventType.CHECKPOINT]
        assert len(ckpt_events2) == 0

    def test_plan_approved_emitted_once(self) -> None:
        """Plan approved message should only be emitted once."""
        thread = ThreadState("test-thread")
        state = {
            "messages": [],
            "plan": [{"id": "step-1", "tool_name": "scan"}],
            "plan_approved": True,
            "current_step": 0,
            "checkpoints": [],
            "awaiting_user": False,
        }

        # First call should emit plan approved message
        events1 = state_to_events(state, "test-thread", thread)
        msg_events1 = [
            e
            for e in events1
            if e.type == SSEEventType.MESSAGE and "approved" in e.data.get("content", "").lower()
        ]
        assert len(msg_events1) == 1

        # Second call should not emit plan approved again
        events2 = state_to_events(state, "test-thread", thread)
        msg_events2 = [
            e
            for e in events2
            if e.type == SSEEventType.MESSAGE and "approved" in e.data.get("content", "").lower()
        ]
        assert len(msg_events2) == 0


class TestThreadState:
    """Tests for ThreadState class."""

    def test_thread_state_initialization(self) -> None:
        """ThreadState should initialize with correct defaults."""
        thread = ThreadState("test-123")

        assert thread.thread_id == "test-123"
        assert thread.connection_count == 0
        assert thread.last_emitted_message_count == 0
        assert thread.last_emitted_plan_hash is None
        assert thread.last_emitted_checkpoint_id is None
        assert not thread.plan_approved_emitted

    def test_thread_state_touch(self) -> None:
        """Touch should update last_activity timestamp."""
        thread = ThreadState("test-123")
        initial_activity = thread.last_activity

        import time

        time.sleep(0.01)
        thread.touch()

        assert thread.last_activity > initial_activity

    def test_thread_state_expiry(self) -> None:
        """is_expired should correctly detect expired threads."""
        thread = ThreadState("test-123")

        # New thread should not be expired
        assert not thread.is_expired()

        # Manually set old timestamp
        import time

        thread.last_activity = time.time() - 7200  # 2 hours ago

        assert thread.is_expired()


class TestProcessAgentRequestStatePersistence:
    """Regression tests for state persistence across approval/resume."""

    @pytest.mark.asyncio
    async def test_resume_rehydrates_persisted_state_after_memory_reset(
        self,
        checkpoint_manager: CheckpointManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Approval resume should load persisted state even after in-memory reset."""
        thread_id = "thread-rehydrate-resume"
        checkpoint_manager.create_thread(thread_id, title="Resume thread")
        checkpoint_manager.save_snapshot(
            thread_id,
            "ckpt-1",
            {
                "channel_values": {
                    "messages": [{"role": "assistant", "content": "Approve this plan"}],
                    "plan": [{"id": "step-1", "tool_name": "scan_directory"}],
                    "plan_approved": False,
                    "current_step": 0,
                    "checkpoints": [],
                    "awaiting_user": True,
                    "metadata": {"pipeline_id": "pipe-rehydrate"},
                }
            },
        )

        _event_queues.clear()
        _thread_states.clear()
        _threads.clear()

        captured_resume_initial: dict[str, object] = {}

        @asynccontextmanager
        async def fake_compile_agent(_db_path: str):
            yield object()

        async def fake_run_agent_resume(_compiled, initial_state, _thread_id):
            captured_resume_initial.update(initial_state)
            yield {
                "plan_approved": True,
                "awaiting_user": False,
                "current_step": 0,
                "plan": initial_state.get("plan", []),
            }

        monkeypatch.setattr(
            "backend.api.streaming.endpoints.compile_agent",
            fake_compile_agent,
        )
        monkeypatch.setattr(
            "backend.api.streaming.endpoints.run_agent",
            fake_run_agent_resume,
        )

        await process_agent_request(
            thread_id,
            "Approved",
            decision=UserDecision.APPROVE,
        )

        assert captured_resume_initial.get("plan")
        assert captured_resume_initial.get("plan_approved") is True
        assert captured_resume_initial.get("awaiting_user") is False

    @pytest.mark.asyncio
    async def test_resume_keeps_plan_after_partial_updates(self, monkeypatch) -> None:
        """Partial node outputs must not drop plan before approval resume."""
        thread_id = "thread-resume"
        _get_or_create_thread(thread_id)

        @asynccontextmanager
        async def fake_compile_agent(_db_path: str):
            yield object()

        first_run_updates = [
            {
                "messages": [{"role": "assistant", "content": "Here's my proposed plan"}],
                "plan": [{"id": "step-1", "tool_name": "scan_directory", "description": "Scan"}],
                "plan_approved": False,
                "current_step": 0,
                "checkpoints": [],
                "awaiting_user": False,
            },
            {
                # Simulate a partial node update that only carries waiting markers.
                "messages": [{"role": "system", "content": "AWAIT_PLAN_APPROVAL"}],
                "awaiting_user": True,
            },
        ]

        async def fake_run_agent_first(_compiled, _initial_state, _thread_id):
            for update in first_run_updates:
                yield update

        monkeypatch.setattr(
            "backend.api.streaming.endpoints.compile_agent",
            fake_compile_agent,
        )
        monkeypatch.setattr(
            "backend.api.streaming.endpoints.run_agent",
            fake_run_agent_first,
        )

        await process_agent_request(thread_id, "user request")

        persisted = _threads[thread_id].pipeline_state
        assert persisted.get("awaiting_user") is True
        assert persisted.get("plan")
        assert persisted["plan"][0]["tool_name"] == "scan_directory"

        captured_resume_initial: dict[str, object] = {}

        async def fake_run_agent_resume(_compiled, initial_state, _thread_id):
            captured_resume_initial.update(initial_state)
            yield {
                "plan_approved": True,
                "awaiting_user": False,
                "current_step": 0,
                "plan": initial_state.get("plan", []),
            }

        monkeypatch.setattr(
            "backend.api.streaming.endpoints.run_agent",
            fake_run_agent_resume,
        )

        await process_agent_request(
            thread_id,
            "Approved",
            decision=UserDecision.APPROVE,
        )

        assert captured_resume_initial.get("plan")
        assert captured_resume_initial.get("plan_approved") is True
        assert captured_resume_initial.get("awaiting_user") is False
