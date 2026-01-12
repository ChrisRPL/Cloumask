"""Tests for SSE streaming endpoints."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.streaming.endpoints import (
    _event_queues,
    _thread_states,
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


class TestCreateThread:
    """Tests for thread creation endpoint."""

    def test_create_thread_returns_200(self, client: TestClient) -> None:
        """Create thread endpoint should return 200."""
        response = client.post("/api/chat/thread/new")
        assert response.status_code == 200

    def test_create_thread_returns_thread_id(self, client: TestClient) -> None:
        """Create thread should return a thread_id."""
        response = client.post("/api/chat/thread/new")
        data = response.json()

        assert "thread_id" in data
        assert len(data["thread_id"]) > 0

    def test_create_thread_marks_created(self, client: TestClient) -> None:
        """Create thread should mark created=True."""
        response = client.post("/api/chat/thread/new")
        data = response.json()

        assert data["created"] is True


class TestGetThreadInfo:
    """Tests for thread info endpoint."""

    def test_get_thread_not_found(self, client: TestClient) -> None:
        """Get thread info should return 404 for unknown thread."""
        response = client.get("/api/chat/thread/unknown-id")
        assert response.status_code == 404

    def test_get_thread_info(self, client: TestClient) -> None:
        """Get thread info should return thread state."""
        # First create a thread
        create_response = client.post("/api/chat/thread/new")
        thread_id = create_response.json()["thread_id"]

        # Then get info
        response = client.get(f"/api/chat/thread/{thread_id}")
        data = response.json()

        assert response.status_code == 200
        assert data["thread_id"] == thread_id
        assert data["created"] is False  # Not a new thread


class TestCloseThread:
    """Tests for thread close endpoint."""

    def test_close_thread(self, client: TestClient) -> None:
        """Close thread should cleanup resources."""
        # Create thread
        create_response = client.post("/api/chat/thread/new")
        thread_id = create_response.json()["thread_id"]

        # Close it
        response = client.delete(f"/api/chat/thread/{thread_id}")
        data = response.json()

        assert response.status_code == 200
        assert data["status"] == "closed"
        assert data["thread_id"] == thread_id

    def test_close_thread_removes_from_queues(self, client: TestClient) -> None:
        """Closing thread should remove it from event queues."""
        create_response = client.post("/api/chat/thread/new")
        thread_id = create_response.json()["thread_id"]

        assert thread_id in _event_queues

        client.delete(f"/api/chat/thread/{thread_id}")

        assert thread_id not in _event_queues


class TestSendMessage:
    """Tests for send message endpoint."""

    def test_send_message_returns_queued(self, client: TestClient) -> None:
        """Send message should return queued status."""
        # Create thread first
        create_response = client.post("/api/chat/thread/new")
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
        create_response = client.post("/api/chat/thread/new")
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
        create_response = client.post("/api/chat/thread/new")
        thread_id = create_response.json()["thread_id"]

        response = client.post(
            f"/api/chat/send/{thread_id}",
            json={"content": "Approved", "decision": "approve"},
        )

        assert response.status_code == 200


class TestSSEStream:
    """Tests for SSE streaming endpoint."""

    def test_sse_endpoint_exists(self, client: TestClient) -> None:
        """SSE endpoint should be defined and accessible."""
        # Create thread first
        create_response = client.post("/api/chat/thread/new")
        thread_id = create_response.json()["thread_id"]

        # Just verify the endpoint is accessible
        # Full SSE testing requires async test client with proper handling
        # This test verifies the route exists and accepts GET requests
        from fastapi import FastAPI
        from backend.api.main import app

        # Verify the route is registered
        route_paths = [route.path for route in app.routes]
        assert "/api/chat/stream/{thread_id}" in route_paths

        # Cleanup
        client.delete(f"/api/chat/thread/{thread_id}")


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
