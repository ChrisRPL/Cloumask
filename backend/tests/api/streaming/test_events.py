"""Tests for SSE event types and builder functions."""

import json

from backend.api.streaming.events import (
    MessageEventData,
    SSEEventType,
    checkpoint_event,
    complete_event,
    connected_event,
    create_event,
    error_event,
    heartbeat_event,
    message_event,
    plan_event,
    thinking_event,
    tool_progress_event,
    tool_result_event,
    tool_start_event,
)


class TestSSEEventType:
    """Tests for SSEEventType enum."""

    def test_all_event_types_defined(self) -> None:
        """All expected event types should be defined."""
        expected_types = [
            "message",
            "thinking",
            "plan",
            "plan_approved",
            "tool_start",
            "tool_progress",
            "tool_result",
            "checkpoint",
            "await_input",
            "step_start",
            "step_complete",
            "complete",
            "error",
            "warning",
            "connected",
            "heartbeat",
        ]
        actual_values = [e.value for e in SSEEventType]
        for expected in expected_types:
            assert expected in actual_values

    def test_event_type_is_string_enum(self) -> None:
        """SSEEventType values should be strings."""
        assert SSEEventType.MESSAGE.value == "message"
        assert isinstance(SSEEventType.MESSAGE, str)


class TestSSEEvent:
    """Tests for SSEEvent class."""

    def test_sse_event_format(self) -> None:
        """SSE event should format correctly."""
        event = message_event("assistant", "Hello!")
        sse_str = event.to_sse()

        assert "event: message" in sse_str
        assert "data:" in sse_str
        assert "Hello!" in sse_str
        assert sse_str.endswith("\n\n")

    def test_sse_event_to_dict(self) -> None:
        """SSE event should convert to dict correctly."""
        event = message_event("assistant", "Test")
        event_dict = event.to_dict()

        assert event_dict["type"] == "message"
        assert "timestamp" in event_dict
        assert event_dict["data"]["role"] == "assistant"
        assert event_dict["data"]["content"] == "Test"

    def test_sse_event_json_serializable(self) -> None:
        """SSE event dict should be JSON serializable."""
        event = error_event("TEST_ERROR", "Something went wrong")
        event_dict = event.to_dict()

        # Should not raise
        json_str = json.dumps(event_dict)
        parsed = json.loads(json_str)

        assert parsed["type"] == "error"
        assert parsed["data"]["error_code"] == "TEST_ERROR"


class TestMessageEvent:
    """Tests for message event builder."""

    def test_message_event_user(self) -> None:
        """Message event should include user role correctly."""
        event = message_event("user", "Hello")

        assert event.type == SSEEventType.MESSAGE
        assert event.data["role"] == "user"
        assert event.data["content"] == "Hello"

    def test_message_event_assistant(self) -> None:
        """Message event should include assistant role correctly."""
        event = message_event("assistant", "Hi there!")

        assert event.data["role"] == "assistant"
        assert event.data["content"] == "Hi there!"

    def test_message_event_with_id(self) -> None:
        """Message event should include optional message_id."""
        event = message_event("assistant", "Response", message_id="msg-123")

        assert event.data["message_id"] == "msg-123"


class TestThinkingEvent:
    """Tests for thinking event builder."""

    def test_thinking_event_default(self) -> None:
        """Thinking event should have default message."""
        event = thinking_event()

        assert event.type == SSEEventType.THINKING
        assert event.data["message"] == "Processing..."

    def test_thinking_event_custom(self) -> None:
        """Thinking event should accept custom message."""
        event = thinking_event("Analyzing your request...")

        assert event.data["message"] == "Analyzing your request..."


class TestPlanEvent:
    """Tests for plan event builder."""

    def test_plan_event_structure(self) -> None:
        """Plan event should include steps and count."""
        steps = [
            {"id": "step-1", "tool_name": "scan", "description": "Scan files"},
            {"id": "step-2", "tool_name": "detect", "description": "Detect objects"},
        ]
        event = plan_event("plan-123", steps)

        assert event.type == SSEEventType.PLAN
        assert event.data["plan_id"] == "plan-123"
        assert event.data["total_steps"] == 2
        assert len(event.data["steps"]) == 2

    def test_plan_event_empty_steps(self) -> None:
        """Plan event should handle empty steps list."""
        event = plan_event("plan-empty", [])

        assert event.data["total_steps"] == 0
        assert event.data["steps"] == []


class TestToolProgressEvent:
    """Tests for tool progress event builder."""

    def test_tool_progress_event_percentage(self) -> None:
        """Progress event should calculate percentage correctly."""
        event = tool_progress_event("scan", 0, 50, 100, "Scanning...")

        assert event.type == SSEEventType.TOOL_PROGRESS
        assert event.data["percentage"] == 50.0
        assert event.data["message"] == "Scanning..."

    def test_tool_progress_event_zero_total(self) -> None:
        """Progress event should handle zero total gracefully."""
        event = tool_progress_event("scan", 0, 0, 0)

        assert event.data["percentage"] == 0

    def test_tool_progress_event_complete(self) -> None:
        """Progress event at 100% should work."""
        event = tool_progress_event("scan", 0, 100, 100)

        assert event.data["percentage"] == 100.0


class TestToolStartEvent:
    """Tests for tool start event builder."""

    def test_tool_start_event(self) -> None:
        """Tool start event should include all parameters."""
        params = {"path": "/data", "recursive": True}
        event = tool_start_event("scan_directory", 0, params)

        assert event.type == SSEEventType.TOOL_START
        assert event.data["tool_name"] == "scan_directory"
        assert event.data["step_index"] == 0
        assert event.data["parameters"] == params


class TestToolResultEvent:
    """Tests for tool result event builder."""

    def test_tool_result_event_success(self) -> None:
        """Tool result event should handle success."""
        result = {"files_found": 10}
        event = tool_result_event("scan", 0, True, result=result)

        assert event.type == SSEEventType.TOOL_RESULT
        assert event.data["success"] is True
        assert event.data["result"] == result
        assert event.data["error"] is None

    def test_tool_result_event_failure(self) -> None:
        """Tool result event should handle failure."""
        event = tool_result_event("scan", 0, False, error="File not found")

        assert event.data["success"] is False
        assert event.data["error"] == "File not found"


class TestCheckpointEvent:
    """Tests for checkpoint event builder."""

    def test_checkpoint_event_structure(self) -> None:
        """Checkpoint event should include all fields."""
        checkpoint = {
            "id": "ckpt-123",
            "step_index": 2,
            "trigger_reason": "percentage",
            "quality_metrics": {"confidence": 0.9},
            "progress_percent": 50.0,
        }
        event = checkpoint_event(checkpoint, "Checkpoint reached")

        assert event.type == SSEEventType.CHECKPOINT
        assert event.data["checkpoint_id"] == "ckpt-123"
        assert event.data["step_index"] == 2
        assert event.data["trigger_reason"] == "percentage"
        assert event.data["message"] == "Checkpoint reached"


class TestErrorEvent:
    """Tests for error event builder."""

    def test_error_event_structure(self) -> None:
        """Error event should include error details."""
        event = error_event("TOOL_FAILED", "Scan failed", recoverable=True)

        assert event.type == SSEEventType.ERROR
        assert event.data["error_code"] == "TOOL_FAILED"
        assert event.data["message"] == "Scan failed"
        assert event.data["recoverable"] is True

    def test_error_event_with_details(self) -> None:
        """Error event should include optional details."""
        details = {"file": "/data/test.jpg", "line": 42}
        event = error_event("PARSE_ERROR", "Failed to parse", details=details)

        assert event.data["details"] == details

    def test_error_event_not_recoverable(self) -> None:
        """Error event should handle non-recoverable errors."""
        event = error_event("FATAL_ERROR", "Critical failure", recoverable=False)

        assert event.data["recoverable"] is False


class TestCompleteEvent:
    """Tests for pipeline complete event builder."""

    def test_complete_event_success(self) -> None:
        """Complete event should indicate success when no failures."""
        stats = {
            "pipeline_id": "pipe-123",
            "total_steps": 5,
            "completed_steps": 5,
            "failed_steps": 0,
            "total_duration_seconds": 10.5,
            "summary": "All steps completed successfully",
        }
        event = complete_event(stats)

        assert event.type == SSEEventType.PIPELINE_COMPLETE
        assert event.data["success"] is True
        assert event.data["total_steps"] == 5
        assert event.data["completed_steps"] == 5

    def test_complete_event_with_failures(self) -> None:
        """Complete event should indicate failure when steps fail."""
        stats = {
            "pipeline_id": "pipe-123",
            "total_steps": 5,
            "completed_steps": 4,
            "failed_steps": 1,
        }
        event = complete_event(stats)

        assert event.data["success"] is False
        assert event.data["failed_steps"] == 1


class TestConnectionEvents:
    """Tests for connection-related events."""

    def test_connected_event(self) -> None:
        """Connected event should include thread_id."""
        event = connected_event("thread-123")

        assert event.type == SSEEventType.CONNECTED
        assert event.data["thread_id"] == "thread-123"

    def test_heartbeat_event(self) -> None:
        """Heartbeat event should include sequence number."""
        event = heartbeat_event(42)

        assert event.type == SSEEventType.HEARTBEAT
        assert event.data["sequence"] == 42


class TestCreateEvent:
    """Tests for generic event creation."""

    def test_create_event_with_dict(self) -> None:
        """create_event should handle dict data."""
        event = create_event(SSEEventType.WARNING, {"message": "Low memory"})

        assert event.type == SSEEventType.WARNING
        assert event.data["message"] == "Low memory"

    def test_create_event_with_dataclass(self) -> None:
        """create_event should handle dataclass data."""
        data = MessageEventData(role="user", content="Hello")
        event = create_event(SSEEventType.MESSAGE, data)

        assert event.data["role"] == "user"
        assert event.data["content"] == "Hello"

    def test_create_event_with_scalar(self) -> None:
        """create_event should wrap scalar values."""
        event = create_event(SSEEventType.THINKING, "Processing")

        assert event.data["value"] == "Processing"

    def test_create_event_has_timestamp(self) -> None:
        """Created events should have ISO format timestamp."""
        event = create_event(SSEEventType.HEARTBEAT, {"seq": 1})

        # Should be ISO format with T separator
        assert "T" in event.timestamp
