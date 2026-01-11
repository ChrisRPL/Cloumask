"""Tests for LangGraph agent state types."""

import json
from datetime import datetime

import pytest

from backend.agent.state import (
    Checkpoint,
    CheckpointTrigger,
    Message,
    MessageRole,
    PipelineMetadata,
    PipelineStep,
    QualityMetrics,
    StepStatus,
    ToolCall,
    UserDecision,
    UserFeedback,
    create_initial_state,
    deserialize_state,
    serialize_state,
)


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_user_role_value(self) -> None:
        """USER role should have correct string value."""
        assert MessageRole.USER.value == "user"

    def test_assistant_role_value(self) -> None:
        """ASSISTANT role should have correct string value."""
        assert MessageRole.ASSISTANT.value == "assistant"

    def test_system_role_value(self) -> None:
        """SYSTEM role should have correct string value."""
        assert MessageRole.SYSTEM.value == "system"

    def test_tool_role_value(self) -> None:
        """TOOL role should have correct string value."""
        assert MessageRole.TOOL.value == "tool"

    def test_enum_is_string(self) -> None:
        """MessageRole should be a string enum for JSON compatibility."""
        assert isinstance(MessageRole.USER, str)


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """All expected statuses should be defined."""
        statuses = {s.value for s in StepStatus}
        expected = {"pending", "running", "completed", "failed", "skipped"}
        assert statuses == expected

    def test_enum_is_string(self) -> None:
        """StepStatus should be a string enum."""
        assert isinstance(StepStatus.PENDING, str)


class TestCheckpointTrigger:
    """Tests for CheckpointTrigger enum."""

    def test_all_triggers_exist(self) -> None:
        """All expected triggers should be defined."""
        triggers = {t.value for t in CheckpointTrigger}
        expected = {"percentage", "quality_drop", "error_rate", "critical_step"}
        assert triggers == expected


class TestUserDecision:
    """Tests for UserDecision enum."""

    def test_all_decisions_exist(self) -> None:
        """All expected decisions should be defined."""
        decisions = {d.value for d in UserDecision}
        expected = {"approve", "edit", "cancel", "retry"}
        assert decisions == expected


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_tool_call(self) -> None:
        """ToolCall should be creatable with valid data."""
        tc = ToolCall(
            id="call-123",
            name="detect_objects",
            arguments={"confidence": 0.8},
        )
        assert tc.id == "call-123"
        assert tc.name == "detect_objects"
        assert tc.arguments == {"confidence": 0.8}

    def test_tool_call_default_arguments(self) -> None:
        """ToolCall should have empty dict as default arguments."""
        tc = ToolCall(id="call-1", name="scan")
        assert tc.arguments == {}

    def test_tool_call_serialization(self) -> None:
        """ToolCall should serialize to dict."""
        tc = ToolCall(id="call-1", name="scan", arguments={"path": "/data"})
        data = tc.model_dump()
        assert data["id"] == "call-1"
        assert data["name"] == "scan"
        assert data["arguments"] == {"path": "/data"}


class TestMessage:
    """Tests for Message model."""

    def test_create_user_message(self) -> None:
        """Message should be creatable with minimal data."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.tool_calls == []
        assert msg.tool_call_id is None

    def test_message_has_timestamp(self) -> None:
        """Message should have auto-generated timestamp."""
        msg = Message(role=MessageRole.USER, content="Test")
        assert isinstance(msg.timestamp, datetime)

    def test_message_to_dict(self) -> None:
        """Message.to_dict should return serializable dict."""
        msg = Message(role=MessageRole.USER, content="Hello")
        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert "timestamp" in data
        assert data["tool_calls"] == []
        assert data["tool_call_id"] is None

    def test_message_with_tool_calls(self) -> None:
        """Message should support tool calls."""
        tc = ToolCall(id="tc-1", name="detect", arguments={"path": "/img.jpg"})
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[tc],
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "detect"

    def test_message_timestamp_iso_format(self) -> None:
        """to_dict should return timestamp in ISO format."""
        msg = Message(role=MessageRole.USER, content="Test")
        data = msg.to_dict()
        # Should be parseable as ISO format
        datetime.fromisoformat(data["timestamp"])

    def test_tool_response_message(self) -> None:
        """Message should support tool_call_id for tool responses."""
        msg = Message(
            role=MessageRole.TOOL,
            content='{"result": "success"}',
            tool_call_id="tc-123",
        )
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "tc-123"


class TestPipelineStep:
    """Tests for PipelineStep model."""

    def test_create_step(self) -> None:
        """PipelineStep should be creatable with required fields."""
        step = PipelineStep(
            id="step-1",
            tool_name="scan_directory",
            parameters={"path": "/data"},
            description="Scan input directory",
        )
        assert step.id == "step-1"
        assert step.status == StepStatus.PENDING

    def test_step_default_values(self) -> None:
        """PipelineStep should have correct default values."""
        step = PipelineStep(
            id="s1",
            tool_name="test",
            parameters={},
            description="Test",
        )
        assert step.status == StepStatus.PENDING
        assert step.result is None
        assert step.error is None
        assert step.started_at is None
        assert step.completed_at is None

    def test_step_duration_none_when_not_started(self) -> None:
        """Duration should be None when step not started."""
        step = PipelineStep(
            id="s1",
            tool_name="test",
            parameters={},
            description="Test",
        )
        assert step.duration_seconds is None

    def test_step_duration_none_when_not_completed(self) -> None:
        """Duration should be None when step not completed."""
        step = PipelineStep(
            id="s1",
            tool_name="test",
            parameters={},
            description="Test",
            started_at=datetime.now(),
        )
        assert step.duration_seconds is None

    def test_step_duration_calculated(self) -> None:
        """Duration should calculate correctly when both times set."""
        start = datetime(2026, 1, 1, 10, 0, 0)
        end = datetime(2026, 1, 1, 10, 0, 30)
        step = PipelineStep(
            id="s1",
            tool_name="test",
            parameters={},
            description="Test",
            started_at=start,
            completed_at=end,
        )
        assert step.duration_seconds == 30.0

    def test_step_with_error(self) -> None:
        """PipelineStep should store error information."""
        step = PipelineStep(
            id="s1",
            tool_name="detect",
            parameters={},
            description="Detect objects",
            status=StepStatus.FAILED,
            error="File not found",
        )
        assert step.status == StepStatus.FAILED
        assert step.error == "File not found"


class TestQualityMetrics:
    """Tests for QualityMetrics model."""

    def test_create_metrics(self) -> None:
        """QualityMetrics should be creatable with valid data."""
        metrics = QualityMetrics(
            average_confidence=0.85,
            error_count=5,
            total_processed=100,
            processing_speed=10.0,
        )
        assert metrics.average_confidence == 0.85
        assert metrics.error_count == 5

    def test_error_rate_calculation(self) -> None:
        """Error rate should calculate correctly."""
        metrics = QualityMetrics(
            average_confidence=0.85,
            error_count=5,
            total_processed=100,
            processing_speed=10.0,
        )
        assert metrics.error_rate == 0.05

    def test_error_rate_zero_when_no_items(self) -> None:
        """Error rate should be 0 when total_processed is 0."""
        metrics = QualityMetrics(
            average_confidence=0.0,
            error_count=0,
            total_processed=0,
            processing_speed=0.0,
        )
        assert metrics.error_rate == 0.0

    def test_confidence_validation_upper_bound(self) -> None:
        """Confidence must be at most 1.0."""
        with pytest.raises(ValueError):
            QualityMetrics(
                average_confidence=1.5,
                error_count=0,
                total_processed=10,
                processing_speed=1.0,
            )

    def test_confidence_validation_lower_bound(self) -> None:
        """Confidence must be at least 0.0."""
        with pytest.raises(ValueError):
            QualityMetrics(
                average_confidence=-0.1,
                error_count=0,
                total_processed=10,
                processing_speed=1.0,
            )

    def test_error_count_non_negative(self) -> None:
        """Error count must be non-negative."""
        with pytest.raises(ValueError):
            QualityMetrics(
                average_confidence=0.9,
                error_count=-1,
                total_processed=10,
                processing_speed=1.0,
            )


class TestCheckpoint:
    """Tests for Checkpoint model."""

    def test_create_checkpoint(self) -> None:
        """Checkpoint should be creatable with required fields."""
        metrics = QualityMetrics(
            average_confidence=0.9,
            error_count=2,
            total_processed=50,
            processing_speed=5.0,
        )
        checkpoint = Checkpoint(
            id="cp-1",
            step_index=5,
            trigger_reason=CheckpointTrigger.PERCENTAGE,
            quality_metrics=metrics,
        )
        assert checkpoint.id == "cp-1"
        assert checkpoint.user_decision is None
        assert checkpoint.resolved_at is None

    def test_checkpoint_with_user_decision(self) -> None:
        """Checkpoint should store user decision."""
        metrics = QualityMetrics(
            average_confidence=0.8,
            error_count=1,
            total_processed=25,
            processing_speed=3.0,
        )
        checkpoint = Checkpoint(
            id="cp-2",
            step_index=3,
            trigger_reason=CheckpointTrigger.QUALITY_DROP,
            quality_metrics=metrics,
            user_decision=UserDecision.APPROVE,
            user_feedback="Looks good",
            resolved_at=datetime.now(),
        )
        assert checkpoint.user_decision == UserDecision.APPROVE
        assert checkpoint.user_feedback == "Looks good"


class TestUserFeedback:
    """Tests for UserFeedback model."""

    def test_create_approve_feedback(self) -> None:
        """UserFeedback should be creatable for approval."""
        feedback = UserFeedback(decision=UserDecision.APPROVE)
        assert feedback.decision == UserDecision.APPROVE
        assert feedback.message is None
        assert feedback.plan_edits is None

    def test_create_edit_feedback(self) -> None:
        """UserFeedback should support plan edits."""
        feedback = UserFeedback(
            decision=UserDecision.EDIT,
            message="Increase confidence threshold",
            plan_edits=[{"step": 2, "parameters": {"confidence": 0.9}}],
        )
        assert feedback.decision == UserDecision.EDIT
        assert feedback.plan_edits is not None
        assert len(feedback.plan_edits) == 1

    def test_create_cancel_feedback(self) -> None:
        """UserFeedback should support cancel decision."""
        feedback = UserFeedback(
            decision=UserDecision.CANCEL,
            message="Wrong folder selected",
        )
        assert feedback.decision == UserDecision.CANCEL


class TestPipelineMetadata:
    """Tests for PipelineMetadata model."""

    def test_create_metadata(self) -> None:
        """PipelineMetadata should be creatable with required fields."""
        metadata = PipelineMetadata(
            pipeline_id="pipe-123",
            created_at=datetime.now(),
        )
        assert metadata.pipeline_id == "pipe-123"
        assert metadata.total_files == 0
        assert metadata.processed_files == 0

    def test_metadata_with_paths(self) -> None:
        """PipelineMetadata should store paths."""
        metadata = PipelineMetadata(
            pipeline_id="pipe-456",
            created_at=datetime.now(),
            input_path="/data/input",
            output_path="/data/output",
            total_files=100,
            processed_files=50,
        )
        assert metadata.input_path == "/data/input"
        assert metadata.output_path == "/data/output"
        assert metadata.total_files == 100
        assert metadata.processed_files == 50

    def test_metadata_json_serialization(self) -> None:
        """PipelineMetadata should serialize to JSON-safe dict."""
        metadata = PipelineMetadata(
            pipeline_id="pipe-789",
            created_at=datetime.now(),
        )
        data = metadata.model_dump(mode="json")
        # Should be JSON-serializable
        json.dumps(data)
        # created_at should be a string, not datetime
        assert isinstance(data["created_at"], str)


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_creates_valid_state(self) -> None:
        """Initial state should have correct structure."""
        state = create_initial_state("Scan /data", "pipe-123")

        assert len(state["messages"]) == 1
        assert state["messages"][0]["role"] == "user"
        assert state["messages"][0]["content"] == "Scan /data"
        assert state["plan"] == []
        assert state["current_step"] == 0
        assert state["awaiting_user"] is False

    def test_initial_state_has_metadata(self) -> None:
        """Initial state should have pipeline metadata."""
        state = create_initial_state("Test", "test-id")
        assert state["metadata"]["pipeline_id"] == "test-id"

    def test_initial_state_not_approved(self) -> None:
        """Initial state should have plan_approved=False."""
        state = create_initial_state("Test", "test-id")
        assert state["plan_approved"] is False

    def test_initial_state_no_error(self) -> None:
        """Initial state should have no errors."""
        state = create_initial_state("Test", "test-id")
        assert state["last_error"] is None
        assert state["retry_count"] == 0

    def test_initial_state_empty_execution_results(self) -> None:
        """Initial state should have empty execution_results."""
        state = create_initial_state("Test", "test-id")
        assert state["execution_results"] == {}

    def test_initial_state_empty_checkpoints(self) -> None:
        """Initial state should have empty checkpoints list."""
        state = create_initial_state("Test", "test-id")
        assert state["checkpoints"] == []


class TestStateSerialization:
    """Tests for serialize_state and deserialize_state functions."""

    def test_serialize_state(self) -> None:
        """State should serialize to JSON string."""
        state = create_initial_state("Test message", "test-id")
        json_str = serialize_state(state)

        assert isinstance(json_str, str)
        # Should be valid JSON
        json.loads(json_str)

    def test_deserialize_state(self) -> None:
        """State should deserialize from JSON string."""
        state = create_initial_state("Test message", "test-id")
        json_str = serialize_state(state)
        restored = deserialize_state(json_str)

        assert restored["messages"] == state["messages"]
        assert restored["plan_approved"] == state["plan_approved"]

    def test_roundtrip_serialization(self) -> None:
        """State should survive serialization roundtrip."""
        original = create_initial_state("Roundtrip test", "round-123")
        json_str = serialize_state(original)
        restored = deserialize_state(json_str)

        assert restored["messages"][0]["content"] == "Roundtrip test"
        assert restored["metadata"]["pipeline_id"] == "round-123"

    def test_serialization_handles_unicode(self) -> None:
        """Serialization should handle unicode content."""
        state = create_initial_state("Hello Welt Bonjour", "unicode-test")
        json_str = serialize_state(state)
        restored = deserialize_state(json_str)
        assert restored["messages"][0]["content"] == "Hello Welt Bonjour"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_message_content(self) -> None:
        """Message should allow empty content (for tool calls)."""
        msg = Message(role=MessageRole.ASSISTANT, content="")
        assert msg.content == ""

    def test_unicode_message_content(self) -> None:
        """Message should handle unicode content."""
        msg = Message(role=MessageRole.USER, content="Hello Welt Bonjour")
        data = msg.to_dict()
        assert data["content"] == "Hello Welt Bonjour"

    def test_long_content(self) -> None:
        """Message should handle very long content."""
        long_content = "x" * 100000
        msg = Message(role=MessageRole.USER, content=long_content)
        assert len(msg.content) == 100000

    def test_state_with_multiple_messages(self) -> None:
        """State should support multiple messages."""
        state = create_initial_state("First", "multi-msg")
        second_msg = Message(role=MessageRole.ASSISTANT, content="Response")
        state["messages"].append(second_msg.to_dict())

        assert len(state["messages"]) == 2

    def test_special_characters_in_content(self) -> None:
        """Message should handle special characters."""
        content = 'Path: /data/test.json "quoted" \'single\' <tag>'
        msg = Message(role=MessageRole.USER, content=content)
        data = msg.to_dict()
        assert data["content"] == content

    def test_nested_tool_arguments(self) -> None:
        """ToolCall should handle nested arguments."""
        tc = ToolCall(
            id="tc-nested",
            name="complex_tool",
            arguments={
                "config": {"threshold": 0.5, "options": ["a", "b"]},
                "paths": ["/path/1", "/path/2"],
            },
        )
        data = tc.model_dump()
        assert data["arguments"]["config"]["threshold"] == 0.5
        assert len(data["arguments"]["paths"]) == 2
