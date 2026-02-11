"""Tests for intent router nodes."""

from __future__ import annotations

import pytest

from backend.agent.nodes.router import (
    chat_reply,
    classify_intent,
    get_latest_user_message,
    route_intent,
)
from backend.agent.state import MessageRole, PipelineState


def _state_with_user_message(content: str) -> PipelineState:
    return PipelineState(
        messages=[
            {
                "role": MessageRole.USER.value,
                "content": content,
                "timestamp": "2024-01-01T00:00:00",
                "tool_calls": [],
                "tool_call_id": None,
            }
        ],
        metadata={},
        plan=[],
        plan_approved=False,
        current_step=0,
        execution_results={},
        checkpoints=[],
        awaiting_user=False,
        last_error=None,
        retry_count=0,
    )


def test_classify_intent_routes_smalltalk_to_chat() -> None:
    assert classify_intent("hi") == "chat"
    assert classify_intent("thanks!") == "chat"


def test_classify_intent_routes_task_requests() -> None:
    assert classify_intent("detect cars in /tmp/images and export yolo") == "task"
    assert classify_intent("scan directory /data") == "task"


def test_get_latest_user_message() -> None:
    state = _state_with_user_message("hello")
    assert get_latest_user_message(state) == "hello"


@pytest.mark.asyncio
async def test_route_intent_sets_metadata_route() -> None:
    state = _state_with_user_message("hello")
    result = await route_intent(state)
    assert result["metadata"]["intent_route"] == "chat"


@pytest.mark.asyncio
async def test_chat_reply_appends_assistant_message() -> None:
    state = _state_with_user_message("hi")
    result = await chat_reply(state)
    messages = result["messages"]
    assert messages[-1]["role"] == MessageRole.ASSISTANT.value
    assert "I can help" in messages[-1]["content"]
