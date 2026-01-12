"""Shared fixtures for LLM tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agent.llm.config import LLMConfig
from backend.agent.llm.provider import OllamaProvider, clear_llm_cache, clear_providers


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset LLM caches before and after each test."""
    clear_llm_cache()
    clear_providers()
    yield
    clear_llm_cache()
    clear_providers()


@pytest.fixture
def test_config() -> LLMConfig:
    """Create a test LLM configuration."""
    return LLMConfig(
        model="test-model",
        temperature=0.1,
        max_tokens=1024,
        timeout=10,
        base_url="http://localhost:11434",
        max_retries=2,
        retry_delay=0.01,  # Fast for tests
        retry_backoff=2.0,
        fallback_models=["fallback-1", "fallback-2"],
    )


@pytest.fixture
def mock_llm():
    """Create a mock ChatOllama instance."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(
        return_value=AIMessage(content="Test response")
    )
    mock.astream = AsyncMock()
    mock.bind_tools = MagicMock(return_value=mock)
    return mock


@pytest.fixture
def provider_with_mock_llm(test_config, mock_llm):
    """Create a provider with mocked LLM."""
    provider = OllamaProvider(test_config)
    provider._llm = mock_llm
    return provider


@pytest.fixture
def sample_messages() -> list:
    """Sample conversation messages for testing."""
    return [
        HumanMessage(content="Hello, scan the /data directory"),
    ]


@pytest.fixture
def sample_tool_schemas() -> list[dict[str, Any]]:
    """Sample tool schemas for testing."""
    return [
        {
            "type": "function",
            "function": {
                "name": "scan_directory",
                "description": "Scan a directory for files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
    ]


@pytest.fixture
def ai_message_with_tool_calls() -> AIMessage:
    """AI message with tool calls in LangChain format."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_123",
                "name": "scan_directory",
                "args": {"path": "/data"},
            },
        ],
    )


@pytest.fixture
def ai_message_with_json_tool_call() -> AIMessage:
    """AI message with tool call in JSON content format."""
    return AIMessage(
        content='{"function": {"name": "scan_directory", "arguments": {"path": "/data"}}}'
    )


@pytest.fixture
def ai_message_plain() -> AIMessage:
    """AI message without tool calls."""
    return AIMessage(content="The directory has been scanned successfully.")


@pytest.fixture
def tool_result_message() -> ToolMessage:
    """Sample tool result message."""
    return ToolMessage(
        content='{"success": true, "data": {"file_count": 10}}',
        tool_call_id="call_123",
    )
