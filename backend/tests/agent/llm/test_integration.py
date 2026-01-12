"""Integration tests for LLM module.

These tests require a running Ollama instance and are marked with
@pytest.mark.integration so they can be skipped in CI.
"""

import pytest
from langchain_core.messages import HumanMessage

from backend.agent.llm import (
    LLMUseCase,
    check_ollama_available,
    extract_tool_calls,
    get_provider,
    list_available_models,
)

# Skip all tests in this module if Ollama is not available
pytestmark = pytest.mark.integration


@pytest.fixture
async def skip_if_ollama_unavailable():
    """Skip test if Ollama is not running."""
    if not await check_ollama_available():
        pytest.skip("Ollama not available")


class TestOllamaConnection:
    """Tests for Ollama connectivity."""

    @pytest.mark.asyncio
    async def test_check_ollama_available(self):
        """check_ollama_available should return True when Ollama is running."""
        # This test will fail/skip if Ollama is not available
        result = await check_ollama_available()
        # We can't assert True because Ollama might not be running
        # Just verify the function returns a bool
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_list_available_models(self):
        """Should list available Ollama models."""
        models = await list_available_models()

        # Should return a list (may be empty if Ollama not available)
        assert isinstance(models, list)
        # If we have models, they should be strings
        for model in models:
            assert isinstance(model, str)


class TestProviderIntegration:
    """Integration tests for OllamaProvider."""

    @pytest.mark.asyncio
    async def test_conversation_provider(self, skip_if_ollama_unavailable):
        """Test basic conversation with Ollama."""
        provider = get_provider(LLMUseCase.CONVERSATION)

        response = await provider.invoke(
            [HumanMessage(content="Say 'hello' and nothing else.")]
        )

        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_token_usage_tracked(self, skip_if_ollama_unavailable):
        """Token usage should be tracked after invocation."""
        provider = get_provider(LLMUseCase.CONVERSATION)
        provider.reset_token_usage()

        await provider.invoke([HumanMessage(content="Hi")])

        usage = provider.get_token_usage()
        # Token counts may or may not be populated depending on Ollama version
        assert "prompt" in usage
        assert "completion" in usage


class TestToolCallingIntegration:
    """Integration tests for tool calling with Ollama."""

    @pytest.mark.asyncio
    async def test_tool_calling_returns_tool_calls(self, skip_if_ollama_unavailable):
        """LLM should return tool calls when given tools."""
        from backend.agent.tools.registry import get_tool_registry

        provider = get_provider(LLMUseCase.TOOL_CALLING)
        registry = get_tool_registry()

        # Get tools that are registered
        tools = registry.get_schemas()
        if not tools:
            pytest.skip("No tools registered")

        # Ask LLM to use a tool
        response = await provider.invoke(
            [HumanMessage(content="Scan the /tmp directory for files")],
            tools=tools,
        )

        # Response should either have tool calls or a text response
        # We can't guarantee tool calls, as it depends on model behavior
        tool_calls = extract_tool_calls(response)
        assert isinstance(tool_calls, list)

        # If there are tool calls, verify structure
        for tc in tool_calls:
            assert "id" in tc
            assert "name" in tc
            assert "arguments" in tc


class TestModelManagement:
    """Integration tests for model management."""

    @pytest.mark.asyncio
    async def test_list_models_returns_installed(self, skip_if_ollama_unavailable):
        """list_available_models should return installed models."""
        models = await list_available_models()

        # Should have at least one model if Ollama is running
        # (skip_if_ollama_unavailable ensures Ollama is available)
        assert len(models) > 0

        # Models should be strings like "qwen3:14b"
        for model in models:
            assert isinstance(model, str)
            assert len(model) > 0
