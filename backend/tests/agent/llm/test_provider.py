"""Tests for LLM provider module."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.llm.config import LLMConfig, LLMUseCase
from backend.agent.llm.provider import (
    OllamaProvider,
    clear_providers,
    get_provider,
    get_provider_with_config,
)


class TestOllamaProvider:
    """Tests for OllamaProvider class."""

    def test_init(self, test_config):
        """Provider should initialize with config."""
        provider = OllamaProvider(test_config)

        assert provider.config == test_config
        assert provider._current_model == test_config.model
        assert provider._llm is None
        assert provider._token_usage == {"prompt": 0, "completion": 0}

    def test_current_model_property(self, test_config):
        """current_model should return the active model."""
        provider = OllamaProvider(test_config)
        assert provider.current_model == test_config.model

    @pytest.mark.asyncio
    async def test_invoke_success(self, provider_with_mock_llm, sample_messages):
        """invoke should return response on success."""
        result = await provider_with_mock_llm.invoke(sample_messages)

        assert isinstance(result, AIMessage)
        assert result.content == "Test response"
        provider_with_mock_llm._llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_with_tools(
        self, provider_with_mock_llm, sample_messages, sample_tool_schemas
    ):
        """invoke with tools should bind tools and invoke."""
        result = await provider_with_mock_llm.invoke(
            sample_messages, tools=sample_tool_schemas
        )

        assert isinstance(result, AIMessage)
        provider_with_mock_llm._llm.bind_tools.assert_called_once_with(
            sample_tool_schemas
        )

    @pytest.mark.asyncio
    async def test_invoke_retry_on_failure(self, test_config, sample_messages):
        """Provider should retry on failure."""
        provider = OllamaProvider(test_config)

        call_count = 0

        async def mock_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Failed")
            return AIMessage(content="Success")

        mock_llm = MagicMock()
        mock_llm.ainvoke = mock_invoke
        provider._llm = mock_llm

        result = await provider.invoke(sample_messages)

        assert call_count == 2
        assert result.content == "Success"

    @pytest.mark.asyncio
    async def test_invoke_fallback_on_all_retries_fail(
        self, test_config, sample_messages
    ):
        """Provider should try fallback models when retries exhausted."""
        provider = OllamaProvider(test_config)

        models_tried = []

        def create_failing_llm(model):
            async def mock_invoke(*args, **kwargs):
                models_tried.append(model)
                if model != "fallback-2":
                    raise RuntimeError(f"Model {model} failed")
                return AIMessage(content="Fallback success")

            mock = MagicMock()
            mock.ainvoke = mock_invoke
            return mock

        provider._create_llm = create_failing_llm
        provider._llm = create_failing_llm(test_config.model)

        result = await provider.invoke(sample_messages)

        assert "test-model" in models_tried
        assert "fallback-1" in models_tried
        assert "fallback-2" in models_tried
        assert provider._current_model == "fallback-2"
        assert result.content == "Fallback success"

    @pytest.mark.asyncio
    async def test_invoke_all_fail_raises_error(self, sample_messages):
        """Provider should raise when all attempts fail."""
        config = LLMConfig(
            model="failing-model",
            max_retries=1,
            retry_delay=0.01,
            fallback_models=[],
        )
        provider = OllamaProvider(config)

        async def always_fail(*args, **kwargs):
            raise RuntimeError("Always fails")

        mock_llm = MagicMock()
        mock_llm.ainvoke = always_fail
        provider._llm = mock_llm

        with pytest.raises(RuntimeError, match="All LLM attempts failed"):
            await provider.invoke(sample_messages)

    def test_token_usage_tracking(self, test_config):
        """Token usage should be tracked."""
        provider = OllamaProvider(test_config)

        assert provider.get_token_usage() == {"prompt": 0, "completion": 0}

        # Simulate token usage update
        provider._token_usage["prompt"] = 100
        provider._token_usage["completion"] = 50

        usage = provider.get_token_usage()
        assert usage == {"prompt": 100, "completion": 50}

    def test_reset_token_usage(self, test_config):
        """reset_token_usage should clear counters."""
        provider = OllamaProvider(test_config)
        provider._token_usage = {"prompt": 100, "completion": 50}

        provider.reset_token_usage()

        assert provider.get_token_usage() == {"prompt": 0, "completion": 0}

    def test_reset_to_primary(self, test_config):
        """reset_to_primary should restore primary model."""
        provider = OllamaProvider(test_config)
        provider._current_model = "fallback-1"
        provider._llm = MagicMock()
        provider._last_error = RuntimeError("test")

        provider.reset_to_primary()

        assert provider._current_model == test_config.model
        assert provider._llm is None
        assert provider._last_error is None

    @pytest.mark.asyncio
    async def test_check_health_success(self, provider_with_mock_llm):
        """check_health should return True when Ollama responds."""
        result = await provider_with_mock_llm.check_health()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self, test_config):
        """check_health should return False on error."""
        provider = OllamaProvider(test_config)

        async def fail(*args, **kwargs):
            raise ConnectionError("Not available")

        mock_llm = MagicMock()
        mock_llm.ainvoke = fail
        provider._llm = mock_llm

        result = await provider.check_health()
        assert result is False


class TestGetProvider:
    """Tests for get_provider function."""

    def test_returns_provider_for_use_case(self):
        """Should return a provider configured for the use case."""
        provider = get_provider(LLMUseCase.TOOL_CALLING)

        assert isinstance(provider, OllamaProvider)
        assert provider.config.temperature == 0.1

    def test_caches_providers(self):
        """Should return the same provider for repeated calls."""
        provider1 = get_provider(LLMUseCase.CONVERSATION)
        provider2 = get_provider(LLMUseCase.CONVERSATION)

        assert provider1 is provider2

    def test_different_providers_for_different_use_cases(self):
        """Different use cases should get different providers."""
        tool_provider = get_provider(LLMUseCase.TOOL_CALLING)
        conv_provider = get_provider(LLMUseCase.CONVERSATION)

        assert tool_provider is not conv_provider
        assert tool_provider.config.temperature != conv_provider.config.temperature

    def test_clear_providers(self):
        """clear_providers should remove cached providers."""
        provider1 = get_provider(LLMUseCase.TOOL_CALLING)
        clear_providers()
        provider2 = get_provider(LLMUseCase.TOOL_CALLING)

        assert provider1 is not provider2


class TestGetProviderWithConfig:
    """Tests for get_provider_with_config function."""

    def test_creates_new_provider(self, test_config):
        """Should create a new provider with custom config."""
        provider = get_provider_with_config(test_config)

        assert isinstance(provider, OllamaProvider)
        assert provider.config == test_config

    def test_does_not_cache(self, test_config):
        """Should create new provider each time (not cached)."""
        provider1 = get_provider_with_config(test_config)
        provider2 = get_provider_with_config(test_config)

        assert provider1 is not provider2


class TestTokenUsageFromResponse:
    """Tests for token usage extraction from response."""

    @pytest.mark.asyncio
    async def test_extracts_token_usage(self, test_config):
        """Should extract token usage from response metadata."""
        provider = OllamaProvider(test_config)

        response = AIMessage(content="Test")
        response.response_metadata = {
            "prompt_eval_count": 100,
            "eval_count": 50,
        }

        async def mock_invoke(*args, **kwargs):
            return response

        mock_llm = MagicMock()
        mock_llm.ainvoke = mock_invoke
        provider._llm = mock_llm

        await provider.invoke([HumanMessage(content="test")])

        usage = provider.get_token_usage()
        assert usage["prompt"] == 100
        assert usage["completion"] == 50

    @pytest.mark.asyncio
    async def test_accumulates_token_usage(self, test_config):
        """Token usage should accumulate across calls."""
        provider = OllamaProvider(test_config)

        call_count = 0

        async def mock_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = AIMessage(content="Test")
            response.response_metadata = {
                "prompt_eval_count": 100,
                "eval_count": 50,
            }
            return response

        mock_llm = MagicMock()
        mock_llm.ainvoke = mock_invoke
        provider._llm = mock_llm

        await provider.invoke([HumanMessage(content="test1")])
        await provider.invoke([HumanMessage(content="test2")])

        usage = provider.get_token_usage()
        assert usage["prompt"] == 200
        assert usage["completion"] == 100
