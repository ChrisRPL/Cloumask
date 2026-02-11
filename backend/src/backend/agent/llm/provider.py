"""
LLM provider using LangChain-Ollama for local inference.

This module provides the OllamaProvider class with retry logic,
model fallback, and token usage tracking. Also includes backward-compatible
functions for simple LLM access.

Implements spec: 08-ollama-integration
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama

from backend.agent.llm.config import LLMConfig, LLMUseCase, get_config_for_use_case
from backend.api.config import settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class OllamaProvider:
    """
    Manages Ollama LLM instances with retry and fallback logic.

    Provides a robust interface to Ollama that handles:
    - Automatic retries with exponential backoff
    - Fallback to alternative models when primary fails
    - Token usage tracking
    - Health checks

    Example:
        provider = OllamaProvider(config)
        response = await provider.invoke([HumanMessage("Hello")])
        print(provider.get_token_usage())
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize the provider with configuration.

        Args:
            config: LLM configuration with model, retry, and fallback settings.
        """
        self.config = config
        self._llm: ChatOllama | None = None
        self._current_model: str = config.model
        self._token_usage: dict[str, int] = {"prompt": 0, "completion": 0}
        self._last_error: Exception | None = None

    @property
    def llm(self) -> ChatOllama:
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = self._create_llm(self._current_model)
        return self._llm

    @property
    def current_model(self) -> str:
        """Get the currently active model name."""
        return self._current_model

    def _create_llm(self, model: str) -> ChatOllama:
        """Create a new ChatOllama instance."""
        return ChatOllama(
            model=model,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
            timeout=self.config.timeout,
            num_ctx=self.config.num_ctx,
        )

    async def invoke(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AIMessage:
        """
        Invoke the LLM with retry and fallback logic.

        Attempts to call the LLM with configured retries. If all retries
        fail, tries each fallback model in order.

        Args:
            messages: Conversation messages.
            tools: Optional tool schemas for tool calling.

        Returns:
            AIMessage with response or tool calls.

        Raises:
            RuntimeError: If all LLM attempts (primary + fallbacks) fail.
        """
        last_error: Exception | None = None

        # Try current model with retries
        for attempt in range(self.config.max_retries):
            try:
                return await self._invoke_with_tools(messages, tools)
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM invoke failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                )

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (
                        self.config.retry_backoff**attempt
                    )
                    logger.debug("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)

        # Try fallback models
        if self.config.fallback_models:
            for fallback_model in self.config.fallback_models:
                logger.info("Trying fallback model: %s", fallback_model)

                try:
                    self._current_model = fallback_model
                    self._llm = self._create_llm(fallback_model)
                    return await self._invoke_with_tools(messages, tools)
                except Exception as e:
                    logger.warning("Fallback model %s failed: %s", fallback_model, e)
                    last_error = e
                    continue

        # All attempts failed
        self._last_error = last_error
        raise RuntimeError(f"All LLM attempts failed. Last error: {last_error}")

    async def _invoke_with_tools(
        self,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]] | None,
    ) -> AIMessage:
        """Invoke LLM with optional tool binding."""
        if tools:
            llm_with_tools = self.llm.bind_tools(tools)
            response = await llm_with_tools.ainvoke(messages)
        else:
            response = await self.llm.ainvoke(messages)

        # Track token usage (if available in response metadata)
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata
            if "prompt_eval_count" in meta:
                self._token_usage["prompt"] += meta["prompt_eval_count"]
            if "eval_count" in meta:
                self._token_usage["completion"] += meta["eval_count"]

        return response

    async def stream(
        self,
        messages: list[BaseMessage],
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens.

        Yields tokens as they are generated. Does not support tool calling.

        Args:
            messages: Conversation messages.

        Yields:
            String chunks of the response.
        """
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    def get_token_usage(self) -> dict[str, int]:
        """
        Get cumulative token usage.

        Returns:
            Dict with "prompt" and "completion" token counts.
        """
        return self._token_usage.copy()

    def reset_token_usage(self) -> None:
        """Reset token counters to zero."""
        self._token_usage = {"prompt": 0, "completion": 0}

    def reset_to_primary(self) -> None:
        """Reset to the primary model (after fallback)."""
        self._current_model = self.config.model
        self._llm = None
        self._last_error = None

    async def check_health(self) -> bool:
        """
        Check if Ollama is available and responding.

        Sends a minimal request to verify connectivity.

        Returns:
            True if Ollama responds successfully, False otherwise.
        """
        try:
            await self.llm.ainvoke([HumanMessage(content="ping")])
            return True
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# Global provider instances by use case
# ---------------------------------------------------------------------------

_providers: dict[LLMUseCase, OllamaProvider] = {}


def get_provider(use_case: LLMUseCase = LLMUseCase.TOOL_CALLING) -> OllamaProvider:
    """
    Get LLM provider for a specific use case.

    Providers are cached by use case and reused across calls.

    Args:
        use_case: The use case for this LLM instance.

    Returns:
        Configured OllamaProvider instance.
    """
    if use_case not in _providers:
        config = get_config_for_use_case(use_case)
        _providers[use_case] = OllamaProvider(config)
    return _providers[use_case]


def get_provider_with_config(config: LLMConfig) -> OllamaProvider:
    """
    Create a provider with custom configuration.

    Creates a new provider instance (not cached).

    Args:
        config: Custom LLM configuration.

    Returns:
        New OllamaProvider instance.
    """
    return OllamaProvider(config)


def clear_providers() -> None:
    """Clear all cached providers (for testing)."""
    _providers.clear()


async def check_ollama_available() -> bool:
    """
    Check if Ollama is running and model is available.

    Uses the conversation provider to perform a health check.

    Returns:
        True if Ollama is available, False otherwise.
    """
    provider = get_provider(LLMUseCase.CONVERSATION)
    return await provider.check_health()


# ---------------------------------------------------------------------------
# Backward-compatible simple LLM access
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimpleLLMConfig:
    """Simple configuration for backward-compatible LLM access."""

    host: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    temperature: float = 0.1
    timeout: int = 120
    num_ctx: int = 8192

    @classmethod
    def from_settings(cls) -> SimpleLLMConfig:
        """Create config from application settings."""
        return cls(
            host=settings.ollama_host,
            model=settings.ollama_model,
        )


class _LLMCache:
    """Simple LRU cache for LLM instances with configurable size."""

    def __init__(self) -> None:
        self._cache: OrderedDict[
            tuple[str, str, float, int, int], BaseChatModel
        ] = OrderedDict()

    @property
    def maxsize(self) -> int:
        """Get max cache size from settings."""
        return settings.llm_cache_size

    def get(
        self,
        host: str,
        model: str,
        temperature: float,
        timeout: int,
        num_ctx: int,
    ) -> BaseChatModel:
        """Get or create a cached LLM instance."""
        key = (host, model, temperature, timeout, num_ctx)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        # Create new instance
        llm = ChatOllama(
            base_url=host,
            model=model,
            temperature=temperature,
            timeout=timeout,
            num_ctx=num_ctx,
        )

        # Add to cache
        self._cache[key] = llm

        # Evict oldest if over capacity
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

        return llm

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


# Global cache instance
_llm_cache = _LLMCache()


def get_llm_with_config(
    host: str,
    model: str,
    temperature: float,
    timeout: int,
    num_ctx: int,
) -> BaseChatModel:
    """
    Get a cached LLM instance with specific configuration.

    Uses LRU cache to avoid recreating clients for the same config.
    Cache size is configurable via CLOUMASK_LLM_CACHE_SIZE env var.

    Args:
        host: Ollama server host URL.
        model: Model name (e.g., "qwen3:14b").
        temperature: Sampling temperature (0.0-1.0).
        timeout: Request timeout in seconds.
        num_ctx: Context window size.

    Returns:
        Configured ChatOllama instance.
    """
    return _llm_cache.get(host, model, temperature, timeout, num_ctx)


def get_llm(
    config: SimpleLLMConfig | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """
    Get a configured LLM instance.

    This is the backward-compatible function for simple LLM access.
    For advanced features (retry, fallback), use get_provider() instead.

    Args:
        config: Optional LLM configuration. If None, uses settings.
        temperature: Optional temperature override.
            Lower values (0.1) for structured output.
            Higher values (0.7) for creative responses.

    Returns:
        Configured ChatOllama instance ready for use.

    Example:
        >>> llm = get_llm()
        >>> response = await llm.ainvoke([HumanMessage("Hello")])

        >>> llm = get_llm(temperature=0.7)  # More creative
        >>> response = await llm.ainvoke(messages)
    """
    if config is None:
        config = SimpleLLMConfig.from_settings()

    final_temperature = temperature if temperature is not None else config.temperature

    return get_llm_with_config(
        host=config.host,
        model=config.model,
        temperature=final_temperature,
        timeout=config.timeout,
        num_ctx=config.num_ctx,
    )


def clear_llm_cache() -> None:
    """Clear the LLM instance cache. Useful for testing."""
    _llm_cache.clear()


__all__ = [
    # New OllamaProvider API
    "OllamaProvider",
    "get_provider",
    "get_provider_with_config",
    "clear_providers",
    "check_ollama_available",
    # Backward-compatible API
    "SimpleLLMConfig",
    "get_llm",
    "get_llm_with_config",
    "clear_llm_cache",
]
