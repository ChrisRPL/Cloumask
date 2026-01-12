"""
LLM provider using LangChain-Ollama for local inference.

This module provides a consistent interface for obtaining configured
LLM instances that connect to a local Ollama server.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_ollama import ChatOllama

from backend.api.config import settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM provider."""

    host: str = "http://localhost:11434"
    model: str = "qwen3:14b"
    temperature: float = 0.1
    timeout: int = 120
    num_ctx: int = 8192

    @classmethod
    def from_settings(cls) -> LLMConfig:
        """Create config from application settings."""
        return cls(
            host=settings.ollama_host,
            model=settings.ollama_model,
        )


class _LLMCache:
    """Simple LRU cache for LLM instances with configurable size."""

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[str, str, float, int, int], BaseChatModel] = (
            OrderedDict()
        )

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
    config: LLMConfig | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """
    Get a configured LLM instance.

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
        config = LLMConfig.from_settings()

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
    "LLMConfig",
    "get_llm",
    "get_llm_with_config",
    "clear_llm_cache",
]
