"""
LLM provider using LangChain-Ollama for local inference.

This module provides a consistent interface for obtaining configured
LLM instances that connect to a local Ollama server.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


@lru_cache(maxsize=4)
def get_llm_with_config(
    host: str,
    model: str,
    temperature: float,
    timeout: int,
    num_ctx: int,
) -> BaseChatModel:
    """
    Get a cached LLM instance with specific configuration.

    Uses lru_cache to avoid recreating clients for the same config.

    Args:
        host: Ollama server host URL.
        model: Model name (e.g., "qwen3:14b").
        temperature: Sampling temperature (0.0-1.0).
        timeout: Request timeout in seconds.
        num_ctx: Context window size.

    Returns:
        Configured ChatOllama instance.
    """
    return ChatOllama(
        base_url=host,
        model=model,
        temperature=temperature,
        timeout=timeout,
        num_ctx=num_ctx,
    )


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
    get_llm_with_config.cache_clear()


__all__ = [
    "LLMConfig",
    "get_llm",
    "get_llm_with_config",
    "clear_llm_cache",
]
