"""
LLM configuration for different use cases.

This module provides configuration classes and presets for different
LLM use cases (tool calling, conversation, planning, JSON output).
Each use case has optimized settings for temperature, token limits,
and fallback models.

Implements spec: 08-ollama-integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LLMUseCase(str, Enum):
    """Different use cases require different LLM settings."""

    TOOL_CALLING = "tool_calling"
    CONVERSATION = "conversation"
    PLANNING = "planning"
    JSON_OUTPUT = "json_output"


@dataclass
class LLMConfig:
    """
    Configuration for LLM instance.

    Includes model settings, retry configuration, and fallback models
    for robust operation when the primary model fails.

    Attributes:
        model: Primary model name (e.g., "qwen3:14b").
        temperature: Sampling temperature (0.0-1.0).
        max_tokens: Maximum tokens for generation.
        timeout: Request timeout in seconds.
        base_url: Ollama server URL.
        num_ctx: Context window size.
        max_retries: Maximum retry attempts on failure.
        retry_delay: Initial delay between retries (seconds).
        retry_backoff: Multiplier for exponential backoff.
        fallback_models: List of fallback models to try if primary fails.
    """

    model: str
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120
    base_url: str = "http://localhost:11434"
    num_ctx: int = 8192

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    # Fallback models
    fallback_models: list[str] = field(default_factory=list)

    def with_temperature(self, temperature: float) -> LLMConfig:
        """Return a copy with different temperature."""
        return LLMConfig(
            model=self.model,
            temperature=temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            base_url=self.base_url,
            num_ctx=self.num_ctx,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            retry_backoff=self.retry_backoff,
            fallback_models=self.fallback_models.copy(),
        )

    def with_model(self, model: str) -> LLMConfig:
        """Return a copy with different model."""
        return LLMConfig(
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            base_url=self.base_url,
            num_ctx=self.num_ctx,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            retry_backoff=self.retry_backoff,
            fallback_models=self.fallback_models.copy(),
        )


# Default configurations by use case
LLM_CONFIGS: dict[LLMUseCase, LLMConfig] = {
    LLMUseCase.TOOL_CALLING: LLMConfig(
        model="qwen3:14b",
        temperature=0.1,  # Low for deterministic tool selection
        max_tokens=2048,
        fallback_models=["qwen3:8b", "llama4:8b"],
    ),
    LLMUseCase.CONVERSATION: LLMConfig(
        model="qwen3:14b",
        temperature=0.7,  # Higher for natural conversation
        max_tokens=4096,
        fallback_models=["qwen3:8b", "llama4:8b"],
    ),
    LLMUseCase.PLANNING: LLMConfig(
        model="qwen3:14b",
        temperature=0.3,  # Moderate for creative but structured plans
        max_tokens=4096,
        fallback_models=["qwen3:8b"],
    ),
    LLMUseCase.JSON_OUTPUT: LLMConfig(
        model="qwen3:14b",
        temperature=0.0,  # Zero for deterministic JSON
        max_tokens=2048,
        fallback_models=["qwen3:8b"],
    ),
}


def get_config_for_use_case(use_case: LLMUseCase) -> LLMConfig:
    """
    Get the default configuration for a specific use case.

    Args:
        use_case: The LLM use case.

    Returns:
        LLMConfig with appropriate settings for the use case.
    """
    return LLM_CONFIGS[use_case]


def get_default_config() -> LLMConfig:
    """
    Get the default configuration (tool calling).

    Returns:
        LLMConfig for tool calling use case.
    """
    return LLM_CONFIGS[LLMUseCase.TOOL_CALLING]


__all__ = [
    "LLMConfig",
    "LLMUseCase",
    "LLM_CONFIGS",
    "get_config_for_use_case",
    "get_default_config",
]
