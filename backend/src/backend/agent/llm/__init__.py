"""
LLM integration module for Cloumask agent.

Provides a unified interface for LLM interactions using LangChain-Ollama.
"""

from backend.agent.llm.provider import (
    LLMConfig,
    get_llm,
    get_llm_with_config,
)

__all__ = [
    "LLMConfig",
    "get_llm",
    "get_llm_with_config",
]
