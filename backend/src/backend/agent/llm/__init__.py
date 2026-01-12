"""
LLM integration module for Cloumask agent.

Provides a unified interface for LLM interactions using LangChain-Ollama,
including retry logic, model fallback, tool calling, and model management.

Implements spec: 08-ollama-integration
"""

from backend.agent.llm.config import (
    LLM_CONFIGS,
    LLMConfig,
    LLMUseCase,
    get_config_for_use_case,
    get_default_config,
)
from backend.agent.llm.models import (
    check_model_supports_tools,
    ensure_model_available,
    get_model_details,
    get_model_info,
    get_running_models,
    list_available_models,
)
from backend.agent.llm.provider import (
    OllamaProvider,
    SimpleLLMConfig,
    check_ollama_available,
    clear_llm_cache,
    clear_providers,
    get_llm,
    get_llm_with_config,
    get_provider,
    get_provider_with_config,
)
from backend.agent.llm.tools import (
    execute_tool_call,
    extract_tool_calls,
    format_tool_result_for_display,
    run_tool_loop,
)

__all__ = [
    # Config
    "LLMConfig",
    "LLMUseCase",
    "LLM_CONFIGS",
    "get_config_for_use_case",
    "get_default_config",
    # Provider (new API)
    "OllamaProvider",
    "get_provider",
    "get_provider_with_config",
    "clear_providers",
    "check_ollama_available",
    # Provider (backward-compatible)
    "SimpleLLMConfig",
    "get_llm",
    "get_llm_with_config",
    "clear_llm_cache",
    # Tool calling
    "extract_tool_calls",
    "execute_tool_call",
    "run_tool_loop",
    "format_tool_result_for_display",
    # Model management
    "list_available_models",
    "ensure_model_available",
    "get_model_info",
    "get_model_details",
    "check_model_supports_tools",
    "get_running_models",
]
