# Ollama Integration

> **Status:** 🔴 Not Started
> **Priority:** P0 (Critical)
> **Dependencies:** 01-foundation (Ollama verified), 06-tool-system
> **Estimated Complexity:** Medium

## Overview

Integrate Ollama LLM (Qwen3-14B) for tool calling and conversation. This includes setting up the LangChain-Ollama connector, implementing the tool calling format, retry logic, and model fallback switching.

## Goals

- [ ] LangChain-Ollama ChatModel configuration
- [ ] Tool calling format for Qwen3
- [ ] Retry logic with exponential backoff
- [ ] Model fallback chain (Qwen3-14B → Qwen3-8B → Llama 4)
- [ ] Token usage tracking
- [ ] Temperature configuration by use case

## Technical Design

### LLM Configuration

```python
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class LLMUseCase(str, Enum):
    """Different use cases require different LLM settings."""
    TOOL_CALLING = "tool_calling"
    CONVERSATION = "conversation"
    PLANNING = "planning"
    JSON_OUTPUT = "json_output"


@dataclass
class LLMConfig:
    """Configuration for LLM instance."""
    model: str
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120
    base_url: str = "http://localhost:11434"

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    # Fallback models
    fallback_models: List[str] = None


# Default configurations by use case
LLM_CONFIGS = {
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
```

### LLM Provider

```python
import asyncio
import logging
from typing import Any, AsyncGenerator, Optional
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from agent.tools.registry import get_tool_registry


logger = logging.getLogger(__name__)


class OllamaProvider:
    """
    Manages Ollama LLM instances with retry and fallback logic.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._llm: Optional[ChatOllama] = None
        self._current_model: str = config.model
        self._fallback_index: int = 0
        self._token_usage: dict = {"prompt": 0, "completion": 0}
        self._last_error: Optional[Exception] = None

    @property
    def llm(self) -> ChatOllama:
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = self._create_llm(self._current_model)
        return self._llm

    def _create_llm(self, model: str) -> ChatOllama:
        """Create a new ChatOllama instance."""
        return ChatOllama(
            model=model,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
            timeout=self.config.timeout,
        )

    async def invoke(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[dict]] = None,
    ) -> AIMessage:
        """
        Invoke the LLM with retry and fallback logic.

        Args:
            messages: Conversation messages
            tools: Optional tool schemas for tool calling

        Returns:
            AIMessage with response or tool calls
        """
        last_error = None

        # Try current model with retries
        for attempt in range(self.config.max_retries):
            try:
                return await self._invoke_with_tools(messages, tools)

            except Exception as e:
                last_error = e
                logger.warning(f"LLM invoke failed (attempt {attempt + 1}): {e}")

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                    await asyncio.sleep(delay)

        # Try fallback models
        if self.config.fallback_models:
            for fallback_model in self.config.fallback_models:
                logger.info(f"Trying fallback model: {fallback_model}")

                try:
                    self._current_model = fallback_model
                    self._llm = self._create_llm(fallback_model)
                    return await self._invoke_with_tools(messages, tools)

                except Exception as e:
                    logger.warning(f"Fallback model {fallback_model} failed: {e}")
                    last_error = e
                    continue

        # All attempts failed
        self._last_error = last_error
        raise RuntimeError(f"All LLM attempts failed. Last error: {last_error}")

    async def _invoke_with_tools(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[dict]],
    ) -> AIMessage:
        """Invoke LLM with optional tool binding."""

        if tools:
            llm_with_tools = self.llm.bind_tools(tools)
            response = await llm_with_tools.ainvoke(messages)
        else:
            response = await self.llm.ainvoke(messages)

        # Track token usage (if available)
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
        """Stream response tokens."""

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    def get_token_usage(self) -> dict:
        """Get cumulative token usage."""
        return self._token_usage.copy()

    def reset_token_usage(self) -> None:
        """Reset token counters."""
        self._token_usage = {"prompt": 0, "completion": 0}

    async def check_health(self) -> bool:
        """Check if Ollama is available."""
        try:
            from langchain_core.messages import HumanMessage
            response = await self.llm.ainvoke([HumanMessage(content="ping")])
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False


# Global provider instances
_providers: dict[LLMUseCase, OllamaProvider] = {}


def get_llm(use_case: LLMUseCase = LLMUseCase.TOOL_CALLING) -> OllamaProvider:
    """Get LLM provider for a specific use case."""
    if use_case not in _providers:
        config = LLM_CONFIGS[use_case]
        _providers[use_case] = OllamaProvider(config)
    return _providers[use_case]


async def check_ollama_available() -> bool:
    """Check if Ollama is running and model is available."""
    provider = get_llm(LLMUseCase.CONVERSATION)
    return await provider.check_health()
```

### Tool Calling

```python
from typing import Optional, Any
import json

from langchain_core.messages import AIMessage, ToolMessage


def extract_tool_calls(response: AIMessage) -> list[dict]:
    """
    Extract tool calls from LLM response.

    Handles both OpenAI-style and Ollama-native formats.
    """
    tool_calls = []

    # Check for standard tool_calls attribute
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            tool_calls.append({
                "id": tc.get("id", f"call_{len(tool_calls)}"),
                "name": tc.get("name"),
                "arguments": tc.get("args", {}),
            })
        return tool_calls

    # Check for Ollama-style function calls in content
    content = response.content
    if isinstance(content, str):
        # Try to parse as JSON (some models return JSON directly)
        try:
            parsed = json.loads(content)
            if "tool_calls" in parsed:
                return parsed["tool_calls"]
            if "function" in parsed:
                return [{
                    "id": "call_0",
                    "name": parsed["function"]["name"],
                    "arguments": parsed["function"].get("arguments", {}),
                }]
        except json.JSONDecodeError:
            pass

    return tool_calls


async def execute_tool_call(
    tool_call: dict,
    registry: Optional[Any] = None,
) -> ToolMessage:
    """
    Execute a tool call and return the result as a message.

    Args:
        tool_call: {id, name, arguments} dict
        registry: Optional tool registry (uses global if not provided)

    Returns:
        ToolMessage with result
    """
    if registry is None:
        registry = get_tool_registry()

    tool_name = tool_call["name"]
    tool_id = tool_call["id"]
    arguments = tool_call.get("arguments", {})

    # Get tool
    tool = registry.get(tool_name)
    if not tool:
        return ToolMessage(
            content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
            tool_call_id=tool_id,
        )

    # Execute tool
    try:
        result = await tool.run(**arguments)
        return ToolMessage(
            content=json.dumps(result.to_dict()),
            tool_call_id=tool_id,
        )
    except Exception as e:
        return ToolMessage(
            content=json.dumps({"error": str(e)}),
            tool_call_id=tool_id,
        )


async def run_tool_loop(
    provider: OllamaProvider,
    messages: list,
    max_iterations: int = 10,
) -> list:
    """
    Run tool calling loop until completion or max iterations.

    Returns updated message list with all tool calls and results.
    """
    tools = get_tool_registry().get_schemas()

    for _ in range(max_iterations):
        response = await provider.invoke(messages, tools=tools)
        messages.append(response)

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            # No more tool calls, done
            break

        # Execute all tool calls
        for tc in tool_calls:
            tool_result = await execute_tool_call(tc)
            messages.append(tool_result)

    return messages
```

### Model Management

```python
import subprocess
import json
from typing import List, Optional


async def list_available_models() -> List[str]:
    """List models available in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        # Parse output (skip header line)
        lines = result.stdout.strip().split("\n")[1:]
        models = []
        for line in lines:
            parts = line.split()
            if parts:
                models.append(parts[0])  # Model name is first column

        return models

    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return []


async def ensure_model_available(model: str) -> bool:
    """Ensure a model is pulled and available."""
    available = await list_available_models()

    if model in available:
        return True

    # Try to pull the model
    logger.info(f"Pulling model: {model}")
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for large models
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to pull model {model}: {e}")
        return False


def get_model_info(model: str) -> Optional[dict]:
    """Get information about a model."""
    try:
        result = subprocess.run(
            ["ollama", "show", model, "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        return None
    except Exception:
        return None
```

## Implementation Tasks

- [ ] Create `backend/agent/llm/__init__.py`
- [ ] Create `backend/agent/llm/config.py`
  - [ ] Define `LLMConfig` dataclass
  - [ ] Define `LLMUseCase` enum
  - [ ] Set up default configurations
- [ ] Create `backend/agent/llm/provider.py`
  - [ ] Implement `OllamaProvider` class
  - [ ] Implement retry logic with backoff
  - [ ] Implement model fallback chain
  - [ ] Add token usage tracking
- [ ] Create `backend/agent/llm/tools.py`
  - [ ] Implement `extract_tool_calls()` function
  - [ ] Implement `execute_tool_call()` function
  - [ ] Implement `run_tool_loop()` function
- [ ] Create `backend/agent/llm/models.py`
  - [ ] Implement `list_available_models()` function
  - [ ] Implement `ensure_model_available()` function
  - [ ] Implement `get_model_info()` function
- [ ] Add health check endpoint integration

## Testing

### Unit Tests

```python
# tests/agent/llm/test_provider.py

def test_llm_config_defaults():
    """Config should have sensible defaults."""
    config = LLMConfig(model="test-model")
    assert config.temperature == 0.1
    assert config.max_retries == 3


def test_use_case_configs():
    """Each use case should have specific settings."""
    tool_config = LLM_CONFIGS[LLMUseCase.TOOL_CALLING]
    conv_config = LLM_CONFIGS[LLMUseCase.CONVERSATION]

    assert tool_config.temperature < conv_config.temperature


@pytest.mark.asyncio
async def test_provider_retry_logic():
    """Provider should retry on failure."""
    config = LLMConfig(model="test", max_retries=3, retry_delay=0.01)
    provider = OllamaProvider(config)

    call_count = 0
    original_invoke = provider._invoke_with_tools

    async def mock_invoke(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Failed")
        return AIMessage(content="Success")

    provider._invoke_with_tools = mock_invoke

    result = await provider.invoke([HumanMessage(content="test")])
    assert call_count == 3
    assert result.content == "Success"


@pytest.mark.asyncio
async def test_provider_fallback():
    """Provider should try fallback models."""
    config = LLMConfig(
        model="primary",
        max_retries=1,
        fallback_models=["fallback1", "fallback2"],
        retry_delay=0.01,
    )
    provider = OllamaProvider(config)

    models_tried = []

    async def mock_invoke(*args, **kwargs):
        models_tried.append(provider._current_model)
        if provider._current_model != "fallback2":
            raise RuntimeError("Not this model")
        return AIMessage(content="Success")

    provider._invoke_with_tools = mock_invoke

    result = await provider.invoke([HumanMessage(content="test")])
    assert "primary" in models_tried
    assert "fallback1" in models_tried
    assert "fallback2" in models_tried
    assert provider._current_model == "fallback2"
```

### Tool Calling Tests

```python
# tests/agent/llm/test_tools.py

def test_extract_tool_calls_standard():
    """Should extract standard tool_calls."""
    response = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "scan_directory", "args": {"path": "/data"}},
        ],
    )
    calls = extract_tool_calls(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "scan_directory"


def test_extract_tool_calls_json_content():
    """Should extract from JSON content."""
    response = AIMessage(
        content='{"function": {"name": "detect", "arguments": {"input_path": "/data"}}}',
    )
    calls = extract_tool_calls(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "detect"


@pytest.mark.asyncio
async def test_execute_tool_call_success():
    """Should execute tool and return message."""
    from agent.tools.base import BaseTool, success_result

    class MockTool(BaseTool):
        name = "mock"
        async def execute(self, **kwargs):
            return success_result({"result": "ok"})

    registry = ToolRegistry()
    registry.clear()
    registry.register(MockTool())

    result = await execute_tool_call(
        {"id": "call_1", "name": "mock", "arguments": {}},
        registry=registry,
    )

    assert result.tool_call_id == "call_1"
    content = json.loads(result.content)
    assert content["success"] == True


@pytest.mark.asyncio
async def test_execute_tool_call_unknown_tool():
    """Should return error for unknown tool."""
    registry = ToolRegistry()
    registry.clear()

    result = await execute_tool_call(
        {"id": "call_1", "name": "unknown", "arguments": {}},
        registry=registry,
    )

    content = json.loads(result.content)
    assert "error" in content
    assert "unknown" in content["error"].lower()
```

### Integration Tests

```python
# tests/agent/llm/test_integration.py

@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_connection():
    """Test actual Ollama connection."""
    if not await check_ollama_available():
        pytest.skip("Ollama not available")

    provider = get_llm(LLMUseCase.CONVERSATION)
    response = await provider.invoke([
        HumanMessage(content="Say 'hello' and nothing else.")
    ])

    assert "hello" in response.content.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_calling_with_ollama():
    """Test tool calling with real Ollama."""
    if not await check_ollama_available():
        pytest.skip("Ollama not available")

    # Register a simple tool
    registry = get_tool_registry()
    registry.clear()

    @register_tool
    class EchoTool(BaseTool):
        name = "echo"
        description = "Echo a message back"
        parameters = [ToolParameter("message", str, "Message to echo")]

        async def execute(self, message: str):
            return success_result({"echoed": message})

    provider = get_llm(LLMUseCase.TOOL_CALLING)
    tools = registry.get_schemas()

    messages = [HumanMessage(content="Use the echo tool to say 'test'")]
    response = await provider.invoke(messages, tools=tools)

    tool_calls = extract_tool_calls(response)
    assert len(tool_calls) > 0
    assert tool_calls[0]["name"] == "echo"
```

## Acceptance Criteria

- [ ] Ollama connection established via LangChain
- [ ] Tool calling works with Qwen3-14B
- [ ] Retry logic handles transient failures
- [ ] Fallback to Qwen3-8B works when primary fails
- [ ] Token usage is tracked
- [ ] Different temperatures used for different use cases
- [ ] Health check endpoint works
- [ ] Model list retrieved from Ollama

## Files to Create/Modify

```
backend/
├── agent/
│   └── llm/
│       ├── __init__.py      # Exports
│       ├── config.py        # LLMConfig, use cases
│       ├── provider.py      # OllamaProvider
│       ├── tools.py         # Tool calling helpers
│       └── models.py        # Model management
└── tests/
    └── agent/
        └── llm/
            ├── test_provider.py
            ├── test_tools.py
            └── test_integration.py
```

## Dependencies

```
# requirements.txt additions
langchain-ollama>=0.2.0
langchain-core>=0.3.0
```

## Notes

- Qwen3 is currently the best open-source model for tool calling
- Temperature 0.0-0.1 for structured outputs, 0.7 for conversation
- Consider caching model info to avoid repeated subprocess calls
- Monitor token usage for cost/performance optimization
- Add metrics/telemetry for model performance tracking
