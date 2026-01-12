"""
Tool calling helpers for LLM integration.

This module provides functions for extracting tool calls from LLM responses,
executing them, and running the tool calling loop.

Implements spec: 08-ollama-integration
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, ToolMessage

from backend.agent.tools.registry import get_tool_registry

if TYPE_CHECKING:
    from backend.agent.llm.provider import OllamaProvider
    from backend.agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def extract_tool_calls(response: AIMessage) -> list[dict[str, Any]]:
    """
    Extract tool calls from LLM response.

    Handles both OpenAI-style tool_calls attribute and Ollama-native
    formats (JSON in content).

    Args:
        response: AIMessage from the LLM.

    Returns:
        List of tool call dicts with keys: id, name, arguments.
        Empty list if no tool calls found.

    Example:
        >>> response = AIMessage(content="", tool_calls=[...])
        >>> calls = extract_tool_calls(response)
        >>> print(calls[0]["name"])  # "scan_directory"
    """
    tool_calls: list[dict[str, Any]] = []

    # Check for standard tool_calls attribute (LangChain format)
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            tool_calls.append(
                {
                    "id": tc.get("id", f"call_{len(tool_calls)}"),
                    "name": tc.get("name"),
                    "arguments": tc.get("args", {}),
                }
            )
        return tool_calls

    # Check for Ollama-style function calls in content
    content = response.content
    if isinstance(content, str) and content.strip():
        # Try to parse as JSON (some models return JSON directly)
        try:
            parsed = json.loads(content)

            # Handle {"tool_calls": [...]} format
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                for tc in parsed["tool_calls"]:
                    tool_calls.append(
                        {
                            "id": tc.get("id", f"call_{len(tool_calls)}"),
                            "name": tc.get("name"),
                            "arguments": tc.get("arguments", tc.get("args", {})),
                        }
                    )
                return tool_calls

            # Handle {"function": {"name": ..., "arguments": ...}} format
            if isinstance(parsed, dict) and "function" in parsed:
                func = parsed["function"]
                tool_calls.append(
                    {
                        "id": "call_0",
                        "name": func.get("name"),
                        "arguments": func.get("arguments", func.get("args", {})),
                    }
                )
                return tool_calls

            # Handle {"name": ..., "arguments": ...} format (direct call)
            if isinstance(parsed, dict) and "name" in parsed:
                tool_calls.append(
                    {
                        "id": "call_0",
                        "name": parsed.get("name"),
                        "arguments": parsed.get("arguments", parsed.get("args", {})),
                    }
                )
                return tool_calls

        except json.JSONDecodeError:
            # Not JSON, no tool calls in content
            pass

    return tool_calls


async def execute_tool_call(
    tool_call: dict[str, Any],
    registry: ToolRegistry | None = None,
) -> ToolMessage:
    """
    Execute a tool call and return the result as a message.

    Looks up the tool by name in the registry, validates parameters,
    executes it, and wraps the result in a ToolMessage.

    Args:
        tool_call: Dict with keys: id, name, arguments.
        registry: Optional tool registry (uses global if not provided).

    Returns:
        ToolMessage with JSON-encoded result.

    Example:
        >>> result = await execute_tool_call({
        ...     "id": "call_1",
        ...     "name": "scan_directory",
        ...     "arguments": {"path": "/data"}
        ... })
        >>> print(result.tool_call_id)  # "call_1"
    """
    if registry is None:
        registry = get_tool_registry()

    tool_name = tool_call.get("name", "")
    tool_id = tool_call.get("id", "call_0")
    arguments = tool_call.get("arguments", {})

    # Handle string arguments (some models return JSON string)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return ToolMessage(
                content=json.dumps({"error": f"Invalid arguments JSON: {arguments}"}),
                tool_call_id=tool_id,
            )

    # Get tool from registry
    tool = registry.get(tool_name)
    if not tool:
        logger.warning("Unknown tool requested: %s", tool_name)
        return ToolMessage(
            content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
            tool_call_id=tool_id,
        )

    # Execute tool
    try:
        logger.info("Executing tool: %s with args: %s", tool_name, arguments)
        result = await tool.run(**arguments)

        return ToolMessage(
            content=json.dumps(result.to_dict()),
            tool_call_id=tool_id,
        )
    except Exception as e:
        logger.exception("Tool %s execution failed: %s", tool_name, e)
        return ToolMessage(
            content=json.dumps({"error": str(e), "tool": tool_name}),
            tool_call_id=tool_id,
        )


async def run_tool_loop(
    provider: OllamaProvider,
    messages: list[Any],
    max_iterations: int = 10,
    registry: ToolRegistry | None = None,
) -> list[Any]:
    """
    Run tool calling loop until completion or max iterations.

    Repeatedly invokes the LLM, extracts tool calls, executes them,
    and adds results to the message list. Stops when the LLM responds
    without any tool calls or max iterations is reached.

    Args:
        provider: OllamaProvider instance for LLM calls.
        messages: Initial message list (will be modified in place).
        max_iterations: Maximum number of LLM invocations.
        registry: Optional tool registry (uses global if not provided).

    Returns:
        Updated message list with all tool calls and results.

    Example:
        >>> provider = get_provider(LLMUseCase.TOOL_CALLING)
        >>> messages = [HumanMessage(content="Scan the /data folder")]
        >>> result = await run_tool_loop(provider, messages)
        >>> print(result[-1].content)  # Final response
    """
    if registry is None:
        registry = get_tool_registry()

    tools = registry.get_schemas()

    for iteration in range(max_iterations):
        logger.debug("Tool loop iteration %d/%d", iteration + 1, max_iterations)

        # Invoke LLM with tools
        response = await provider.invoke(messages, tools=tools)
        messages.append(response)

        # Extract tool calls
        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            # No more tool calls, LLM is done
            logger.debug("No tool calls, ending loop")
            break

        logger.info("Executing %d tool calls", len(tool_calls))

        # Execute all tool calls
        for tc in tool_calls:
            tool_result = await execute_tool_call(tc, registry=registry)
            messages.append(tool_result)

    return messages


def format_tool_result_for_display(tool_message: ToolMessage) -> str:
    """
    Format a tool message for human-readable display.

    Parses the JSON content and formats it nicely for chat display.

    Args:
        tool_message: ToolMessage with JSON content.

    Returns:
        Human-readable string representation.
    """
    try:
        content = json.loads(tool_message.content)
    except json.JSONDecodeError:
        return tool_message.content

    if content.get("error"):
        return f"Error: {content['error']}"

    if content.get("success"):
        data = content.get("data", {})
        # Format key details
        parts = []
        for key, value in data.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (list, dict)):
                parts.append(f"{key}: {len(value)} items")
            else:
                parts.append(f"{key}: {value}")
        return ", ".join(parts) if parts else "Success"

    return str(content)


__all__ = [
    "extract_tool_calls",
    "execute_tool_call",
    "run_tool_loop",
    "format_tool_result_for_display",
]
