"""
Understand node for parsing user requests and extracting intent.

This node uses the LLM to analyze natural language requests and
extract structured information about what the user wants to do.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.llm import get_llm
from backend.agent.prompts import load_prompt
from backend.agent.state import MessageRole, PipelineState

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls
MAX_RETRIES = 3


def _extract_json_from_response(content: str) -> dict[str, Any] | None:
    """
    Extract JSON from LLM response, handling markdown code blocks.

    Args:
        content: Raw LLM response content.

    Returns:
        Parsed JSON dict, or None if parsing fails.
    """

    def _parse_and_validate(text: str) -> dict[str, Any] | None:
        """Parse JSON and ensure it's a dict."""
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        return None

    # Try direct parsing first
    if result := _parse_and_validate(content):
        return result

    # Try extracting from markdown code block
    if "```" in content:
        # Find JSON between code blocks
        parts = content.split("```")
        for part in parts:
            # Remove 'json' language identifier if present
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

            if result := _parse_and_validate(cleaned):
                return result

    # Try finding JSON-like content with braces
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        if result := _parse_and_validate(content[start_idx : end_idx + 1]):
            return result

    return None


async def understand(state: PipelineState) -> dict[str, Any]:
    """
    Analyze user request and extract structured intent.

    This node:
    1. Gets the latest user message
    2. Sends it to the LLM with the understand prompt
    3. Parses the JSON response to extract intent
    4. Stores understanding in metadata or asks for clarification

    Args:
        state: Current pipeline state.

    Returns:
        State update dict with understanding in metadata or clarification message.
    """
    # Get the latest user message
    messages = state.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == MessageRole.USER.value]

    if not user_messages:
        logger.warning("No user message found in state")
        return {"last_error": "No user message found"}

    latest_user_msg = user_messages[-1]["content"]
    logger.info(f"Understanding request: {latest_user_msg[:100]}...")

    # Load the understand prompt
    try:
        understand_prompt = load_prompt("understand")
    except FileNotFoundError:
        logger.error("Failed to load understand prompt")
        return {"last_error": "Failed to load understand prompt"}

    # Build LLM messages
    llm_messages = [
        SystemMessage(content=understand_prompt),
        HumanMessage(content=latest_user_msg),
    ]

    # Get LLM and call with retries
    llm = get_llm(temperature=0.1)  # Low temperature for structured output

    last_error: str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await llm.ainvoke(llm_messages)
            response_content = response.content

            # Handle string or list content
            if isinstance(response_content, list):
                response_content = " ".join(
                    str(item) for item in response_content if item
                )

            logger.debug(f"LLM response (attempt {attempt + 1}): {response_content}")

            # Parse JSON from response
            understanding = _extract_json_from_response(str(response_content))

            if understanding is None:
                last_error = f"Failed to parse JSON from response: {response_content[:200]}"
                logger.warning(last_error)
                continue

            # Check if clarification is needed
            clarification = understanding.get("clarification_needed")
            if clarification:
                logger.info(f"Clarification needed: {clarification}")
                new_message = {
                    "role": MessageRole.ASSISTANT.value,
                    "content": clarification,
                    "timestamp": datetime.now().isoformat(),
                    "tool_calls": [],
                    "tool_call_id": None,
                }
                return {
                    "messages": [*messages, new_message],
                    "awaiting_user": True,
                }

            # Store understanding in metadata
            metadata = state.get("metadata", {})
            metadata["understanding"] = understanding
            logger.info(f"Understood intent: {understanding.get('intent')}")

            # Build confirmation message
            operations = understanding.get("operations", [])
            if not operations and understanding.get("intent"):
                operations = [understanding["intent"]]

            input_path = understanding.get("input_path", "your data")
            operation_text = ", ".join(operations) if operations else "process"

            confirmation = (
                f"I understand you want to {operation_text} on {input_path}. "
                "Let me create a plan."
            )

            new_message = {
                "role": MessageRole.ASSISTANT.value,
                "content": confirmation,
                "timestamp": datetime.now().isoformat(),
                "tool_calls": [],
                "tool_call_id": None,
            }

            return {
                "messages": [*messages, new_message],
                "metadata": metadata,
            }

        except Exception as e:
            last_error = f"LLM call failed: {e}"
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                break

    # All retries failed
    logger.error(f"All {MAX_RETRIES} attempts failed: {last_error}")
    error_message: dict[str, Any] = {
        "role": MessageRole.ASSISTANT.value,
        "content": "I'm having trouble understanding your request. Could you please rephrase it?",
        "timestamp": datetime.now().isoformat(),
        "tool_calls": [],
        "tool_call_id": None,
    }

    return {
        "messages": [*messages, error_message],
        "last_error": last_error,
        "awaiting_user": True,
    }


__all__ = ["understand"]
