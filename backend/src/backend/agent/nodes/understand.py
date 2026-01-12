"""
Understand node for parsing user requests and extracting intent.

This node uses the LLM to analyze natural language requests and
extract structured information about what the user wants to do.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.llm import get_llm
from backend.agent.prompts import load_prompt
from backend.agent.state import MessageRole, PipelineState
from backend.agent.utils import extract_json_object

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls
MAX_RETRIES = 3


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
            understanding = extract_json_object(str(response_content))

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

            # Store understanding in metadata (avoid mutating original state)
            metadata = {**state.get("metadata", {}), "understanding": understanding}
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
