"""
Plan node for generating execution plans from understood requests.

This node uses the LLM to create a structured execution plan
based on the user's intent extracted by the understand node.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.llm import get_llm
from backend.agent.prompts import load_prompt
from backend.agent.state import MessageRole, PipelineState, StepStatus

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls
MAX_RETRIES = 3

# Valid tool names that can appear in plans
VALID_TOOLS = frozenset(["scan_directory", "anonymize", "detect", "segment", "export"])


def validate_plan(plan: list[dict[str, Any]]) -> str | None:
    """
    Validate a generated plan for correctness.

    Checks:
    - Plan is not empty
    - All tools are known
    - Each step has required parameters
    - Tool-specific parameter validation

    Args:
        plan: List of plan step dictionaries.

    Returns:
        Error message string if invalid, None if valid.
    """
    if not plan:
        return "Plan is empty"

    for i, step in enumerate(plan):
        step_num = i + 1

        # Check tool name exists and is valid
        tool_name = step.get("tool_name")
        if not tool_name:
            return f"Step {step_num} has no tool_name"

        if tool_name not in VALID_TOOLS:
            return f"Step {step_num} uses unknown tool: {tool_name}"

        # Check parameters exist
        parameters = step.get("parameters")
        if not parameters:
            return f"Step {step_num} ({tool_name}) has no parameters"

        # Tool-specific validation
        if tool_name == "scan_directory":
            if "path" not in parameters:
                return f"Step {step_num} (scan_directory) missing required 'path' parameter"

        elif tool_name == "anonymize":
            if "input_path" not in parameters:
                return f"Step {step_num} (anonymize) missing required 'input_path' parameter"
            if "output_path" not in parameters:
                return f"Step {step_num} (anonymize) missing required 'output_path' parameter"

        elif tool_name == "detect":
            if "input_path" not in parameters:
                return f"Step {step_num} (detect) missing required 'input_path' parameter"
            if "classes" not in parameters:
                return f"Step {step_num} (detect) missing required 'classes' parameter"

        elif tool_name == "segment":
            if "input_path" not in parameters:
                return f"Step {step_num} (segment) missing required 'input_path' parameter"
            if "prompt" not in parameters:
                return f"Step {step_num} (segment) missing required 'prompt' parameter"

        elif tool_name == "export":
            if "input_path" not in parameters:
                return f"Step {step_num} (export) missing required 'input_path' parameter"
            if "output_path" not in parameters:
                return f"Step {step_num} (export) missing required 'output_path' parameter"
            if "format" not in parameters:
                return f"Step {step_num} (export) missing required 'format' parameter"

    return None


def format_plan_for_display(plan: list[dict[str, Any]]) -> str:
    """
    Format a plan as a human-readable string for chat display.

    Args:
        plan: List of plan step dictionaries.

    Returns:
        Formatted string with numbered steps, tools, and parameters.
    """
    if not plan:
        return "No steps in plan."

    lines: list[str] = []

    for i, step in enumerate(plan, 1):
        status = step.get("status", StepStatus.PENDING.value)

        # Status icons
        status_icons = {
            StepStatus.PENDING.value: "[ ]",
            StepStatus.RUNNING.value: "[>]",
            StepStatus.COMPLETED.value: "[x]",
            StepStatus.FAILED.value: "[!]",
            StepStatus.SKIPPED.value: "[-]",
        }
        icon = status_icons.get(status, "[ ]")

        description = step.get("description", f"Step {i}")
        tool_name = step.get("tool_name", "unknown")
        parameters = step.get("parameters", {})

        lines.append(f"{icon} **Step {i}: {description}**")
        lines.append(f"    Tool: `{tool_name}`")

        if parameters:
            param_items = []
            for k, v in parameters.items():
                v_str = ", ".join(str(item) for item in v) if isinstance(v, list) else str(v)
                param_items.append(f"{k}={v_str}")
            lines.append(f"    Parameters: {', '.join(param_items)}")

        lines.append("")  # Blank line between steps

    return "\n".join(lines)


def _extract_json_array_from_response(content: str) -> list[dict[str, Any]] | None:
    """
    Extract JSON array from LLM response, handling markdown code blocks.

    Args:
        content: Raw LLM response content.

    Returns:
        Parsed JSON list, or None if parsing fails.
    """
    # Try direct parsing first
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

            try:
                result = json.loads(cleaned)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue

    # Try finding JSON-like content with brackets
    start_idx = content.find("[")
    end_idx = content.rfind("]")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            result = json.loads(content[start_idx : end_idx + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


async def generate_plan(state: PipelineState) -> dict[str, Any]:
    """
    Generate an execution plan from the understood request.

    This node:
    1. Gets the understanding from metadata
    2. Sends it to the LLM with the planning prompt
    3. Parses the JSON array response into pipeline steps
    4. Validates the plan
    5. Formats and displays the plan for user approval

    Args:
        state: Current pipeline state with understanding in metadata.

    Returns:
        State update dict with plan, or error message.
    """
    messages = state.get("messages", [])
    metadata = state.get("metadata", {})
    understanding = metadata.get("understanding", {})

    if not understanding:
        logger.warning("No understanding found in metadata")
        return {"last_error": "No understanding found, cannot plan"}

    logger.info(f"Generating plan for intent: {understanding.get('intent')}")

    # Load the planning prompt
    try:
        planning_prompt = load_prompt("planning")
    except FileNotFoundError:
        logger.error("Failed to load planning prompt")
        return {"last_error": "Failed to load planning prompt"}

    # Build the context for planning
    planning_context = f"""Create a plan for the following request:

Intent: {understanding.get('intent', 'unknown')}
Input path: {understanding.get('input_path', 'not specified')}
Input type: {understanding.get('input_type', 'unknown')}
Operations: {understanding.get('operations', [])}
Parameters: {understanding.get('parameters', {})}
Output path: {understanding.get('output_path', 'not specified')}

Generate a JSON array of steps to accomplish this."""

    llm_messages = [
        SystemMessage(content=planning_prompt),
        HumanMessage(content=planning_context),
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

            # Parse JSON array from response
            steps_raw = _extract_json_array_from_response(str(response_content))

            if steps_raw is None:
                last_error = f"Failed to parse JSON array from response: {response_content[:200]}"
                logger.warning(last_error)
                continue

            # Convert to PipelineStep format
            plan: list[dict[str, Any]] = []
            for i, step in enumerate(steps_raw):
                plan.append({
                    "id": f"step-{uuid4().hex[:8]}",
                    "tool_name": step.get("tool_name", "unknown"),
                    "parameters": step.get("parameters", {}),
                    "description": step.get("description", f"Step {i + 1}"),
                    "status": StepStatus.PENDING.value,
                    "result": None,
                    "error": None,
                    "started_at": None,
                    "completed_at": None,
                })

            # Validate the plan
            validation_error = validate_plan(plan)
            if validation_error:
                logger.warning(f"Plan validation failed: {validation_error}")

                # Try to fix common issues on retry
                last_error = validation_error
                if attempt < MAX_RETRIES - 1:
                    # Add validation error to context for next attempt
                    llm_messages.append(
                        HumanMessage(
                            content=f"The plan had an error: {validation_error}. "
                            "Please fix and try again."
                        )
                    )
                    continue
                else:
                    # Final attempt failed, report error
                    error_message: dict[str, Any] = {
                        "role": MessageRole.ASSISTANT.value,
                        "content": f"I had trouble creating a valid plan: {validation_error}. Could you rephrase your request?",
                        "timestamp": datetime.now().isoformat(),
                        "tool_calls": [],
                        "tool_call_id": None,
                    }
                    return {
                        "messages": [*messages, error_message],
                        "last_error": validation_error,
                        "awaiting_user": True,
                    }

            # Plan is valid, format for display
            plan_display = format_plan_for_display(plan)
            logger.info(f"Generated plan with {len(plan)} steps")

            plan_message: dict[str, Any] = {
                "role": MessageRole.ASSISTANT.value,
                "content": (
                    f"Here's my proposed plan:\n\n{plan_display}\n"
                    "Do you want to proceed, or would you like to make changes?"
                ),
                "timestamp": datetime.now().isoformat(),
                "tool_calls": [],
                "tool_call_id": None,
            }

            return {
                "messages": [*messages, plan_message],
                "plan": plan,
                "current_step": 0,
                "plan_approved": False,
            }

        except Exception as e:
            last_error = f"LLM call failed: {e}"
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                break

    # All retries failed
    logger.error(f"All {MAX_RETRIES} attempts failed: {last_error}")
    error_message = {
        "role": MessageRole.ASSISTANT.value,
        "content": "I had trouble creating a structured plan. Could you rephrase your request?",
        "timestamp": datetime.now().isoformat(),
        "tool_calls": [],
        "tool_call_id": None,
    }

    return {
        "messages": [*messages, error_message],
        "last_error": last_error,
        "awaiting_user": True,
    }


__all__ = [
    "generate_plan",
    "validate_plan",
    "format_plan_for_display",
    "VALID_TOOLS",
]
