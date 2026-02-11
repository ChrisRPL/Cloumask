"""
Understand node for parsing user requests and extracting intent.

This node uses the LLM to analyze natural language requests and
extract structured information about what the user wants to do.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.llm import SimpleLLMConfig, get_llm
from backend.agent.prompts import load_prompt
from backend.agent.state import MessageRole, PipelineState
from backend.agent.utils import extract_json_object
from backend.api.config import settings

logger = logging.getLogger(__name__)

# Maximum retries for LLM calls
MAX_RETRIES = 1
FAST_LLM_TIMEOUT_SECONDS = 45
FAST_LLM_CONTEXT_TOKENS = 4096

PATH_PATTERN = re.compile(r"(/[\w\-.~/]+|~[/\w\-.]+|[A-Za-z]:\\[^\s]+)")


def _extract_path(content: str) -> str | None:
    match = PATH_PATTERN.search(content)
    return match.group(0) if match else None


def _extract_operations(content: str) -> list[str]:
    normalized = content.lower()
    # Track first occurrence position to preserve user-requested order.
    operation_hits: list[tuple[int, str]] = []
    operation_patterns = [
        ("scan", ["scan", "inspect", "list files"]),
        ("detect", ["detect", "find objects", "object detection"]),
        ("segment", ["segment", "segmentation", "mask"]),
        ("anonymize", ["anonymize", "blur", "redact", "privacy"]),
        ("export", ["export", "convert", "save as"]),
        ("label", ["label", "annotate", "annotation"]),
    ]

    for operation, aliases in operation_patterns:
        for alias in aliases:
            idx = normalized.find(alias)
            if idx != -1:
                operation_hits.append((idx, operation))
                break

    operation_hits.sort(key=lambda hit: hit[0])
    operations: list[str] = []
    for _, operation in operation_hits:
        if operation not in operations:
            operations.append(operation)

    # Labeling implies detect + export workflow.
    if "label" in operations:
        if "detect" not in operations:
            operations.append("detect")
        if "export" not in operations:
            operations.append("export")

    return operations


def _extract_parameters(content: str) -> dict[str, Any]:
    normalized = content.lower()
    parameters: dict[str, Any] = {}

    # Common detection classes for quick heuristic extraction.
    class_terms = {
        "person": ["person", "people", "pedestrian", "pedestrians"],
        "car": ["car", "cars", "vehicle", "vehicles"],
        "truck": ["truck", "trucks"],
        "bus": ["bus", "buses"],
        "traffic light": ["traffic light", "traffic lights"],
        "road sign": ["road sign", "road signs", "sign", "signs"],
    }
    classes: list[str] = []
    for canonical, aliases in class_terms.items():
        if any(alias in normalized for alias in aliases):
            classes.append(canonical)
    if classes:
        parameters["classes"] = classes

    confidence_match = re.search(r"confidence\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)", normalized)
    if confidence_match:
        parameters["confidence"] = float(confidence_match.group(1))

    format_match = re.search(
        r"\b(yolo|coco|kitti|voc|pascal|cvat|nuscenes|openlabel)\b",
        normalized,
    )
    if format_match:
        fmt = format_match.group(1)
        parameters["format"] = "voc" if fmt == "pascal" else fmt

    if "face" in normalized and "plate" in normalized:
        parameters["target"] = "all"
    elif "face" in normalized:
        parameters["target"] = "faces"
    elif "plate" in normalized:
        parameters["target"] = "plates"

    return parameters


def _infer_input_type(content: str, input_path: str | None) -> str:
    normalized = content.lower()
    if input_path:
        lower_path = input_path.lower()
        if any(ext in lower_path for ext in (".pcd", ".ply", ".las", ".laz", ".bin")):
            return "pointcloud"
        if any(ext in lower_path for ext in (".mp4", ".mov", ".avi", ".mkv")):
            return "video"
    if "point cloud" in normalized or "lidar" in normalized:
        return "pointcloud"
    if "video" in normalized:
        return "video"
    return "images"


def _build_fast_understanding(content: str) -> dict[str, Any] | None:
    """
    Build understanding without LLM for clear multi-operation task requests.

    We deliberately avoid fast-path for pure "scan" requests to keep broad
    compatibility with existing planning tests and behavior.
    """
    input_path = _extract_path(content)
    operations = _extract_operations(content)
    if not input_path or not operations:
        return None

    if len(operations) == 1 and operations[0] == "scan":
        return None

    parameters = _extract_parameters(content)

    return {
        "intent": operations[0],
        "input_path": input_path,
        "input_type": _infer_input_type(content, input_path),
        "operations": operations,
        "parameters": parameters,
        "output_path": None,
        "clarification_needed": None,
    }


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

    # Fast-path deterministic understanding for clear task requests.
    fast_understanding = _build_fast_understanding(latest_user_msg)
    if fast_understanding is not None:
        metadata = {**state.get("metadata", {}), "understanding": fast_understanding}
        operations = fast_understanding.get("operations", [])
        input_path = fast_understanding.get("input_path", "your data")
        operation_text = ", ".join(operations) if operations else "process"

        new_message = {
            "role": MessageRole.ASSISTANT.value,
            "content": (
                f"I understand you want to {operation_text} on {input_path}. "
                "Let me create a plan."
            ),
            "timestamp": datetime.now().isoformat(),
            "tool_calls": [],
            "tool_call_id": None,
        }

        return {
            "messages": [*messages, new_message],
            "metadata": metadata,
        }

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
    llm = get_llm(
        config=SimpleLLMConfig(
            host=settings.ollama_host,
            model=settings.ollama_model,
            temperature=0.1,
            timeout=FAST_LLM_TIMEOUT_SECONDS,
            num_ctx=FAST_LLM_CONTEXT_TOKENS,
        ),
        temperature=0.1,  # Low temperature for structured output
    )

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
