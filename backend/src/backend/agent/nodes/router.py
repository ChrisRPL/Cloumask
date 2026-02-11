"""
Intent routing nodes for fast-path chat vs task execution.

This module provides a lightweight deterministic router to avoid expensive
LLM planning calls for simple conversational messages (e.g., "hi", "thanks").
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from backend.agent.state import MessageRole, PipelineState

# Conservative task hints: if any of these appear, route to planning flow.
TASK_HINTS = {
    "scan",
    "anonymize",
    "detect",
    "segment",
    "export",
    "convert",
    "duplicate",
    "qa",
    "split",
    "dataset",
    "label",
    "yolo",
    "coco",
    "kitti",
    "cvat",
    "openlabel",
    "image",
    "images",
    "video",
    "point",
    "pointcloud",
    "rosbag",
    "folder",
    "directory",
    "path",
    "file",
    "files",
    "process",
    "pipeline",
    "blur",
    "face",
    "plate",
}

CHAT_PATTERNS = (
    re.compile(r"^\s*(hi|hello|hey|yo)\s*!*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(thanks|thank you|thx)\s*!*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(good morning|good afternoon|good evening)\s*!*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(ok|okay|cool|nice)\s*!*\s*$", re.IGNORECASE),
)

# UNIX, home, or Windows-style path snippets.
PATH_PATTERN = re.compile(r"(/[\w\-.~/]+|~[/\w\-.]+|[A-Za-z]:\\[^\s]+)")


def get_latest_user_message(state: PipelineState) -> str:
    """Return the latest user message content, or empty string."""
    messages = state.get("messages", [])
    for message in reversed(messages):
        if message.get("role") == MessageRole.USER.value:
            return str(message.get("content", "")).strip()
    return ""


def classify_intent(content: str) -> Literal["chat", "task"]:
    """
    Deterministically route a message to chat or task flow.

    The classifier is intentionally conservative: any task-like signal routes
    to the planning flow to avoid false "chat" classifications.
    """
    normalized = content.strip().lower()
    if not normalized:
        return "task"

    if PATH_PATTERN.search(normalized):
        return "task"

    # Symbol-heavy requests often include classes, paths, formats, etc.
    if any(char in normalized for char in ("/", "\\", ":", ".", "_")):
        if any(ext in normalized for ext in (".jpg", ".jpeg", ".png", ".mp4", ".pcd", ".ply", ".bag")):
            return "task"

    if any(hint in normalized for hint in TASK_HINTS):
        return "task"

    for pattern in CHAT_PATTERNS:
        if pattern.match(normalized):
            return "chat"

    if normalized in {"what can you do?", "what can you do", "help"}:
        return "chat"

    # Default to task so requests are not accidentally dropped from planning flow.
    return "task"


async def route_intent(state: PipelineState) -> dict[str, Any]:
    """
    Lightweight routing node.

    Writes metadata.intent_route as "chat" or "task" for downstream routing.
    """
    metadata = dict(state.get("metadata", {}))
    latest_user_message = get_latest_user_message(state)
    metadata["intent_route"] = classify_intent(latest_user_message)
    return {"metadata": metadata}


async def chat_reply(state: PipelineState) -> dict[str, Any]:
    """
    Fast-path assistant reply for simple conversational messages.

    Keeps response cheap and immediate without triggering planning.
    """
    messages = list(state.get("messages", []))
    metadata = dict(state.get("metadata", {}))
    metadata["intent_route"] = "chat"

    messages.append({
        "role": MessageRole.ASSISTANT.value,
        "content": (
            "Hi! I can help you scan datasets, detect/segment objects, anonymize faces and plates, "
            "and export annotations. Share your data path and what output you need."
        ),
        "timestamp": datetime.now().isoformat(),
        "tool_calls": [],
        "tool_call_id": None,
    })

    return {
        "messages": messages,
        "metadata": metadata,
        "awaiting_user": False,
    }


__all__ = [
    "classify_intent",
    "get_latest_user_message",
    "route_intent",
    "chat_reply",
]
