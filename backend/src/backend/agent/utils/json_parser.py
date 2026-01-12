"""
JSON extraction utilities for parsing LLM responses.

Handles various formats including raw JSON, markdown code blocks,
and JSON embedded within other text.
"""

from __future__ import annotations

import json
from typing import Any


def _extract_from_code_blocks(content: str) -> str | None:
    """
    Try to extract JSON from markdown code blocks.

    Args:
        content: Raw content that may contain code blocks.

    Returns:
        Extracted content from code block, or None.
    """
    if "```" not in content:
        return None

    parts = content.split("```")
    for part in parts:
        cleaned = part.strip()
        # Remove 'json' language identifier if present
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

        # Check if it looks like JSON
        if cleaned.startswith(("{", "[")):
            return cleaned

    return None


def extract_json_object(content: str) -> dict[str, Any] | None:
    """
    Extract a JSON object from LLM response, handling various formats.

    Tries multiple strategies:
    1. Direct JSON parsing
    2. Extract from markdown code blocks
    3. Find JSON-like content with braces

    Args:
        content: Raw LLM response content.

    Returns:
        Parsed JSON dict, or None if parsing fails.
    """

    def _try_parse(text: str) -> dict[str, Any] | None:
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        return None

    # Try direct parsing first
    if result := _try_parse(content):
        return result

    # Try extracting from markdown code block
    code_block = _extract_from_code_blocks(content)
    if code_block and (result := _try_parse(code_block)):
        return result

    # Try finding JSON-like content with braces
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if (
        start_idx != -1
        and end_idx != -1
        and end_idx > start_idx
        and (result := _try_parse(content[start_idx : end_idx + 1]))
    ):
        return result

    return None


def extract_json_array(content: str) -> list[dict[str, Any]] | None:
    """
    Extract a JSON array from LLM response, handling various formats.

    Tries multiple strategies:
    1. Direct JSON parsing
    2. Extract from markdown code blocks
    3. Find JSON-like content with brackets

    Args:
        content: Raw LLM response content.

    Returns:
        Parsed JSON list of dicts, or None if parsing fails.
    """

    def _try_parse(text: str) -> list[dict[str, Any]] | None:
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        return None

    # Try direct parsing first
    if result := _try_parse(content):
        return result

    # Try extracting from markdown code block
    code_block = _extract_from_code_blocks(content)
    if code_block and (result := _try_parse(code_block)):
        return result

    # Try finding JSON-like content with brackets
    start_idx = content.find("[")
    end_idx = content.rfind("]")
    if (
        start_idx != -1
        and end_idx != -1
        and end_idx > start_idx
        and (result := _try_parse(content[start_idx : end_idx + 1]))
    ):
        return result

    return None


__all__ = [
    "extract_json_object",
    "extract_json_array",
]
