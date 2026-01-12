"""
Prompt management module for Cloumask agent.

Provides utilities for loading and managing prompts from markdown files.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    """
    Load a prompt from a markdown file.

    Prompts are cached for performance. Use clear_prompt_cache()
    to reload prompts during development.

    Args:
        name: Name of the prompt file (without .md extension).

    Returns:
        The prompt content as a string.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.

    Example:
        >>> system = load_prompt("system")
        >>> planning = load_prompt("planning")
    """
    prompt_file = PROMPTS_DIR / f"{name}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt not found: {name} (looked for {prompt_file})")
    return prompt_file.read_text(encoding="utf-8")


def clear_prompt_cache() -> None:
    """Clear the prompt cache. Useful for development and testing."""
    load_prompt.cache_clear()


__all__ = [
    "PROMPTS_DIR",
    "load_prompt",
    "clear_prompt_cache",
]
