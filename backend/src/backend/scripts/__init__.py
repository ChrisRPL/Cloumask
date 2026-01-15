"""
Script generation and management module.

Provides AI-powered script generation using Ollama coding models,
script validation, and persistence to user scripts directory.
"""

from backend.scripts.generator import ScriptGeneratorService
from backend.scripts.templates import SCRIPT_GENERATION_PROMPT
from backend.scripts.validator import ScriptValidator

__all__ = [
    "ScriptGeneratorService",
    "ScriptValidator",
    "SCRIPT_GENERATION_PROMPT",
]
