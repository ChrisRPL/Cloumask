"""
Script generation service using Ollama coding models.

Generates Python scripts from natural language descriptions using
AI-powered code generation with the CODE_GENERATION use case.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

from backend.agent.llm.config import LLMUseCase
from backend.agent.llm.provider import OllamaProvider, get_provider
from backend.scripts.templates import (
    format_explanation_prompt,
    format_generation_prompt,
    format_refinement_prompt,
)
from backend.scripts.validator import ScriptValidator, ValidationResult

logger = logging.getLogger(__name__)


# Default scripts directory in user home
DEFAULT_SCRIPTS_DIR = Path.home() / ".cloumask" / "scripts"


class ScriptGeneratorService:
    """
    Generates Python scripts using Ollama coding models.

    Uses the CODE_GENERATION use case configuration with fallback
    to smaller models if the primary model is unavailable.

    Attributes:
        provider: The Ollama provider for LLM calls.
        validator: Script validator for syntax checking.
        scripts_dir: Directory for saving generated scripts.
    """

    def __init__(
        self,
        provider: OllamaProvider | None = None,
        scripts_dir: Path | None = None,
    ) -> None:
        """
        Initialize the script generator.

        Args:
            provider: Optional custom provider. If None, uses CODE_GENERATION default.
            scripts_dir: Directory for saving scripts. Defaults to ~/.cloumask/scripts/
        """
        self.provider = provider or get_provider(LLMUseCase.CODE_GENERATION)
        self.validator = ScriptValidator()
        self.scripts_dir = scripts_dir or DEFAULT_SCRIPTS_DIR

        # Ensure scripts directory exists
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

    async def generate(
        self,
        prompt: str,
        context: dict | None = None,
    ) -> tuple[str, str | None]:
        """
        Generate a Python script from natural language description.

        Args:
            prompt: Natural language description of what the script should do.
            context: Optional context dict with:
                - existing_code: Code to refine instead of generating new
                - refinement: Whether this is a refinement request

        Returns:
            Tuple of (script_code, explanation or None).

        Raises:
            Exception: If generation fails after all retries and fallbacks.
        """
        # Build the prompt
        if context and context.get("existing_code"):
            full_prompt = format_refinement_prompt(
                existing_code=context["existing_code"],
                refinement_request=prompt,
            )
        else:
            full_prompt = format_generation_prompt(prompt)

        # Generate using the provider
        messages = [HumanMessage(content=full_prompt)]

        logger.info("Generating script with model: %s", self.provider.current_model)
        response = await self.provider.invoke(messages)

        # Extract code from response
        script = self._extract_code(response.content)
        explanation = self._extract_explanation(response.content, script)

        logger.info("Generated script with %d lines", script.count("\n") + 1)
        return script, explanation

    async def explain(self, script: str) -> str:
        """
        Generate an explanation for a script.

        Args:
            script: Python script code to explain.

        Returns:
            Brief explanation of what the script does.
        """
        prompt = format_explanation_prompt(script)
        messages = [HumanMessage(content=prompt)]

        response = await self.provider.invoke(messages)
        return response.content.strip()

    def validate(self, script: str) -> ValidationResult:
        """
        Validate a script for syntax and interface compliance.

        Args:
            script: Python script code to validate.

        Returns:
            ValidationResult with errors and warnings.
        """
        return self.validator.validate(script)

    def save(
        self,
        name: str,
        content: str,
        description: str | None = None,
        overwrite: bool = False,
    ) -> tuple[Path, bool]:
        """
        Save a script to the scripts directory.

        Args:
            name: Script name (without .py extension).
            content: Script content.
            description: Optional description to prepend as docstring.
            overwrite: Whether to overwrite existing file.

        Returns:
            Tuple of (file_path, created_new).

        Raises:
            FileExistsError: If file exists and overwrite is False.
        """
        # Sanitize name
        safe_name = self._sanitize_filename(name)
        if not safe_name.endswith(".py"):
            safe_name += ".py"

        filepath = self.scripts_dir / safe_name
        created = not filepath.exists()

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"Script already exists: {filepath}")

        # Add description docstring if provided
        if description and not content.strip().startswith('"""'):
            content = f'"""\n{description}\n"""\n{content}'

        filepath.write_text(content, encoding="utf-8")
        logger.info("Saved script to: %s", filepath)

        return filepath, created

    def list_scripts(self) -> list[dict]:
        """
        List all saved scripts.

        Returns:
            List of dicts with script info (name, path, modified_at, size).
        """
        scripts = []
        for filepath in self.scripts_dir.glob("*.py"):
            stat = filepath.stat()
            scripts.append({
                "name": filepath.stem,
                "path": str(filepath),
                "modified_at": stat.st_mtime,
                "size_bytes": stat.st_size,
            })
        return sorted(scripts, key=lambda s: s["modified_at"], reverse=True)

    def load(self, name: str) -> str | None:
        """
        Load a script by name.

        Args:
            name: Script name (with or without .py extension).

        Returns:
            Script content or None if not found.
        """
        if not name.endswith(".py"):
            name += ".py"

        filepath = self.scripts_dir / name
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")
        return None

    def delete(self, name: str) -> bool:
        """
        Delete a script by name.

        Args:
            name: Script name (with or without .py extension).

        Returns:
            True if deleted, False if not found.
        """
        if not name.endswith(".py"):
            name += ".py"

        filepath = self.scripts_dir / name
        if filepath.exists():
            filepath.unlink()
            logger.info("Deleted script: %s", filepath)
            return True
        return False

    def _extract_code(self, content: str) -> str:
        """
        Extract Python code from LLM response.

        Handles markdown code blocks and raw code.

        Args:
            content: LLM response content.

        Returns:
            Extracted Python code.
        """
        # Try to find markdown code block
        match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try without 'python' specifier
        match = re.search(r"```\n(.*?)```", content, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Verify it looks like Python
            if "def " in code or "import " in code:
                return code

        # Return raw content as fallback
        return content.strip()

    def _extract_explanation(self, content: str, code: str) -> str | None:
        """
        Extract explanation text from response.

        Looks for text before or after the code block.

        Args:
            content: Full LLM response.
            code: Extracted code.

        Returns:
            Explanation text or None.
        """
        # Remove code block from content
        cleaned = re.sub(r"```python\n.*?```", "", content, flags=re.DOTALL)
        cleaned = re.sub(r"```\n.*?```", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()

        # Return if there's meaningful explanation text
        if len(cleaned) > 20:
            # Take first paragraph
            lines = cleaned.split("\n\n")
            return lines[0].strip() if lines else None

        return None

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize a filename to be safe for the filesystem.

        Args:
            name: Original filename.

        Returns:
            Safe filename.
        """
        # Remove or replace unsafe characters
        safe = re.sub(r"[^\w\-_.]", "_", name)
        # Remove leading/trailing underscores and dots
        safe = safe.strip("_.")
        # Ensure not empty
        return safe or "script"


# Module-level convenience function
async def generate_script(prompt: str) -> tuple[str, str | None]:
    """
    Generate a script using the default generator.

    Args:
        prompt: Natural language description.

    Returns:
        Tuple of (script_code, explanation).
    """
    generator = ScriptGeneratorService()
    return await generator.generate(prompt)
