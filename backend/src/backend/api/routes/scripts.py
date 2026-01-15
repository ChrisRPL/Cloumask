"""
Script generation and management API endpoints.

Provides REST API for AI-powered script generation, validation,
and persistence using Ollama coding models.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.scripts.generator import ScriptGeneratorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scripts", tags=["Scripts"])

# Shared generator instance
_generator: ScriptGeneratorService | None = None


def get_generator() -> ScriptGeneratorService:
    """Get or create the script generator instance."""
    global _generator
    if _generator is None:
        _generator = ScriptGeneratorService()
    return _generator


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class ScriptGenerateRequest(BaseModel):
    """Request to generate a script from natural language."""

    prompt: str = Field(
        description="Natural language description of what the script should do",
        min_length=10,
        max_length=2000,
    )
    model: str | None = Field(
        default=None,
        description="Specific model to use (defaults to configured code model)",
    )
    context: dict | None = Field(
        default=None,
        description="Additional context (existing_code for refinement)",
    )


class ScriptGenerateResponse(BaseModel):
    """Response from script generation."""

    script: str = Field(description="Generated Python script")
    model: str = Field(description="Model used for generation")
    explanation: str | None = Field(
        default=None,
        description="Brief explanation of what the script does",
    )


class ScriptValidateRequest(BaseModel):
    """Request to validate a script."""

    content: str = Field(
        description="Python script content to validate",
        min_length=1,
    )


class ScriptValidateResponse(BaseModel):
    """Response from script validation."""

    valid: bool = Field(description="Whether the script is valid")
    errors: list[str] = Field(
        default_factory=list,
        description="Validation errors",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Validation warnings",
    )
    has_process_function: bool = Field(
        description="Whether script has required process() function",
    )


class ScriptSaveRequest(BaseModel):
    """Request to save a script."""

    name: str = Field(
        description="Script name (without .py extension)",
        min_length=1,
        max_length=100,
    )
    content: str = Field(
        description="Python script content",
        min_length=1,
    )
    description: str | None = Field(
        default=None,
        description="Optional description to add as docstring",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing script",
    )


class ScriptSaveResponse(BaseModel):
    """Response from saving a script."""

    path: str = Field(description="Full path to saved script")
    name: str = Field(description="Final script name")
    created: bool = Field(description="Whether a new file was created")


class ScriptInfo(BaseModel):
    """Information about a saved script."""

    name: str = Field(description="Script name without extension")
    path: str = Field(description="Full path to script")
    modified_at: str = Field(description="Last modified timestamp (ISO 8601)")
    size_bytes: int = Field(description="File size in bytes")


class ScriptListResponse(BaseModel):
    """Response listing all saved scripts."""

    scripts: list[ScriptInfo] = Field(description="List of saved scripts")
    scripts_dir: str = Field(description="Scripts directory path")


class ScriptLoadResponse(BaseModel):
    """Response from loading a script."""

    name: str = Field(description="Script name")
    content: str = Field(description="Script content")
    path: str = Field(description="Full path to script")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/generate", response_model=ScriptGenerateResponse)
async def generate_script(request: ScriptGenerateRequest) -> ScriptGenerateResponse:
    """
    Generate a Python script from natural language description.

    Uses AI-powered code generation with Ollama coding models.
    The generated script will conform to the Cloumask custom step interface.
    """
    generator = get_generator()

    try:
        script, explanation = await generator.generate(
            prompt=request.prompt,
            context=request.context,
        )

        return ScriptGenerateResponse(
            script=script,
            model=generator.provider.current_model,
            explanation=explanation,
        )

    except Exception as e:
        logger.exception("Script generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Script generation failed: {e}",
        ) from e


@router.post("/validate", response_model=ScriptValidateResponse)
async def validate_script(request: ScriptValidateRequest) -> ScriptValidateResponse:
    """
    Validate a Python script for syntax and interface compliance.

    Checks that the script:
    - Has valid Python syntax
    - Defines a process(input_path, output_path, config) function
    - Uses available imports
    """
    generator = get_generator()
    result = generator.validate(request.content)

    return ScriptValidateResponse(
        valid=result.valid,
        errors=result.errors,
        warnings=result.warnings,
        has_process_function=result.has_process_function,
    )


@router.post("/save", response_model=ScriptSaveResponse)
async def save_script(request: ScriptSaveRequest) -> ScriptSaveResponse:
    """
    Save a script to the user's scripts directory.

    Scripts are saved to ~/.cloumask/scripts/ by default.
    """
    generator = get_generator()

    try:
        filepath, created = generator.save(
            name=request.name,
            content=request.content,
            description=request.description,
            overwrite=request.overwrite,
        )

        return ScriptSaveResponse(
            path=str(filepath),
            name=filepath.stem,
            created=created,
        )

    except FileExistsError:
        raise HTTPException(
            status_code=409,
            detail=f"Script '{request.name}' already exists. Use overwrite=true to replace.",
        ) from None


@router.get("/list", response_model=ScriptListResponse)
async def list_scripts() -> ScriptListResponse:
    """
    List all saved scripts in the scripts directory.

    Returns scripts sorted by modification time (newest first).
    """
    generator = get_generator()
    scripts_data = generator.list_scripts()

    scripts = [
        ScriptInfo(
            name=s["name"],
            path=s["path"],
            modified_at=datetime.fromtimestamp(s["modified_at"]).isoformat(),
            size_bytes=s["size_bytes"],
        )
        for s in scripts_data
    ]

    return ScriptListResponse(
        scripts=scripts,
        scripts_dir=str(generator.scripts_dir),
    )


@router.get("/{name}", response_model=ScriptLoadResponse)
async def load_script(name: str) -> ScriptLoadResponse:
    """
    Load a script by name.

    The name can include or exclude the .py extension.
    """
    generator = get_generator()
    content = generator.load(name)

    if content is None:
        raise HTTPException(
            status_code=404,
            detail=f"Script '{name}' not found",
        )

    # Build path for response
    script_name = name if name.endswith(".py") else f"{name}.py"
    filepath = generator.scripts_dir / script_name

    return ScriptLoadResponse(
        name=filepath.stem,
        content=content,
        path=str(filepath),
    )


@router.delete("/{name}")
async def delete_script(name: str) -> dict:
    """
    Delete a script by name.

    The name can include or exclude the .py extension.
    """
    generator = get_generator()
    deleted = generator.delete(name)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Script '{name}' not found",
        )

    return {"deleted": True, "name": name}
