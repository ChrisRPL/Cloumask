"""
Model management utilities for Ollama.

This module provides functions for listing available models, checking
model availability, pulling models, and retrieving model information.

Implements spec: 08-ollama-integration
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from backend.api.config import settings

logger = logging.getLogger(__name__)

# Default timeout for model operations (in seconds)
DEFAULT_TIMEOUT = 10
PULL_TIMEOUT = 600  # 10 minutes for pulling large models


async def list_available_models() -> list[str]:
    """
    List models available in Ollama.

    Calls `ollama list` and parses the output to extract model names.

    Returns:
        List of model names (e.g., ["qwen3:14b", "llama3:8b"]).
        Empty list if Ollama is not available or an error occurs.

    Example:
        >>> models = await list_available_models()
        >>> print(models)  # ["qwen3:14b", "qwen3:8b", "llama3:8b"]
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "ollama",
            "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=DEFAULT_TIMEOUT
        )

        if process.returncode != 0:
            logger.warning("ollama list failed: %s", stderr.decode())
            return []

        # Parse output (skip header line)
        output = stdout.decode()
        lines = output.strip().split("\n")[1:]  # Skip "NAME..." header

        models = []
        for line in lines:
            parts = line.split()
            if parts:
                models.append(parts[0])  # Model name is first column

        return models

    except FileNotFoundError:
        logger.error("ollama command not found. Is Ollama installed?")
        return []
    except TimeoutError:
        logger.error("ollama list timed out")
        return []
    except Exception as e:
        logger.exception("Failed to list Ollama models: %s", e)
        return []


async def ensure_model_available(model: str) -> bool:
    """
    Ensure a model is pulled and available.

    Checks if the model is already available; if not, attempts to pull it.

    Args:
        model: Model name to check/pull (e.g., "qwen3:14b").

    Returns:
        True if model is now available, False otherwise.

    Example:
        >>> if await ensure_model_available("qwen3:14b"):
        ...     print("Model ready!")
    """
    available = await list_available_models()

    # Check for exact match or tag-less match
    model_base = model.split(":")[0]
    for available_model in available:
        if available_model == model or available_model.startswith(f"{model_base}:"):
            logger.debug("Model %s is already available", model)
            return True

    # Try to pull the model
    logger.info("Pulling model: %s", model)
    try:
        process = await asyncio.create_subprocess_exec(
            "ollama",
            "pull",
            model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=PULL_TIMEOUT
        )

        if process.returncode == 0:
            logger.info("Successfully pulled model: %s", model)
            return True
        else:
            logger.error("Failed to pull model %s: %s", model, stderr.decode())
            return False

    except TimeoutError:
        logger.error("Model pull timed out for %s", model)
        return False
    except Exception as e:
        logger.exception("Failed to pull model %s: %s", model, e)
        return False


async def get_model_info(model: str) -> dict[str, Any] | None:
    """
    Get information about a model.

    Retrieves model metadata including parameters, size, and configuration.

    Args:
        model: Model name to query.

    Returns:
        Dict with model information, or None if not available.

    Example:
        >>> info = await get_model_info("qwen3:14b")
        >>> if info:
        ...     print(info.get("parameters"))
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "ollama",
            "show",
            model,
            "--modelfile",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=DEFAULT_TIMEOUT
        )

        if process.returncode != 0:
            logger.warning("ollama show failed for %s: %s", model, stderr.decode())
            return None

        # Parse modelfile output
        output = stdout.decode()
        info: dict[str, Any] = {
            "name": model,
            "modelfile": output,
        }

        # Extract key parameters from modelfile
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("PARAMETER"):
                parts = line.split(None, 2)
                if len(parts) >= 3:
                    param_name = parts[1]
                    param_value = parts[2]
                    info.setdefault("parameters", {})[param_name] = param_value

        return info

    except FileNotFoundError:
        logger.error("ollama command not found")
        return None
    except TimeoutError:
        logger.error("ollama show timed out for %s", model)
        return None
    except Exception as e:
        logger.exception("Failed to get model info for %s: %s", model, e)
        return None


async def get_model_details(model: str) -> dict[str, Any] | None:
    """
    Get detailed model information in JSON format.

    Uses the Ollama API to get comprehensive model details.

    Args:
        model: Model name to query.

    Returns:
        Dict with detailed model information, or None if not available.
    """
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.ollama_host}/api/show",
                json={"name": model},
                timeout=DEFAULT_TIMEOUT,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(
                    "API show failed for %s: %d", model, response.status_code
                )
                return None

    except ImportError:
        # Fall back to CLI if httpx not available
        return await get_model_info(model)
    except Exception as e:
        logger.exception("Failed to get model details for %s: %s", model, e)
        return None


async def check_model_supports_tools(model: str) -> bool:
    """
    Check if a model supports tool calling.

    Some models may not support the tool/function calling format.
    This performs a simple check to verify compatibility.

    Args:
        model: Model name to check.

    Returns:
        True if model likely supports tools, False otherwise.
    """
    # Known models that support tool calling
    TOOL_CAPABLE_MODELS = [
        "qwen",
        "llama3",
        "llama4",
        "mistral",
        "mixtral",
        "command-r",
        "gemma2",
    ]

    model_lower = model.lower()
    for capable in TOOL_CAPABLE_MODELS:
        if capable in model_lower:
            return True

    # Try to get model info and check for tool support indicators
    details = await get_model_details(model)
    if details:
        template = details.get("template", "")
        if "tools" in template.lower() or "function" in template.lower():
            return True

    return False


async def get_running_models() -> list[dict[str, Any]]:
    """
    Get currently loaded/running models.

    Returns information about models currently loaded in Ollama's memory.

    Returns:
        List of dicts with model info (name, size, etc.).
    """
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ollama_host}/api/ps",
                timeout=DEFAULT_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            else:
                return []

    except Exception as e:
        logger.debug("Failed to get running models: %s", e)
        return []


__all__ = [
    "list_available_models",
    "ensure_model_available",
    "get_model_info",
    "get_model_details",
    "check_model_supports_tools",
    "get_running_models",
]
