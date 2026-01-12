"""
Tool discovery and initialization for the Cloumask agent.

This module provides functions to auto-discover and register tools
from Python packages at startup.

Implements spec: 06-tool-system
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
from pathlib import Path

from backend.agent.tools.registry import get_tool_registry

logger = logging.getLogger(__name__)


def discover_tools(package_path: str = "backend.agent.tools") -> list[str]:
    """
    Auto-discover and register tools from a package.

    Scans the specified package for modules and imports them.
    Tools decorated with @register_tool are automatically registered
    when their modules are imported.

    Args:
        package_path: Dot-separated package path to scan for tools.

    Returns:
        List of module names that were discovered and imported.

    Example:
        # Discovers all tools in backend/agent/tools/
        discovered = discover_tools("backend.agent.tools")
    """
    discovered_modules: list[str] = []

    try:
        package = importlib.import_module(package_path)
    except ImportError as e:
        logger.warning("Could not import package %s: %s", package_path, e)
        return discovered_modules

    # Get package directory
    if not hasattr(package, "__file__") or package.__file__ is None:
        logger.warning("Package %s has no __file__ attribute", package_path)
        return discovered_modules

    package_dir = Path(package.__file__).parent

    # Modules that should never be reloaded (would break singletons or cause issues)
    skip_modules = {"registry", "base", "constants", "discovery"}

    # Iterate through all modules in the package
    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        # Skip private modules, sub-packages, and system modules
        if module_name.startswith("_") or module_name in skip_modules:
            continue

        full_name = f"{package_path}.{module_name}"

        try:
            # Use reload() for already-imported modules to re-trigger decorators
            if full_name in sys.modules:
                importlib.reload(sys.modules[full_name])
            else:
                importlib.import_module(full_name)
            discovered_modules.append(module_name)
            logger.debug("Discovered tool module: %s", full_name)

            # Recursively discover tools in sub-packages
            if is_pkg:
                sub_modules = discover_tools(full_name)
                discovered_modules.extend([f"{module_name}.{m}" for m in sub_modules])

        except ImportError as e:
            logger.warning("Could not import tool module %s: %s", full_name, e)
        except Exception as e:
            logger.exception("Error importing tool module %s: %s", full_name, e)

    return discovered_modules


def initialize_tools(
    package_paths: list[str] | None = None,
    clear_existing: bool = False,
) -> int:
    """
    Initialize the tool system on startup.

    Optionally clears existing registrations and discovers tools from
    specified packages. Default package is "backend.agent.tools".

    Args:
        package_paths: List of package paths to scan. If None, uses default.
        clear_existing: If True, clears existing registrations first.

    Returns:
        Number of tools registered after initialization.

    Example:
        # Initialize with defaults
        count = initialize_tools()

        # Initialize with custom packages
        count = initialize_tools(["backend.agent.tools", "custom.tools"])
    """
    registry = get_tool_registry()

    if clear_existing:
        registry.clear()

    if package_paths is None:
        package_paths = ["backend.agent.tools"]

    for package_path in package_paths:
        discover_tools(package_path)

    tool_count = len(registry)
    logger.info("Tool system initialized with %d tools", tool_count)

    return tool_count


def reload_tools(package_paths: list[str] | None = None) -> int:
    """
    Reload all tools by clearing and re-discovering.

    Useful for development when tool modules have been modified.

    Args:
        package_paths: List of package paths to scan. If None, uses default.

    Returns:
        Number of tools registered after reload.
    """
    return initialize_tools(package_paths, clear_existing=True)


def list_available_tools() -> list[dict[str, str]]:
    """
    List all registered tools with their descriptions.

    Returns:
        List of dicts with 'name', 'description', and 'category'.
    """
    registry = get_tool_registry()
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
        }
        for tool in registry.get_all()
    ]
