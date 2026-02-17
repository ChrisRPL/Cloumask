"""
Custom script execution tool for user-defined processing.

Allows users to run custom Python scripts that conform to the
Cloumask script interface as pipeline steps.

Supports two execution modes:
1. Docker execution (secure, isolated) - preferred when Docker is available
2. Local execution (fallback) - for development or when Docker is unavailable
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.registry import register_tool
from backend.scripts.docker_executor import DockerScriptExecutor, ExecutionConfig

logger = logging.getLogger(__name__)


@register_tool
class CustomScriptTool(BaseTool):
    """Execute user-defined Python scripts for custom processing."""

    name = "custom_script"
    description = """Execute a custom Python script for specialized data processing.
The script must define a process(input_path, output_path, config) function.
Use this for custom transformations not covered by built-in tools."""
    category = ToolCategory.UTILITY

    parameters = [
        ToolParameter(
            name="script_path",
            type=str,
            description="Path to the Python script to execute",
            required=True,
        ),
        ToolParameter(
            name="input_path",
            type=str,
            description="Path to input file or directory to process",
            required=True,
        ),
        ToolParameter(
            name="output_path",
            type=str,
            description="Path where output should be written",
            required=True,
        ),
        ToolParameter(
            name="config",
            type=dict,
            description="Configuration parameters to pass to the script",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="use_docker",
            type=bool,
            description="Whether to use Docker for isolated execution (default: True)",
            required=False,
            default=True,
        ),
    ]

    def __init__(self) -> None:
        """Initialize with Docker executor."""
        super().__init__()
        self._docker_executor: DockerScriptExecutor | None = None

    @property
    def docker_executor(self) -> DockerScriptExecutor:
        """Get or create Docker executor (lazy initialization)."""
        if self._docker_executor is None:
            self._docker_executor = DockerScriptExecutor()
        return self._docker_executor

    async def execute(
        self,
        script_path: str,
        input_path: str,
        output_path: str,
        config: dict[str, Any] | None = None,
        use_docker: bool = True,
    ) -> ToolResult:
        """
        Execute the custom script.

        Tries Docker execution first (if enabled and available), then falls
        back to local execution. Docker provides isolated, secure execution.

        Args:
            script_path: Path to Python script.
            input_path: Path to input file/directory.
            output_path: Path for output.
            config: Script configuration.
            use_docker: Whether to try Docker execution (default: True).

        Returns:
            ToolResult with execution status and output.
        """
        script_p = Path(script_path)
        input_p = Path(input_path)
        output_p = Path(output_path)

        # Validate script exists
        if not script_p.exists():
            return error_result(f"Script not found: {script_path}")

        if script_p.suffix != ".py":
            return error_result("Script must be a .py file")

        # Validate input exists
        if not input_p.exists():
            return error_result(f"Input not found: {input_path}")

        # Ensure output directory exists
        if output_p.suffix:
            # output_path is a file, ensure parent exists
            output_p.parent.mkdir(parents=True, exist_ok=True)
        else:
            # output_path is a directory
            output_p.mkdir(parents=True, exist_ok=True)

        # Try Docker execution first (preferred for security)
        if use_docker:
            docker_result = await self._execute_in_docker(
                script_p, input_p, output_p, config
            )
            if docker_result is not None:
                return docker_result
            logger.info("Docker unavailable, falling back to local execution")

        # Fall back to local execution
        try:
            result = await asyncio.to_thread(
                self._execute_script,
                script_p,
                input_p,
                output_p,
                config,
            )
            return result

        except Exception as e:
            logger.exception("Custom script execution failed: %s", e)
            return error_result(f"Script execution failed: {e}")

    async def _execute_in_docker(
        self,
        script_path: Path,
        input_path: Path,
        output_path: Path,
        config: dict[str, Any] | None,
    ) -> ToolResult | None:
        """
        Execute script in Docker container.

        Returns:
            ToolResult if Docker execution succeeded or failed definitively.
            None if Docker is unavailable (caller should fall back to local).
        """
        # Check if Docker is available
        if not await self.docker_executor.check_docker_available():
            return None

        self.report_progress(0, 100, "Starting script in Docker...")

        exec_config = ExecutionConfig(
            memory_limit="2g",
            cpu_limit="2",
            timeout=300,
        )

        result = await self.docker_executor.execute(
            script_path=script_path,
            input_path=input_path,
            output_path=output_path,
            config=config,
            execution_config=exec_config,
        )

        self.report_progress(100, 100, "Docker execution completed")

        if result.success:
            return success_result(
                {
                    "output": result.output,
                    "script_name": script_path.name,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "execution_time": result.execution_time,
                    "execution_mode": "docker",
                },
                script=script_path.name,
            )
        else:
            return error_result(
                f"Docker execution failed: {result.error}"
                + (f"\nstderr: {result.stderr}" if result.stderr else "")
            )

    def _execute_script(
        self,
        script_path: Path,
        input_path: Path,
        output_path: Path,
        config: dict[str, Any] | None,
    ) -> ToolResult:
        """
        Execute the script synchronously (called from thread pool).

        Loads the script as a module and calls its process() function.
        """
        module_name = f"cloumask_custom_{script_path.stem}"

        try:
            # Load script as module
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None or spec.loader is None:
                return error_result(f"Failed to load script: {script_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Verify interface
            if not hasattr(module, "process"):
                return error_result(
                    "Script must define a 'process(input_path, output_path, config)' function"
                )

            process_func = module.process
            if not callable(process_func):
                return error_result("'process' must be a callable function")

            # Report progress start
            self.report_progress(0, 100, "Starting custom script...")

            # Execute the script
            logger.info(
                "Executing custom script: %s (input=%s, output=%s)",
                script_path.name,
                input_path,
                output_path,
            )

            result = process_func(
                str(input_path),
                str(output_path),
                config,
            )

            # Report progress complete
            self.report_progress(100, 100, "Script completed")

            # Validate result
            if not isinstance(result, dict):
                return error_result(
                    f"Script must return a dict, got {type(result).__name__}"
                )

            # Check for success flag
            if not result.get("success", False):
                error_msg = result.get("error", "Script reported failure")
                return error_result(error_msg)

            # Build success result with script metadata
            return success_result(
                {
                    **result,
                    "script_name": script_path.name,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                },
                script=script_path.name,
            )

        finally:
            # Clean up loaded module
            if module_name in sys.modules:
                del sys.modules[module_name]


def validate_script_interface(script_path: str | Path) -> tuple[bool, str | None]:
    """
    Validate that a script has the required interface.

    Args:
        script_path: Path to the script to validate.

    Returns:
        Tuple of (is_valid, error_message or None).
    """
    script_p = Path(script_path)

    if not script_p.exists():
        return False, f"Script not found: {script_path}"

    if script_p.suffix != ".py":
        return False, "Script must be a .py file"

    try:
        # Read and compile to check syntax
        code = script_p.read_text(encoding="utf-8")
        compile(code, str(script_path), "exec")

        # Check for process function definition
        if "def process(" not in code:
            return False, "Script must define a process() function"

        return True, None

    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Validation failed: {e}"


@register_tool
class RunScriptToolAlias(CustomScriptTool):
    """Backward-compatible alias for planner steps that still use `run_script`."""

    name = "run_script"
