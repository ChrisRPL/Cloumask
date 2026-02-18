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
import logging
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
from backend.scripts.env_executor import ScriptEnvironmentExecutor

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
            name="dependencies",
            type=list,
            description="Optional Python dependencies to install in isolated runtime",
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
        self._env_executor: ScriptEnvironmentExecutor | None = None

    @property
    def docker_executor(self) -> DockerScriptExecutor:
        """Get or create Docker executor (lazy initialization)."""
        if self._docker_executor is None:
            self._docker_executor = DockerScriptExecutor()
        return self._docker_executor

    @property
    def env_executor(self) -> ScriptEnvironmentExecutor:
        """Get or create isolated env executor (lazy initialization)."""
        if self._env_executor is None:
            self._env_executor = ScriptEnvironmentExecutor()
        return self._env_executor

    async def execute(
        self,
        script_path: str,
        input_path: str,
        output_path: str,
        config: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
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
            dependencies: Optional dependency list for isolated env install.
            use_docker: Whether to try Docker execution (default: True).

        Returns:
            ToolResult with execution status and output.
        """
        script_p = Path(script_path)
        input_p = Path(input_path)
        output_p = Path(output_path)
        runtime_dependencies = self._extract_dependencies(config, dependencies)

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
        if use_docker and not runtime_dependencies:
            docker_result = await self._execute_in_docker(
                script_p, input_p, output_p, config
            )
            if docker_result is not None:
                return docker_result
            logger.info("Docker unavailable, falling back to local execution")
        elif runtime_dependencies:
            logger.info(
                "Skipping Docker path for %s: runtime dependencies requested",
                script_p.name,
            )

        # Fall back to local execution
        try:
            result = await asyncio.to_thread(
                self._execute_script,
                script_p,
                input_p,
                output_p,
                config,
                runtime_dependencies,
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
        dependencies: list[str],
    ) -> ToolResult:
        """
        Execute script in a per-context isolated virtualenv.

        Called from thread pool to avoid blocking the event loop.
        """
        self.report_progress(0, 100, "Starting isolated custom script runtime...")
        logger.info(
            "Executing custom script in isolated env: %s (input=%s, output=%s)",
            script_path.name,
            input_path,
            output_path,
        )

        runtime_result = self.env_executor.execute(
            script_path=script_path,
            input_path=input_path,
            output_path=output_path,
            config=config,
            dependencies=dependencies,
        )

        self.report_progress(100, 100, "Isolated script execution completed")

        if not runtime_result.success:
            return error_result(
                runtime_result.error or "Custom script execution failed in isolated environment"
            )

        result = runtime_result.output or {}
        if not isinstance(result, dict):
            return error_result(f"Script must return a dict, got {type(result).__name__}")

        if not result.get("success", False):
            error_msg = result.get("error", "Script reported failure")
            return error_result(error_msg)

        return success_result(
            {
                **result,
                "script_name": script_path.name,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "execution_time": runtime_result.execution_time,
                "execution_mode": "isolated_env",
                "environment_path": runtime_result.environment_path,
                "dependencies": dependencies,
            },
            script=script_path.name,
        )

    def _extract_dependencies(
        self,
        config: dict[str, Any] | None,
        dependencies: list[str] | None,
    ) -> list[str]:
        """Collect dependency hints from explicit args and config payload."""
        collected: list[str] = []

        if dependencies:
            collected.extend(str(dep).strip() for dep in dependencies if str(dep).strip())

        if config:
            for key in ("dependencies", "python_dependencies", "requirements"):
                value = config.get(key)
                if isinstance(value, list):
                    collected.extend(str(dep).strip() for dep in value if str(dep).strip())
                elif isinstance(value, str):
                    # Support comma/newline-separated dependency strings.
                    for dep in value.replace(",", "\n").splitlines():
                        dep_value = dep.strip()
                        if dep_value:
                            collected.append(dep_value)

        return sorted(set(collected))


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
