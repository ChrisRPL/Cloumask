"""
Docker-based script execution for isolated, secure processing.

Runs user scripts inside Docker containers with:
- No network access (security isolation)
- Memory/CPU limits
- Pre-installed CV packages
- Read-only script mounting
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Docker image for script execution
EXECUTOR_IMAGE = "cloumask-executor:latest"

# Resource limits
DEFAULT_MEMORY_LIMIT = "2g"
DEFAULT_CPU_LIMIT = "2"
DEFAULT_TIMEOUT = 300  # 5 minutes


@dataclass
class ExecutionResult:
    """Result of a Docker script execution."""

    success: bool
    """Whether the script executed successfully."""

    output: dict[str, Any] | None = None
    """Script output data (if successful)."""

    error: str | None = None
    """Error message (if failed)."""

    stdout: str = ""
    """Standard output from the container."""

    stderr: str = ""
    """Standard error from the container."""

    exit_code: int = -1
    """Container exit code."""

    execution_time: float = 0.0
    """Execution time in seconds."""


@dataclass
class ExecutionConfig:
    """Configuration for Docker script execution."""

    memory_limit: str = DEFAULT_MEMORY_LIMIT
    """Memory limit (e.g., '2g', '512m')."""

    cpu_limit: str = DEFAULT_CPU_LIMIT
    """CPU limit (e.g., '2' for 2 cores)."""

    timeout: int = DEFAULT_TIMEOUT
    """Execution timeout in seconds."""

    enable_gpu: bool = False
    """Whether to enable GPU access."""

    environment: dict[str, str] = field(default_factory=dict)
    """Additional environment variables."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        import re

        # Validate memory limit format (e.g., '2g', '512m', '1024k')
        if not re.match(r"^\d+[kmgKMG]?$", self.memory_limit):
            raise ValueError(f"Invalid memory limit format: {self.memory_limit}")

        # Validate CPU limit format (e.g., '2', '0.5', '1.5')
        if not re.match(r"^\d+(\.\d+)?$", self.cpu_limit):
            raise ValueError(f"Invalid CPU limit format: {self.cpu_limit}")

        # Validate timeout is positive
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive: {self.timeout}")

        # Validate environment variable names
        for key in self.environment:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                raise ValueError(f"Invalid environment variable name: {key}")


class DockerScriptExecutor:
    """
    Executes Python scripts inside isolated Docker containers.

    Provides secure execution environment with:
    - Network isolation (--network none)
    - Resource limits (memory, CPU)
    - Read-only script access
    - Temporary output directory

    Example:
        executor = DockerScriptExecutor()
        result = await executor.execute(
            script_path=Path("/scripts/process.py"),
            input_path=Path("/data/input"),
            output_path=Path("/data/output"),
        )
        if result.success:
            print(result.output)
    """

    def __init__(self, image: str = EXECUTOR_IMAGE) -> None:
        """
        Initialize the executor.

        Args:
            image: Docker image to use for execution.
        """
        self.image = image
        self._docker_available: bool | None = None

    async def check_docker_available(self) -> bool:
        """
        Check if Docker is available and the executor image exists.

        Returns:
            True if Docker is ready for script execution.
        """
        if self._docker_available is not None:
            return self._docker_available

        try:
            # Check Docker daemon
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

            if proc.returncode != 0:
                logger.warning("Docker daemon not available")
                self._docker_available = False
                return False

            # Check if executor image exists
            proc = await asyncio.create_subprocess_exec(
                "docker", "image", "inspect", self.image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

            if proc.returncode != 0:
                logger.warning("Executor image not found: %s", self.image)
                self._docker_available = False
                return False

            self._docker_available = True
            return True

        except FileNotFoundError:
            logger.warning("Docker command not found")
            self._docker_available = False
            return False

    async def build_image(self, dockerfile_path: Path | None = None) -> bool:
        """
        Build the executor Docker image.

        Args:
            dockerfile_path: Path to Dockerfile directory.
                           If None, uses default location.

        Returns:
            True if build succeeded.
        """
        if dockerfile_path is None:
            # Default to project's docker/executor directory
            dockerfile_path = Path(__file__).parents[4] / "docker" / "executor"

        if not dockerfile_path.exists():
            logger.error("Dockerfile not found at: %s", dockerfile_path)
            return False

        logger.info("Building executor image: %s", self.image)

        proc = await asyncio.create_subprocess_exec(
            "docker", "build",
            "-t", self.image,
            str(dockerfile_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error("Image build failed: %s", stderr.decode())
            return False

        logger.info("Executor image built successfully")
        self._docker_available = None  # Reset cache
        return True

    async def execute(
        self,
        script_path: Path,
        input_path: Path,
        output_path: Path,
        config: dict[str, Any] | None = None,
        execution_config: ExecutionConfig | None = None,
    ) -> ExecutionResult:
        """
        Execute a script inside a Docker container.

        Args:
            script_path: Path to Python script to execute.
            input_path: Path to input file or directory.
            output_path: Path where output should be written.
            config: Configuration parameters to pass to script.
            execution_config: Docker execution settings.

        Returns:
            ExecutionResult with success/failure and output data.
        """
        import time
        start_time = time.monotonic()

        if execution_config is None:
            execution_config = ExecutionConfig()

        # Validate paths - prevent path traversal and symlink attacks
        if not script_path.exists():
            return ExecutionResult(
                success=False,
                error=f"Script not found: {script_path}",
            )

        if not input_path.exists():
            return ExecutionResult(
                success=False,
                error=f"Input not found: {input_path}",
            )

        # Security: Reject symlinks to prevent escape attacks
        if script_path.is_symlink():
            return ExecutionResult(
                success=False,
                error="Script path cannot be a symlink",
            )

        if input_path.is_symlink():
            return ExecutionResult(
                success=False,
                error="Input path cannot be a symlink",
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check Docker availability
        if not await self.check_docker_available():
            return ExecutionResult(
                success=False,
                error="Docker not available. Run setup wizard to configure.",
            )

        # Create temporary directory for container communication
        with tempfile.TemporaryDirectory(prefix="cloumask_exec_") as temp_dir:
            temp_path = Path(temp_dir)

            # Write config to temp file
            config_file = temp_path / "config.json"
            config_file.write_text(json.dumps(config or {}))

            # Create output dir inside temp for container to write to
            container_output = temp_path / "output"
            container_output.mkdir()

            # Build Docker command
            container_name = f"cloumask-exec-{uuid.uuid4().hex[:8]}"
            cmd = self._build_docker_command(
                container_name=container_name,
                script_path=script_path,
                input_path=input_path,
                config_file=config_file,
                output_dir=container_output,
                execution_config=execution_config,
            )

            logger.info("Executing script in Docker: %s", script_path.name)
            logger.debug("Docker command: %s", " ".join(cmd))

            try:
                # Run container with timeout
                # Note: Using create_subprocess_exec (not shell) for security
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=execution_config.timeout,
                    )
                except TimeoutError:
                    # Kill the container
                    await self._kill_container(container_name)
                    return ExecutionResult(
                        success=False,
                        error=f"Execution timed out after {execution_config.timeout}s",
                        exit_code=-1,
                        execution_time=time.monotonic() - start_time,
                    )

                execution_time = time.monotonic() - start_time

                # Read result file if it exists
                result_file = container_output / "result.json"
                output_data = None

                if result_file.exists():
                    try:
                        output_data = json.loads(result_file.read_text())
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse result.json: %s", e)

                # Copy output files to final destination
                if proc.returncode == 0:
                    await self._copy_outputs(container_output, output_path)

                return ExecutionResult(
                    success=proc.returncode == 0,
                    output=output_data,
                    error=stderr.decode() if proc.returncode != 0 else None,
                    stdout=stdout.decode(),
                    stderr=stderr.decode(),
                    exit_code=proc.returncode or 0,
                    execution_time=execution_time,
                )

            except Exception as e:
                logger.exception("Docker execution failed: %s", e)
                return ExecutionResult(
                    success=False,
                    error=str(e),
                    execution_time=time.monotonic() - start_time,
                )

    def _build_docker_command(
        self,
        container_name: str,
        script_path: Path,
        input_path: Path,
        config_file: Path,
        output_dir: Path,
        execution_config: ExecutionConfig,
    ) -> list[str]:
        """Build the docker run command."""
        cmd = [
            "docker", "run",
            "--rm",  # Remove container after exit
            "--name", container_name,
            "--network", "none",  # No network access (security)
            "--memory", execution_config.memory_limit,
            "--cpus", execution_config.cpu_limit,
            # Mount script as read-only
            "-v", f"{script_path.resolve()}:/workspace/script.py:ro",
            # Mount input as read-only
            "-v", f"{input_path.resolve()}:/workspace/input:ro",
            # Mount config as read-only
            "-v", f"{config_file.resolve()}:/workspace/config.json:ro",
            # Mount output directory (writable)
            "-v", f"{output_dir.resolve()}:/workspace/output",
            # Set working directory
            "-w", "/workspace",
        ]

        # Add environment variables
        for key, value in execution_config.environment.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Add GPU support if enabled
        if execution_config.enable_gpu:
            cmd.extend(["--gpus", "all"])

        # Image and command
        cmd.extend([
            self.image,
            "python", "/workspace/script.py",
            "--input", "/workspace/input",
            "--output", "/workspace/output",
            "--config", "/workspace/config.json",
        ])

        return cmd

    async def _kill_container(self, container_name: str) -> None:
        """Kill a running container."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "kill", container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception as e:
            logger.warning("Failed to kill container %s: %s", container_name, e)

    async def _copy_outputs(self, source: Path, dest: Path) -> None:
        """Copy output files from temp directory to final destination."""
        import shutil

        # Skip result.json (internal use only)
        for item in source.iterdir():
            if item.name == "result.json":
                continue

            dest_item = dest / item.name
            if item.is_dir():
                shutil.copytree(item, dest_item, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest_item)


# Convenience function
async def execute_script_in_docker(
    script_path: Path | str,
    input_path: Path | str,
    output_path: Path | str,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ExecutionResult:
    """
    Execute a script in Docker (convenience function).

    Args:
        script_path: Path to Python script.
        input_path: Path to input data.
        output_path: Path for output.
        config: Script configuration.
        **kwargs: Additional ExecutionConfig parameters.

    Returns:
        ExecutionResult with execution outcome.
    """
    executor = DockerScriptExecutor()
    exec_config = ExecutionConfig(**kwargs) if kwargs else None

    return await executor.execute(
        script_path=Path(script_path),
        input_path=Path(input_path),
        output_path=Path(output_path),
        config=config,
        execution_config=exec_config,
    )
