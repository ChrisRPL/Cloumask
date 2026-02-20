"""
Isolated virtualenv runtime for custom script execution.

Creates and reuses per-context environments, installs requested
dependencies, and executes scripts out-of-process.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

DEFAULT_ENV_ROOT = Path.home() / ".cloumask" / "script-envs"
RESULT_SENTINEL = "__CLOUMASK_RESULT__"

_SCRIPT_RUNNER = (
    "from __future__ import annotations\n"
    "import importlib.util\n"
    "import json\n"
    "import pathlib\n"
    "import traceback\n"
    "import sys\n"
    "\n"
    "script_path = pathlib.Path(sys.argv[1])\n"
    "input_path = sys.argv[2]\n"
    "output_path = sys.argv[3]\n"
    "config_path = pathlib.Path(sys.argv[4])\n"
    "\n"
    "config = {}\n"
    "if config_path.exists():\n"
    "    config = json.loads(config_path.read_text(encoding='utf-8'))\n"
    "\n"
    "spec = importlib.util.spec_from_file_location('cloumask_runtime_script', script_path)\n"
    "if spec is None or spec.loader is None:\n"
    "    print(f'Failed to load script: {script_path}', file=sys.stderr)\n"
    "    sys.exit(1)\n"
    "\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "try:\n"
    "    spec.loader.exec_module(module)\n"
    "    process_func = getattr(module, 'process', None)\n"
    "    if process_func is None or not callable(process_func):\n"
    "        print('Script must define a callable process(input_path, output_path, config)', file=sys.stderr)\n"
    "        sys.exit(1)\n"
    "\n"
    "    result = process_func(input_path, output_path, config)\n"
    "    if not isinstance(result, dict):\n"
    "        print(f\"Script must return a dict, got {type(result).__name__}\", file=sys.stderr)\n"
    "        sys.exit(1)\n"
    "\n"
    f"    print('{RESULT_SENTINEL}' + json.dumps(result))\n"
    "except Exception:\n"
    "    traceback.print_exc()\n"
    "    sys.exit(1)\n"
)


@dataclass
class ScriptExecutionResult:
    """Execution result for isolated script runtime."""

    success: bool
    output: dict[str, Any] | None = None
    error: str | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    execution_time: float = 0.0
    environment_path: str | None = None


class ScriptEnvironmentExecutor:
    """Manage and execute custom scripts in per-context virtualenvs."""

    def __init__(
        self,
        env_root: Path | None = None,
        python_executable: str | None = None,
        install_timeout: int = 300,
    ) -> None:
        self.env_root = (env_root or DEFAULT_ENV_ROOT).expanduser()
        self.python_executable = python_executable or sys.executable
        self.install_timeout = install_timeout
        self.env_root.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        script_path: Path,
        input_path: Path,
        output_path: Path,
        config: dict[str, Any] | None = None,
        dependencies: Sequence[str] | None = None,
        timeout: int = 300,
    ) -> ScriptExecutionResult:
        """Execute a script in an isolated environment."""
        start_time = time.monotonic()
        normalized_deps = self._normalize_dependencies(dependencies)

        try:
            env_path = self.ensure_environment(script_path, normalized_deps)
        except RuntimeError as exc:
            return ScriptExecutionResult(
                success=False,
                error=str(exc),
                execution_time=time.monotonic() - start_time,
            )

        python_bin = self._venv_bin(env_path, "python")
        if not python_bin.exists():
            return ScriptExecutionResult(
                success=False,
                error=f"Isolated environment is missing python executable: {python_bin}",
                execution_time=time.monotonic() - start_time,
                environment_path=str(env_path),
            )

        with tempfile.TemporaryDirectory(prefix="cloumask_script_config_") as temp_dir:
            config_file = Path(temp_dir) / "config.json"
            config_file.write_text(json.dumps(config or {}), encoding="utf-8")

            cmd = [
                str(python_bin),
                "-c",
                _SCRIPT_RUNNER,
                str(script_path),
                str(input_path),
                str(output_path),
                str(config_file),
            ]

            try:
                proc = self._run_command(cmd, timeout=timeout)
            except RuntimeError as exc:
                return ScriptExecutionResult(
                    success=False,
                    error=str(exc),
                    execution_time=time.monotonic() - start_time,
                    environment_path=str(env_path),
                )

        stdout = proc.stdout
        stderr = proc.stderr
        execution_time = time.monotonic() - start_time

        if proc.returncode != 0:
            message = stderr.strip() or stdout.strip() or "Script execution failed"
            return ScriptExecutionResult(
                success=False,
                error=message,
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                execution_time=execution_time,
                environment_path=str(env_path),
            )

        output_payload, output_error = self._parse_output(stdout)
        if output_error:
            return ScriptExecutionResult(
                success=False,
                error=output_error,
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                execution_time=execution_time,
                environment_path=str(env_path),
            )

        return ScriptExecutionResult(
            success=True,
            output=output_payload,
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
            execution_time=execution_time,
            environment_path=str(env_path),
        )

    def ensure_environment(
        self,
        script_path: Path,
        dependencies: Sequence[str] | None = None,
    ) -> Path:
        """Create/reuse isolated environment for a script execution context."""
        normalized_deps = self._normalize_dependencies(dependencies)
        env_path = self.env_path_for(script_path, normalized_deps)

        if not env_path.exists():
            logger.info("Creating isolated script environment: %s", env_path)
            create_proc = self._run_command(
                [self.python_executable, "-m", "venv", str(env_path)],
                timeout=self.install_timeout,
            )
            if create_proc.returncode != 0:
                error = create_proc.stderr.strip() or create_proc.stdout.strip() or "unknown error"
                raise RuntimeError(f"Failed to create isolated environment: {error}")

        if normalized_deps:
            self._install_dependencies(env_path, normalized_deps)

        return env_path

    def env_path_for(
        self,
        script_path: Path,
        dependencies: Sequence[str] | None = None,
    ) -> Path:
        """Compute deterministic env path for a script+dependencies context."""
        normalized_deps = self._normalize_dependencies(dependencies)
        context_material = "\n".join(
            [str(script_path.resolve()), *normalized_deps]
        ).encode("utf-8")
        context_id = hashlib.sha256(context_material).hexdigest()[:16]
        return self.env_root / context_id

    def _install_dependencies(self, env_path: Path, dependencies: Sequence[str]) -> None:
        """Install required dependencies into an environment."""
        if not dependencies:
            return

        deps_hash = hashlib.sha256("\n".join(dependencies).encode("utf-8")).hexdigest()
        hash_marker = env_path / ".deps.hash"

        if hash_marker.exists() and hash_marker.read_text(encoding="utf-8").strip() == deps_hash:
            return

        pip_bin = self._venv_bin(env_path, "pip")
        if not pip_bin.exists():
            raise RuntimeError(f"Isolated environment is missing pip executable: {pip_bin}")

        install_proc = self._run_command(
            [str(pip_bin), "install", "--disable-pip-version-check", *dependencies],
            timeout=self.install_timeout,
        )
        if install_proc.returncode != 0:
            error = install_proc.stderr.strip() or install_proc.stdout.strip() or "unknown error"
            raise RuntimeError(f"Failed to install dependencies in isolated environment: {error}")

        hash_marker.write_text(deps_hash, encoding="utf-8")

    def _run_command(
        self,
        cmd: list[str],
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run command and return completed process with text output."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Command not found: {cmd[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd)}") from exc

    def _parse_output(self, stdout: str) -> tuple[dict[str, Any] | None, str | None]:
        """Parse sentinel result payload from script stdout."""
        for line in reversed(stdout.splitlines()):
            if not line.startswith(RESULT_SENTINEL):
                continue
            payload = line[len(RESULT_SENTINEL):]
            try:
                result = json.loads(payload)
            except json.JSONDecodeError as exc:
                return None, f"Script produced invalid JSON result: {exc}"

            if not isinstance(result, dict):
                return None, "Script result payload must be a JSON object"
            return result, None

        return None, "Script did not emit a result payload"

    def _normalize_dependencies(self, dependencies: Sequence[str] | None) -> list[str]:
        """Normalize dependency strings for deterministic env reuse."""
        if not dependencies:
            return []

        normalized: set[str] = set()
        for dep in dependencies:
            dep_value = str(dep).strip()
            if dep_value:
                normalized.add(dep_value)
        return sorted(normalized)

    def _venv_bin(self, env_path: Path, binary_name: str) -> Path:
        """Resolve binary location in a virtualenv."""
        if sys.platform == "win32":
            return env_path / "Scripts" / f"{binary_name}.exe"
        return env_path / "bin" / binary_name
