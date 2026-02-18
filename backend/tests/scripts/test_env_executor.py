"""Tests for isolated custom script environment execution."""

from __future__ import annotations

import subprocess
from pathlib import Path

from backend.scripts.env_executor import RESULT_SENTINEL, ScriptEnvironmentExecutor


def _write_process_script(path: Path) -> None:
    path.write_text(
        "from typing import Any\n"
        "def process(input_path: str, output_path: str, config: dict[str, Any] | None = None) -> dict[str, Any]:\n"
        "    return {'success': True, 'files_processed': 1}\n",
        encoding="utf-8",
    )


def _create_fake_venv(env_path: Path) -> None:
    bin_dir = env_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "python").write_text("", encoding="utf-8")
    (bin_dir / "pip").write_text("", encoding="utf-8")


def test_execute_reuses_existing_environment_for_same_context(tmp_path, monkeypatch) -> None:
    """Same script+deps context should reuse env and skip reinstall."""
    script = tmp_path / "custom.py"
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    output_path.mkdir()
    _write_process_script(script)

    executor = ScriptEnvironmentExecutor(
        env_root=tmp_path / "envs",
        python_executable="/opt/python/bin/python3",
    )

    venv_calls = 0
    pip_calls = 0
    script_calls = 0

    def fake_run(
        cmd: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal venv_calls, pip_calls, script_calls
        del capture_output, text, check, timeout

        if cmd[:3] == ["/opt/python/bin/python3", "-m", "venv"]:
            venv_calls += 1
            _create_fake_venv(Path(cmd[3]))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        if cmd[0].endswith("/pip"):
            pip_calls += 1
            return subprocess.CompletedProcess(cmd, 0, "installed", "")

        if cmd[0].endswith("/python") and "-c" in cmd:
            script_calls += 1
            return subprocess.CompletedProcess(
                cmd,
                0,
                f"{RESULT_SENTINEL}{{\"success\": true, \"files_processed\": 1}}\n",
                "",
            )

        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    first = executor.execute(
        script_path=script,
        input_path=input_path,
        output_path=output_path,
        dependencies=["numpy==1.26.4"],
    )
    second = executor.execute(
        script_path=script,
        input_path=input_path,
        output_path=output_path,
        dependencies=["numpy==1.26.4"],
    )

    assert first.success
    assert second.success
    assert venv_calls == 1
    assert pip_calls == 1
    assert script_calls == 2


def test_environment_context_hash_normalizes_dependency_order(tmp_path, monkeypatch) -> None:
    """Dependency order changes should reuse env; changed deps should create a new env."""
    script = tmp_path / "custom.py"
    _write_process_script(script)

    executor = ScriptEnvironmentExecutor(
        env_root=tmp_path / "envs",
        python_executable="/opt/python/bin/python3",
    )

    venv_calls = 0
    pip_calls = 0

    def fake_run(
        cmd: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal venv_calls, pip_calls
        del capture_output, text, check, timeout

        if cmd[:3] == ["/opt/python/bin/python3", "-m", "venv"]:
            venv_calls += 1
            _create_fake_venv(Path(cmd[3]))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        if cmd[0].endswith("/pip"):
            pip_calls += 1
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    env_a = executor.ensure_environment(script, ["numpy==1.26.4", "pandas==2.2.2"])
    env_b = executor.ensure_environment(script, ["pandas==2.2.2", "numpy==1.26.4"])
    env_c = executor.ensure_environment(script, ["numpy==1.26.4"])

    assert env_a == env_b
    assert env_c != env_a
    assert venv_calls == 2
    assert pip_calls == 2


def test_execute_reports_dependency_install_failure(tmp_path, monkeypatch) -> None:
    """Dependency install failure should surface as execution failure."""
    script = tmp_path / "custom.py"
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    output_path.mkdir()
    _write_process_script(script)

    executor = ScriptEnvironmentExecutor(
        env_root=tmp_path / "envs",
        python_executable="/opt/python/bin/python3",
    )

    def fake_run(
        cmd: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, check, timeout

        if cmd[:3] == ["/opt/python/bin/python3", "-m", "venv"]:
            _create_fake_venv(Path(cmd[3]))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        if cmd[0].endswith("/pip"):
            return subprocess.CompletedProcess(
                cmd,
                1,
                "",
                "No matching distribution found for not-a-real-package==0.0.1",
            )

        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = executor.execute(
        script_path=script,
        input_path=input_path,
        output_path=output_path,
        dependencies=["not-a-real-package==0.0.1"],
    )

    assert not result.success
    assert result.error is not None
    assert "Failed to install dependencies" in result.error


def test_execute_reports_script_runtime_error(tmp_path, monkeypatch) -> None:
    """Script process failures should return stderr as error."""
    script = tmp_path / "custom.py"
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    output_path.mkdir()
    _write_process_script(script)

    executor = ScriptEnvironmentExecutor(
        env_root=tmp_path / "envs",
        python_executable="/opt/python/bin/python3",
    )

    def fake_run(
        cmd: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, check, timeout

        if cmd[:3] == ["/opt/python/bin/python3", "-m", "venv"]:
            _create_fake_venv(Path(cmd[3]))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        if cmd[0].endswith("/python") and "-c" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                1,
                "",
                "Traceback (most recent call last):\nRuntimeError: boom",
            )

        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = executor.execute(
        script_path=script,
        input_path=input_path,
        output_path=output_path,
    )

    assert not result.success
    assert result.error is not None
    assert "boom" in result.error
