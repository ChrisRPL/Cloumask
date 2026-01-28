#!/usr/bin/env python3
"""
Cloumask script executor entrypoint.

Wraps user scripts to provide a standard interface:
- Parses CLI arguments (--input, --output, --config)
- Calls the script's process() function
- Writes result.json with execution status
"""

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path


def load_script(script_path: Path):
    """Load a Python script as a module."""
    spec = importlib.util.spec_from_file_location("user_script", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_script"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Execute Cloumask script")
    parser.add_argument("script", help="Path to Python script")
    parser.add_argument("--input", required=True, help="Input path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--config", required=True, help="Config JSON path")
    args = parser.parse_args()

    script_path = Path(args.script)
    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)

    # Load config
    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        write_result(output_path, success=False, error=f"Failed to load config: {e}")
        sys.exit(1)

    # Load and execute script
    try:
        module = load_script(script_path)

        if not hasattr(module, "process"):
            write_result(
                output_path,
                success=False,
                error="Script must define a process(input_path, output_path, config) function",
            )
            sys.exit(1)

        # Call the script's process function
        result = module.process(str(input_path), str(output_path), config)

        # Validate result
        if not isinstance(result, dict):
            write_result(
                output_path,
                success=False,
                error=f"process() must return a dict, got {type(result).__name__}",
            )
            sys.exit(1)

        # Write result
        success = result.get("success", False)
        write_result(output_path, success=success, data=result)
        sys.exit(0 if success else 1)

    except Exception as e:
        tb = traceback.format_exc()
        write_result(output_path, success=False, error=str(e), traceback=tb)
        sys.exit(1)


def write_result(
    output_path: Path,
    success: bool,
    data: dict | None = None,
    error: str | None = None,
    traceback: str | None = None,
):
    """Write execution result to result.json."""
    result = {
        "success": success,
        **(data or {}),
    }
    if error:
        result["error"] = error
    if traceback:
        result["traceback"] = traceback

    result_file = output_path / "result.json"
    result_file.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
