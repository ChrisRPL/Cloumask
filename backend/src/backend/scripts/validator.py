"""
Script validation using Python's AST module.

Validates script syntax, checks for required interface, and
identifies potential issues before execution.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of script validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_process_function: bool = False
    imports: list[str] = field(default_factory=list)


class ScriptValidator:
    """
    Validates Python scripts for the Cloumask custom step interface.

    Performs:
    - Syntax validation via AST parsing
    - Interface compliance checking (process function)
    - Import analysis
    - Common issue detection
    """

    # Required parameters for the process function
    REQUIRED_PARAMS = {"input_path", "output_path"}

    # Common safe imports that are available
    AVAILABLE_IMPORTS = {
        "pathlib",
        "typing",
        "cv2",
        "numpy",
        "PIL",
        "os",
        "json",
        "logging",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "re",
        "math",
        "shutil",
    }

    # Patterns to warn about (for detection, not execution)
    WARN_PATTERNS = {"subprocess", "eval", "exec", "compile", "importlib"}

    def validate(self, code: str) -> ValidationResult:
        """
        Validate a Python script.

        Args:
            code: Python source code to validate.

        Returns:
            ValidationResult with validity status, errors, and warnings.
        """
        result = ValidationResult(valid=True)

        # Parse the AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result.valid = False
            result.errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return result

        # Analyze the AST
        self._check_process_function(tree, result)
        self._check_imports(tree, result)
        self._check_dangerous_patterns(tree, result)

        # Final validity check
        if not result.has_process_function:
            result.valid = False
            result.errors.append(
                "Script must define a 'process(input_path, output_path, config)' function"
            )

        return result

    def _check_process_function(self, tree: ast.AST, result: ValidationResult) -> None:
        """Check for required process function with correct signature."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "process":
                result.has_process_function = True

                # Check parameters
                param_names = {arg.arg for arg in node.args.args}

                missing_params = self.REQUIRED_PARAMS - param_names
                if missing_params:
                    result.errors.append(
                        f"process() missing required parameters: {', '.join(missing_params)}"
                    )
                    result.valid = False

                # Check for config parameter (optional but recommended)
                if "config" not in param_names:
                    result.warnings.append(
                        "process() should accept a 'config' parameter for step configuration"
                    )

                # Check for return statement
                has_return = any(
                    isinstance(child, ast.Return) for child in ast.walk(node)
                )
                if not has_return:
                    result.warnings.append(
                        "process() should return a dict with 'success' and 'files_processed' keys"
                    )

                break

    def _check_imports(self, tree: ast.AST, result: ValidationResult) -> None:
        """Analyze imports and check availability."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    result.imports.append(module_name)

                    if module_name not in self.AVAILABLE_IMPORTS:
                        result.warnings.append(
                            f"Import '{module_name}' may not be available - "
                            "ensure it's installed in the environment"
                        )

            elif isinstance(node, ast.ImportFrom) and node.module:
                module_name = node.module.split(".")[0]
                result.imports.append(module_name)

                if module_name not in self.AVAILABLE_IMPORTS:
                    result.warnings.append(
                        f"Import from '{module_name}' may not be available"
                    )

    def _check_dangerous_patterns(self, tree: ast.AST, result: ValidationResult) -> None:
        """Check for potentially dangerous code patterns."""
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name in self.WARN_PATTERNS:
                    result.warnings.append(
                        f"Potentially dangerous function call: {func_name}"
                    )

            # Check for dynamic code execution
            if isinstance(node, ast.Name) and node.id in {"exec", "eval"}:
                result.warnings.append(
                    f"Use of '{node.id}' detected - review carefully"
                )

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""

    def validate_syntax_only(self, code: str) -> tuple[bool, str | None]:
        """
        Quick syntax-only validation.

        Args:
            code: Python source code.

        Returns:
            Tuple of (is_valid, error_message or None).
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"


def validate_script(code: str) -> ValidationResult:
    """
    Convenience function to validate a script.

    Args:
        code: Python source code to validate.

    Returns:
        ValidationResult with validation status.
    """
    return ScriptValidator().validate(code)
