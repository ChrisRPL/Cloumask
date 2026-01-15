"""
Prompt templates for AI-powered script generation.

These templates guide the LLM to generate Python scripts that conform
to the Cloumask custom step interface.
"""

SCRIPT_GENERATION_PROMPT = '''You are an expert Python developer creating image/video processing scripts for Cloumask, a computer vision data processing application.

## Script Interface Requirements
Every script MUST follow this exact interface:

```python
"""
{Brief description of what the script does}
"""
from pathlib import Path
from typing import Any

def process(
    input_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Process input file(s) and write results to output.

    Args:
        input_path: Path to input file or directory
        output_path: Path where output should be written
        config: Optional configuration dict from step params

    Returns:
        Dict with at least:
        - success: bool - Whether processing succeeded
        - files_processed: int - Number of files processed
        - Any additional metrics relevant to the operation
    """
    # Implementation here
    pass
```

## Available Libraries
You can use these pre-installed libraries:
- opencv-python (cv2) - Image/video processing, color conversion, filtering
- numpy - Array operations, mathematical computations
- Pillow (PIL) - Image manipulation, format conversion
- pathlib - File path handling (already imported)
- typing - Type hints (already imported)

## Guidelines
1. Always handle both single file and directory inputs:
   - If input_path is a file, process just that file
   - If input_path is a directory, iterate over supported files
2. Preserve original files - write results to output_path only
3. Include proper error handling with try/except blocks
4. Return meaningful metrics in the result dict
5. Use type hints and docstrings
6. Be efficient - avoid loading entire videos into memory
7. Support common image formats: .jpg, .jpeg, .png, .bmp, .tiff
8. Support common video formats: .mp4, .avi, .mov, .mkv (when applicable)

## Config Parameter Usage
The config dict may contain step-specific parameters. Use .get() with defaults:
```python
threshold = config.get("threshold", 0.5) if config else 0.5
```

## Output Format
Return a dict like:
```python
return {
    "success": True,
    "files_processed": count,
    "files_failed": failed_count,
    "output_path": str(output_path),
    # Add any operation-specific metrics
}
```

Generate ONLY the Python script with no additional text before or after.
Wrap the code in ```python ... ``` markers.
'''

SCRIPT_REFINEMENT_PROMPT = '''You are refining an existing Python script for Cloumask.

## Current Script
```python
{existing_code}
```

## Refinement Request
{refinement_request}

## Guidelines
- Maintain the same interface (process function signature)
- Keep existing functionality unless explicitly asked to change it
- Add or modify only what's requested
- Preserve error handling and type hints

Generate the complete refined script wrapped in ```python ... ``` markers.
'''

SCRIPT_EXPLANATION_PROMPT = '''Explain what this script does in 2-3 sentences:

```python
{script_code}
```

Focus on:
1. What transformation/processing it performs
2. What input it expects
3. What output it produces
'''


def format_generation_prompt(user_prompt: str) -> str:
    """
    Format the complete prompt for script generation.

    Args:
        user_prompt: User's natural language description of the script.

    Returns:
        Complete prompt with system instructions and user request.
    """
    return f"{SCRIPT_GENERATION_PROMPT}\n\nCreate a script that: {user_prompt}"


def format_refinement_prompt(existing_code: str, refinement_request: str) -> str:
    """
    Format the prompt for script refinement.

    Args:
        existing_code: The current script code.
        refinement_request: What the user wants to change.

    Returns:
        Complete refinement prompt.
    """
    return SCRIPT_REFINEMENT_PROMPT.format(
        existing_code=existing_code,
        refinement_request=refinement_request,
    )


def format_explanation_prompt(script_code: str) -> str:
    """
    Format the prompt for script explanation.

    Args:
        script_code: The script to explain.

    Returns:
        Explanation prompt.
    """
    return SCRIPT_EXPLANATION_PROMPT.format(script_code=script_code)
