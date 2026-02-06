---
name: langgraph-tool-development
description: Guide for creating LangGraph agent tools following Cloumask patterns. Use when adding new agent capabilities, creating tools for the agent, or debugging tool execution.
---

# LangGraph Tool Development

## Quick Start

When creating a new agent tool:

1. Create tool class inheriting from `BaseTool`
2. Define `name`, `description`, `category`, and `parameters`
3. Implement `execute()` method
4. Use `@register_tool` decorator for auto-registration
5. Return `ToolResult` with structured data

## BaseTool Pattern

All tools must inherit from `BaseTool`:

```python
from backend.agent.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolResult,
    error_result,
    success_result,
)
from backend.agent.tools.registry import register_tool

@register_tool
class MyTool(BaseTool):
    """Brief description of what this tool does."""
    
    name = "my_tool"
    description = """Detailed description shown to the LLM.
    
    Explain what the tool does, when to use it, and any important notes.
    Include examples if helpful."""
    category = ToolCategory.UTILITY
    
    parameters = [
        ToolParameter(
            name="input_path",
            type=str,
            description="Path to input file or directory",
            required=True,
        ),
        ToolParameter(
            name="confidence",
            type=float,
            description="Confidence threshold (0-1)",
            required=False,
            default=0.5,
        ),
    ]
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with provided parameters."""
        try:
            # Validate inputs
            input_path = kwargs.get("input_path")
            if not input_path:
                return error_result("input_path is required")
            
            confidence = kwargs.get("confidence", 0.5)
            
            # Perform work
            result_data = self._do_work(input_path, confidence)
            
            # Return success result
            return success_result({
                "count": result_data.count,
                "items": result_data.items,
            })
        except Exception as e:
            logger.error("Tool execution failed: %s", e)
            return error_result(f"Execution failed: {e}")
```

## Tool Parameters

Define parameters with `ToolParameter`:

```python
parameters = [
    ToolParameter(
        name="param_name",
        type=str,  # str, int, float, bool, list, dict
        description="What this parameter does",
        required=True,  # or False
        default=None,  # Default value if not required
        enum_values=["option1", "option2"],  # Optional enum constraint
    ),
]
```

## Tool Categories

Use appropriate category from `ToolCategory`:

- `ToolCategory.SCAN` - Directory scanning, file discovery
- `ToolCategory.DETECTION` - Object detection
- `ToolCategory.SEGMENTATION` - Image segmentation
- `ToolCategory.ANONYMIZATION` - Privacy/anonymization
- `ToolCategory.EXPORT` - Data export, format conversion
- `ToolCategory.UTILITY` - General utilities

## Result Format

Always return `ToolResult`:

```python
# Success result
return success_result({
    "key": "value",
    "count": 42,
    "items": [...],
})

# Error result
return error_result("Descriptive error message")
```

## Progress Reporting

For long-running operations, use progress callbacks:

```python
def execute(self, progress_callback: ProgressCallback | None = None, **kwargs) -> ToolResult:
    total = 100
    
    if progress_callback:
        progress_callback(0, total, "Starting...")
    
    # Do work
    for i, item in enumerate(items):
        process_item(item)
        
        if progress_callback:
            progress_callback(i + 1, total, f"Processing {i+1}/{total}")
    
    if progress_callback:
        progress_callback(total, total, "Complete")
    
    return success_result({"processed": len(items)})
```

## Error Handling

Always catch exceptions and return error results:

```python
def execute(self, **kwargs) -> ToolResult:
    try:
        # Validate inputs
        path = kwargs.get("path")
        if not path or not Path(path).exists():
            return error_result(f"Path does not exist: {path}")
        
        # Perform operation
        result = risky_operation(path)
        
        return success_result({"result": result})
    except FileNotFoundError as e:
        return error_result(f"File not found: {e}")
    except ValueError as e:
        return error_result(f"Invalid input: {e}")
    except Exception as e:
        logger.exception("Unexpected error in tool execution")
        return error_result(f"Unexpected error: {e}")
```

## Using CV Models

Access models through `ModelManager`:

```python
from backend.cv import ModelManager

def execute(self, **kwargs) -> ToolResult:
    manager = ModelManager()
    
    try:
        # Get model (lazy-loaded)
        model = manager.get("yolo11m")
        
        # Run inference
        result = model.predict(kwargs["image_path"])
        
        return success_result({
            "detections": result.detections,
            "count": result.count,
        })
    finally:
        # Cleanup
        manager.unload("yolo11m")
```

## File Path Validation

Always validate file paths:

```python
from pathlib import Path

def execute(self, **kwargs) -> ToolResult:
    input_path = Path(kwargs.get("input_path", ""))
    
    if not input_path.exists():
        return error_result(f"Path does not exist: {input_path}")
    
    if not input_path.is_file():
        return error_result(f"Path is not a file: {input_path}")
    
    # Process file
    ...
```

## Batch Processing

For processing multiple items:

```python
def execute(self, **kwargs) -> ToolResult:
    input_paths = kwargs.get("input_paths", [])
    
    if not input_paths:
        return error_result("No input paths provided")
    
    results = []
    errors = []
    
    for path in input_paths:
        try:
            result = process_single(path)
            results.append(result)
        except Exception as e:
            errors.append({"path": path, "error": str(e)})
    
    return success_result({
        "processed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    })
```

## Tool Registration

Tools are auto-registered with `@register_tool` decorator. Tools are discovered automatically when their module is imported.

For manual registration:

```python
from backend.agent.tools.registry import get_tool_registry

registry = get_tool_registry()
registry.register(MyTool())
```

## Testing Tools

Test tools independently:

```python
def test_my_tool():
    tool = MyTool()
    
    result = tool.execute(input_path="test.jpg", confidence=0.5)
    
    assert result.success
    assert "count" in result.data
    assert result.data["count"] > 0
```

## Tool Schema Generation

Tool schemas are automatically generated for LLM tool calling:

```python
# Get tool schema
tool = MyTool()
schema = tool.get_schema()

# Schema includes:
# - name: tool name
# - description: tool description
# - parameters: JSON schema for parameters
```

## Common Patterns

### Directory Scanning

```python
@register_tool
class ScanDirectoryTool(BaseTool):
    name = "scan_directory"
    description = "Scan directory for image files"
    category = ToolCategory.SCAN
    
    parameters = [
        ToolParameter("path", str, "Directory to scan", required=True),
        ToolParameter("recursive", bool, "Scan subdirectories", default=True),
    ]
    
    def execute(self, **kwargs) -> ToolResult:
        path = Path(kwargs["path"])
        recursive = kwargs.get("recursive", True)
        
        if not path.is_dir():
            return error_result(f"Not a directory: {path}")
        
        pattern = "**/*" if recursive else "*"
        images = list(path.glob(f"{pattern}.jpg")) + list(path.glob(f"{pattern}.png"))
        
        return success_result({
            "path": str(path),
            "count": len(images),
            "files": [str(f) for f in images],
        })
```

### Model Selection

```python
def _select_model(self, classes: list[str] | None) -> str:
    """Select appropriate model based on classes."""
    from backend.cv.detection import COCO_CLASSES
    
    coco_set = set(c.lower() for c in COCO_CLASSES)
    
    if not classes:
        return "yolo11m"  # Default
    
    # Check if all classes are COCO
    if all(c.lower() in coco_set for c in classes):
        return "yolo11m"
    
    # Need open-vocab
    return "yolo_world"
```

## Best Practices

1. **Always validate inputs** - Fail fast with clear errors
2. **Use structured results** - Return dicts, not strings
3. **Handle exceptions** - Never let exceptions propagate
4. **Provide progress** - Use callbacks for long operations
5. **Clean up resources** - Unload models, close files
6. **Write descriptive docstrings** - Help LLM understand when to use tool
7. **Use appropriate categories** - Organize tools logically
8. **Test independently** - Tools should work standalone

## Additional Resources

- See `backend/src/backend/agent/tools/detect.py` for detection tool example
- See `backend/src/backend/agent/tools/scan.py` for scanning tool example
- See `backend/src/backend/agent/tools/base.py` for BaseTool API
