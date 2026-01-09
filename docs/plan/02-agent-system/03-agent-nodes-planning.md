# Agent Nodes: Planning

> **Status:** 🔴 Not Started
> **Priority:** P0 (Critical)
> **Dependencies:** 01-state-types, 02-langgraph-core, 08-ollama-integration
> **Estimated Complexity:** High

## Overview

Implement the `understand` and `plan` nodes that process user input and generate execution plans. These nodes use the LLM (Qwen3 via Ollama) to interpret natural language requests and create structured pipeline steps.

## Goals

- [ ] `understand` node: Parse user intent and extract parameters
- [ ] `plan` node: Generate ordered list of pipeline steps
- [ ] System and planning prompts with examples
- [ ] Structured output parsing from LLM
- [ ] Validation of generated plans

## Technical Design

### Understand Node

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import PipelineState, MessageRole


UNDERSTAND_SYSTEM_PROMPT = """You are an AI assistant for Cloumask, a computer vision data processing application.

Your task is to understand the user's request and extract:
1. **Intent**: What operation do they want? (scan, anonymize, detect, segment, export, etc.)
2. **Input**: What data are they working with? (directory path, file type, etc.)
3. **Parameters**: Any specific settings? (confidence threshold, output format, etc.)
4. **Output**: Where should results go? (directory, format, etc.)

Respond in this JSON format:
{
    "intent": "primary_operation",
    "input_path": "/path/to/data",
    "input_type": "images|video|pointcloud|mixed",
    "operations": ["operation1", "operation2"],
    "parameters": {
        "confidence": 0.5,
        "format": "yolo"
    },
    "output_path": "/path/to/output",
    "clarification_needed": null | "question if unclear"
}

Common intents:
- scan: Analyze directory contents
- anonymize: Blur/mask faces, license plates
- detect: Find objects (vehicles, pedestrians, etc.)
- segment: Instance/semantic segmentation
- label: Auto-label for training data
- export: Convert to specific format (YOLO, COCO, etc.)

If the request is unclear, set clarification_needed with a specific question."""


async def understand_node(state: PipelineState) -> PipelineState:
    """
    Parse user request and extract structured intent.

    Updates state with:
    - Parsed intent in metadata
    - Assistant message with understanding confirmation
    """
    from agent.llm import get_llm
    import json

    llm = get_llm()

    # Get the latest user message
    messages = state.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        state["last_error"] = "No user message found"
        return state

    latest_user_msg = user_messages[-1]["content"]

    # Build conversation for LLM
    llm_messages = [
        SystemMessage(content=UNDERSTAND_SYSTEM_PROMPT),
        HumanMessage(content=latest_user_msg),
    ]

    # Get LLM response
    response = await llm.ainvoke(llm_messages)

    try:
        # Parse JSON from response
        understanding = json.loads(response.content)

        # Check if clarification needed
        if understanding.get("clarification_needed"):
            state["messages"].append({
                "role": "assistant",
                "content": understanding["clarification_needed"],
                "timestamp": datetime.now().isoformat(),
            })
            state["awaiting_user"] = True
            return state

        # Store understanding in metadata
        metadata = state.get("metadata", {})
        metadata["understanding"] = understanding
        state["metadata"] = metadata

        # Add confirmation message
        operations = understanding.get("operations", [understanding.get("intent")])
        state["messages"].append({
            "role": "assistant",
            "content": f"I understand you want to {', '.join(operations)} on {understanding.get('input_path', 'your data')}. Let me create a plan.",
            "timestamp": datetime.now().isoformat(),
        })

    except json.JSONDecodeError:
        # Fallback: treat response as natural language
        state["messages"].append({
            "role": "assistant",
            "content": response.content,
            "timestamp": datetime.now().isoformat(),
        })
        state["last_error"] = "Failed to parse LLM response as JSON"

    return state
```

### Plan Node

```python
PLANNING_SYSTEM_PROMPT = """You are a pipeline planner for Cloumask, a computer vision application.

Given the user's understood intent, create an execution plan as a list of steps.

Available tools:
- scan_directory: Analyze folder contents (count files, detect formats)
  Parameters: path (str), recursive (bool)

- anonymize: Blur faces and license plates
  Parameters: input_path (str), output_path (str), target (faces|plates|all)

- detect: Object detection
  Parameters: input_path (str), classes (list[str]), confidence (float)

- segment: Instance segmentation
  Parameters: input_path (str), prompt (str), model (sam3|sam2)

- export: Convert annotations to format
  Parameters: input_path (str), output_path (str), format (yolo|coco|pascal)

Respond with a JSON array of steps:
[
    {
        "tool_name": "tool_name",
        "parameters": {...},
        "description": "Human-readable description"
    },
    ...
]

Guidelines:
1. Always start with scan_directory to verify input
2. Order operations logically (detect before export)
3. Include reasonable defaults for missing parameters
4. Keep plans focused and minimal"""


async def plan_node(state: PipelineState) -> PipelineState:
    """
    Generate execution plan based on understood intent.

    Updates state with:
    - plan: List of PipelineStep dicts
    - Assistant message showing the plan
    """
    from agent.llm import get_llm
    from uuid import uuid4
    import json

    llm = get_llm()

    # Get understanding from metadata
    metadata = state.get("metadata", {})
    understanding = metadata.get("understanding", {})

    if not understanding:
        state["last_error"] = "No understanding found, cannot plan"
        return state

    # Build planning prompt
    llm_messages = [
        SystemMessage(content=PLANNING_SYSTEM_PROMPT),
        HumanMessage(content=f"""Create a plan for:
Intent: {understanding.get('intent')}
Input: {understanding.get('input_path')}
Type: {understanding.get('input_type')}
Operations: {understanding.get('operations')}
Parameters: {understanding.get('parameters')}
Output: {understanding.get('output_path')}"""),
    ]

    response = await llm.ainvoke(llm_messages)

    try:
        # Parse plan from response
        steps_raw = json.loads(response.content)

        # Convert to PipelineStep format
        plan = []
        for i, step in enumerate(steps_raw):
            plan.append({
                "id": f"step-{uuid4().hex[:8]}",
                "tool_name": step["tool_name"],
                "parameters": step.get("parameters", {}),
                "description": step.get("description", f"Step {i+1}"),
                "status": "pending",
                "result": None,
                "error": None,
            })

        # Validate plan
        validation_error = validate_plan(plan)
        if validation_error:
            state["last_error"] = validation_error
            state["messages"].append({
                "role": "assistant",
                "content": f"I encountered an issue creating the plan: {validation_error}. Let me try again.",
                "timestamp": datetime.now().isoformat(),
            })
            return state

        state["plan"] = plan
        state["current_step"] = 0
        state["plan_approved"] = False

        # Format plan for display
        plan_display = format_plan_for_display(plan)
        state["messages"].append({
            "role": "assistant",
            "content": f"Here's my proposed plan:\n\n{plan_display}\n\nDo you want to proceed, or would you like to make changes?",
            "timestamp": datetime.now().isoformat(),
        })

    except json.JSONDecodeError as e:
        state["last_error"] = f"Failed to parse plan: {e}"
        state["messages"].append({
            "role": "assistant",
            "content": "I had trouble creating a structured plan. Could you rephrase your request?",
            "timestamp": datetime.now().isoformat(),
        })

    return state


def validate_plan(plan: list[dict]) -> Optional[str]:
    """Validate a generated plan."""

    if not plan:
        return "Plan is empty"

    valid_tools = ["scan_directory", "anonymize", "detect", "segment", "export"]

    for i, step in enumerate(plan):
        tool = step.get("tool_name")
        if tool not in valid_tools:
            return f"Step {i+1} uses unknown tool: {tool}"

        if not step.get("parameters"):
            return f"Step {i+1} ({tool}) has no parameters"

    return None


def format_plan_for_display(plan: list[dict]) -> str:
    """Format plan as numbered list for chat display."""

    lines = []
    for i, step in enumerate(plan, 1):
        status_icon = "⏳"
        lines.append(f"{status_icon} **Step {i}: {step['description']}**")
        lines.append(f"   Tool: `{step['tool_name']}`")
        params = step.get('parameters', {})
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"   Parameters: {param_str}")
        lines.append("")

    return "\n".join(lines)
```

### Prompts Module

```python
# backend/agent/prompts/__init__.py

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt from markdown file."""
    prompt_file = PROMPTS_DIR / f"{name}.md"
    if prompt_file.exists():
        return prompt_file.read_text()
    raise FileNotFoundError(f"Prompt not found: {name}")
```

### System Prompt (Markdown)

```markdown
<!-- backend/agent/prompts/system.md -->

# Cloumask AI Agent

You are the AI agent for Cloumask, a local-first desktop application for computer vision data processing.

## Your Capabilities

- **Scan directories** to analyze image, video, and point cloud datasets
- **Anonymize** faces and license plates in visual data
- **Detect** objects using YOLO11 and other models
- **Segment** objects using SAM3 with text prompts
- **Export** annotations to YOLO, COCO, and other formats

## Communication Style

- Be concise and direct
- Use bullet points for lists
- Show progress percentages when available
- Ask for clarification when the request is ambiguous
- Explain what you're doing at each step

## Important Rules

1. Always scan the input directory first to verify it exists and contains valid data
2. Never proceed without user approval of the plan
3. Pause at checkpoints to show progress and quality metrics
4. If confidence drops significantly, ask the user if they want to continue

## Data Privacy

- All processing happens locally on the user's machine
- No data is sent to external servers
- Model inference uses local Ollama instance
```

## Implementation Tasks

- [ ] Create `backend/agent/nodes/understand.py`
- [ ] Create `backend/agent/nodes/plan.py`
- [ ] Create `backend/agent/prompts/` directory
- [ ] Write `system.md` prompt
- [ ] Write `planning.md` prompt
- [ ] Implement `understand_node()` function
- [ ] Implement `plan_node()` function
- [ ] Implement `validate_plan()` helper
- [ ] Implement `format_plan_for_display()` helper
- [ ] Add JSON parsing with error handling
- [ ] Add retry logic for LLM failures

## Testing

### Unit Tests

```python
# tests/agent/nodes/test_planning.py

def test_validate_plan_empty():
    """Empty plan should fail validation."""
    assert validate_plan([]) == "Plan is empty"


def test_validate_plan_unknown_tool():
    """Unknown tool should fail validation."""
    plan = [{"tool_name": "unknown_tool", "parameters": {}}]
    result = validate_plan(plan)
    assert "unknown tool" in result.lower()


def test_validate_plan_valid():
    """Valid plan should pass."""
    plan = [
        {"tool_name": "scan_directory", "parameters": {"path": "/data"}},
        {"tool_name": "detect", "parameters": {"input_path": "/data", "classes": ["car"]}},
    ]
    assert validate_plan(plan) is None


def test_format_plan_display():
    """Plan should format correctly for display."""
    plan = [
        {"tool_name": "scan_directory", "parameters": {"path": "/data"}, "description": "Scan input"},
    ]
    output = format_plan_for_display(plan)
    assert "Step 1" in output
    assert "scan_directory" in output


@pytest.mark.asyncio
async def test_understand_node_basic():
    """Understand node should parse simple request."""
    state = {
        "messages": [{"role": "user", "content": "scan /data/images"}],
        "metadata": {},
    }

    # Mock the LLM
    with patch("agent.llm.get_llm") as mock_llm:
        mock_llm.return_value.ainvoke.return_value.content = json.dumps({
            "intent": "scan",
            "input_path": "/data/images",
            "input_type": "images",
            "operations": ["scan"],
            "parameters": {},
            "output_path": None,
            "clarification_needed": None,
        })

        result = await understand_node(state)

        assert "understanding" in result["metadata"]
        assert result["metadata"]["understanding"]["intent"] == "scan"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_understand_to_plan_flow():
    """Test full flow from understand to plan."""
    # Requires running Ollama with Qwen3
    if not ollama_available():
        pytest.skip("Ollama not available")

    state = create_initial_state("Anonymize all faces in /data/dashcam", "test")

    state = await understand_node(state)
    assert "understanding" in state["metadata"]

    state = await plan_node(state)
    assert len(state["plan"]) > 0
    assert state["plan"][0]["tool_name"] in ["scan_directory", "anonymize"]
```

### Edge Cases

- [ ] Completely ambiguous request ("do something with my data")
- [ ] Request in non-English language
- [ ] Very long request with multiple operations
- [ ] Request referencing non-existent paths
- [ ] LLM returns malformed JSON
- [ ] LLM returns valid JSON with missing fields

## Acceptance Criteria

- [ ] `understand` node correctly parses "scan /path/to/folder"
- [ ] `understand` node asks clarification for ambiguous requests
- [ ] `plan` node generates valid steps for common operations
- [ ] Plans always start with `scan_directory`
- [ ] Plans are displayed in human-readable format
- [ ] Invalid plans are caught and reported
- [ ] Prompts are loaded from markdown files

## Files to Create/Modify

```
backend/
├── agent/
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── understand.py
│   │   └── plan.py
│   └── prompts/
│       ├── __init__.py
│       ├── system.md
│       └── planning.md
└── tests/
    └── agent/
        └── nodes/
            └── test_planning.py
```

## Notes

- Prompts will need iteration based on Qwen3's behavior
- Consider few-shot examples in prompts for better consistency
- JSON mode may be available in newer Ollama versions
- Track token usage for optimization
