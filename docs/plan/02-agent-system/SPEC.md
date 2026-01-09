# Agent System Module

> **Status:** 🔴 Not Started
> **Priority:** P0 (Critical)
> **Dependencies:** 01-foundation

## Overview

Implement the conversational AI agent using LangGraph. The agent understands natural language requests, creates execution plans, and orchestrates CV tools with human-in-the-loop checkpoints.

## Goals

- [ ] LangGraph state machine with plan → execute → checkpoint flow
- [ ] Tool calling via Ollama (Qwen3-14B)
- [ ] Human-in-the-loop approval at checkpoints
- [ ] SSE streaming for real-time updates
- [ ] Persistent checkpoints for resume capability

## Technical Design

### State Machine Flow
```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    ▼                                     │
START ──> understand ──> plan ──> await_approval ──>──────┤
                              │         │                 │
                         [cancel]    [approve]            │
                              │         │                 │
                              ▼         ▼                 │
                            END    execute_step ──>───────┤
                                        │                 │
                                        ▼                 │
                              ┌──> checkpoint ──>─────────┘
                              │         │
                              │    [needs_review]
                              │         │
                              │         ▼
                              │   await_approval
                              │         │
                              │    [continue]
                              │         │
                              └─────────┘
                                        │
                                   [complete]
                                        │
                                        ▼
                                    complete ──> END
```

### Checkpoint Triggers
- **Percentage-based:** 10%, 25%, 50% progress
- **Quality-based:** Confidence drop >15%, error rate >5%
- **Critical steps:** Always after anonymize, segment, detect_3d

### LLM Configuration
- **Primary:** Qwen3-14B via Ollama (best tool calling)
- **Fallback:** Qwen3-8B (lighter), Llama 4 (alternative)
- **Temperature:** 0.1 for tool calling, 0.7 for conversation

---

## Sub-Specs

This module is broken down into 10 atomic, implementable sub-specifications:

| # | Spec | Description | Dependencies | Complexity |
|---|------|-------------|--------------|------------|
| 01 | [State Types](./01-state-types.md) | PipelineState, Message, Checkpoint types | 01-foundation | Low |
| 02 | [LangGraph Core](./02-langgraph-core.md) | Graph definition, nodes, conditional edges | 01 | Medium |
| 03 | [Agent Nodes: Planning](./03-agent-nodes-planning.md) | understand + plan nodes with prompts | 01, 02, 08 | High |
| 04 | [Agent Nodes: Execution](./04-agent-nodes-execution.md) | execute_step + complete nodes | 01, 02, 06 | Medium |
| 05 | [Human-in-the-Loop](./05-human-in-the-loop.md) | await_approval + checkpoint nodes | 01, 02 | Medium |
| 06 | [Tool System](./06-tool-system.md) | BaseTool, registry, result types | 01 | Medium |
| 07 | [Tool Implementations](./07-tool-implementations.md) | scan_directory + stub tools | 06 | Medium |
| 08 | [Ollama Integration](./08-ollama-integration.md) | LangChain-Ollama, tool calling, retry | 01-foundation, 06 | Medium |
| 09 | [SSE Streaming](./09-sse-streaming.md) | Event schema, FastAPI endpoints | 02, 04 | Medium |
| 10 | [Checkpoint Persistence](./10-checkpoint-persistence.md) | SQLite storage, resume capability | 01, 02 | Medium |

### Dependency Graph

```
01-state-types ──┬──> 02-langgraph-core ──┬──> 03-agent-nodes-planning
                 │                        ├──> 04-agent-nodes-execution
                 │                        └──> 05-human-in-the-loop
                 │
                 └──> 06-tool-system ──────> 07-tool-implementations
                                              ↓
                                     08-ollama-integration
                                              ↓
                                     (integrates with 02-04)

10-checkpoint-persistence ←── 05-human-in-the-loop
                          ←── 02-langgraph-core

09-sse-streaming ←── 04-agent-nodes-execution
                 ←── 05-human-in-the-loop
```

### Implementation Order

**Phase 1: Foundation (parallel)**
- 01-state-types
- 06-tool-system

**Phase 2: Core Graph**
- 02-langgraph-core

**Phase 3: Nodes (parallel after Phase 2)**
- 03-agent-nodes-planning
- 04-agent-nodes-execution
- 05-human-in-the-loop

**Phase 4: Integration (parallel)**
- 07-tool-implementations
- 08-ollama-integration
- 09-sse-streaming
- 10-checkpoint-persistence

---

## Acceptance Criteria

- [ ] Agent responds to "scan /path/to/folder" with directory analysis
- [ ] Plan is generated and shown before execution
- [ ] User can approve, edit, or cancel the plan
- [ ] Execution pauses at checkpoints for review
- [ ] Pipeline can be resumed after app restart
- [ ] SSE stream shows real-time progress

---

## Files to Create/Modify

```
backend/
├── agent/
│   ├── __init__.py
│   ├── graph.py              # LangGraph definition (02)
│   ├── state.py              # State types (01)
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── understand.py     # (03)
│   │   ├── plan.py           # (03)
│   │   ├── execute.py        # (04)
│   │   ├── complete.py       # (04)
│   │   ├── approval.py       # (05)
│   │   └── checkpoint.py     # (05)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py           # Tool base class (06)
│   │   ├── registry.py       # Tool registry (06)
│   │   ├── scan.py           # scan_directory (07)
│   │   ├── anonymize.py      # anonymize stub (07)
│   │   ├── detect.py         # detect stub (07)
│   │   └── export.py         # export stub (07)
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── config.py         # LLM configuration (08)
│   │   ├── provider.py       # Ollama provider (08)
│   │   ├── tools.py          # Tool calling (08)
│   │   └── models.py         # Model management (08)
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── system.md         # System prompt (03)
│   │   └── planning.md       # Planning prompt (03)
│   └── checkpoints/
│       ├── __init__.py
│       ├── saver.py          # SQLite saver (10)
│       ├── manager.py        # Checkpoint manager (10)
│       └── schema.sql        # DB schema (10)
├── api/
│   ├── routes/
│   │   ├── chat.py           # Chat endpoint
│   │   └── threads.py        # Thread management (10)
│   └── streaming/
│       ├── __init__.py
│       ├── events.py         # Event types (09)
│       ├── endpoints.py      # SSE endpoints (09)
│       └── batching.py       # Rate limiting (09)
└── tests/
    └── agent/
        ├── test_state.py
        ├── test_graph.py
        ├── nodes/
        │   ├── test_planning.py
        │   ├── test_execution.py
        │   └── test_hitl.py
        ├── tools/
        │   ├── test_base.py
        │   ├── test_registry.py
        │   ├── test_scan.py
        │   └── test_stubs.py
        ├── llm/
        │   ├── test_provider.py
        │   ├── test_tools.py
        │   └── test_integration.py
        └── checkpoints/
            ├── test_saver.py
            ├── test_manager.py
            └── test_integration.py
```

---

## Dependencies

```
# requirements.txt additions for 02-agent-system
langgraph>=0.2.0
langgraph-checkpoint>=0.2.0
langchain-ollama>=0.2.0
langchain-core>=0.3.0
sse-starlette>=1.6.0
```

---

## Quick Reference

Each sub-spec includes:
- **Overview**: 2-3 sentence description
- **Goals**: Specific deliverables with checkboxes
- **Technical Design**: Types, code examples, diagrams
- **Implementation Tasks**: Detailed checkbox list
- **Testing**: Unit tests, integration tests, edge cases
- **Acceptance Criteria**: Testable requirements
- **Files to Create/Modify**: Specific file structure

---

## Notes

- All sub-specs follow the same template for consistency
- Each spec is designed to be implementable in isolation
- Stub tools (07) enable end-to-end testing before CV models (03-cv-models)
- SSE events have strict JSON schema for frontend integration
- SQLite is sufficient for single-user desktop app checkpointing
