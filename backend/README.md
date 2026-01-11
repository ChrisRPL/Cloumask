# Cloumask Backend

Python FastAPI sidecar for Cloumask desktop application.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Run development server (after main.py is created in task 06)
uvicorn backend.api.main:app --reload --port 8765
```

## Structure

```
src/backend/
├── api/     - FastAPI application and routes
├── agent/   - LangGraph agent implementation
├── cv/      - Computer vision model wrappers
└── data/    - Data loaders and exporters
```
