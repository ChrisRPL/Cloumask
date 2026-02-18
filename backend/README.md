# Cloumask Backend

Python FastAPI sidecar for Cloumask desktop application.

## Requirements

- Python 3.11 or 3.12
- pip (latest version recommended)

## Quick Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip

# Install runtime dependencies
pip install -r requirements.txt

# OR install with development tools
pip install -r requirements-dev.txt

# Install the package itself (for imports)
pip install -e .
```

## Running the Server

```bash
# Development mode (with auto-reload)
source .venv/bin/activate
uvicorn backend.api.main:app --reload --port 8765

# Production mode
uvicorn backend.api.main:app --host 127.0.0.1 --port 8765
```

## Verification

```bash
# Test import
python -c "from backend.api.main import app; print('OK')"

# Test health endpoint
curl http://localhost:8765/health
```

## Using Make (optional)

```bash
make dev      # Install dev dependencies
make run      # Start dev server
make test     # Run tests with coverage
make lint     # Check code style
make format   # Auto-format code
```

## Dependency Groups

| File | Purpose |
|------|---------|
| `requirements.txt` | Core runtime dependencies |
| `requirements-dev.txt` | Development and testing tools |
| `requirements-agent.txt` | LangGraph agent (02-agent-system) |
| `requirements-cv.txt` | Computer vision models (03-cv-models) |

## Structure

```
backend/
├── src/backend/
│   ├── __init__.py       # Package version
│   ├── api/
│   │   ├── main.py       # App factory and entry point
│   │   ├── config.py     # Settings management
│   │   └── routes/       # API route handlers
│   ├── agent/            # LangGraph agent (02-agent-system)
│   ├── cv/               # CV model wrappers (03-cv-models)
│   └── data/             # Data loaders (06-data-pipeline)
├── tests/                # Pytest test suite
├── requirements*.txt     # Dependency files
├── pyproject.toml        # Project metadata
└── Makefile              # Common task shortcuts
```

## Development Notes

- The package uses a `src` layout - source code is in `src/backend/`
- `pip install -e .` is required for imports to work correctly
- pyproject.toml uses version ranges, requirements.txt uses pinned versions
- Use `ruff` for linting and `mypy` for type checking
- Review queue items are isolated by `execution_id` and optional `project_id` in `/api/review/items`
