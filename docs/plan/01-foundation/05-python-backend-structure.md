# Python Backend Scaffolding

> **Parent:** 01-foundation
> **Depends on:** None (can be done in parallel with frontend)
> **Blocks:** 06-fastapi-app-health

## Objective

Create the directory structure and package configuration for the Python FastAPI backend that will serve as the Cloumask sidecar.

## Acceptance Criteria

- [ ] `backend/` directory structure created
- [ ] All `__init__.py` files in place
- [ ] `pyproject.toml` configured with project metadata
- [ ] Package is importable (`from backend.api import main`)
- [ ] Virtual environment setup documented

## Implementation Steps

1. **Create directory structure**
   ```bash
   cd /Users/krzysztof/Cloumask
   mkdir -p backend/api/routes
   mkdir -p backend/agent
   mkdir -p backend/cv
   mkdir -p backend/data
   ```

2. **Create root __init__.py**
   Create `backend/__init__.py`:
   ```python
   """Cloumask Python Backend - FastAPI sidecar for CV processing."""

   __version__ = "0.1.0"
   __app_name__ = "cloumask-backend"
   ```

3. **Create api package __init__.py**
   Create `backend/api/__init__.py`:
   ```python
   """FastAPI application and API routes."""
   ```

4. **Create routes package __init__.py**
   Create `backend/api/routes/__init__.py`:
   ```python
   """API route handlers."""
   ```

5. **Create agent package __init__.py**
   Create `backend/agent/__init__.py`:
   ```python
   """LangGraph agent implementation (to be implemented in 02-agent-system)."""
   ```

6. **Create cv package __init__.py**
   Create `backend/cv/__init__.py`:
   ```python
   """Computer vision model wrappers (to be implemented in 03-cv-models)."""
   ```

7. **Create data package __init__.py**
   Create `backend/data/__init__.py`:
   ```python
   """Data loaders and exporters (to be implemented in 06-data-pipeline)."""
   ```

8. **Create pyproject.toml**
   Create `backend/pyproject.toml`:
   ```toml
   [project]
   name = "cloumask-backend"
   version = "0.1.0"
   description = "Python sidecar for Cloumask - CV processing and LangGraph agent"
   readme = "README.md"
   license = { text = "MIT" }
   requires-python = ">=3.11"
   authors = [
       { name = "Cloumask Team" }
   ]
   keywords = ["computer-vision", "llm", "agent", "fastapi", "tauri"]
   classifiers = [
       "Development Status :: 3 - Alpha",
       "Framework :: FastAPI",
       "Intended Audience :: Developers",
       "License :: OSI Approved :: MIT License",
       "Programming Language :: Python :: 3.11",
       "Programming Language :: Python :: 3.12",
       "Topic :: Scientific/Engineering :: Artificial Intelligence",
       "Topic :: Scientific/Engineering :: Image Processing",
   ]

   dependencies = [
       "fastapi>=0.115.0",
       "uvicorn[standard]>=0.32.0",
       "sse-starlette>=2.1.0",
       "pydantic>=2.9.0",
       "pydantic-settings>=2.6.0",
       "httpx>=0.27.0",
   ]

   [project.optional-dependencies]
   dev = [
       "pytest>=8.0.0",
       "pytest-asyncio>=0.24.0",
       "pytest-cov>=5.0.0",
       "ruff>=0.7.0",
       "mypy>=1.12.0",
   ]
   agent = [
       "langgraph>=0.2.0",
       "langchain-ollama>=0.2.0",
   ]
   cv = [
       "torch>=2.4.0",
       "torchvision>=0.19.0",
       "opencv-python>=4.10.0",
       "numpy>=1.26.0",
       "pillow>=10.4.0",
   ]
   all = [
       "cloumask-backend[dev,agent,cv]",
   ]

   [project.scripts]
   cloumask-backend = "backend.api.main:start"

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build.targets.wheel]
   packages = ["backend"]

   [tool.ruff]
   target-version = "py311"
   line-length = 100

   [tool.ruff.lint]
   select = ["E", "F", "I", "UP", "B", "SIM", "PTH"]
   ignore = ["E501"]

   [tool.mypy]
   python_version = "3.11"
   warn_return_any = true
   warn_unused_configs = true
   disallow_untyped_defs = true

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   asyncio_mode = "auto"
   ```

9. **Create a simple README**
   Create `backend/README.md`:
   ```markdown
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

   # Run development server
   uvicorn api.main:app --reload --port 8765
   ```

   ## Structure

   - `api/` - FastAPI application and routes
   - `agent/` - LangGraph agent implementation
   - `cv/` - Computer vision model wrappers
   - `data/` - Data loaders and exporters
   ```

10. **Add backend to .gitignore**
    Ensure `.gitignore` includes:
    ```
    # Python
    __pycache__/
    *.py[cod]
    *$py.class
    *.so
    .Python
    .venv/
    venv/
    ENV/
    *.egg-info/
    dist/
    build/
    .mypy_cache/
    .ruff_cache/
    .pytest_cache/
    ```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/__init__.py` | Create | Root package with version |
| `backend/api/__init__.py` | Create | API package marker |
| `backend/api/routes/__init__.py` | Create | Routes package marker |
| `backend/agent/__init__.py` | Create | Agent package placeholder |
| `backend/cv/__init__.py` | Create | CV package placeholder |
| `backend/data/__init__.py` | Create | Data package placeholder |
| `backend/pyproject.toml` | Create | Project configuration |
| `backend/README.md` | Create | Backend documentation |
| `.gitignore` | Modify | Add Python ignores |

## Verification

```bash
cd /Users/krzysztof/Cloumask/backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Test import
python -c "from backend import __version__; print(f'Backend version: {__version__}')"
# Should print: Backend version: 0.1.0

# Test package structure
python -c "from backend.api import routes; print('Imports work!')"
```

## Notes

- Python 3.11+ is required for modern type hints and performance
- The `pyproject.toml` uses hatchling as the build backend (PEP 517)
- Optional dependency groups allow installing only what's needed
- The `agent/`, `cv/`, and `data/` packages are placeholders for future modules
- Development dependencies include ruff (linting), mypy (type checking), pytest (testing)
