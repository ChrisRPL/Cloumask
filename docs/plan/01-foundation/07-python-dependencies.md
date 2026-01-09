# Python Dependencies Management

> **Parent:** 01-foundation
> **Depends on:** 06-fastapi-app-health
> **Blocks:** 08-rust-sidecar-module (Python must be runnable)

## Objective

Create requirements files with pinned versions for reproducible Python environment setup.

## Acceptance Criteria

- [ ] `requirements.txt` contains all runtime dependencies with pinned versions
- [ ] `requirements-dev.txt` contains development dependencies
- [ ] All dependencies install successfully
- [ ] `python -c "from backend.api.main import app"` works
- [ ] Python version requirement documented

## Implementation Steps

1. **Create requirements.txt**
   Create `backend/requirements.txt`:
   ```
   # Cloumask Backend - Runtime Dependencies
   # Python 3.11+ required

   # Web framework
   fastapi==0.115.6
   uvicorn[standard]==0.32.1
   starlette==0.41.3

   # SSE for streaming responses
   sse-starlette==2.1.3

   # Data validation
   pydantic==2.10.3
   pydantic-settings==2.6.1

   # HTTP client (for Ollama, external APIs)
   httpx==0.28.1

   # Async support
   anyio==4.7.0

   # Type hints backports (if needed)
   typing-extensions==4.12.2
   ```

2. **Create requirements-dev.txt**
   Create `backend/requirements-dev.txt`:
   ```
   # Cloumask Backend - Development Dependencies
   # Install with: pip install -r requirements-dev.txt

   -r requirements.txt

   # Testing
   pytest==8.3.4
   pytest-asyncio==0.24.0
   pytest-cov==6.0.0
   httpx==0.28.1  # For TestClient

   # Linting and formatting
   ruff==0.8.3

   # Type checking
   mypy==1.13.0

   # Development utilities
   watchfiles==1.0.0  # For uvicorn --reload
   python-dotenv==1.0.1

   # Type stubs
   types-requests==2.32.0
   ```

3. **Create requirements-agent.txt (for future use)**
   Create `backend/requirements-agent.txt`:
   ```
   # Cloumask Backend - Agent Dependencies
   # Install with: pip install -r requirements-agent.txt
   # Used in 02-agent-system module

   -r requirements.txt

   # LangGraph and LangChain
   langgraph==0.2.60
   langchain-core==0.3.28
   langchain-ollama==0.2.2

   # Additional agent utilities
   tenacity==9.0.0  # Retry logic
   ```

4. **Create requirements-cv.txt (for future use)**
   Create `backend/requirements-cv.txt`:
   ```
   # Cloumask Backend - Computer Vision Dependencies
   # Install with: pip install -r requirements-cv.txt
   # Used in 03-cv-models module

   -r requirements.txt

   # PyTorch (CUDA 12.x)
   torch==2.5.1
   torchvision==0.20.1

   # OpenCV
   opencv-python==4.10.0.84

   # Image processing
   numpy==2.1.3
   pillow==11.0.0

   # Model loading
   huggingface-hub==0.26.5
   safetensors==0.4.5

   # ONNX Runtime (optional, for optimized inference)
   # onnxruntime-gpu==1.20.0
   ```

5. **Create .python-version file**
   Create `backend/.python-version`:
   ```
   3.11
   ```

6. **Update backend README with setup instructions**
   Update `backend/README.md`:
   ```markdown
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
   ```

   ## Running the Server

   ```bash
   # Development mode (with auto-reload)
   cd /Users/krzysztof/Cloumask/backend
   source .venv/bin/activate
   uvicorn api.main:app --reload --port 8765

   # Production mode
   uvicorn api.main:app --host 127.0.0.1 --port 8765
   ```

   ## Verification

   ```bash
   # Test import
   python -c "from backend.api.main import app; print('OK')"

   # Test health endpoint
   curl http://localhost:8765/health
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
   ├── api/              # FastAPI application
   │   ├── main.py       # App factory and entry point
   │   ├── config.py     # Settings management
   │   └── routes/       # API route handlers
   ├── agent/            # LangGraph agent (02-agent-system)
   ├── cv/               # CV model wrappers (03-cv-models)
   ├── data/             # Data loaders (06-data-pipeline)
   ├── requirements*.txt # Dependency files
   └── pyproject.toml    # Project metadata
   ```
   ```

7. **Create a Makefile for common tasks (optional)**
   Create `backend/Makefile`:
   ```makefile
   .PHONY: install dev test lint type-check clean

   # Install runtime dependencies
   install:
   	pip install -r requirements.txt

   # Install development dependencies
   dev:
   	pip install -r requirements-dev.txt

   # Run tests
   test:
   	pytest tests/ -v --cov=backend --cov-report=term-missing

   # Run linter
   lint:
   	ruff check .
   	ruff format --check .

   # Run type checker
   type-check:
   	mypy backend/

   # Format code
   format:
   	ruff format .
   	ruff check --fix .

   # Run development server
   run:
   	uvicorn api.main:app --reload --port 8765

   # Clean build artifacts
   clean:
   	find . -type d -name __pycache__ -exec rm -rf {} +
   	find . -type d -name .pytest_cache -exec rm -rf {} +
   	find . -type d -name .mypy_cache -exec rm -rf {} +
   	find . -type d -name .ruff_cache -exec rm -rf {} +
   	find . -type d -name "*.egg-info" -exec rm -rf {} +
   ```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `backend/requirements.txt` | Create | Runtime dependencies |
| `backend/requirements-dev.txt` | Create | Development dependencies |
| `backend/requirements-agent.txt` | Create | Agent dependencies (placeholder) |
| `backend/requirements-cv.txt` | Create | CV dependencies (placeholder) |
| `backend/.python-version` | Create | Python version specification |
| `backend/README.md` | Modify | Setup instructions |
| `backend/Makefile` | Create | Common task shortcuts |

## Verification

```bash
cd /Users/krzysztof/Cloumask/backend

# Create fresh virtual environment
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -E "fastapi|uvicorn|pydantic"
# Should show:
# fastapi         0.115.6
# uvicorn         0.32.1
# pydantic        2.10.3

# Verify imports
python -c "from backend.api.main import app; print('Import successful!')"

# Start server and test
uvicorn api.main:app --port 8765 &
sleep 2
curl http://localhost:8765/health
kill %1
```

## Notes

- All versions are pinned for reproducibility
- Requirements files use `-r` to include base dependencies
- The agent and CV requirements are placeholders for future modules
- `uvicorn[standard]` includes `watchfiles` for better reload performance
- Python 3.11+ is required for modern features (ExceptionGroup, tomllib, etc.)
- Consider using `uv` for faster dependency installation in the future
