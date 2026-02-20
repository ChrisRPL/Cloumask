.PHONY: install dev test lint type-check format run clean

# Install sidecar runtime dependencies (core + agent + CV)
install:
	pip install -r requirements-agent.txt -r requirements-cv.txt
	pip install -e .

# Install development dependencies
dev:
	pip install -r requirements-dev.txt
	pip install -e .

# Run tests
test:
	pytest tests/ -v --cov=backend --cov-report=term-missing

# Run linter
lint:
	ruff check .
	ruff format --check .

# Run type checker
type-check:
	mypy src/backend/

# Format code
format:
	ruff format .
	ruff check --fix .

# Run development server
run:
	uvicorn backend.api.main:app --reload --port 8765

# Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
