---
name: verification-workflow
description: Guide for running verification checks before commits including linting, type checking, and tests. Use when preparing to commit, debugging test failures, or ensuring code quality.
---

# Verification Workflow

## Quick Start

Before committing, run the full verification suite:

```bash
# Full verification
cd src-tauri && cargo clippy -- -D warnings
cd backend && ruff check . && mypy .
npm run check && npm run lint
cd backend && pytest tests/ -v
```

## Rust Verification

### Clippy (Linting)

Run clippy with warnings as errors:

```bash
cd src-tauri
cargo clippy -- -D warnings
```

Common issues:
- Unused imports/variables
- Unnecessary clones
- Missing error handling
- Inefficient patterns

### Tests

Run Rust tests:

```bash
cd src-tauri
cargo test
```

Run specific test:

```bash
cargo test test_name
```

## Python Verification

### Ruff (Linting)

Run ruff checks:

```bash
cd backend
ruff check .
```

Auto-fix issues:

```bash
ruff check . --fix
```

### MyPy (Type Checking)

Run type checking:

```bash
cd backend
mypy .
```

Check specific module:

```bash
mypy backend/agent/tools/
```

### Pytest (Tests)

Run all tests:

```bash
cd backend
pytest tests/ -v
```

Run specific test file:

```bash
pytest tests/test_detection.py -v
```

Run with coverage:

```bash
pytest tests/ --cov=backend --cov-report=html
```

## Frontend Verification

### Type Checking

Run TypeScript type checking:

```bash
npm run check
```

### Linting

Run ESLint:

```bash
npm run lint
```

Auto-fix issues:

```bash
npm run lint -- --fix
```

### Formatting

Check formatting:

```bash
npm run format:check
```

Format files:

```bash
npm run format
```

## Pre-Commit Checklist

Before committing, verify:

- [ ] `cargo clippy -- -D warnings` passes
- [ ] `ruff check .` passes
- [ ] `mypy .` passes
- [ ] `npm run check` passes
- [ ] `npm run lint` passes
- [ ] `pytest tests/` passes
- [ ] New features have tests
- [ ] Sidecar imports work: `python -c "from backend.api.main import app"`

## Common Issues

### Rust: Unused Variable

```rust
// Bad
let _unused = value;

// Good
let unused = value;
// Or remove if truly unused
```

### Rust: Unnecessary Clone

```rust
// Bad
let cloned = data.clone();

// Good
let cloned = data;  // Move if possible
// Or use reference: &data
```

### Python: Missing Type Hints

```python
# Bad
def process(data):
    return data

# Good
def process(data: dict[str, Any]) -> dict[str, Any]:
    return data
```

### Python: Import Order

```python
# Bad
import os
from backend.cv import ModelManager
import sys

# Good
import os
import sys

from backend.cv import ModelManager
```

### TypeScript: Any Type

```typescript
// Bad
function process(data: any) {
  return data;
}

// Good
function process(data: unknown) {
  if (typeof data === 'string') {
    return data;
  }
  throw new Error('Invalid data');
}
```

## CI/CD Checks

The project uses GitHub Actions for CI. Checks include:

1. Rust clippy
2. Python ruff + mypy
3. Frontend type check + lint
4. All test suites

## Debugging Test Failures

### Rust Tests

```bash
# Run with output
cargo test -- --nocapture

# Run single test
cargo test test_name -- --nocapture

# Run with debug logs
RUST_LOG=debug cargo test
```

### Python Tests

```bash
# Run with output
pytest tests/ -v -s

# Run single test
pytest tests/test_file.py::test_function -v -s

# Run with debug logs
pytest tests/ -v --log-cli-level=DEBUG
```

### Frontend Tests

```bash
# Run tests
npm run test

# Run with coverage
npm run test -- --coverage
```

## Sidecar Verification

Verify sidecar can be imported:

```bash
cd backend
python -c "from backend.api.main import app"
```

Check sidecar health:

```bash
curl http://localhost:8765/health
```

## Model Verification

Verify models can be loaded:

```bash
cd backend
python -c "from backend.cv import ModelManager; m = ModelManager(); print(m.registered_models)"
```

## Quick Verification Script

Create a script for quick checks:

```bash
#!/bin/bash
set -e

echo "Running Rust checks..."
cd src-tauri && cargo clippy -- -D warnings

echo "Running Python checks..."
cd ../backend && ruff check . && mypy .

echo "Running Frontend checks..."
cd .. && npm run check && npm run lint

echo "All checks passed!"
```

## Additional Resources

- See `CLAUDE.md` for verification checklist
- See `.github/workflows/` for CI configuration
- See `backend/pyproject.toml` for Python tool configs
- See `package.json` for frontend scripts
