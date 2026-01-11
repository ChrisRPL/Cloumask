#!/bin/bash
# Verification script for Foundation module
# This script checks all components required for the Foundation module

set -e

echo "=== Foundation Module Verification ==="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

check() {
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} $1"
  else
    echo -e "${RED}✗${NC} $1"
    exit 1
  fi
}

warn() {
  echo -e "${YELLOW}!${NC} $1"
}

echo "1. Checking Node.js dependencies..."
npm list @tauri-apps/api > /dev/null 2>&1
check "Node dependencies installed"

echo ""
echo "2. Checking Rust compilation..."
cd src-tauri && cargo check > /dev/null 2>&1
check "Rust code compiles"
cd ..

echo ""
echo "3. Checking Python backend..."
cd backend
if [ -d ".venv" ]; then
  source .venv/bin/activate 2>/dev/null || true
fi
python -c "from backend.api.main import app" 2>/dev/null
check "Python backend imports"
cd ..

echo ""
echo "4. Checking Svelte compilation..."
npm run check > /dev/null 2>&1
check "Svelte type check passes"

echo ""
echo "5. Starting backend for health check..."
cd backend
if [ -d ".venv" ]; then
  source .venv/bin/activate 2>/dev/null || true
fi
uvicorn backend.api.main:app --port 8765 &
BACKEND_PID=$!
sleep 3

curl -s http://localhost:8765/health | grep -q "healthy"
check "Backend health endpoint works"

curl -s http://localhost:8765/ready | grep -q "ready"
check "Backend ready endpoint works"

kill $BACKEND_PID 2>/dev/null || true
wait $BACKEND_PID 2>/dev/null || true
cd ..

echo ""
echo "6. Checking Ollama connectivity (required)..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo -e "${GREEN}✓${NC} Ollama is running"

  # Check for at least one model
  MODEL_COUNT=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data.get('models', [])))" 2>/dev/null || echo "0")
  if [ "$MODEL_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Ollama has $MODEL_COUNT model(s) available"
  else
    echo -e "${RED}✗${NC} No Ollama models found. Run: ollama pull qwen3:14b"
    exit 1
  fi
else
  echo -e "${RED}✗${NC} Ollama not running. Start with: ollama serve"
  exit 1
fi

echo ""
echo "7. Checking Rust linting..."
cd src-tauri && cargo clippy -- -D warnings > /dev/null 2>&1
check "Cargo clippy passes"
cd ..

echo ""
echo "8. Checking Python linting..."
cd backend
if [ -d ".venv" ]; then
  source .venv/bin/activate 2>/dev/null || true
fi
ruff check . > /dev/null 2>&1
check "Ruff linting passes"
cd ..

echo ""
echo -e "${GREEN}=== All Foundation checks passed! ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Run 'cargo tauri dev' to start the full application"
echo "  2. Verify the UI shows all components as 'healthy'"
echo "  3. Test IPC commands in browser devtools"
echo "  4. Verify Ollama Models card shows available models"
