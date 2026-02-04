---
name: sidecar-debugging
description: Guide for debugging Python sidecar issues including startup failures, import errors, API connectivity, and SSE streaming problems. Use when troubleshooting sidecar issues or debugging agent/API problems.
---

# Sidecar Debugging

## Quick Start

When debugging sidecar issues:

1. Check if sidecar is running: `curl http://localhost:8765/health`
2. Check sidecar logs in Tauri console
3. Verify Python imports: `python -c "from backend.api.main import app"`
4. Check port availability: `lsof -i :8765`
5. Review FastAPI logs for errors

## Sidecar Status

Check sidecar status from Rust:

```rust
use crate::commands::sidecar_status;

let status = sidecar_status(state)?;
println!("Running: {}, URL: {}", status.running, status.url);
```

Check from frontend:

```typescript
import { invoke } from '@tauri-apps/api/core';

const status = await invoke<SidecarStatus>('sidecar_status');
console.log('Sidecar running:', status.running);
```

## Health Check

Test sidecar health endpoint:

```bash
curl http://localhost:8765/health
```

Expected response:

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Import Verification

Verify Python imports work:

```bash
cd backend
python -c "from backend.api.main import app"
```

Check for import errors:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from backend.api.main import app"
```

## Port Conflicts

Check if port 8765 is in use:

```bash
# macOS/Linux
lsof -i :8765

# Windows
netstat -ano | findstr :8765
```

Kill process if needed:

```bash
# macOS/Linux
kill -9 $(lsof -t -i:8765)

# Windows
taskkill /PID <pid> /F
```

## Sidecar Logs

View sidecar logs:

1. **Tauri Dev Console**: Check terminal where `cargo tauri dev` is running
2. **Python Logging**: Check for Python log output
3. **FastAPI Logs**: Check uvicorn logs

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Common Issues

### Sidecar Won't Start

**Symptoms**: `start_sidecar` returns error, status shows `running: false`

**Debugging**:
1. Check Python executable path
2. Verify `backend/` directory exists
3. Check for Python syntax errors
4. Verify dependencies installed

```bash
# Check Python version
python --version

# Verify backend directory
ls -la backend/

# Check for syntax errors
python -m py_compile backend/api/main.py
```

### Import Errors

**Symptoms**: `ModuleNotFoundError` or `ImportError`

**Debugging**:
1. Check PYTHONPATH
2. Verify module structure
3. Check for circular imports
4. Verify dependencies installed

```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Verify imports
cd backend/src
python -c "from backend.api.main import app"

# Check dependencies
pip list | grep fastapi
```

### Port Already in Use

**Symptoms**: `Address already in use` error

**Solution**:
1. Kill existing process on port 8765
2. Or change port in configuration

```python
# backend/api/config.py
SIDECAR_PORT = 8766  # Change port
```

### API Endpoints Not Found

**Symptoms**: 404 errors on API calls

**Debugging**:
1. Check route registration
2. Verify endpoint paths
3. Check FastAPI app setup

```python
# Check registered routes
from backend.api.main import app
for route in app.routes:
    print(route.path, route.methods)
```

### SSE Streaming Issues

**Symptoms**: No SSE events received, connection drops

**Debugging**:
1. Check SSE endpoint implementation
2. Verify EventSource connection
3. Check for CORS issues
4. Review SSE event format

```typescript
// Frontend: Check SSE connection
const eventSource = new EventSource('http://localhost:8765/api/stream');
eventSource.onmessage = (event) => {
  console.log('SSE event:', event.data);
};
eventSource.onerror = (error) => {
  console.error('SSE error:', error);
};
```

### Model Loading Failures

**Symptoms**: Models fail to load, GPU errors

**Debugging**:
1. Check GPU availability
2. Verify model files exist
3. Check VRAM usage
4. Review model loading logs

```python
# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Check VRAM
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

## Debugging Tools

### Python Debugger

Use pdb for debugging:

```python
import pdb

def my_function():
    pdb.set_trace()  # Breakpoint
    # ... code ...
```

### FastAPI Debug Mode

Enable debug mode:

```python
# backend/api/main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="debug")
```

### Request Logging

Log all requests:

```python
from fastapi import Request
import logging

logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url}")
    response = await call_next(request)
    return response
```

## Testing Sidecar Manually

### Start Sidecar Manually

```bash
cd backend
uvicorn api.main:app --reload --port 8765
```

### Test Endpoints

```bash
# Health check
curl http://localhost:8765/health

# LLM status
curl http://localhost:8765/llm/status

# List models
curl http://localhost:8765/llm/models
```

### Test SSE Stream

```bash
curl -N http://localhost:8765/api/stream
```

## Sidecar Restart

Restart sidecar from Rust:

```rust
// Stop and start
stop_sidecar(state)?;
start_sidecar(state)?;
```

Restart from frontend:

```typescript
await invoke('stop_sidecar');
await invoke('start_sidecar');
```

## Environment Variables

Check environment variables:

```bash
# Python
python -c "import os; print(os.environ)"

# Rust
env::var("VAR_NAME")
```

Set environment variables:

```rust
// In sidecar spawn
.env("VAR_NAME", "value")
```

## Dependency Issues

Check Python dependencies:

```bash
cd backend
pip list
pip check
```

Install missing dependencies:

```bash
pip install -r requirements.txt
```

## Best Practices

1. **Always check health endpoint first** - Quick way to verify sidecar is running
2. **Enable debug logging** - Helps identify issues
3. **Test endpoints manually** - Use curl to verify API
4. **Check logs** - Review both Rust and Python logs
5. **Verify imports** - Ensure Python modules can be imported
6. **Check port availability** - Ensure port isn't in use
7. **Test SSE separately** - Verify streaming works independently

## Additional Resources

- See `src-tauri/src/sidecar.rs` for sidecar management
- See `backend/api/main.py` for FastAPI setup
- See `backend/api/streaming/` for SSE implementation
- See `CLAUDE.md` for common mistakes
