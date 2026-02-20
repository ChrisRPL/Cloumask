set dotenv-load := true

dev-tauri:
    npm run tauri:dev

dev-frontend:
    npm run dev

dev-backend:
    npm run backend:dev

health:
    curl -fsS http://127.0.0.1:8765/health | jq .

ports:
    lsof -nP -iTCP:5173 -sTCP:LISTEN || true
    lsof -nP -iTCP:8765 -sTCP:LISTEN || true
    lsof -nP -iTCP:11434 -sTCP:LISTEN || true

check-frontend:
    npm run check

test-frontend:
    npm test -- --run

test-backend:
    cd backend && PYTHONPATH=src pytest -q

test-rust:
    cd src-tauri && cargo test

ci: check-frontend test-frontend test-backend test-rust
