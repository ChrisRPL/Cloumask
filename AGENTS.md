READ ~/Projects/agent-scripts/AGENTS.MD BEFORE ANYTHING (skip if missing).

Repo quick facts:
- Desktop dev: npm run tauri:dev (spawns sidecar)
- Web dev: npm run backend:dev + npm run dev
- Preferred shortcuts: just dev-backend, just dev-frontend, just dev-tauri, just health, just ports, just ci
- Verify: backend pytest, cargo test, npm run check, npm test -- --run
- Packaging/desktop verify: npm run tauri:build
- Backend helpers: cd backend && make run|test|lint|type-check|format

## Git Strategy
- Start each work item on a new branch (`feat/<short-topic>`).
- When you change code, you MUST create atomic commits.
- Finish with a PR and code review before merge.

### Atomic commit rule
- One commit = one logical change / intention.
- Commit message must be describable in one sentence without "and".
- If task includes multiple intentions, split into multiple commits.

### How to split (common patterns)
- Refactor vs behavior change: commit refactor first (no behavior change), then commit the behavior change.
- Tests vs fix/feature: add or adjust tests in one commit, then implement the fix/feature in the next.
- Mechanical changes (rename/move/format): keep in their own commit, separate from logic changes.
