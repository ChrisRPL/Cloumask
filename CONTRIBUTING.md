# Contributing to Cloumask

Thanks for contributing.

## Who This Guide Is For

- Users improving workflows for local-first CV dataset processing.
- Contributors adding features, bug fixes, tests, and docs.

## Development Setup

### Prerequisites

- Node.js 20+
- Rust 1.75+
- Python 3.11+

### Install

```bash
npm install
npm run backend:install
```

### Run Locally

```bash
# Terminal 1
npm run backend:dev

# Terminal 2
npm run dev
```

Desktop mode:

```bash
npm run tauri:dev
```

## Required Validation Before PR

Run all relevant checks for your change:

```bash
# Frontend static checks
npm run check

# Frontend tests
npm test -- --run

# Backend tests
cd backend && PYTHONPATH=src pytest -q

# Rust tests
cd src-tauri && cargo test
```

If you touched packaging/desktop delivery, also run:

```bash
npm run tauri:build
```

## User-Flow Expectations

When changing UX or workflow logic, validate from user perspective:

- First-run setup wizard
- Chat -> Plan -> Execute -> Review navigation
- Point cloud open/export flow
- Settings/system status visibility
- Desktop first-run model setup behavior (no manual CLI steps)

Prefer adding or updating tests for these paths.

## PR Guidelines

- Keep PRs focused and small.
- Include:
  - Problem statement
  - User impact
  - What changed
  - Validation evidence (commands + results)
- Update docs when behavior changes.
- Add tests for bug fixes and user-flow changes.

## Code Style

- Match existing project patterns.
- Keep interfaces typed and explicit.
- Prefer readable, maintainable code over clever shortcuts.
- Avoid unrelated refactors in feature/fix PRs.

## Reporting Issues

When filing bugs, include:

- Repro steps
- Expected vs actual behavior
- Environment (OS, web vs desktop, app version)
- Logs/screenshots when available
