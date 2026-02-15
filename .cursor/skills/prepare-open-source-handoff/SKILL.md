---
name: prepare-open-source-handoff
description: Prepare a repository for public open-source handoff by removing unneeded development artifacts and upgrading contributor-facing documentation. Use when finishing development, before release/tag, before publishing a repo, or when asked to clean project structure and make README and contribution guidance ready for external maintainers and contributors.
---

# Prepare Open-Source Handoff

## Workflow

1. Run a non-destructive cleanup scan with `scripts/cleanup_candidates.py`.
2. Remove only safe clutter candidates (untracked/ignored) and never remove tracked files unless explicitly requested.
3. Audit README coverage for open-source handoff with `scripts/readme_handoff_check.py`.
4. Append missing README sections when needed, then manually refine wording for project-specific details.
5. Validate that contributor and governance links are clear (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `LICENSE`).

## Quick Commands

```bash
# 1) Preview cleanup candidates
python3 scripts/cleanup_candidates.py --root .

# 2) Apply safe cleanup (untracked/ignored only)
python3 scripts/cleanup_candidates.py --root . --apply --yes

# 3) Audit README handoff readiness
python3 scripts/readme_handoff_check.py --readme README.md --strict

# 4) Append missing README stubs, then refine manually
python3 scripts/readme_handoff_check.py --readme README.md --append-missing --append-recommended
```

## Cleanup Rules

1. Prefer deletion candidates that are clearly generated artifacts (caches, coverage outputs, build outputs, OS junk).
2. Keep source, tests, docs, licenses, and contributor policy files.
3. Treat tracked files as protected by default.
4. Use `--include-tracked` only with explicit user confirmation.
5. Re-run tests/lint after cleanup if build artifacts may affect local workflows.

## README Rules

1. Ensure README explains purpose, setup, usage, development workflow, and contribution path.
2. Ensure contributor-facing sections exist: `Contributing`, `Code of Conduct`, `Security`, and `License`.
3. Prefer links to dedicated policy files when present.
4. If a section is missing, append a clear stub and leave project-specific TODOs.
5. Keep language concrete, with runnable commands and repository-relative paths.

## Expected Deliverable

A handoff-ready repository state with:
- Cleanup report (or applied cleanup summary)
- README audit report
- Updated README that is understandable for external contributors
- Clear references to contributor governance docs

## References

- Use `references/handoff-checklist.md` for final acceptance checks.
- Use `references/readme-template.md` when README requires major restructuring.

## Scripts

### `scripts/cleanup_candidates.py`
Scan for removable clutter and optionally delete safe candidates.

### `scripts/readme_handoff_check.py`
Audit README section coverage for open-source handoff and optionally append missing section stubs.
