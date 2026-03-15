# Cloumask Full QA Scenarios (Playwright Manual Runbook)

Last updated: 2026-03-14

## Scope

This file is the manual QA target, not a claim that all flows are fully automated today.

Current automated Playwright coverage is still smoke-level for many browser flows:
- setup/onboarding visibility and skip path
- keyboard navigation shell checks
- chat shell rendering and basic input presence
- review seeding/load smoke
- settings shell smoke
- project selector creation path
- backend API smoke checks

Separate non-Playwright app-userflow automation now covers these resume/hydration paths in the Svelte test harness:
- latest thread auto-resume selection priority (`awaiting review` before `in progress` before completed)
- visible resumed-thread breadcrumb with backend thread id plus backend-provided resume summary text
- temporary auto-resume note that reuses the backend thread-list `summary` field instead of recomputing copy in the frontend
- persistent resumed-thread strip after hydration, including completed-thread restores
- resumed-thread strip dismissal via button and `Escape`
- completed-thread resume breadcrumb coverage using backend-provided summary text
- frontend fallback copy when older/backend-misaligned thread summaries are missing
- checkpoint resume hydration after startup
- completed execution hydration after startup, including stats and preview tiles

These flows still require manual verification before release because the current browser suite does not yet prove end-user success end-to-end:
- project persistence after reload
- chat -> plan generation -> approval transitions
- plan editing correctness and execution parity
- real-browser reload/remount behavior for auto-resume breadcrumb and thread selection
- real-browser checkpoint/completed resume parity after full page refresh
- review editing/undo/redo fidelity
- point cloud load/export behavior
- responsive/runtime parity across Tauri and web

## Environment

- frontend: `http://127.0.0.1:5173`
- backend: `http://127.0.0.1:8765`
- browser: Chromium (headed)
- Playwright projects:
  - `chromium`: browser UX smoke
  - `api`: backend-only HTTP smoke
- clear localStorage before first run
- reset setup state between major flows as needed

## Test Data

- image folder for review seeding: `tmp/test-images/`
- download point cloud fixtures: `./scripts/download-pointcloud-samples.sh`
- point cloud fixtures:
  - `tmp/pointclouds/lamppost.pcd`
  - `tmp/pointclouds/milk.pcd`

## Scenario Matrix

### A. Project Creation

| ID | Scenario | Steps | Expected |
|---|---|---|---|
| A1 | Create project from header selector | Open project selector -> click `New Project...` -> enter name/path -> confirm | New project is created, selected, and visible in selector |
| A2 | Persistence after reload | Create/select a project -> reload page | Last selected project remains selected |
| A3 | Recent projects ordering | Select project X then Y | Recent list keeps Y before X |
| A4 | Validation | Attempt create with empty name/path | Validation error shown, project not created |

### B. Chat and Agent Planning

| ID | Scenario | Steps | Expected |
|---|---|---|---|
| B1 | Chat fast-path response | Send `hello` | Agent returns chat help response without planning flow |
| B2 | Plan generation for multi-action task | Send task with path and actions (`detect`, `anonymize`, `export`) | Plan is generated and appears in plan preview + plan view |
| B3 | Class/action extraction | Ask for specific classes (for example person/car) | Plan step parameters include requested classes/actions |
| B4 | Approval controls | Use approve/edit/cancel in clarification form | Decision is sent and UI phase changes correctly |

### C. Plan Editing (Including Custom Script Steps)

| ID | Scenario | Steps | Expected |
|---|---|---|---|
| C1 | Toggle edit and modify step config | Open plan -> press `E` or click edit -> update step values -> apply | Step config updates and remains after panel close |
| C2 | Reorder and enable/disable steps | Drag/drop steps -> toggle step enabled state | Order/status updates in step list and visual state |
| C3 | Add custom step and script | Add `Custom` step -> open config -> generate script from prompt -> validate -> save/apply | Script saved path is attached to custom step config |
| C4 | Execution uses edited plan | Edit plan then start execution | Runtime executes edited order/config (not original pre-edit plan) |

### D. Execution View

| ID | Scenario | Steps | Expected |
|---|---|---|---|
| D1 | Start execution from plan | From awaiting approval, start execution | App moves to `Execute` view and status becomes running |
| D2 | Progress and step transitions | Observe running job events | Step statuses move pending -> running -> completed/failed |
| D3 | Checkpoint handling | Trigger/receive checkpoint -> continue/review | Banner appears and actions behave correctly |
| D4 | Pause/resume/cancel | Use controls + keyboard (`Space`, `Esc`) | State transitions are correct and cancel confirmation works |

### E. Review Queue

| ID | Scenario | Steps | Expected |
|---|---|---|---|
| E1 | Load review items | Seed backend items -> open review view | Item list loads and first item is selectable |
| E2 | Preview and annotation details | Select item | Canvas preview appears with annotation metadata |
| E3 | Approve/reject flows | Use action bar buttons and keyboard (`A`, `R`) | Status updates and next-item auto-advance works |
| E4 | Edit operations | Enter edit mode -> move/resize/delete/add annotation | Annotation changes are reflected in details and canvas |
| E5 | Undo/redo | Perform edit -> `Ctrl+Z` -> `Ctrl+Y` | Changes undo/redo correctly |
| E6 | Filtering/search | Apply status/confidence/search filters | List updates correctly |

### F. Point Cloud Viewer and User Flow

| ID | Scenario | Steps | Expected |
|---|---|---|---|
| F1 | Load point cloud file | Open point cloud view -> `Load` -> choose `.pcd`/`.ply` | File metadata and scene load without crash |
| F2 | Navigate scene | Orbit/pan/zoom and reset camera | Camera controls are responsive and predictable |
| F3 | Visual settings | Change color mode/point size/grid/axes | Rendering updates match selected controls |
| F4 | Export flow | Export loaded cloud to new format | Export completes and output file is written |
| F5 | Error handling | Load unsupported/bad file | User sees clear error message and app remains usable |

### G. UX/UI Quality

| ID | Scenario | Steps | Expected |
|---|---|---|---|
| G1 | Keyboard-first flow | Navigate views with `1-5`, `,`, `Ctrl+K` | Navigation works without mouse |
| G2 | Responsive behavior | Test desktop + mobile viewport | Layout remains usable, no clipped critical controls |
| G3 | State messaging | Trigger loading/offline/error states | Messaging is actionable and non-blocking where possible |
| G4 | Visual consistency | Review typography, spacing, contrast, affordances | UI is coherent and readable across all main views |

### H. Runtime Parity (Desktop vs Web)

| ID | Scenario | Steps | Expected |
|---|---|---|---|
| H1 | Core flow parity across runtimes | Run once in Tauri (`npm run tauri:dev`) and once in Web+backend (`npm run backend:dev` + `npm run dev`) -> create thread -> send task -> review generated plan | Chat/planning flow works in both modes; runtime-specific limitations are explicit (not silent failure) |

## Latest Run Results (2026-02-11)

### Passed

- Project creation from selector works and persists after reload.
- Chat task prompts produce deterministic multi-step plans with extracted classes/actions.
- Plan edit interactions work (configure panel opens reliably; skip/edit state is preserved).
- Edited plan payload is applied on execution start.
- Custom script step path now serializes to `custom_script` and executes (no parameter-leak crash).
- Review queue loads items, previews images, and supports approve/reject actions.
- Review filtering/search now keeps selection/index consistent (`1/1` instead of `0/1` under active filters).
- Point cloud web mode fails gracefully with a clear desktop-only message (no runtime crash).
- Mobile default sidebar state is collapsed for better first-load usability.
- Point cloud processing endpoints execute successfully with downloaded fixtures (`lamppost.pcd`, `milk.pcd`).
- Point cloud normals endpoint accepts both `search_radius` and legacy `radius`.

### Fixed Defects (This Run)

- `BUG-101` (execute): detect step failed after plan edit because `classes` was serialized as string.
  - Fix: backend execution node now coerces list-typed parameters from comma-separated strings.
  - Retest: backend execution tests pass with string->list coercion.
- `BUG-102` (review): current index displayed as `0/x` when current item was filtered out.
  - Fix: review queue now reselects first visible filtered item when filters/search change.
  - Retest: Playwright shows `1/1` for filtered result set.
- `BUG-103` (execution UI): checkpoint banner could render `NaN%` confidence and blank processed count.
  - Fix: checkpoint banner now sanitizes non-finite metrics to `0`.
- `BUG-104` (pointcloud API): route-level `HTTPException`s were converted to 500 due broad exception handling.
  - Fix: routes now re-raise `HTTPException` before generic exception handling.
  - Retest: API tests pass for expected 4xx paths.

### Notes

- Custom script generation falls back when configured coding models are missing, which can slow generation.
- Export flow now uses detect-generated YOLO annotations by default when detect+export are in the same plan.
- Browser (non-Tauri) mode still cannot open local point cloud files directly; expected desktop-only limitation.

## Bug Reporting Template

Use for every issue found:
- ID: `BUG-XXX`
- Area: chat/plan/execute/review/pointcloud/project/ui
- Steps to reproduce
- Actual result
- Expected result
- Severity: blocker/high/medium/low
- Evidence: screenshot + console/network notes
- Fix status: open/fixed/retested
