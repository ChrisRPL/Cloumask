# Full QA Playwright Suite

This directory contains the broad UI/API smoke suite in `full-qa.spec.ts`.

## Run

```bash
npx playwright test tests/e2e/full-qa.spec.ts
```

## Environment Variables

- `CLOUMASK_BACKEND_URL`: Backend base URL. Default: `http://localhost:8765`
- `CLOUMASK_TEST_IMAGE_DIR`: Absolute path to images used by review seeding tests. Default: repository `assets/`
- `CLOUMASK_E2E_SCREENSHOT_DIR`: Directory for suite screenshots. Default: `test-results/full-qa-screenshots`
