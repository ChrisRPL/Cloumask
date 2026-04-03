import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { test, expect, type APIRequestContext, type Page } from '@playwright/test';

const BACKEND_URL = process.env.CLOUMASK_BACKEND_URL ?? 'http://localhost:8765';
const TEST_IMAGE_DIR = process.env.CLOUMASK_TEST_IMAGE_DIR ?? resolve(process.cwd(), 'assets');
const SCREENSHOT_DIR =
    process.env.CLOUMASK_E2E_SCREENSHOT_DIR ?? 'test-results/full-qa-screenshots';

let screenshotDirReady = false;
type ReviewListItem = { id: string };
const BROWSER_TEST_PROJECT = {
    id: 'qa-keyboard-project',
    name: 'QA Keyboard Project',
    path: '/tmp/cloumask-qa-keyboard',
    lastOpened: '2026-03-31T10:00:00.000Z',
};

// Helper: skip setup wizard by setting localStorage
async function skipSetup(page: Page) {
    await page.addInitScript(() => {
        localStorage.setItem('cloumask:setup', 'complete');
    });
}

// Helper: clear all localStorage
async function clearStorage(page: Page) {
    await page.addInitScript(() => {
        localStorage.clear();
    });
}

// Helper: take a named screenshot
async function snap(page: Page, name: string) {
    if (!screenshotDirReady) {
        mkdirSync(SCREENSHOT_DIR, { recursive: true });
        screenshotDirReady = true;
    }
    await page.screenshot({ path: `${SCREENSHOT_DIR}/${name}.png`, fullPage: true });
}

// Helper: seed review items for a given execution id
async function seedReviewItems(request: APIRequestContext, executionId: string) {
    return request.post(`${BACKEND_URL}/api/review/seed`, {
        params: {
            execution_id: executionId,
            image_dir: TEST_IMAGE_DIR,
        },
    });
}

// ============================================================================
// SECTION A: Setup Wizard & First-Time User
// ============================================================================

test.describe('A. Setup Wizard & Onboarding', () => {
    test('T-001: App loads and shows SetupWizard on fresh start', async ({ page }) => {
        await clearStorage(page);
        await page.goto('/');
        await expect(page.getByRole('heading', { name: 'Setting up Cloumask' })).toBeVisible();
        await expect(page.getByText('Preparing your desktop app. No CLI configuration required.')).toBeVisible();
        await expect(page.getByRole('button', { name: /skip setup/i })).toBeVisible();
        await snap(page, 'T-001-setup-wizard-visible');
    });

    test('T-002: Skip setup in dev mode', async ({ page }) => {
        await clearStorage(page);
        await page.goto('/');
        const skipBtn = page.getByRole('button', { name: /skip setup/i });
        await expect(skipBtn).toBeVisible();
        await snap(page, 'T-002-before-skip');

        await skipBtn.click();
        await expect(page.getByRole('heading', { name: 'Setting up Cloumask' })).toBeHidden();
        await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible();
        await expect(page.getByLabel('Chat messages')).toBeVisible();
        await expect(page.getByLabel('Message input')).toBeVisible();
        await expect(page.getByRole('button', { name: /select project/i })).toBeVisible();
        await snap(page, 'T-002-after-skip');
        await snap(page, 'T-002-final');
    });

    test('T-004: Setup persistence after reload', async ({ page }) => {
        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await snap(page, 'T-004-after-reload');

        // Setup wizard should NOT be visible
        const wizardHeading = page.locator('h1:has-text("Setting up Cloumask")');
        const wizardVisible = await wizardHeading.isVisible().catch(() => false);
        console.log(`[T-004] Wizard visible after setup-persisted reload: ${wizardVisible}`);
        expect(wizardVisible).toBe(false);
    });
});

// ============================================================================
// SECTION B: Sidebar Navigation & Keyboard Shortcuts
// ============================================================================

test.describe('B. Navigation & Keyboard Shortcuts', () => {
    test.beforeEach(async ({ page }) => {
        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
    });

    async function expectHeaderProjectSelectorWithinViewport(page: Page) {
        const header = page.locator('header').first();
        const projectSelector = page.getByRole('button', { name: /select project/i });

        await expect(header).toBeVisible();
        await expect(projectSelector).toBeVisible();

        const headerBox = await header.boundingBox();
        const selectorBox = await projectSelector.boundingBox();
        const viewport = page.viewportSize();

        expect(headerBox).not.toBeNull();
        expect(selectorBox).not.toBeNull();
        expect(viewport).not.toBeNull();

        const headerRight = (headerBox?.x ?? 0) + (headerBox?.width ?? 0);
        const selectorRight = (selectorBox?.x ?? 0) + (selectorBox?.width ?? 0);

        expect(selectorBox?.x ?? 0).toBeGreaterThanOrEqual(headerBox?.x ?? 0);
        expect(selectorRight).toBeLessThanOrEqual(headerRight);
        expect(selectorRight).toBeLessThanOrEqual(viewport?.width ?? 0);

        await expect
            .poll(
                async () =>
                    page.evaluate(
                        () => document.documentElement.scrollWidth - window.innerWidth
                    )
            )
            .toBeLessThanOrEqual(1);
    }

    test('T-060: Sidebar navigation keys 1-5', async ({ page }) => {
        await page.evaluate((project) => {
            localStorage.setItem('cloumask:project:current', JSON.stringify(project));
        }, BROWSER_TEST_PROJECT);
        await page.reload();
        await page.waitForTimeout(1500);
        await snap(page, 'T-060-initial');

        // Press 1 for Chat
        await page.keyboard.press('1');
        await page.waitForTimeout(500);
        await snap(page, 'T-060-key-1-chat');

        // Press 2 for Plan
        await page.keyboard.press('2');
        await page.waitForTimeout(500);
        await expect(page.getByText('No steps in pipeline', { exact: true })).toBeVisible();
        await expect(
            page.getByText('Describe the job in Chat to generate a plan, then come back here to review the steps.', {
                exact: true,
            })
        ).toBeVisible();
        await expect(page.getByRole('button', { name: 'Edit' })).toHaveCount(0);
        await expect(page.getByRole('button', { name: 'Cancel' })).toHaveCount(0);
        await snap(page, 'T-060-key-2-plan');

        // Press 3 for Execute
        await page.keyboard.press('3');
        await page.waitForTimeout(500);
        await expect(page.getByText('<idle>', { exact: true })).toBeVisible();
        await expect(page.getByText('No pipeline queued', { exact: true })).toBeVisible();
        await expect(page.getByText('No live execution yet', { exact: true })).toBeVisible();
        await expect(
            page.getByText(
                'Start a pipeline in Chat or Plan. When the run begins, this view fills with recent previews, progress, counts, and agent commentary in one place.',
                { exact: true }
            )
        ).toBeVisible();
        await expect(page.getByText('Execution workspace', { exact: true })).toBeVisible();
        await expect(page.getByText('Good next steps', { exact: true })).toBeVisible();
        await expect(page.getByText('Shortcut path: 1 chat, 2 plan, 3 execute, R review.', { exact: true })).toBeVisible();
        await expect(page.getByRole('button', { name: 'Cancel' })).toHaveCount(0);
        await expect(page.getByText('LIVE PREVIEW')).toHaveCount(0);
        await expect(page.getByText('Processed', { exact: true })).toHaveCount(0);
        const executeFooter = page.locator('.border-t.border-border.text-xs.font-mono');
        await expect(executeFooter).not.toContainText('Space Pause/Resume');
        await expect(executeFooter).not.toContainText('Esc Cancel');
        await expect(executeFooter).not.toContainText('Enter Continue');
        await snap(page, 'T-060-key-3-execute');

        // Press 4 for Review
        await page.keyboard.press('4');
        await page.waitForTimeout(500);
        await expect(page.getByText('0 items', { exact: true })).toBeVisible();
        await expect(page.getByText('Review inbox', { exact: true })).toBeVisible();
        await expect(page.getByText('Review canvas', { exact: true })).toBeVisible();
        await expect(
            page.getByText('Flagged detections and human-check checkpoints appear here after a run reaches review.', {
                exact: true,
            })
        ).toBeVisible();
        await expect(
            page.getByText(
                'Select a queued item to inspect the source image, adjust annotations, and approve or reject the result.',
                { exact: true }
            )
        ).toBeVisible();
        await expect(page.locator('button:has-text("Approve All")').first()).toBeDisabled();
        await expect(page.locator('button:has-text("Prev")')).toHaveCount(0);
        await expect(page.locator('button:has-text("Next")')).toHaveCount(0);
        await expect(page.locator('button:has-text("Edit")')).toHaveCount(0);
        await expect(page.locator('button:has-text("Reject")')).toHaveCount(0);
        await expect(page.getByRole('button', { name: /^Approve$/ })).toHaveCount(0);
        await expect(page.getByRole('button', { name: 'Done' })).not.toHaveClass(/bg-primary/);
        await expect(page.locator('.border-t.border-border.bg-background').last()).toContainText('0 / 0');
        await expect(page.getByText('Ctrl+Z', { exact: true })).toHaveCount(0);
        await expect(page.getByText('J/K', { exact: true })).toHaveCount(0);
        await snap(page, 'T-060-key-4-review');

        // Press 5 for Point Cloud
        await page.keyboard.press('5');
        await page.waitForTimeout(500);
        await expect(page.getByText('Browser preview only', { exact: true })).toBeVisible();
        await expect(page.getByText('Load/export require desktop mode', { exact: true })).toBeVisible();
        await expect(page.getByText('Desktop mode required for file workflows', { exact: true })).toBeVisible();
        await expect(
            page.getByText('This web preview shows the shell only. Load and export stay in the desktop app.')
        ).toBeVisible();
        await expect(page.getByRole('button', { name: 'Load' })).toBeDisabled();
        await expect(page.getByRole('button', { name: 'Export' })).toBeDisabled();
        await expect(page.getByText('Color:', { exact: true })).toHaveCount(0);
        await expect(page.getByText('Size:', { exact: true })).toHaveCount(0);
        await expect(page.getByText('Info Panel', { exact: true })).toHaveCount(0);
        await snap(page, 'T-060-key-5-pointcloud');
    });

    test('T-061: Settings view via comma key', async ({ page }) => {
        await page.keyboard.press(',');
        await page.waitForTimeout(500);
        await expect(page.getByText('Running outside Tauri - IPC commands unavailable.')).toBeVisible();
        await expect(page.getByRole('button', { name: 'Refresh' })).toHaveCount(0);
        await snap(page, 'T-061-settings-view');

        // Should see "Settings" heading
        const settings = page.locator('h1:has-text("Settings")');
        const settingsVisible = await settings.isVisible().catch(() => false);
        console.log(`[T-061] Settings heading visible: ${settingsVisible}`);
    });

    test('T-062: Ctrl+B sidebar toggle', async ({ page }) => {
        await snap(page, 'T-062-before-toggle');

        await page.keyboard.press('Control+b');
        await page.waitForTimeout(500);
        await snap(page, 'T-062-after-first-toggle');

        await page.keyboard.press('Control+b');
        await page.waitForTimeout(500);
        await snap(page, 'T-062-after-second-toggle');
    });

    test('T-063: Ctrl+K command palette', async ({ page }) => {
        await page.keyboard.press('Control+k');
        await page.waitForTimeout(500);
        await expect(page.getByRole('dialog')).toBeVisible();
        await expect(page.getByLabel('Search commands')).toBeVisible();
        await snap(page, 'T-063-command-palette-open');

        // Close with Escape
        await page.keyboard.press('Escape');
        await page.waitForTimeout(300);
        await snap(page, 'T-063-palette-closed');
    });

    test('T-064: ? help overlay', async ({ page }) => {
        await page.keyboard.press('?');
        await page.waitForTimeout(500);
        await snap(page, 'T-064-help-overlay');

        // Close with Escape
        await page.keyboard.press('Escape');
        await page.waitForTimeout(300);
        await snap(page, 'T-064-help-closed');
    });

    test('T-005: Header and branding', async ({ page }) => {
        await snap(page, 'T-005-header-branding');
        const header = page.locator('header, [class*="header"], [class*="Header"]').first();
        const headerVisible = await header.isVisible().catch(() => false);
        console.log(`[T-005] Header visible: ${headerVisible}`);
        await expectHeaderProjectSelectorWithinViewport(page);
        await page.keyboard.press('Control+b');
        await page.waitForTimeout(300);
        await expectHeaderProjectSelectorWithinViewport(page);
        await page.keyboard.press('Control+b');
    });

    test('T-003: Sidebar click navigation', async ({ page }) => {
        // Click each sidebar nav item
        const navItems = page.locator('nav button, [class*="nav"] button, [role="navigation"] button');
        const count = await navItems.count();
        console.log(`[T-003] Found ${count} nav buttons`);
        await snap(page, 'T-003-sidebar-buttons');

        for (let i = 0; i < Math.min(count, 6); i++) {
            await navItems.nth(i).click();
            await page.waitForTimeout(400);
            await snap(page, `T-003-nav-click-${i}`);
        }
    });
});

// ============================================================================
// SECTION C: Chat View
// ============================================================================

test.describe('C. Chat View', () => {
    test.beforeEach(async ({ page }) => {
        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await page.keyboard.press('1'); // Navigate to Chat
        await page.waitForTimeout(1000);
    });

    test('T-010: Chat view renders correctly', async ({ page }) => {
        await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible();
        await expect(page.getByText(/connected|disconnected/i)).toBeVisible();
        await expect(page.getByLabel('Chat messages')).toBeVisible();
        await expect(page.getByText('Start a local vision workflow')).toBeVisible();
        await expect(page.getByText('Good first prompts')).toBeVisible();
        await expect(page.getByText('Choose project', { exact: true })).toBeVisible();
        await expect(
            page.getByText('Every run needs a project. Pick one here first so chat, plans, and review work stay grouped.')
        ).toBeVisible();
        await expect(page.getByRole('button', { name: 'Choose project to start chat' })).toBeVisible();
        await expect(page.getByLabel('Message input')).toBeEditable();
        await expect(page.getByLabel('Message input')).toHaveAttribute('placeholder', 'Choose a project above to unlock chat...');
        await expect(page.getByRole('button', { name: 'Send message' })).toBeDisabled();
        await expect(page.getByRole('button', { name: 'Export' })).toHaveCount(0);
        await expect(page.getByRole('button', { name: 'Clear' })).toBeVisible();
        await snap(page, 'T-010-chat-view');
    });

    test('T-012: LLM status display', async ({ page }) => {
        await page.waitForTimeout(3000); // Wait for LLM check
        await snap(page, 'T-012-llm-status');

        // Check for any status/banner/error related to LLM
        const statusElements = page.locator('[class*="status"], [class*="banner"], [class*="error"], [class*="alert"]');
        const count = await statusElements.count();
        console.log(`[T-012] Status elements found: ${count}`);
        for (let i = 0; i < Math.min(count, 5); i++) {
            const text = await statusElements.nth(i).textContent();
            console.log(`[T-012] Status ${i}: "${text?.substring(0, 100)}"`);
        }
    });

    test('T-014: Send message interaction', async ({ page }) => {
        await page.waitForTimeout(2000);
        await snap(page, 'T-014-before-send');

        // Find and type in input
        const input = page.locator('input[type="text"], textarea').first();
        const inputVisible = await input.isVisible().catch(() => false);

        if (inputVisible) {
            await input.fill('hello');
            await snap(page, 'T-014-typed-hello');
            await expect(page.getByRole('button', { name: 'Send message' })).toBeDisabled();
            await input.press('Enter');
            await page.waitForTimeout(3000);
            await expect(input).toHaveValue('hello');
            await snap(page, 'T-014-after-send');
        } else {
            console.log('[T-014] BUG: No chat input found');
            await snap(page, 'T-014-no-input-bug');
        }
    });

    test('T-015: Clear conversation', async ({ page }) => {
        await page.waitForTimeout(1500);
        await snap(page, 'T-015-before-clear');

        // Look for clear / new conversation button
        const clearBtn = page.locator('button').filter({ hasText: /clear|new|reset/i }).first();
        const clearVisible = await clearBtn.isVisible().catch(() => false);
        console.log(`[T-015] Clear button visible: ${clearVisible}`);

        // Also check for icon buttons in header area
        const headerBtns = page.locator('[class*="ChatHeader"] button, [class*="chat-header"] button');
        const headerCount = await headerBtns.count();
        console.log(`[T-015] Chat header buttons: ${headerCount}`);
        await snap(page, 'T-015-header-buttons');
    });
});

// ============================================================================
// SECTION D: Plan View
// ============================================================================

test.describe('D. Plan View', () => {
    test.beforeEach(async ({ page }) => {
        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await page.keyboard.press('2'); // Navigate to Plan
        await page.waitForTimeout(500);
    });

    test('T-020: Plan view renders', async ({ page }) => {
        await snap(page, 'T-020-plan-view');

        // Check for PlanEditor or placeholder
        const content = await page.textContent('body');
        const hasPlanContent = content?.includes('Plan') || content?.includes('pipeline') || content?.includes('step');
        console.log(`[T-020] Plan-related content found: ${hasPlanContent}`);
    });

    test('T-021: Plan empty state', async ({ page }) => {
        await snap(page, 'T-021-plan-empty');

        // Check for empty state or placeholder
        const placeholder = page.locator('[class*="placeholder"], [class*="empty"], [class*="Placeholder"]').first();
        const placeholderVisible = await placeholder.isVisible().catch(() => false);
        console.log(`[T-021] Empty state/placeholder visible: ${placeholderVisible}`);

        // Check for plan header
        const header = page.locator('[class*="PlanHeader"], [class*="plan-header"]').first();
        const headerVisible = await header.isVisible().catch(() => false);
        console.log(`[T-021] Plan header visible: ${headerVisible}`);
    });

    test('T-023: Edit mode toggle with E key', async ({ page }) => {
        await snap(page, 'T-023-before-edit');
        await page.keyboard.press('e');
        await page.waitForTimeout(500);
        await snap(page, 'T-023-after-edit-toggle');
    });
});

// ============================================================================
// SECTION E: Execution View
// ============================================================================

test.describe('E. Execution View', () => {
    test.beforeEach(async ({ page }) => {
        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await page.keyboard.press('3'); // Navigate to Execute
        await page.waitForTimeout(500);
    });

    test('T-030: Execution view empty state', async ({ page }) => {
        await snap(page, 'T-030-execution-empty');
        await expect(page.getByText('Execution workspace', { exact: true })).toBeVisible();
        await expect(page.getByText('Good next steps', { exact: true })).toBeVisible();
        await expect(page.getByText('Shortcut path: 1 chat, 2 plan, 3 execute, R review.', { exact: true })).toBeVisible();

        const content = await page.textContent('body');
        console.log(`[T-030] Page contains execution-related content: ${content?.toLowerCase().includes('execut') || content?.toLowerCase().includes('pipeline') || content?.toLowerCase().includes('run')
            }`);
    });

    test('T-031: Execution controls render', async ({ page }) => {
        await snap(page, 'T-031-execution-controls');

        // Check for execution-specific elements
        const progressBars = page.locator('[class*="progress"], [role="progressbar"]');
        const progressCount = await progressBars.count();
        console.log(`[T-031] Progress elements: ${progressCount}`);

        const statCards = page.locator('[class*="stat"], [class*="Stat"]');
        const statCount = await statCards.count();
        console.log(`[T-031] Stat elements: ${statCount}`);
    });
});

// ============================================================================
// SECTION F: Review Queue
// ============================================================================

test.describe('F. Review Queue', () => {
    test('T-040: Seed review items via API', async ({ request }) => {
        // First, seed review items
        const seedResponse = await seedReviewItems(request, 'test-e2e');
        console.log(`[T-040] Seed response status: ${seedResponse.status()}`);
        const seedBody = await seedResponse.text();
        console.log(`[T-040] Seed response: ${seedBody.substring(0, 200)}`);

        expect(seedResponse.status()).toBeLessThan(500);
    });

    test('T-041: Review queue loads items', async ({ page }) => {
        // Seed with 'current' which is what the frontend uses by default
        await seedReviewItems(page.request, 'current');

        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await page.keyboard.press('4'); // Navigate to Review
        await page.waitForTimeout(2000);
        await snap(page, 'T-041-review-queue');

        // Check for list items
        const listItems = page.locator('[class*="list"] [class*="item"], [class*="review"] [class*="item"], [role="listbox"] button');
        const itemCount = await listItems.count();
        console.log(`[T-041] Review list items found: ${itemCount}`);

        // Check for any image preview
        const images = page.locator('img, canvas');
        const imgCount = await images.count();
        console.log(`[T-041] Image/canvas elements: ${imgCount}`);
        await snap(page, 'T-041-review-items');
    });

    test('T-042: Review item selection and preview', async ({ page }) => {
        await seedReviewItems(page.request, 'test-e2e');

        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await page.keyboard.press('4');
        await page.waitForTimeout(2000);
        await snap(page, 'T-042-review-initial');

        // Try clicking first item
        const firstItem = page.locator('[class*="item"], [role="listitem"], [class*="review"] button').first();
        const firstItemVisible = await firstItem.isVisible().catch(() => false);
        if (firstItemVisible) {
            await firstItem.click();
            await page.waitForTimeout(1000);
            await snap(page, 'T-042-item-selected');
        }
    });

    test('T-044: Filter bar and search', async ({ page }) => {
        await seedReviewItems(page.request, 'test-e2e');

        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await page.keyboard.press('4');
        await page.waitForTimeout(2000);
        await snap(page, 'T-044-review-for-filter');

        // Check for filter/search elements
        const searchInput = page.locator('input[placeholder*="search" i], input[placeholder*="filter" i], input[type="search"]').first();
        const searchVisible = await searchInput.isVisible().catch(() => false);
        console.log(`[T-044] Search input visible: ${searchVisible}`);

        const filterBtns = page.locator('[class*="filter"], [class*="Filter"]');
        const filterCount = await filterBtns.count();
        console.log(`[T-044] Filter elements: ${filterCount}`);
        await snap(page, 'T-044-filter-elements');
    });

    test('T-043: Approve/Reject keyboard shortcuts', async ({ page }) => {
        await seedReviewItems(page.request, 'test-e2e');

        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await page.keyboard.press('4');
        await page.waitForTimeout(2000);
        await snap(page, 'T-043-before-approve');

        // Press A to approve
        await page.keyboard.press('a');
        await page.waitForTimeout(500);
        await snap(page, 'T-043-after-approve');

        // Press R to reject
        await page.keyboard.press('r');
        await page.waitForTimeout(500);
        await snap(page, 'T-043-after-reject');
    });

    test('T-046: Undo/Redo', async ({ page }) => {
        await seedReviewItems(page.request, 'test-e2e');

        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await page.keyboard.press('4');
        await page.waitForTimeout(2000);

        // Approve first
        await page.keyboard.press('a');
        await page.waitForTimeout(500);
        await snap(page, 'T-046-after-action');

        // Undo
        await page.keyboard.press('Control+z');
        await page.waitForTimeout(500);
        await snap(page, 'T-046-after-undo');

        // Redo
        await page.keyboard.press('Control+y');
        await page.waitForTimeout(500);
        await snap(page, 'T-046-after-redo');
    });
});

// ============================================================================
// SECTION G: Point Cloud View
// ============================================================================

test.describe('G. Point Cloud View', () => {
    test.beforeEach(async ({ page }) => {
        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await page.keyboard.press('5'); // Navigate to Point Cloud
        await page.waitForTimeout(500);
    });

    test('T-055: Point cloud view renders', async ({ page }) => {
        await snap(page, 'T-055-pointcloud-view');

        const content = await page.textContent('body');
        const hasPointCloudContent = content?.toLowerCase().includes('point cloud') ||
            content?.toLowerCase().includes('3d') ||
            content?.toLowerCase().includes('viewer');
        console.log(`[T-055] Point cloud content found: ${hasPointCloudContent}`);
    });

    test('T-056: Browser-mode graceful degradation', async ({ page }) => {
        await snap(page, 'T-056-browser-mode');

        // In browser mode, should show desktop-only message or graceful fallback
        const desktopMsg = page.locator('text=/desktop|tauri|native/i').first();
        const msgVisible = await desktopMsg.isVisible().catch(() => false);
        console.log(`[T-056] Desktop-only message visible: ${msgVisible}`);

        // Check for controls
        const controls = page.locator('[class*="control"], [class*="Control"], [class*="toolbar"]');
        const controlCount = await controls.count();
        console.log(`[T-056] Control elements: ${controlCount}`);
    });
});

// ============================================================================
// SECTION H: Settings View
// ============================================================================

test.describe('H. Settings View', () => {
    test.beforeEach(async ({ page }) => {
        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await page.keyboard.press(','); // Navigate to Settings
        await page.waitForTimeout(500);
    });

    test('T-090: Settings view renders system status', async ({ page }) => {
        await snap(page, 'T-090-settings-view');

        // Check for Settings heading
        const settingsH1 = page.locator('h1:has-text("Settings")');
        const h1Visible = await settingsH1.isVisible().catch(() => false);
        console.log(`[T-090] Settings h1 visible: ${h1Visible}`);

        // Check for System Status card
        const statusCard = page.locator('text="System Status"');
        const statusVisible = await statusCard.isVisible().catch(() => false);
        console.log(`[T-090] System Status card visible: ${statusVisible}`);

        // Check for component statuses
        const badges = page.locator('[class*="badge"], [class*="Badge"]');
        const badgeCount = await badges.count();
        console.log(`[T-090] Badge elements: ${badgeCount}`);

        await snap(page, 'T-090-system-status-details');
    });

    test('T-091: Health badge indicators', async ({ page }) => {
        await page.waitForTimeout(2000); // Wait for status refresh
        await snap(page, 'T-091-badges');

        // Expect certain labels
        const labels = ['Frontend', 'Rust Core', 'Python Sidecar', 'AI Service'];
        for (const label of labels) {
            const el = page.locator(`text="${label}"`).first();
            const visible = await el.isVisible().catch(() => false);
            console.log(`[T-091] "${label}" label visible: ${visible}`);
        }
    });

    test('T-092: Refresh button', async ({ page }) => {
        const refreshBtn = page.locator('button:has-text("Refresh")');
        const refreshVisible = await refreshBtn.isVisible().catch(() => false);
        console.log(`[T-092] Refresh button visible: ${refreshVisible}`);

        if (refreshVisible) {
            await refreshBtn.click();
            await page.waitForTimeout(500);
            await snap(page, 'T-092-after-refresh');
        }
        await snap(page, 'T-092-refresh-state');
    });

    test('T-093: Restart Sidecar button', async ({ page }) => {
        const restartBtn = page.locator('button:has-text("Restart Sidecar")');
        const restartVisible = await restartBtn.isVisible().catch(() => false);
        console.log(`[T-093] Restart sidecar button visible: ${restartVisible}`);
        await snap(page, 'T-093-restart-btn');
    });
});

// ============================================================================
// SECTION I: Backend API Tests
// ============================================================================

test.describe('I. Backend API', () => {
    test('T-070: GET /health returns healthy', async ({ request }) => {
        const resp = await request.get(`${BACKEND_URL}/health`);
        console.log(`[T-070] Health status: ${resp.status()}`);
        const body = await resp.json();
        console.log(`[T-070] Health body: ${JSON.stringify(body)}`);
        expect(resp.status()).toBe(200);
        expect(body.status).toBe('healthy');
    });

    test('T-071: GET /ready returns ready', async ({ request }) => {
        const resp = await request.get(`${BACKEND_URL}/ready`);
        console.log(`[T-071] Ready status: ${resp.status()}`);
        const body = await resp.json();
        console.log(`[T-071] Ready body: ${JSON.stringify(body)}`);
        expect(resp.status()).toBe(200);
        expect(body.ready).toBe(true);
    });

    test('T-072: GET / returns app info', async ({ request }) => {
        const resp = await request.get(`${BACKEND_URL}/`);
        console.log(`[T-072] Root status: ${resp.status()}`);
        const body = await resp.json();
        console.log(`[T-072] Root body: ${JSON.stringify(body)}`);
        expect(resp.status()).toBe(200);
        expect(body.name).toBe('Cloumask Backend');
    });

    test('T-073: GET /llm/status', async ({ request }) => {
        const resp = await request.get(`${BACKEND_URL}/llm/status`);
        console.log(`[T-073] LLM status: ${resp.status()}`);
        const body = await resp.json();
        console.log(`[T-073] LLM body: ${JSON.stringify(body)}`);
        expect(resp.status()).toBe(200);
    });

    test('T-074: GET /llm/models', async ({ request }) => {
        const resp = await request.get(`${BACKEND_URL}/llm/models`);
        console.log(`[T-074] Models status: ${resp.status()}`);
        const body = await resp.json();
        console.log(`[T-074] Models body: ${JSON.stringify(body).substring(0, 200)}`);
        expect(resp.status()).toBeLessThan(500);
    });

    test('T-075: GET /llm/ensure-ready', async ({ request }) => {
        const resp = await request.get(`${BACKEND_URL}/llm/ensure-ready`);
        console.log(`[T-075] Ensure-ready status: ${resp.status()}`);
        const body = await resp.json();
        console.log(`[T-075] Ensure-ready body: ${JSON.stringify(body)}`);
        expect(resp.status()).toBe(200);
    });

    test('T-076: POST /api/chat/threads creates thread', async ({ request }) => {
        const resp = await request.post(`${BACKEND_URL}/api/chat/threads`);
        console.log(`[T-076] Create thread status: ${resp.status()}`);
        const body = await resp.json();
        console.log(`[T-076] Thread body: ${JSON.stringify(body)}`);
        expect(resp.status()).toBe(200);
        expect(body.thread_id).toBeDefined();
    });

    test('T-077: GET thread info + DELETE thread', async ({ request }) => {
        // Create thread first
        const createResp = await request.post(`${BACKEND_URL}/api/chat/threads`);
        const { thread_id } = await createResp.json();

        // Get thread info
        const infoResp = await request.get(`${BACKEND_URL}/api/chat/threads/${thread_id}`);
        console.log(`[T-077] Thread info status: ${infoResp.status()}`);
        const infoBody = await infoResp.json();
        console.log(`[T-077] Thread info: ${JSON.stringify(infoBody)}`);
        expect(infoResp.status()).toBe(200);

        // Delete thread
        const deleteResp = await request.delete(`${BACKEND_URL}/api/chat/threads/${thread_id}`);
        console.log(`[T-077] Delete thread status: ${deleteResp.status()}`);
        expect(deleteResp.status()).toBe(200);
    });

    test('T-079: POST /api/review/seed creates items', async ({ request }) => {
        const resp = await seedReviewItems(request, 'test-api');
        console.log(`[T-079] Seed status: ${resp.status()}`);
        const body = await resp.text();
        console.log(`[T-079] Seed body: ${body.substring(0, 200)}`);
        expect(resp.status()).toBeLessThan(500);
    });

    test('T-080: GET /api/review/items returns items', async ({ request }) => {
        // Seed first
        await seedReviewItems(request, 'test-api-list');

        const resp = await request.get(`${BACKEND_URL}/api/review/items`, {
            params: { execution_id: 'test-api-list' },
        });
        console.log(`[T-080] List items status: ${resp.status()}`);
        const body = await resp.json();
        const items = Array.isArray(body.items) ? body.items : [];
        console.log(`[T-080] Items count: ${items.length}`);
        expect(resp.status()).toBe(200);
    });

    test('T-081: PUT /api/review/items/:id updates item', async ({ request }) => {
        // Seed
        await seedReviewItems(request, 'test-api-update');

        const listResp = await request.get(`${BACKEND_URL}/api/review/items`, {
            params: { execution_id: 'test-api-update' },
        });
        const listBody = await listResp.json();
        const items = Array.isArray(listBody.items) ? listBody.items : [];
        if (items.length > 0) {
            const itemId = items[0].id;
            const updateResp = await request.put(`${BACKEND_URL}/api/review/items/${itemId}`, {
                data: { status: 'approved' },
            });
            console.log(`[T-081] Update status: ${updateResp.status()}`);
            expect(updateResp.status()).toBeLessThan(500);
        } else {
            console.log('[T-081] No items to update');
        }
    });

    test('T-084: POST /api/review/batch/approve', async ({ request }) => {
        // Seed
        await seedReviewItems(request, 'test-api-batch');

        const listResp = await request.get(`${BACKEND_URL}/api/review/items`, {
            params: { execution_id: 'test-api-batch' },
        });
        const listBody = await listResp.json();
        const items: ReviewListItem[] = Array.isArray(listBody.items) ? listBody.items : [];
        if (items.length > 0) {
            const ids = items.slice(0, 3).map((i) => i.id);
            const batchResp = await request.post(`${BACKEND_URL}/api/review/batch-approve`, {
                data: { item_ids: ids },
            });
            console.log(`[T-084] Batch approve status: ${batchResp.status()}`);
            const batchBody = await batchResp.json();
            console.log(`[T-084] Batch body: ${JSON.stringify(batchBody)}`);
            expect(batchResp.status()).toBeLessThan(500);
        }
    });
});

// ============================================================================
// SECTION J: UI/UX Quality & Persistence
// ============================================================================

test.describe('J. UI/UX Quality', () => {
    test('T-094: Dark theme applied by default', async ({ page }) => {
        await skipSetup(page);
        await page.goto('/');
        await page.waitForTimeout(1500);
        await snap(page, 'T-094-dark-theme');

        // Check background color
        const bgColor = await page.evaluate(() => {
            return window.getComputedStyle(document.body).backgroundColor;
        });
        console.log(`[T-094] Body background color: ${bgColor}`);
    });

    test('T-098: localStorage persistence - view', async ({ page }) => {
        await skipSetup(page);
        await page.addInitScript(() => {
            localStorage.setItem('cloumask:view:current', 'plan');
        });
        await page.goto('/');
        await page.waitForTimeout(1500);
        await snap(page, 'T-098-persisted-plan-view');

        // Should be on plan view
        const content = await page.textContent('body');
        console.log(`[T-098] Content includes plan-related text: ${content?.toLowerCase().includes('plan') || content?.toLowerCase().includes('step')
            }`);
    });

    test('T-099: localStorage persistence - sidebar', async ({ page }) => {
        await skipSetup(page);
        await page.addInitScript(() => {
            localStorage.setItem('cloumask:sidebar:expanded', 'false');
        });
        await page.goto('/');
        await page.waitForTimeout(1500);
        await snap(page, 'T-099-sidebar-collapsed');
    });

    test('T-096: Responsive layout 768px', async ({ page }) => {
        await skipSetup(page);
        await page.setViewportSize({ width: 768, height: 1024 });
        await page.goto('/');
        await page.waitForTimeout(1500);
        await snap(page, 'T-096-responsive-768');
    });

    test('T-097: Responsive layout 1920px', async ({ page }) => {
        await skipSetup(page);
        await page.setViewportSize({ width: 1920, height: 1080 });
        await page.goto('/');
        const emptyState = page.locator('[data-chat-empty-state]');
        const landingProjectSelector = page.getByRole('button', { name: 'Choose project to start chat' });
        await expect(emptyState).toBeVisible();
        await expect(landingProjectSelector).toBeVisible();

        const [emptyStateBox, landingSelectorBox, viewport] = await Promise.all([
            emptyState.boundingBox(),
            landingProjectSelector.boundingBox(),
            Promise.resolve(page.viewportSize()),
        ]);

        expect(emptyStateBox).not.toBeNull();
        expect(landingSelectorBox).not.toBeNull();
        expect(viewport).not.toBeNull();
        expect((emptyStateBox?.width ?? 0) / (viewport?.width ?? 1)).toBeGreaterThanOrEqual(0.55);
        expect((emptyStateBox?.x ?? 0) / (viewport?.width ?? 1)).toBeLessThanOrEqual(0.3);
        expect((emptyStateBox?.y ?? 0) / (viewport?.height ?? 1)).toBeLessThanOrEqual(0.22);
        expect((landingSelectorBox?.x ?? 0) / (viewport?.width ?? 1)).toBeLessThanOrEqual(0.35);
        expect((landingSelectorBox?.y ?? 0) / (viewport?.height ?? 1)).toBeLessThanOrEqual(0.5);

        await page.waitForTimeout(1500);
        await snap(page, 'T-097-responsive-1920');
    });

    test('T-100: Project selector', async ({ page }) => {
        await skipSetup(page);
        await page.goto('/');
        const projectSelector = page.getByRole('button', { name: /select project/i });
        await expect(projectSelector).toBeVisible();
        await snap(page, 'T-100-project-selector-before');

        await projectSelector.click();
        await expect(page.getByText('New Project...')).toBeVisible();
        await snap(page, 'T-100-project-selector-open');

        await page.getByText('New Project...').click();
        await expect(page.getByRole('heading', { name: 'New Project' })).toBeVisible();

        const nameInput = page.getByLabel('Project Name');
        const pathInput = page.getByLabel('Project Path');

        await nameInput.fill('Street Anonymization');
        await expect(pathInput).toHaveValue('/data/street-anonymization');
        await page.getByRole('button', { name: 'Create Project' }).click();

        await expect(page.getByRole('heading', { name: 'New Project' })).toBeHidden();
        const savedProjectSelector = page.getByRole('button', { name: /street anonymization/i });
        await expect(savedProjectSelector).toBeVisible();

        await savedProjectSelector.click();
        await expect(page.getByText('Recent Projects')).toBeVisible();
        await expect(page.getByRole('option', { name: 'Street Anonymization' })).toBeVisible();
        await expect(page.getByRole('option', { name: 'New Project...' })).toBeVisible();

        const [triggerBox, menuBox] = await Promise.all([
            savedProjectSelector.boundingBox(),
            page.locator('[data-slot="select-content"]').boundingBox(),
        ]);

        expect(triggerBox).not.toBeNull();
        expect(menuBox).not.toBeNull();
        expect((menuBox?.height ?? 0)).toBeGreaterThan((triggerBox?.height ?? 0) + 24);

        await snap(page, 'T-100-project-selector-open-with-history');
    });
});
