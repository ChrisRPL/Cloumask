import { expect, test, type Page } from '@playwright/test';

async function skipSetup(page: Page) {
	await page.addInitScript(() => {
		localStorage.setItem('cloumask:setup', 'complete');
	});
}

async function assertShellFitsViewport(page: Page) {
	const header = page.locator('header').first();
	const projectSelector = page.getByRole('button', { name: /select project/i });
	const messageInput = page.getByLabel('Message input');

	await expect(header).toBeVisible();
	await expect(projectSelector).toBeVisible();
	await expect(messageInput).toBeVisible();

	const [headerBox, projectBox, inputBox, viewport] = await Promise.all([
		header.boundingBox(),
		projectSelector.boundingBox(),
		messageInput.boundingBox(),
		Promise.resolve(page.viewportSize()),
	]);

	expect(headerBox).not.toBeNull();
	expect(projectBox).not.toBeNull();
	expect(inputBox).not.toBeNull();
	expect(viewport).not.toBeNull();

	expect(projectBox!.x).toBeGreaterThanOrEqual(headerBox!.x - 1);
	expect(projectBox!.x + projectBox!.width).toBeLessThanOrEqual(headerBox!.x + headerBox!.width + 1);
	expect(projectBox!.x + projectBox!.width).toBeLessThanOrEqual(viewport!.width + 1);
	expect(inputBox!.y + inputBox!.height).toBeLessThanOrEqual(viewport!.height + 1);

	await expect
		.poll(() => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth))
		.toBeLessThanOrEqual(1);
}

test.describe('App shell layout', () => {
	test.beforeEach(async ({ page }) => {
		await skipSetup(page);
		await page.goto('/');
		await expect(page.getByRole('heading', { name: 'Chat' })).toBeVisible();
	});

	test('keeps header controls and chat input inside the viewport', async ({ page }) => {
		await assertShellFitsViewport(page);

		await page.keyboard.press('Control+b');
		await assertShellFitsViewport(page);

		await page.keyboard.press('Control+b');
		await assertShellFitsViewport(page);
	});
});
