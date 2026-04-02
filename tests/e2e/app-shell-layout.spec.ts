import { expect, test, type Page } from '@playwright/test';

async function skipSetup(page: Page) {
	await page.addInitScript(() => {
		localStorage.setItem('cloumask:setup', 'complete');
	});
}

async function assertShellFitsViewport(page: Page) {
	const header = page.locator('header').first();
	const connectionStatus = page.locator('[data-slot="connection-status"]').first();
	const projectSelector = page.getByRole('button', { name: /select project/i });
	const messageInput = page.getByLabel('Message input');

	await expect(header).toBeVisible();
	await expect(connectionStatus).toBeVisible();
	await expect(connectionStatus).toContainText(/live|booting|retry|offline|error/i);
	await expect(projectSelector).toBeVisible();
	await expect(messageInput).toBeVisible();

	const [headerBox, connectionBox, projectBox, inputBox, viewport] = await Promise.all([
		header.boundingBox(),
		connectionStatus.boundingBox(),
		projectSelector.boundingBox(),
		messageInput.boundingBox(),
		Promise.resolve(page.viewportSize()),
	]);

	expect(headerBox).not.toBeNull();
	expect(connectionBox).not.toBeNull();
	expect(projectBox).not.toBeNull();
	expect(inputBox).not.toBeNull();
	expect(viewport).not.toBeNull();

	expect(connectionBox!.x).toBeGreaterThanOrEqual(headerBox!.x - 1);
	expect(connectionBox!.x + connectionBox!.width).toBeLessThanOrEqual(headerBox!.x + headerBox!.width + 1);
	expect(projectBox!.x).toBeGreaterThanOrEqual(headerBox!.x - 1);
	expect(projectBox!.x + projectBox!.width).toBeLessThanOrEqual(headerBox!.x + headerBox!.width + 1);
	expect(projectBox!.x + projectBox!.width).toBeLessThanOrEqual(viewport!.width + 1);
	expect(viewport!.width - (projectBox!.x + projectBox!.width)).toBeGreaterThanOrEqual(8);
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
		for (const viewport of [
			{ width: 1232, height: 793 },
			{ width: 1100, height: 760 },
			{ width: 960, height: 720 },
			{ width: 820, height: 700 },
			{ width: 760, height: 680 },
		]) {
			await page.setViewportSize(viewport);
			await assertShellFitsViewport(page);

			await page.keyboard.press('Control+b');
			await assertShellFitsViewport(page);

			await page.keyboard.press('Control+b');
			await assertShellFitsViewport(page);
		}
	});
});
