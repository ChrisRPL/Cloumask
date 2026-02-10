import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AppTestHost from '$lib/test-utils/AppTestHost.svelte';

function jsonResponse(body: unknown, init?: ResponseInit): Response {
	return new Response(JSON.stringify(body), {
		status: 200,
		headers: { 'content-type': 'application/json' },
		...init,
	});
}

function createFetchMock(options?: { llmReady?: boolean }) {
	const llmReady = options?.llmReady ?? true;

	return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
		const url =
			typeof input === 'string'
				? input
				: input instanceof URL
					? input.toString()
					: input.url;
		const method = (init?.method ?? 'GET').toUpperCase();

		if (url.endsWith('/llm/ensure-ready') && method === 'GET') {
			return jsonResponse({
				ready: llmReady,
				service_running: true,
				required_model: 'qwen3:14b',
				model_available: llmReady,
				error: llmReady ? null : 'Model missing',
			});
		}

		if (url.endsWith('/api/chat/thread/new') && method === 'POST') {
			return jsonResponse({
				thread_id: 'thread-test-1',
				created: true,
				awaiting_user: false,
				current_step: 0,
				total_steps: 0,
			});
		}

		if (url.includes('/api/review/items') && method === 'GET') {
			return jsonResponse({
				items: [],
				total: 0,
				skip: 0,
				limit: 50,
			});
		}

		if (url.includes('/api/chat/send/') && method === 'POST') {
			return jsonResponse({
				status: 'queued',
				thread_id: 'thread-test-1',
				message_id: 'message-1',
			});
		}

		return jsonResponse({});
	});
}

describe('App user flows', () => {
	beforeEach(() => {
		localStorage.clear();
		vi.stubGlobal('fetch', createFetchMock());
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it('allows first-time users to skip setup and enter the app shell', async () => {
		render(AppTestHost);

		expect(screen.getByText('Setting up Cloumask')).toBeTruthy();

		await fireEvent.click(screen.getByText('Skip setup (development only)'));

		await waitFor(() => {
			expect(screen.queryByText('Setting up Cloumask')).toBeNull();
		});

		expect(screen.getAllByText('Chat').length).toBeGreaterThan(0);
	});

	it('supports primary navigation via keyboard shortcuts for key user flows', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		render(AppTestHost);

		expect(screen.getAllByText('Chat').length).toBeGreaterThan(0);

		await fireEvent.keyDown(window, { key: '2' });
		await waitFor(() => {
			expect(screen.getByText('# PIPELINE')).toBeTruthy();
		});

		await fireEvent.keyDown(window, { key: '3' });
		await waitFor(() => {
			expect(screen.getByText('> LIVE PREVIEW')).toBeTruthy();
		});

		await fireEvent.keyDown(window, { key: '4' });
		await waitFor(() => {
			expect(screen.getByText('Review Queue')).toBeTruthy();
		});

		await fireEvent.keyDown(window, { key: ',' });
		await waitFor(() => {
			expect(screen.getByText('System Status')).toBeTruthy();
		});
	});

	it('offers a friendly first-run choice when AI model is not yet installed', async () => {
		vi.stubGlobal('fetch', createFetchMock({ llmReady: false }));
		render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('AI model setup')).toBeTruthy();
		});
		expect(screen.getByText('Download now (recommended)')).toBeTruthy();
		expect(screen.getByText('Continue without model')).toBeTruthy();

		await fireEvent.click(screen.getByText('Continue without model'));

		await waitFor(() => {
			expect(screen.queryByText('Setting up Cloumask')).toBeNull();
		});
		expect(screen.getAllByText('Chat').length).toBeGreaterThan(0);
	});
});
