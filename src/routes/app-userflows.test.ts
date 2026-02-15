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

function createFetchMock(options?: {
	llmReady?: boolean;
	llmReadyAfterPull?: boolean;
	failThreadCreate?: boolean;
}) {
	const llmReady = options?.llmReady ?? true;
	const llmReadyAfterPull = options?.llmReadyAfterPull ?? llmReady;
	const failThreadCreate = options?.failThreadCreate ?? false;
	let modelPulled = false;

	return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
		const url =
			typeof input === 'string'
				? input
				: input instanceof URL
					? input.toString()
					: input.url;
		const method = (init?.method ?? 'GET').toUpperCase();

		if (url.endsWith('/llm/ensure-ready') && method === 'GET') {
			const ready = llmReady || (llmReadyAfterPull && modelPulled);
			return jsonResponse({
				ready,
				service_running: true,
				required_model: 'qwen3:8b',
				model_available: ready,
				error: ready ? null : 'Model missing',
			});
		}

		if (url.endsWith('/llm/ensure-ready') && method === 'POST') {
			return jsonResponse({
				ready: llmReadyAfterPull,
				service_running: true,
				required_model: 'qwen3:8b',
				model_available: llmReadyAfterPull,
				error: llmReadyAfterPull ? null : 'Model download failed',
			});
		}

		if (url.endsWith('/llm/pull/stream') && method === 'POST') {
			modelPulled = true;
			return new Response(
				'data: {"status":"pulling manifest"}\n\n' +
					'data: {"status":"downloading","total":1000,"completed":1000}\n\n' +
					'data: {"status":"success"}\n\n',
				{
					status: 200,
					headers: { 'content-type': 'text/event-stream' },
				}
			);
		}

		if (url.endsWith('/api/chat/threads') && method === 'POST') {
			if (failThreadCreate) {
				throw new TypeError('Load failed');
			}
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

	it('automatically bootstraps AI model on first run when model is missing', async () => {
		const fetchMock = createFetchMock({ llmReady: false, llmReadyAfterPull: true });
		vi.stubGlobal('fetch', fetchMock);
		render(AppTestHost);

		await waitFor(() => {
			expect(screen.queryByText('Setting up Cloumask')).toBeNull();
		}, { timeout: 4000 });
		expect(screen.getAllByText('Chat').length).toBeGreaterThan(0);
		expect(
			fetchMock.mock.calls.some(([url, init]) => {
				const requestUrl = typeof url === 'string' ? url : url.toString();
				const method = (init?.method ?? 'GET').toUpperCase();
				return requestUrl.endsWith('/llm/pull/stream') && method === 'POST';
			})
		).toBe(true);
	});

	it('keeps the message textbox editable when initial backend connection fails', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		vi.stubGlobal('fetch', createFetchMock({ failThreadCreate: true }));
		render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Retry')).toBeTruthy();
		});

		const input = screen.getByLabelText('Message input') as HTMLTextAreaElement;
		expect(input.disabled).toBe(false);
		expect(input.placeholder).toContain('Reconnecting');
	});
});
