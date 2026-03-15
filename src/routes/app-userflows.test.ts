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
	threadStateDelayMs?: number;
	threadList?: Array<{
		thread_id: string;
		awaiting_user?: boolean;
		current_step?: number;
		total_steps?: number;
	}>;
	threadStates?: Record<string, Record<string, unknown>>;
}) {
	const llmReady = options?.llmReady ?? true;
	const llmReadyAfterPull = options?.llmReadyAfterPull ?? llmReady;
	const failThreadCreate = options?.failThreadCreate ?? false;
	const threadStateDelayMs = options?.threadStateDelayMs ?? 0;
	const threadList = options?.threadList ?? [];
	const threadStates = options?.threadStates ?? {};
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

		if (url.includes('/api/chat/threads') && !url.includes('/api/chat/threads/') && method === 'GET') {
			return jsonResponse({
				threads: threadList.map((thread) => ({
					title: null,
					status: 'active',
					last_message: '',
					updated_at: '2026-03-13T12:00:00.000Z',
					created_at: '2026-03-13T12:00:00.000Z',
					awaiting_user: false,
					current_step: 0,
					total_steps: 0,
					...thread,
				})),
			});
		}

		if (url.includes('/api/chat/threads/') && url.endsWith('/state') && method === 'GET') {
			const threadId = url.split('/api/chat/threads/')[1]?.replace('/state', '') ?? '';
			if (threadStateDelayMs > 0) {
				await new Promise((resolve) => setTimeout(resolve, threadStateDelayMs));
			}
			return jsonResponse({
				thread_id: threadId,
				state: threadStates[threadId] ?? {},
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
		Object.defineProperty(HTMLElement.prototype, 'scrollTo', {
			value: vi.fn(),
			writable: true,
			configurable: true,
		});
		// @ts-expect-error test cleanup
		delete window.__TAURI__;
		// @ts-expect-error test cleanup
		delete window.__TAURI_INTERNALS__;
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

	it('shows welcome setup on tauri startup even if setup was completed before', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		Object.defineProperty(window, '__TAURI_INTERNALS__', {
			value: {},
			writable: true,
			configurable: true,
		});

		render(AppTestHost);

		expect(screen.getByText('Setting up Cloumask')).toBeTruthy();
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

	it('restores unresolved checkpoint details when resuming the latest thread', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-checkpoint-1',
					awaiting_user: true,
					current_step: 1,
					total_steps: 2,
				},
			],
			threadStates: {
				'thread-checkpoint-1': {
					messages: [
						{
							role: 'user',
							content: 'resume processing',
							timestamp: '2026-03-14T10:00:00.000Z',
						},
						{
							role: 'assistant',
							content: 'Quality dip detected after batch 12.',
							timestamp: '2026-03-14T10:00:05.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan input',
							parameters: { path: '/data/input' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'detect',
							description: 'Detect people',
							parameters: { classes: ['person'] },
							status: 'running',
						},
					],
					plan_approved: true,
					awaiting_user: true,
					current_step: 1,
					metadata: {
						pipeline_id: 'pipe-checkpoint-1',
						progress_percent: 50,
						processed_files: 24,
					},
					checkpoints: [
						{
							id: 'ckpt-1',
							step_index: 1,
							trigger_reason: 'percentage',
							progress_percent: 50,
							quality_metrics: {
								average_confidence: 0.64,
								error_count: 2,
								total_processed: 24,
							},
							created_at: '2026-03-14T10:00:05.000Z',
							resolved_at: null,
						},
					],
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getAllByText('Chat').length).toBeGreaterThan(0);
		});
		await waitFor(() => {
			expect(screen.getByText('Resume from the saved checkpoint when you are ready.')).toBeTruthy();
		});

		await fireEvent.keyDown(window, { key: '3' });
		await waitFor(() => {
			expect(screen.getByText('[CHECKPOINT]')).toBeTruthy();
		});

		expect(screen.getByText('Progress milestone reached')).toBeTruthy();
		expect(screen.getAllByText('Quality dip detected after batch 12.').length).toBeGreaterThan(0);
		expect(screen.getByRole('button', { name: 'Continue' })).toBeTruthy();
		expect(screen.getAllByText('50%').length).toBeGreaterThan(0);
		expect(screen.getByText('64%')).toBeTruthy();
		expect(screen.getAllByText('24').length).toBeGreaterThan(0);
		expect(screen.getAllByText('2').length).toBeGreaterThan(0);

		const createdThreadCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return requestUrl.endsWith('/api/chat/threads') && method === 'POST';
		});

		expect(createdThreadCalls).toHaveLength(0);
		view.unmount();
	});

	it('restores completed execution stats and previews when resuming the latest thread', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-complete-1',
					awaiting_user: false,
					current_step: 2,
					total_steps: 2,
				},
			],
			threadStates: {
				'thread-complete-1': {
					messages: [
						{
							role: 'user',
							content: 'finish processing this batch',
							timestamp: '2026-03-14T11:00:00.000Z',
						},
						{
							role: 'assistant',
							content: 'Pipeline finished successfully.',
							timestamp: '2026-03-14T11:00:08.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan input',
							parameters: { path: '/data/input' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'detect',
							description: 'Detect people',
							parameters: { classes: ['person'] },
							status: 'completed',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 2,
					metadata: {
						pipeline_id: 'pipe-complete-1',
						created_at: '2026-03-14T11:00:00.000Z',
					},
					checkpoints: [],
					execution_results: {
						'step-1': {
							total_files: 12,
						},
						'step-2': {
							count: 4,
							preview_items: [
								{
									image_path: '/tmp/frame-001.jpg',
									annotations: [
										{
											label: 'person',
											confidence: 0.92,
											bbox: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 },
										},
									],
								},
								{
									image_path: '/tmp/frame-002.jpg',
									annotations: [
										{
											label: 'person',
											confidence: 0.88,
											bbox: { x: 0.2, y: 0.25, width: 0.25, height: 0.35 },
										},
									],
								},
							],
						},
					},
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getAllByText('Chat').length).toBeGreaterThan(0);
		});
		await waitFor(() => {
			expect(screen.getByText('Pipeline finished successfully.')).toBeTruthy();
		});

		await fireEvent.keyDown(window, { key: '3' });
		await waitFor(() => {
			expect(screen.getByText('<complete>')).toBeTruthy();
		});

		expect(screen.queryByText('[CHECKPOINT]')).toBeNull();
		expect(screen.getByText('Step 2/2')).toBeTruthy();
		expect(screen.getByText('Detect people')).toBeTruthy();
		expect(screen.getByText('2 recent')).toBeTruthy();
		expect(screen.getByText('Processed').parentElement?.textContent).toContain('12');
		expect(screen.getByText('Detected').parentElement?.textContent).toContain('4');

		const createdThreadCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return requestUrl.endsWith('/api/chat/threads') && method === 'POST';
		});

		expect(createdThreadCalls).toHaveLength(0);
		view.unmount();
	});

	it('prefers awaiting review threads and shows a resume summary breadcrumb', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-complete-older',
					awaiting_user: false,
					current_step: 2,
					total_steps: 2,
				},
				{
					thread_id: 'thread-awaiting-review',
					awaiting_user: true,
					current_step: 1,
					total_steps: 3,
				},
			],
			threadStates: {
				'thread-awaiting-review': {
					messages: [
						{
							role: 'user',
							content: 'continue the latest run',
							timestamp: '2026-03-14T12:00:00.000Z',
						},
						{
							role: 'assistant',
							content: 'Review this saved plan before continuing.',
							timestamp: '2026-03-14T12:00:02.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan input',
							parameters: { path: '/data/inbox' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'detect',
							description: 'Detect people',
							parameters: { classes: ['person'] },
							status: 'pending',
						},
						{
							id: 'step-3',
							tool_name: 'export',
							description: 'Export labels',
							parameters: { output_path: '/data/out' },
							status: 'pending',
						},
					],
					plan_approved: false,
					awaiting_user: true,
					current_step: 1,
					metadata: { pipeline_id: 'pipe-awaiting-review' },
				},
				'thread-complete-older': {
					messages: [
						{
							role: 'assistant',
							content: 'Older completed run',
							timestamp: '2026-03-14T11:55:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-a',
							tool_name: 'scan_directory',
							description: 'Scan old input',
							parameters: { path: '/data/old' },
							status: 'completed',
						},
						{
							id: 'step-b',
							tool_name: 'detect',
							description: 'Detect old people',
							parameters: { classes: ['person'] },
							status: 'completed',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 2,
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Review this saved plan before continuing.')).toBeTruthy();
		});

		expect(
			screen.getByText(
				'Resumed backend thread thread-awaiting-review. Status: awaiting review. Progress: 1/3 steps.'
			)
		).toBeTruthy();
		expect(screen.queryByText('Older completed run')).toBeNull();

		const stateCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return requestUrl.includes('/api/chat/threads/') &&
				requestUrl.endsWith('/state') &&
				method === 'GET';
		});

		expect(
			stateCalls.some(([url]) => {
				const requestUrl = typeof url === 'string' ? url : url.toString();
				return requestUrl.includes('/api/chat/threads/thread-awaiting-review/state');
			})
		).toBe(true);
		expect(
			stateCalls.some(([url]) => {
				const requestUrl = typeof url === 'string' ? url : url.toString();
				return requestUrl.includes('/api/chat/threads/thread-complete-older/state');
			})
		).toBe(false);

		view.unmount();
	});

	it('does not duplicate an existing resumed thread breadcrumb on hydration', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const resumeMessage =
			'Resumed backend thread thread-deduped. Status: awaiting review. Progress: 1/2 steps.';
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-deduped',
					awaiting_user: true,
					current_step: 1,
					total_steps: 2,
				},
			],
			threadStates: {
				'thread-deduped': {
					messages: [
						{
							role: 'system',
							content: resumeMessage,
							timestamp: '2026-03-14T12:30:00.000Z',
						},
						{
							role: 'assistant',
							content: 'Pick up where you left off.',
							timestamp: '2026-03-14T12:30:01.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan inbox',
							parameters: { path: '/data/inbox' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'detect',
							description: 'Detect objects',
							parameters: { classes: ['person'] },
							status: 'pending',
						},
					],
					plan_approved: false,
					awaiting_user: true,
					current_step: 1,
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Pick up where you left off.')).toBeTruthy();
		});
		expect(screen.getAllByText(resumeMessage)).toHaveLength(1);

		view.unmount();
	});

	it('shows a temporary auto-resume note while loading the selected thread', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadStateDelayMs: 200,
			threadList: [
				{
					thread_id: 'thread-loading-note',
					awaiting_user: true,
					current_step: 1,
					total_steps: 3,
				},
			],
			threadStates: {
				'thread-loading-note': {
					messages: [
						{
							role: 'assistant',
							content: 'Thread hydration finished.',
							timestamp: '2026-03-14T12:45:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan inbox',
							parameters: { path: '/data/inbox' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'detect',
							description: 'Detect people',
							parameters: { classes: ['person'] },
							status: 'pending',
						},
						{
							id: 'step-3',
							tool_name: 'export',
							description: 'Export labels',
							parameters: { output_path: '/data/out' },
							status: 'pending',
						},
					],
					plan_approved: false,
					awaiting_user: true,
					current_step: 1,
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);
		const note = 'Resuming thread-loading-note: awaiting review (1/3 steps)';

		await waitFor(() => {
			expect(screen.getByText(note)).toBeTruthy();
		});
		await waitFor(() => {
			expect(screen.getByText('Thread hydration finished.')).toBeTruthy();
		});
		await waitFor(() => {
			expect(screen.queryByText(note)).toBeNull();
		});

		view.unmount();
	});

	it('reuses latest resumable backend thread before creating a new one', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-existing-1',
					awaiting_user: true,
					current_step: 1,
					total_steps: 3,
				},
			],
			threadStates: {
				'thread-existing-1': {
					messages: [
						{
							role: 'user',
							content: 'process this folder',
							timestamp: '2026-03-14T10:00:00.000Z',
						},
						{
							role: 'assistant',
							content: "Here's a plan",
							timestamp: '2026-03-14T10:00:01.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan input',
							parameters: { path: '/data/input' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'detect',
							description: 'Detect people',
							parameters: { classes: ['person'] },
							status: 'pending',
						},
					],
					plan_approved: false,
					awaiting_user: true,
					current_step: 1,
					metadata: { pipeline_id: 'pipe-existing-1' },
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getAllByText('Chat').length).toBeGreaterThan(0);
		});
		await waitFor(() => {
			expect(screen.getByText("Here's a plan")).toBeTruthy();
		});
		expect(screen.getByText('Scan input')).toBeTruthy();
		expect(screen.getByText('Review the saved plan and choose how to continue.')).toBeTruthy();

		const createdThreadCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return requestUrl.endsWith('/api/chat/threads') && method === 'POST';
		});

		const listedThreadCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return requestUrl.includes('/api/chat/threads') &&
				!requestUrl.includes('/api/chat/threads/') &&
				method === 'GET';
		});

		expect(listedThreadCalls.length).toBeGreaterThan(0);
		expect(createdThreadCalls).toHaveLength(0);

		await fireEvent.keyDown(window, { key: '2' });
		await waitFor(() => {
			expect(screen.getByText('Detect people')).toBeTruthy();
		});

		view.unmount();
	});
});
