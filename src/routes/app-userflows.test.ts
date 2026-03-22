import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AppTestHost from '$lib/test-utils/AppTestHost.svelte';
import { getSSEManager } from '$lib/utils/sse';
import { mockInvoke } from '$lib/test-utils';

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
		title?: string | null;
		resume_status?: string;
		awaiting_user?: boolean;
		current_step?: number;
		total_steps?: number;
		summary?: string;
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
		mockInvoke.mockReset();
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
		expect(screen.getByText('Resumed:')).toBeTruthy();
		expect(screen.getByText('thread-complete-1')).toBeTruthy();
		expect(
			screen.getByText((_, element) =>
				element?.textContent === '• completed. Progress: 2/2 steps.'
			)
		).toBeTruthy();

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
					summary: 'completed. Progress: 2/2 steps.',
				},
				{
					thread_id: 'thread-awaiting-review',
					awaiting_user: true,
					current_step: 0,
					total_steps: 3,
					summary: 'awaiting review. Progress: 1/3 steps.',
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
					current_step: 0,
					total_steps: 3,
					summary: 'awaiting review. Progress: 1/3 steps.',
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

	it('shows thread title in the temporary auto-resume note when available', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadStateDelayMs: 200,
			threadList: [
				{
					thread_id: 'thread-title-note',
					title: 'Inbox Review',
					awaiting_user: true,
					current_step: 0,
					total_steps: 3,
					summary: 'awaiting review. Progress: 1/3 steps.',
				},
			],
			threadStates: {
				'thread-title-note': {
					messages: [
						{
							role: 'assistant',
							content: 'Titled thread hydration finished.',
							timestamp: '2026-03-15T12:45:00.000Z',
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
							tool_name: 'review',
							description: 'Review detections',
							parameters: {},
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
		const note = 'Resuming Inbox Review (thread-title-note): awaiting review (1/3 steps)';

		await waitFor(() => {
			expect(screen.getByText(note)).toBeTruthy();
		});
		await waitFor(() => {
			expect(screen.getByText('Titled thread hydration finished.')).toBeTruthy();
		});
		await waitFor(() => {
			expect(screen.queryByText(note)).toBeNull();
		});
		expect(screen.getByText('Resumed:')).toBeTruthy();
		expect(screen.getByText('Inbox Review (thread-title-note)')).toBeTruthy();
		expect(
			screen.getByText((_, element) =>
				element?.textContent === '• awaiting review. Progress: 1/3 steps.'
			)
		).toBeTruthy();
		await fireEvent.click(screen.getByRole('button', { name: 'Dismiss resumed thread summary' }));
		expect(screen.queryByText('Resumed:')).toBeNull();

		view.unmount();
	});

	it('dismisses the resumed thread strip with Escape', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-escape-strip',
					title: 'Escape Review',
					awaiting_user: true,
					current_step: 0,
					total_steps: 2,
					summary: 'awaiting review. Progress: 1/2 steps.',
				},
			],
			threadStates: {
				'thread-escape-strip': {
					messages: [
						{
							role: 'assistant',
							content: 'Escape strip thread restored.',
							timestamp: '2026-03-15T13:10:00.000Z',
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
							tool_name: 'review',
							description: 'Review detections',
							parameters: {},
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
			expect(screen.getByText('Escape strip thread restored.')).toBeTruthy();
		});
		expect(screen.getByText('Resumed:')).toBeTruthy();
		expect(screen.getByText('Escape Review (thread-escape-strip)')).toBeTruthy();

		await fireEvent.keyDown(window, { key: 'Escape' });

		expect(screen.queryByText('Resumed:')).toBeNull();

		view.unmount();
	});

	it('keeps the resumed thread strip dismissed after reconnecting', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-dismissed-strip',
					title: 'Dismissed Review',
					awaiting_user: true,
					current_step: 0,
					total_steps: 2,
					summary: 'awaiting review. Progress: 1/2 steps.',
				},
			],
			threadStates: {
				'thread-dismissed-strip': {
					messages: [
						{
							role: 'assistant',
							content: 'Dismissed strip thread restored.',
							timestamp: '2026-03-15T13:20:00.000Z',
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
							tool_name: 'review',
							description: 'Review detections',
							parameters: {},
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
			expect(screen.getByText('Dismissed strip thread restored.')).toBeTruthy();
		});
		expect(screen.getByText('Resumed:')).toBeTruthy();

		await fireEvent.click(screen.getByRole('button', { name: 'Dismiss resumed thread summary' }));
		expect(screen.queryByText('Resumed:')).toBeNull();

		(getSSEManager() as unknown as { updateState(state: 'disconnected' | 'connected'): void }).updateState(
			'disconnected'
		);
		(getSSEManager() as unknown as { updateState(state: 'disconnected' | 'connected'): void }).updateState(
			'connected'
		);

		expect(screen.queryByText('Resumed:')).toBeNull();

		view.unmount();
	});

	it('keeps the resumed thread strip cleared after sending a new message and reconnecting', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-send-clears-strip',
					title: 'Follow-up Review',
					awaiting_user: true,
					current_step: 0,
					total_steps: 2,
					summary: 'awaiting review. Progress: 1/2 steps.',
				},
			],
			threadStates: {
				'thread-send-clears-strip': {
					messages: [
						{
							role: 'assistant',
							content: 'Ready for your follow-up.',
							timestamp: '2026-03-15T13:15:00.000Z',
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
							tool_name: 'review',
							description: 'Review detections',
							parameters: {},
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
			expect(screen.getByText('Ready for your follow-up.')).toBeTruthy();
		});
		expect(screen.getByText('Resumed:')).toBeTruthy();
		(getSSEManager() as unknown as { updateState(state: 'connected'): void }).updateState(
			'connected'
		);

		const input = screen.getByLabelText('Message input') as HTMLTextAreaElement;
		await fireEvent.input(input, { target: { value: 'Continue the run' } });
		await fireEvent.keyDown(input, { key: 'Enter' });

		await waitFor(() => {
			expect(screen.queryByText('Resumed:')).toBeNull();
		});
		(getSSEManager() as unknown as { updateState(state: 'disconnected' | 'connected'): void }).updateState(
			'disconnected'
		);
		(getSSEManager() as unknown as { updateState(state: 'disconnected' | 'connected'): void }).updateState(
			'connected'
		);
		expect(screen.queryByText('Resumed:')).toBeNull();

		const sendCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return requestUrl.includes('/api/chat/send/thread-send-clears-strip') && method === 'POST';
		});
		expect(sendCalls).toHaveLength(1);

		view.unmount();
	});

	it('clears resumed thread context when starting a fresh conversation', async () => {
		localStorage.setItem('cloumask:setup', 'complete');

		let availableThreads = [
			{
				thread_id: 'thread-clear-strip',
				title: 'Clear Review',
				status: 'active',
				last_message: '',
				updated_at: '2026-03-15T13:20:00.000Z',
				created_at: '2026-03-15T13:20:00.000Z',
				awaiting_user: true,
				current_step: 0,
				total_steps: 2,
				summary: 'awaiting review. Progress: 1/2 steps.',
			},
		];
		const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
			const url =
				typeof input === 'string'
					? input
					: input instanceof URL
						? input.toString()
						: input.url;
			const method = (init?.method ?? 'GET').toUpperCase();

			if (url.endsWith('/llm/ensure-ready') && method === 'GET') {
				return jsonResponse({
					ready: true,
					service_running: true,
					required_model: 'qwen3:8b',
					model_available: true,
					error: null,
				});
			}

			if (url.includes('/api/chat/threads?limit=') && method === 'GET') {
				return jsonResponse({ threads: availableThreads });
			}

			if (url.endsWith('/api/chat/threads/thread-clear-strip/state') && method === 'GET') {
				return jsonResponse({
					thread_id: 'thread-clear-strip',
					state: {
						messages: [
							{
								role: 'assistant',
								content: 'Clear strip thread restored.',
								timestamp: '2026-03-15T13:20:01.000Z',
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
								tool_name: 'review',
								description: 'Review detections',
								parameters: {},
								status: 'pending',
							},
						],
						plan_approved: false,
						awaiting_user: true,
						current_step: 1,
					},
				});
			}

			if (url.endsWith('/api/chat/threads/thread-clear-strip') && method === 'DELETE') {
				availableThreads = [];
				return jsonResponse({});
			}

			if (url.endsWith('/api/chat/threads') && method === 'POST') {
				return jsonResponse({
					thread_id: 'thread-fresh-after-clear',
					created: true,
					awaiting_user: false,
					current_step: 0,
					total_steps: 0,
				});
			}

			return jsonResponse({});
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Clear strip thread restored.')).toBeTruthy();
		});
		expect(screen.getByText('Resumed:')).toBeTruthy();

		await fireEvent.click(screen.getByRole('button', { name: 'Clear' }));

		await waitFor(() => {
			expect(screen.queryByText('Resumed:')).toBeNull();
		});
		expect(screen.queryByText('Clear strip thread restored.')).toBeNull();

		const closeCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return requestUrl.endsWith('/api/chat/threads/thread-clear-strip') && method === 'DELETE';
		});
		expect(closeCalls).toHaveLength(1);

		await waitFor(() => {
			const createCalls = fetchMock.mock.calls.filter(([url, init]) => {
				const requestUrl = typeof url === 'string' ? url : url.toString();
				const method = (init?.method ?? 'GET').toUpperCase();
				return requestUrl.endsWith('/api/chat/threads') && method === 'POST';
			});
			expect(createCalls).toHaveLength(1);
		});

		view.unmount();
	});

	it('keeps cleared resume UI from returning after disconnect and reconnect', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		let availableThreads = [
			{
				thread_id: 'thread-clear-reconnect',
				title: 'Reconnect Review',
				status: 'active',
				last_message: '',
				updated_at: '2026-03-15T13:30:00.000Z',
				created_at: '2026-03-15T13:30:00.000Z',
				awaiting_user: true,
				current_step: 0,
				total_steps: 2,
				summary: 'awaiting review. Progress: 1/2 steps.',
			},
		];
		const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
			const url =
				typeof input === 'string'
					? input
					: input instanceof URL
						? input.toString()
						: input.url;
			const method = (init?.method ?? 'GET').toUpperCase();

			if (url.endsWith('/llm/ensure-ready') && method === 'GET') {
				return jsonResponse({
					ready: true,
					service_running: true,
					required_model: 'qwen3:8b',
					model_available: true,
					error: null,
				});
			}

			if (url.includes('/api/chat/threads?limit=') && method === 'GET') {
				return jsonResponse({ threads: availableThreads });
			}

			if (url.endsWith('/api/chat/threads/thread-clear-reconnect/state') && method === 'GET') {
				return jsonResponse({
					thread_id: 'thread-clear-reconnect',
					state: {
						messages: [
							{
								role: 'assistant',
								content: 'Reconnect clear thread restored.',
								timestamp: '2026-03-15T13:30:01.000Z',
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
								tool_name: 'review',
								description: 'Review detections',
								parameters: {},
								status: 'pending',
							},
						],
						plan_approved: false,
						awaiting_user: true,
						current_step: 1,
					},
				});
			}

			if (url.endsWith('/api/chat/threads/thread-clear-reconnect') && method === 'DELETE') {
				availableThreads = [];
				return jsonResponse({});
			}

			if (url.endsWith('/api/chat/threads') && method === 'POST') {
				return jsonResponse({
					thread_id: 'thread-fresh-after-clear-reconnect',
					created: true,
					awaiting_user: false,
					current_step: 0,
					total_steps: 0,
				});
			}

			return jsonResponse({});
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Reconnect clear thread restored.')).toBeTruthy();
		});
		expect(screen.getByText('Resumed:')).toBeTruthy();

		(getSSEManager() as unknown as { updateState(state: 'disconnected'): void }).updateState(
			'disconnected'
		);
		await fireEvent.click(screen.getByRole('button', { name: 'Clear' }));

		await waitFor(() => {
			expect(screen.queryByText('Resumed:')).toBeNull();
		});

		(getSSEManager() as unknown as { updateState(state: 'connected'): void }).updateState(
			'connected'
		);

		expect(screen.queryByText('Resumed:')).toBeNull();
		expect(screen.queryByText('Reconnect clear thread restored.')).toBeNull();

		view.unmount();
	});

	it('handles repeated clear presses while disconnected without reviving old resume state', async () => {
		localStorage.setItem('cloumask:setup', 'complete');

		let availableThreads = [
			{
				thread_id: 'thread-repeat-clear',
				title: 'Repeat Clear Review',
				status: 'active',
				last_message: '',
				updated_at: '2026-03-15T13:40:00.000Z',
				created_at: '2026-03-15T13:40:00.000Z',
				awaiting_user: true,
				current_step: 0,
				total_steps: 2,
				summary: 'awaiting review. Progress: 1/2 steps.',
			},
		];
		let oldStateFetches = 0;
		let createCount = 0;
		const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
			const url =
				typeof input === 'string'
					? input
					: input instanceof URL
						? input.toString()
						: input.url;
			const method = (init?.method ?? 'GET').toUpperCase();

			if (url.endsWith('/llm/ensure-ready') && method === 'GET') {
				return jsonResponse({
					ready: true,
					service_running: true,
					required_model: 'qwen3:8b',
					model_available: true,
					error: null,
				});
			}

			if (url.includes('/api/chat/threads?limit=') && method === 'GET') {
				return jsonResponse({ threads: availableThreads });
			}

			if (url.endsWith('/api/chat/threads/thread-repeat-clear/state') && method === 'GET') {
				oldStateFetches += 1;
				return jsonResponse({
					thread_id: 'thread-repeat-clear',
					state: {
						messages: [
							{
								role: 'assistant',
								content: 'Repeat clear thread restored.',
								timestamp: '2026-03-15T13:40:01.000Z',
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
								tool_name: 'review',
								description: 'Review detections',
								parameters: {},
								status: 'pending',
							},
						],
						plan_approved: false,
						awaiting_user: true,
						current_step: 1,
					},
				});
			}

			if (url.endsWith('/api/chat/threads/thread-repeat-clear') && method === 'DELETE') {
				availableThreads = [];
				return jsonResponse({});
			}

			if (url.endsWith('/api/chat/threads') && method === 'POST') {
				createCount += 1;
				return jsonResponse({
					thread_id: 'thread-fresh-after-repeat-clear',
					created: true,
					awaiting_user: false,
					current_step: 0,
					total_steps: 0,
				});
			}

			return jsonResponse({});
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Repeat clear thread restored.')).toBeTruthy();
		});
		expect(screen.getByText('Resumed:')).toBeTruthy();
		expect(oldStateFetches).toBe(1);

		(getSSEManager() as unknown as { updateState(state: 'disconnected'): void }).updateState(
			'disconnected'
		);
		await fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
		await fireEvent.click(screen.getByRole('button', { name: 'Clear' }));

		await waitFor(() => {
			expect(screen.queryByText('Resumed:')).toBeNull();
		});
		expect(screen.queryByText('Repeat clear thread restored.')).toBeNull();

		await waitFor(() => {
			expect(createCount).toBe(1);
		});
		expect(oldStateFetches).toBe(1);

		view.unmount();
	});

	it('does not refetch dismissed resume state after reconnecting', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-dismiss-send-reconnect',
					title: 'Dismiss Reconnect Review',
					awaiting_user: true,
					current_step: 0,
					total_steps: 2,
					summary: 'awaiting review. Progress: 1/2 steps.',
				},
			],
			threadStates: {
				'thread-dismiss-send-reconnect': {
					messages: [
						{
							role: 'assistant',
							content: 'Dismiss reconnect thread restored.',
							timestamp: '2026-03-15T13:45:00.000Z',
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
							tool_name: 'review',
							description: 'Review detections',
							parameters: {},
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
			expect(screen.getByText('Dismiss reconnect thread restored.')).toBeTruthy();
		});
		expect(screen.getByText('Resumed:')).toBeTruthy();

		await fireEvent.click(screen.getByRole('button', { name: 'Dismiss resumed thread summary' }));
		expect(screen.queryByText('Resumed:')).toBeNull();

		(getSSEManager() as unknown as { updateState(state: 'disconnected' | 'connected'): void }).updateState(
			'disconnected'
		);
		(getSSEManager() as unknown as { updateState(state: 'disconnected' | 'connected'): void }).updateState(
			'connected'
		);
		expect(screen.queryByText('Resumed:')).toBeNull();

		const stateCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return requestUrl.includes('/api/chat/threads/thread-dismiss-send-reconnect/state') && method === 'GET';
		});
		expect(stateCalls).toHaveLength(1);

		view.unmount();
	});

	it('falls back to locally computed resume copy when backend summary is missing', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-fallback-summary',
					awaiting_user: true,
					current_step: 2,
					total_steps: 4,
				},
			],
			threadStates: {
				'thread-fallback-summary': {
					messages: [
						{
							role: 'assistant',
							content: 'Fallback summary thread restored.',
							timestamp: '2026-03-15T10:00:00.000Z',
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
							status: 'completed',
						},
						{
							id: 'step-3',
							tool_name: 'export',
							description: 'Export labels',
							parameters: { output_path: '/data/out' },
							status: 'pending',
						},
						{
							id: 'step-4',
							tool_name: 'review',
							description: 'Review output',
							parameters: {},
							status: 'pending',
						},
					],
					plan_approved: false,
					awaiting_user: true,
					current_step: 2,
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Fallback summary thread restored.')).toBeTruthy();
		});
		expect(
			screen.getByText(
				'Resumed backend thread thread-fallback-summary. Status: awaiting review. Progress: 2/4 steps.'
			)
		).toBeTruthy();

		view.unmount();
	});

	it('falls back to local counters when backend resume status is unknown', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-unknown-resume-status',
					resume_status: 'mystery',
					awaiting_user: false,
					current_step: 1,
					total_steps: 3,
				},
			],
			threadStates: {
				'thread-unknown-resume-status': {
					messages: [
						{
							role: 'assistant',
							content: 'Unknown status thread restored.',
							timestamp: '2026-03-15T10:05:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan unknown status inbox',
							parameters: { path: '/data/unknown-status' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'detect',
							description: 'Detect unknown status people',
							parameters: { classes: ['person'] },
							status: 'pending',
						},
						{
							id: 'step-3',
							tool_name: 'export',
							description: 'Export unknown status labels',
							parameters: { output_path: '/data/unknown-status-out' },
							status: 'pending',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 1,
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Unknown status thread restored.')).toBeTruthy();
		});
		expect(
			screen.getByText(
				'Resumed backend thread thread-unknown-resume-status. Status: in progress. Progress: 1/3 steps.'
			)
		).toBeTruthy();

		view.unmount();
	});

	it('uses backend completed summary copy for resumed completed threads', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-completed-summary',
					awaiting_user: false,
					current_step: 99,
					total_steps: 2,
					summary: 'completed. Progress: 2/2 steps.',
				},
			],
			threadStates: {
				'thread-completed-summary': {
					messages: [
						{
							role: 'assistant',
							content: 'Completed thread restored.',
							timestamp: '2026-03-15T12:00:00.000Z',
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
							tool_name: 'export',
							description: 'Export labels',
							parameters: { output_path: '/data/out' },
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
			expect(screen.getByText('Completed thread restored.')).toBeTruthy();
		});
		expect(
			screen.getByText(
				'Resumed backend thread thread-completed-summary. Status: completed. Progress: 2/2 steps.'
			)
		).toBeTruthy();

		view.unmount();
	});

	it('uses backend in-progress summary copy when raw counters look completed', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-stale-summary',
					awaiting_user: false,
					current_step: 99,
					total_steps: 2,
					summary: 'in progress. Progress: 1/2 steps.',
				},
			],
			threadStates: {
				'thread-stale-summary': {
					messages: [
						{
							role: 'assistant',
							content: 'Stale summary thread restored.',
							timestamp: '2026-03-15T12:10:00.000Z',
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
							tool_name: 'export',
							description: 'Export labels',
							parameters: { output_path: '/data/out' },
							status: 'pending',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 1,
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Stale summary thread restored.')).toBeTruthy();
		});
		expect(
			screen.getByText(
				'Resumed backend thread thread-stale-summary. Status: in progress. Progress: 1/2 steps.'
			)
		).toBeTruthy();

		view.unmount();
	});

	it('keeps resumed execution running when persisted state has stale completed counters', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-stale-hydration',
					awaiting_user: false,
					current_step: 99,
					total_steps: 2,
					summary: 'in progress. Progress: 1/2 steps.',
				},
			],
			threadStates: {
				'thread-stale-hydration': {
					messages: [
						{
							role: 'assistant',
							content: 'Stale hydration thread restored.',
							timestamp: '2026-03-15T12:15:00.000Z',
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
							tool_name: 'export',
							description: 'Export labels',
							parameters: { output_path: '/data/out' },
							status: 'pending',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 99,
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Stale hydration thread restored.')).toBeTruthy();
		});

		await fireEvent.click(screen.getByRole('button', { name: /Execute/ }));
		await waitFor(() => {
			expect(screen.getByRole('button', { name: 'Pause' })).toBeTruthy();
		});

		expect(screen.getByText('Step 2/2')).toBeTruthy();
		expect(screen.getByText('Export labels')).toBeTruthy();

		view.unmount();
	});

	it('shows resumed failed execution without a running pause action', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-failed-hydration',
					awaiting_user: false,
					current_step: 99,
					total_steps: 2,
					summary: 'failed. Progress: 1/2 steps.',
				},
			],
			threadStates: {
				'thread-failed-hydration': {
					messages: [
						{
							role: 'assistant',
							content: 'Failed hydration thread restored.',
							timestamp: '2026-03-15T12:20:00.000Z',
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
							tool_name: 'export',
							description: 'Export labels',
							parameters: { output_path: '/data/out' },
							status: 'failed',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 99,
					execution_results: {
						'step-2': {
							error: 'Disk full',
						},
					},
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Failed hydration thread restored.')).toBeTruthy();
		});
		expect(
			screen.getByText(
				'Resumed backend thread thread-failed-hydration. Status: failed. Progress: 1/2 steps.'
			)
		).toBeTruthy();

		await fireEvent.click(screen.getByRole('button', { name: /Execute/ }));
		await waitFor(() => {
			expect(screen.getByText('Disk full')).toBeTruthy();
		});

		expect(screen.queryByRole('button', { name: 'Pause' })).toBeNull();
		expect(screen.getByText('Step 2/2')).toBeTruthy();
		expect(screen.getByText('Export labels')).toBeTruthy();

		view.unmount();
	});

	it('prefers in-progress threads over newer failed resumes', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-failed-newest',
					awaiting_user: false,
					current_step: 1,
					total_steps: 2,
					summary: 'failed. Progress: 1/2 steps.',
				},
				{
					thread_id: 'thread-progress-older',
					awaiting_user: false,
					current_step: 1,
					total_steps: 3,
					summary: 'in progress. Progress: 1/3 steps.',
				},
			],
			threadStates: {
				'thread-failed-newest': {
					messages: [
						{
							role: 'assistant',
							content: 'Failed newest thread restored.',
							timestamp: '2026-03-15T12:25:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan newest',
							parameters: { path: '/data/newest' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'export',
							description: 'Export newest labels',
							parameters: { output_path: '/data/newest-out' },
							status: 'failed',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 99,
					execution_results: {
						'step-2': {
							error: 'Newest thread failed',
						},
					},
				},
				'thread-progress-older': {
					messages: [
						{
							role: 'assistant',
							content: 'Progress older thread restored.',
							timestamp: '2026-03-15T12:20:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan older',
							parameters: { path: '/data/older' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'detect',
							description: 'Detect older people',
							parameters: { classes: ['person'] },
							status: 'pending',
						},
						{
							id: 'step-3',
							tool_name: 'export',
							description: 'Export older labels',
							parameters: { output_path: '/data/older-out' },
							status: 'pending',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 1,
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Progress older thread restored.')).toBeTruthy();
		});
		expect(
			screen.getByText(
				'Resumed backend thread thread-progress-older. Status: in progress. Progress: 1/3 steps.'
			)
		).toBeTruthy();
		expect(screen.queryByText('Failed newest thread restored.')).toBeNull();

		const stateCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return requestUrl.includes('/api/chat/threads/thread-progress-older/state') && method === 'GET';
		});
		expect(stateCalls).toHaveLength(1);

		view.unmount();
	});

	it('prefers in-progress threads over newer failed resumes without summary text', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-failed-nosummary-newest',
					resume_status: 'failed',
					awaiting_user: false,
					current_step: 1,
					total_steps: 2,
				},
				{
					thread_id: 'thread-progress-summary-older',
					awaiting_user: false,
					current_step: 1,
					total_steps: 3,
					summary: 'in progress. Progress: 1/3 steps.',
				},
			],
			threadStates: {
				'thread-failed-nosummary-newest': {
					messages: [
						{
							role: 'assistant',
							content: 'Failed without summary thread restored.',
							timestamp: '2026-03-15T12:30:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan failed latest',
							parameters: { path: '/data/failed-latest' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'export',
							description: 'Export failed latest labels',
							parameters: { output_path: '/data/failed-latest-out' },
							status: 'failed',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 99,
					execution_results: {
						'step-2': {
							error: 'Failed without summary thread errored',
						},
					},
				},
				'thread-progress-summary-older': {
					messages: [
						{
							role: 'assistant',
							content: 'Progress with summary thread restored.',
							timestamp: '2026-03-15T12:10:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan progress older',
							parameters: { path: '/data/progress-older' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'detect',
							description: 'Detect progress older people',
							parameters: { classes: ['person'] },
							status: 'pending',
						},
						{
							id: 'step-3',
							tool_name: 'export',
							description: 'Export progress older labels',
							parameters: { output_path: '/data/progress-older-out' },
							status: 'pending',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 1,
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Progress with summary thread restored.')).toBeTruthy();
		});
		expect(
			screen.getByText(
				'Resumed backend thread thread-progress-summary-older. Status: in progress. Progress: 1/3 steps.'
			)
		).toBeTruthy();
		expect(screen.queryByText('Failed without summary thread restored.')).toBeNull();

		const stateCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return (
				requestUrl.includes('/api/chat/threads/thread-progress-summary-older/state') &&
				method === 'GET'
			);
		});
		expect(stateCalls).toHaveLength(1);

		view.unmount();
	});

	it('prefers failed threads over newer completed resumes without summary text', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-completed-nosummary-newest',
					resume_status: 'completed',
					awaiting_user: false,
					current_step: 2,
					total_steps: 2,
				},
				{
					thread_id: 'thread-failed-nosummary-older',
					resume_status: 'failed',
					awaiting_user: false,
					current_step: 1,
					total_steps: 2,
				},
			],
			threadStates: {
				'thread-completed-nosummary-newest': {
					messages: [
						{
							role: 'assistant',
							content: 'Completed without summary thread restored.',
							timestamp: '2026-03-15T12:35:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan completed latest',
							parameters: { path: '/data/completed-latest' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'export',
							description: 'Export completed latest labels',
							parameters: { output_path: '/data/completed-latest-out' },
							status: 'completed',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 99,
				},
				'thread-failed-nosummary-older': {
					messages: [
						{
							role: 'assistant',
							content: 'Failed fallback thread restored.',
							timestamp: '2026-03-15T12:05:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan failed older',
							parameters: { path: '/data/failed-older' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'export',
							description: 'Export failed older labels',
							parameters: { output_path: '/data/failed-older-out' },
							status: 'failed',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 99,
					execution_results: {
						'step-2': {
							error: 'Failed fallback thread errored',
						},
					},
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Failed fallback thread restored.')).toBeTruthy();
		});
		expect(
			screen.getByText(
				'Resumed backend thread thread-failed-nosummary-older. Status: failed. Progress: 1/2 steps.'
			)
		).toBeTruthy();
		expect(screen.queryByText('Completed without summary thread restored.')).toBeNull();

		const stateCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return (
				requestUrl.includes('/api/chat/threads/thread-failed-nosummary-older/state') &&
				method === 'GET'
			);
		});
		expect(stateCalls).toHaveLength(1);

		view.unmount();
	});

	it('prefers awaiting review threads over newer failed resumes without summary text', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-failed-review-fallback-newest',
					resume_status: 'failed',
					awaiting_user: false,
					current_step: 1,
					total_steps: 2,
				},
				{
					thread_id: 'thread-review-fallback-older',
					resume_status: 'awaiting review',
					awaiting_user: true,
					current_step: 1,
					total_steps: 3,
				},
			],
			threadStates: {
				'thread-failed-review-fallback-newest': {
					messages: [
						{
							role: 'assistant',
							content: 'Failed review fallback thread restored.',
							timestamp: '2026-03-15T12:40:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan failed review fallback',
							parameters: { path: '/data/failed-review-fallback' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'export',
							description: 'Export failed review fallback labels',
							parameters: { output_path: '/data/failed-review-fallback-out' },
							status: 'failed',
						},
					],
					plan_approved: true,
					awaiting_user: false,
					current_step: 99,
					execution_results: {
						'step-2': {
							error: 'Failed review fallback thread errored',
						},
					},
				},
				'thread-review-fallback-older': {
					messages: [
						{
							role: 'assistant',
							content: 'Awaiting review fallback thread restored.',
							timestamp: '2026-03-15T12:00:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan review fallback',
							parameters: { path: '/data/review-fallback' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'detect',
							description: 'Detect review fallback people',
							parameters: { classes: ['person'] },
							status: 'completed',
						},
						{
							id: 'step-3',
							tool_name: 'review',
							description: 'Review review fallback items',
							parameters: {},
							status: 'pending',
						},
					],
					plan_approved: true,
					awaiting_user: true,
					current_step: 1,
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Awaiting review fallback thread restored.')).toBeTruthy();
		});
		expect(
			screen.getByText(
				'Resumed backend thread thread-review-fallback-older. Status: awaiting review. Progress: 1/3 steps.'
			)
		).toBeTruthy();
		expect(screen.queryByText('Failed review fallback thread restored.')).toBeNull();

		const stateCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return (
				requestUrl.includes('/api/chat/threads/thread-review-fallback-older/state') &&
				method === 'GET'
			);
		});
		expect(stateCalls).toHaveLength(1);

		view.unmount();
	});

	it('keeps backend recency order for fallback-only awaiting review threads', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-review-fallback-newest',
					resume_status: 'awaiting review',
					awaiting_user: true,
					current_step: 1,
					total_steps: 4,
				},
				{
					thread_id: 'thread-review-fallback-older',
					resume_status: 'awaiting review',
					awaiting_user: true,
					current_step: 1,
					total_steps: 3,
				},
			],
			threadStates: {
				'thread-review-fallback-newest': {
					messages: [
						{
							role: 'assistant',
							content: 'Newest fallback review thread restored.',
							timestamp: '2026-03-15T12:45:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan newest fallback review',
							parameters: { path: '/data/review-fallback-newest' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'detect',
							description: 'Detect newest fallback review people',
							parameters: { classes: ['person'] },
							status: 'completed',
						},
						{
							id: 'step-3',
							tool_name: 'review',
							description: 'Review newest fallback items',
							parameters: {},
							status: 'pending',
						},
						{
							id: 'step-4',
							tool_name: 'export',
							description: 'Export newest fallback labels',
							parameters: { output_path: '/data/review-fallback-newest-out' },
							status: 'pending',
						},
					],
					plan_approved: true,
					awaiting_user: true,
					current_step: 1,
				},
				'thread-review-fallback-older': {
					messages: [
						{
							role: 'assistant',
							content: 'Older fallback review thread restored.',
							timestamp: '2026-03-15T12:15:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan older fallback review',
							parameters: { path: '/data/review-fallback-older' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'review',
							description: 'Review older fallback items',
							parameters: {},
							status: 'pending',
						},
						{
							id: 'step-3',
							tool_name: 'export',
							description: 'Export older fallback labels',
							parameters: { output_path: '/data/review-fallback-older-out' },
							status: 'pending',
						},
					],
					plan_approved: true,
					awaiting_user: true,
					current_step: 1,
				},
			},
		});
		vi.stubGlobal('fetch', fetchMock);

		const view = render(AppTestHost);

		await waitFor(() => {
			expect(screen.getByText('Newest fallback review thread restored.')).toBeTruthy();
		});
		expect(
			screen.getByText(
				'Resumed backend thread thread-review-fallback-newest. Status: awaiting review. Progress: 1/4 steps.'
			)
		).toBeTruthy();
		expect(screen.queryByText('Older fallback review thread restored.')).toBeNull();

		const stateCalls = fetchMock.mock.calls.filter(([url, init]) => {
			const requestUrl = typeof url === 'string' ? url : url.toString();
			const method = (init?.method ?? 'GET').toUpperCase();
			return (
				requestUrl.includes('/api/chat/threads/thread-review-fallback-newest/state') &&
				method === 'GET'
			);
		});
		expect(stateCalls).toHaveLength(1);

		view.unmount();
	});

	it('keeps backend recency order when resume priorities tie', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadList: [
				{
					thread_id: 'thread-review-newest',
					awaiting_user: true,
					current_step: 0,
					total_steps: 4,
					summary: 'awaiting review. Progress: 2/4 steps.',
				},
				{
					thread_id: 'thread-review-older',
					awaiting_user: true,
					current_step: 0,
					total_steps: 3,
					summary: 'awaiting review. Progress: 1/3 steps.',
				},
			],
			threadStates: {
				'thread-review-newest': {
					messages: [
						{
							role: 'assistant',
							content: 'Newest review thread restored.',
							timestamp: '2026-03-15T12:15:00.000Z',
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
							status: 'completed',
						},
						{
							id: 'step-3',
							tool_name: 'review',
							description: 'Review detections',
							parameters: {},
							status: 'pending',
						},
						{
							id: 'step-4',
							tool_name: 'export',
							description: 'Export labels',
							parameters: { output_path: '/data/out' },
							status: 'pending',
						},
					],
					plan_approved: false,
					awaiting_user: true,
					current_step: 2,
				},
				'thread-review-older': {
					messages: [
						{
							role: 'assistant',
							content: 'Older review thread restored.',
							timestamp: '2026-03-15T12:10:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan older inbox',
							parameters: { path: '/data/older' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'review',
							description: 'Review older detections',
							parameters: {},
							status: 'pending',
						},
						{
							id: 'step-3',
							tool_name: 'export',
							description: 'Export older labels',
							parameters: { output_path: '/data/older-out' },
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
			expect(screen.getByText('Newest review thread restored.')).toBeTruthy();
		});
		expect(screen.queryByText('Older review thread restored.')).toBeNull();
		expect(
			screen.getByText(
				'Resumed backend thread thread-review-newest. Status: awaiting review. Progress: 2/4 steps.'
			)
		).toBeTruthy();

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
				return requestUrl.includes('/api/chat/threads/thread-review-newest/state');
			})
		).toBe(true);
		expect(
			stateCalls.some(([url]) => {
				const requestUrl = typeof url === 'string' ? url : url.toString();
				return requestUrl.includes('/api/chat/threads/thread-review-older/state');
			})
		).toBe(false);

		view.unmount();
	});

	it('uses the newest titled thread in the temporary note when priorities tie', async () => {
		localStorage.setItem('cloumask:setup', 'complete');
		const fetchMock = createFetchMock({
			threadStateDelayMs: 200,
			threadList: [
				{
					thread_id: 'thread-title-newest',
					title: 'Newest Review',
					awaiting_user: true,
					current_step: 0,
					total_steps: 4,
					summary: 'awaiting review. Progress: 2/4 steps.',
				},
				{
					thread_id: 'thread-title-older',
					title: 'Older Review',
					awaiting_user: true,
					current_step: 0,
					total_steps: 3,
					summary: 'awaiting review. Progress: 1/3 steps.',
				},
			],
			threadStates: {
				'thread-title-newest': {
					messages: [
						{
							role: 'assistant',
							content: 'Newest titled thread restored.',
							timestamp: '2026-03-15T13:00:00.000Z',
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
							tool_name: 'review',
							description: 'Review detections',
							parameters: {},
							status: 'completed',
						},
						{
							id: 'step-3',
							tool_name: 'export',
							description: 'Export labels',
							parameters: { output_path: '/data/out' },
							status: 'pending',
						},
						{
							id: 'step-4',
							tool_name: 'notify',
							description: 'Notify user',
							parameters: {},
							status: 'pending',
						},
					],
					plan_approved: false,
					awaiting_user: true,
					current_step: 2,
				},
				'thread-title-older': {
					messages: [
						{
							role: 'assistant',
							content: 'Older titled thread restored.',
							timestamp: '2026-03-15T12:55:00.000Z',
						},
					],
					plan: [
						{
							id: 'step-1',
							tool_name: 'scan_directory',
							description: 'Scan older inbox',
							parameters: { path: '/data/older' },
							status: 'completed',
						},
						{
							id: 'step-2',
							tool_name: 'review',
							description: 'Review older detections',
							parameters: {},
							status: 'pending',
						},
						{
							id: 'step-3',
							tool_name: 'export',
							description: 'Export older labels',
							parameters: { output_path: '/data/older-out' },
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
		const note = 'Resuming Newest Review (thread-title-newest): awaiting review (2/4 steps)';

		await waitFor(() => {
			expect(screen.getByText(note)).toBeTruthy();
		});
		await waitFor(() => {
			expect(screen.getByText('Newest titled thread restored.')).toBeTruthy();
		});
		expect(screen.queryByText('Older titled thread restored.')).toBeNull();

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
