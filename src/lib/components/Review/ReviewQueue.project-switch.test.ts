import { render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ReviewQueueTestHost from '$lib/test-utils/ReviewQueueTestHost.svelte';

function jsonResponse(body: unknown, init?: ResponseInit): Response {
	return new Response(JSON.stringify(body), {
		status: 200,
		headers: { 'content-type': 'application/json' },
		...init
	});
}

function createBackendItem(id: string, fileName: string) {
	return {
		id,
		file_path: `/tmp/${fileName}`,
		file_name: fileName,
		dimensions: { width: 1280, height: 720 },
		thumbnail_url: `/thumbs/${fileName}`,
		annotations: [],
		original_annotations: [],
		status: 'pending'
	};
}

function getProjectIdFromUrl(input: RequestInfo | URL): string | null {
	const raw =
		typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
	const url = new URL(raw, 'http://localhost');
	return url.searchParams.get('project_id');
}

describe('ReviewQueue project switching', () => {
	beforeEach(() => {
		localStorage.clear();
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it('reloads review items when project changes even if execution id stays current', async () => {
		const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
			const method = (init?.method ?? 'GET').toUpperCase();
			const url =
				typeof input === 'string'
					? input
					: input instanceof URL
						? input.toString()
						: input.url;

			if (method === 'GET' && url.includes('/api/review/items')) {
				return jsonResponse({
					items: [],
					total: 0,
					skip: 0,
					limit: 50
				});
			}

			return jsonResponse({});
		});

		vi.stubGlobal('fetch', fetchMock);

		const { rerender } = render(ReviewQueueTestHost, {
			executionId: 'current',
			projectId: 'project-a'
		});

		await waitFor(() => {
			expect(fetchMock).toHaveBeenCalledTimes(1);
		});

		await rerender({
			executionId: 'current',
			projectId: 'project-b'
		});

		await waitFor(() => {
			expect(fetchMock).toHaveBeenCalledTimes(2);
		});
	});

	it('clears old project items before loading the next project', async () => {
		const projectAItem = createBackendItem('item-a', 'project-a-image.jpg');
		const projectBItem = createBackendItem('item-b', 'project-b-image.jpg');
		let resolveProjectBLoad: ((response: Response) => void) | null = null;

		const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
			const method = (init?.method ?? 'GET').toUpperCase();
			const projectId = getProjectIdFromUrl(input);

			if (method === 'GET' && projectId === 'project-a') {
				return Promise.resolve(
					jsonResponse({ items: [projectAItem], total: 1, skip: 0, limit: 50 })
				);
			}

			if (method === 'GET' && projectId === 'project-b') {
				return new Promise<Response>((resolve) => {
					resolveProjectBLoad = resolve;
				});
			}

			return Promise.resolve(jsonResponse({}));
		});

		vi.stubGlobal('fetch', fetchMock);

		const { rerender } = render(ReviewQueueTestHost, {
			executionId: 'current',
			projectId: 'project-a'
		});

		await waitFor(() => {
			expect(screen.getByText('project-a-image.jpg')).toBeTruthy();
		});

		await rerender({
			executionId: 'current',
			projectId: 'project-b'
		});

		await waitFor(() => {
			expect(screen.queryByText('project-a-image.jpg')).toBeNull();
			expect(screen.getByText('Loading...')).toBeTruthy();
		});

		resolveProjectBLoad?.(
			jsonResponse({ items: [projectBItem], total: 1, skip: 0, limit: 50 })
		);

		await waitFor(() => {
			expect(screen.getByText('project-b-image.jpg')).toBeTruthy();
			expect(screen.queryByText('project-a-image.jpg')).toBeNull();
		});
	});

	it('ignores stale responses from the previous project after a switch', async () => {
		const projectAItem = createBackendItem('item-a-stale', 'project-a-stale.jpg');
		const projectBItem = createBackendItem('item-b-fresh', 'project-b-fresh.jpg');
		let resolveProjectALoad: ((response: Response) => void) | null = null;
		let resolveProjectBLoad: ((response: Response) => void) | null = null;

		const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
			const method = (init?.method ?? 'GET').toUpperCase();
			const projectId = getProjectIdFromUrl(input);

			if (method === 'GET' && projectId === 'project-a') {
				return new Promise<Response>((resolve) => {
					resolveProjectALoad = resolve;
				});
			}

			if (method === 'GET' && projectId === 'project-b') {
				return new Promise<Response>((resolve) => {
					resolveProjectBLoad = resolve;
				});
			}

			return Promise.resolve(jsonResponse({}));
		});

		vi.stubGlobal('fetch', fetchMock);

		const { rerender } = render(ReviewQueueTestHost, {
			executionId: 'current',
			projectId: 'project-a'
		});

		await waitFor(() => {
			expect(fetchMock).toHaveBeenCalledTimes(1);
		});

		await rerender({
			executionId: 'current',
			projectId: 'project-b'
		});

		await waitFor(() => {
			expect(fetchMock).toHaveBeenCalledTimes(2);
		});

		resolveProjectBLoad?.(
			jsonResponse({ items: [projectBItem], total: 1, skip: 0, limit: 50 })
		);

		await waitFor(() => {
			expect(screen.getByText('project-b-fresh.jpg')).toBeTruthy();
		});

		resolveProjectALoad?.(
			jsonResponse({ items: [projectAItem], total: 1, skip: 0, limit: 50 })
		);

		await waitFor(() => {
			expect(screen.getByText('project-b-fresh.jpg')).toBeTruthy();
			expect(screen.queryByText('project-a-stale.jpg')).toBeNull();
		});
	});
});
