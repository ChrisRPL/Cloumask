import { render, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ReviewQueueTestHost from '$lib/test-utils/ReviewQueueTestHost.svelte';

function jsonResponse(body: unknown, init?: ResponseInit): Response {
	return new Response(JSON.stringify(body), {
		status: 200,
		headers: { 'content-type': 'application/json' },
		...init
	});
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
});
