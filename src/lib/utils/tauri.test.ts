import { describe, expect, it } from 'vitest';
import { parseLLMPullEventData } from './tauri';

describe('parseLLMPullEventData', () => {
	it('parses JSON progress payloads and computes percent', () => {
		const event = parseLLMPullEventData(
			JSON.stringify({
				status: 'downloading',
				total: 2_000,
				completed: 500,
				digest: 'sha256:abc',
			}),
			'qwen3:14b'
		);

		expect(event.model).toBe('qwen3:14b');
		expect(event.status).toBe('downloading');
		expect(event.totalBytes).toBe(2_000);
		expect(event.completedBytes).toBe(500);
		expect(event.progressPercent).toBe(25);
		expect(event.digest).toBe('sha256:abc');
	});

	it('parses plain status payloads', () => {
		const event = parseLLMPullEventData('pulling manifest', 'qwen3:14b');
		expect(event.status).toBe('pulling manifest');
		expect(event.progressPercent).toBeNull();
	});

	it('marks done payloads as complete', () => {
		const event = parseLLMPullEventData('[DONE]', 'qwen3:14b');
		expect(event.status).toBe('done');
		expect(event.progressPercent).toBe(100);
	});

	it('sets 100% for success statuses without byte counters', () => {
		const event = parseLLMPullEventData(JSON.stringify({ status: 'success' }), 'qwen3:14b');
		expect(event.progressPercent).toBe(100);
	});
});
