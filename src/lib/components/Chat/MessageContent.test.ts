import { render } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import MessageContent from './MessageContent.svelte';

describe('MessageContent typing animation', () => {
	beforeEach(() => {
		vi.useFakeTimers();
	});

	afterEach(async () => {
		await vi.runOnlyPendingTimersAsync();
		vi.useRealTimers();
	});

	it('reruns typing animation for a new assistant turn even if message id repeats', async () => {
		const { container, rerender } = render(MessageContent, {
			messageId: 'assistant-message',
			messageTimestamp: '2026-02-18T10:00:00.000Z',
			content: 'First assistant response.',
			role: 'assistant',
			isStreaming: false
		});

		await vi.advanceTimersByTimeAsync(1000);

		expect(container.textContent).toContain('First assistant response.');
		expect(container.querySelector('.animate-blink')).toBeNull();

		await rerender({
			messageId: 'assistant-message',
			messageTimestamp: '2026-02-18T10:00:05.000Z',
			content: 'Second assistant response.',
			role: 'assistant',
			isStreaming: false
		});

		await Promise.resolve();

		expect(container.querySelector('.animate-blink')).not.toBeNull();
		expect(container.textContent).not.toContain('Second assistant response.');

		await vi.advanceTimersByTimeAsync(1000);

		expect(container.textContent).toContain('Second assistant response.');
		expect(container.querySelector('.animate-blink')).toBeNull();
	});
});
