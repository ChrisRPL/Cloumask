<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { MessageRole } from '$lib/types/agent';
	import { marked } from 'marked';
	import DOMPurify from 'dompurify';

	export interface MessageContentProps {
		messageId?: string;
		messageTimestamp?: string;
		content: string;
		role: MessageRole;
		isStreaming?: boolean;
		class?: string;
	}
</script>

<script lang="ts">
	let {
		messageId = '',
		messageTimestamp = '',
		content,
		role,
		isStreaming = false,
		class: className
	}: MessageContentProps = $props();

	let renderedContent = $state('');
	let isAnimating = $state(false);
	let animatedTurnKey = $state<string | null>(null);

	// Configure marked for inline rendering (no wrapping <p> tags)
	marked.use({ breaks: true, gfm: true });

	// Parse markdown reactively from renderedContent
	const htmlContent = $derived.by(() => {
		if (!renderedContent || role === 'tool') return renderedContent;
		const raw = marked.parse(renderedContent, { async: false }) as string;
		return DOMPurify.sanitize(raw);
	});

	function shouldAnimateMessage(text: string): boolean {
		// Skip very large payloads to keep UI responsive.
		return text.length > 0 && text.length <= 2400;
	}

	function calculateChunkSize(textLength: number): number {
		// Keep animation time roughly between 350ms and 3.5s so typing is visible.
		const targetDurationMs = Math.max(350, Math.min(3500, textLength * 15));
		const frameMs = 24;
		const frames = Math.max(1, Math.floor(targetDurationMs / frameMs));
		return Math.max(1, Math.ceil(textLength / frames));
	}

	function resolveTurnKey(id: string, timestamp: string): string {
		if (id && timestamp) return `${id}:${timestamp}`;
		return id;
	}

	$effect(() => {
		const trimmed = content.trim();
		const turnKey = resolveTurnKey(messageId, messageTimestamp);

		if (!trimmed) {
			renderedContent = '';
			isAnimating = false;
			return;
		}

		const shouldAnimate =
			role === 'assistant' &&
			shouldAnimateMessage(trimmed) &&
			turnKey !== '' &&
			animatedTurnKey !== turnKey;

		if (!shouldAnimate) {
			renderedContent = trimmed;
			isAnimating = false;
			if (turnKey && role === 'assistant') {
				animatedTurnKey = turnKey;
			}
			return;
		}

		renderedContent = '';
		isAnimating = true;
		const chunkSize = calculateChunkSize(trimmed.length);
		let cursor = 0;
		const timer = setInterval(() => {
			cursor = Math.min(trimmed.length, cursor + chunkSize);
			renderedContent = trimmed.slice(0, cursor);
			if (cursor >= trimmed.length) {
				clearInterval(timer);
				isAnimating = false;
				animatedTurnKey = turnKey;
			}
		}, 24);

		return () => clearInterval(timer);
	});
</script>

<div
	class={cn(
		'text-sm leading-relaxed whitespace-pre-wrap break-words',
		role === 'system' && 'text-muted-foreground italic',
		role === 'tool' && 'font-mono text-xs bg-muted/30 p-2 rounded',
		className
	)}
>
	{#if role === 'tool'}
		{renderedContent}
	{:else}
		{@html htmlContent}
	{/if}
	{#if isStreaming || isAnimating}
		<span class="animate-blink text-forest-light ml-0.5">|</span>
	{/if}
</div>

<style>
	@keyframes blink {
		0%,
		50% {
			opacity: 1;
		}
		51%,
		100% {
			opacity: 0;
		}
	}

	.animate-blink {
		animation: blink 1s step-end infinite;
	}

	/* Markdown content styling */
	div :global(p) {
		margin: 0.25em 0;
	}
	div :global(p:first-child) {
		margin-top: 0;
	}
	div :global(p:last-child) {
		margin-bottom: 0;
	}
	div :global(code) {
		font-size: 0.85em;
		background: var(--muted);
		padding: 0.15em 0.35em;
		border-radius: 0.25em;
		font-family: 'JetBrains Mono', 'Fira Code', monospace;
	}
	div :global(pre) {
		background: var(--muted);
		padding: 0.75em;
		border-radius: 0.5em;
		overflow-x: auto;
		margin: 0.5em 0;
	}
	div :global(pre code) {
		background: none;
		padding: 0;
	}
	div :global(ul), div :global(ol) {
		padding-left: 1.5em;
		margin: 0.25em 0;
	}
	div :global(strong) {
		font-weight: 600;
	}
</style>
