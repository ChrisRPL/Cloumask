<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { MessageRole } from '$lib/types/agent';

	export interface MessageContentProps {
		content: string;
		role: MessageRole;
		isStreaming?: boolean;
		class?: string;
	}
</script>

<script lang="ts">
	let { content, role, isStreaming = false, class: className }: MessageContentProps = $props();

	// Simple text processing: preserve newlines and basic formatting
	const processedContent = $derived(content.trim());
</script>

<div
	class={cn(
		'text-sm leading-relaxed whitespace-pre-wrap break-words',
		role === 'system' && 'text-muted-foreground italic',
		role === 'tool' && 'font-mono text-xs bg-muted/30 p-2 rounded',
		className
	)}
>
	{processedContent}
	{#if isStreaming}
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
</style>
