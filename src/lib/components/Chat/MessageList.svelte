<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { Message } from '$lib/types/agent';

	export interface MessageListProps {
		messages: Message[];
		isStreaming?: boolean;
		class?: string;
	}
</script>

<script lang="ts">
	import { ArrowDown } from 'lucide-svelte';
	import { Button } from '$lib/components/ui/button';
	import MessageBubble from './MessageBubble.svelte';

	let { messages, isStreaming = false, class: className }: MessageListProps = $props();

	// Refs and scroll state
	let viewport: HTMLDivElement | null = $state(null);
	let isAtBottom = $state(true);
	let shouldAutoScroll = $state(true);

	// Check if user is near bottom (within threshold)
	function checkIsAtBottom(): boolean {
		if (!viewport) return true;
		const { scrollTop, scrollHeight, clientHeight } = viewport;
		return scrollHeight - scrollTop - clientHeight < 100;
	}

	// Handle scroll events
	function handleScroll() {
		isAtBottom = checkIsAtBottom();
		// Re-enable auto-scroll if user scrolls to bottom
		if (isAtBottom) {
			shouldAutoScroll = true;
		}
	}

	// Handle wheel/touch to disable auto-scroll
	function handleUserScroll() {
		if (!checkIsAtBottom()) {
			shouldAutoScroll = false;
		}
	}

	// Scroll to bottom
	function scrollToBottom(smooth = true) {
		if (!viewport) return;
		viewport.scrollTo({
			top: viewport.scrollHeight,
			behavior: smooth ? 'smooth' : 'instant'
		});
		shouldAutoScroll = true;
		isAtBottom = true;
	}

	// Auto-scroll when messages change
	$effect(() => {
		// Trigger on message count change
		const _ = messages.length;
		if (shouldAutoScroll && viewport) {
			// Use requestAnimationFrame to ensure DOM is updated
			requestAnimationFrame(() => {
				scrollToBottom(true);
			});
		}
	});
</script>

<div class={cn('relative flex-1 min-h-0', className)}>
	<!-- Scrollable container -->
	<div
		bind:this={viewport}
		class="h-full overflow-y-auto px-4 py-4"
		onscroll={handleScroll}
		onwheel={handleUserScroll}
		ontouchmove={handleUserScroll}
		role="log"
		aria-live="polite"
		aria-label="Chat messages"
	>
		<!-- Empty state -->
		{#if messages.length === 0}
			<div class="flex flex-col items-center justify-center h-full text-muted-foreground">
				<span class="text-2xl mb-2 opacity-30">&lt;&gt;</span>
				<span class="text-sm">No messages yet</span>
				<span class="text-xs opacity-60 mt-1">Type a message to start</span>
			</div>
		{:else}
			<!-- Messages -->
			<div class="flex flex-col gap-4">
				{#each messages as message, index (message.id)}
					<MessageBubble
						{message}
						isLatest={index === messages.length - 1}
					/>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Scroll to bottom button -->
	{#if !isAtBottom && messages.length > 0}
		<div class="absolute bottom-4 left-1/2 -translate-x-1/2">
			<Button
				variant="outline"
				size="icon"
				class="h-8 w-8 rounded-full shadow-md bg-background/80 backdrop-blur-sm"
				onclick={() => scrollToBottom()}
			>
				<ArrowDown class="h-4 w-4" />
				<span class="sr-only">Scroll to bottom</span>
			</Button>
		</div>
	{/if}
</div>
