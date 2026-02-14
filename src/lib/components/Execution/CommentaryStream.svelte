<script lang="ts" module>
	export interface CommentaryStreamProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { getAgentState } from '$lib/stores/agent.svelte';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { marked } from 'marked';
	import DOMPurify from 'dompurify';

	let { class: className }: CommentaryStreamProps = $props();

	const agent = getAgentState();

	marked.use({ breaks: true, gfm: true });

	function renderMarkdown(text: string): string {
		const raw = marked.parse(text, { async: false }) as string;
		return DOMPurify.sanitize(raw);
	}

	// Filter to only show assistant messages
	const commentaryMessages = $derived(
		agent.messages.filter((m) => m.role === 'assistant').slice(-20)
	);

	// Scroll anchor
	let scrollAnchor: HTMLDivElement | null = $state(null);

	// Auto-scroll on new messages
	$effect(() => {
		const _ = commentaryMessages.length;
		if (scrollAnchor) {
			scrollAnchor.scrollIntoView({ behavior: 'smooth', block: 'end' });
		}
	});
</script>

<div class={cn('flex flex-col h-full', className)}>
	<!-- Header -->
	<div class="px-3 py-2 border-b border-border">
		<span class="text-xs font-mono text-muted-foreground">&gt; AGENT OUTPUT</span>
	</div>

	<!-- Message stream -->
	<ScrollArea class="flex-1">
		<div class="p-3 space-y-2">
			{#if commentaryMessages.length === 0}
				<div class="text-xs text-muted-foreground/60 font-mono">
					<span class="text-forest-light">&gt;</span> Waiting for execution to start...
				</div>
			{:else}
				{#each commentaryMessages as msg (msg.id)}
					<div class="text-sm text-muted-foreground animate-fade-in commentary-msg">
						<span class="text-forest-light font-mono">&gt;</span>
						<span class="ml-1">{@html renderMarkdown(msg.content)}</span>
					</div>
				{/each}
			{/if}
			<div bind:this={scrollAnchor}></div>
		</div>
	</ScrollArea>
</div>
