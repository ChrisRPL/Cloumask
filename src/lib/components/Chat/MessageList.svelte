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
	import { ArrowDown } from '@lucide/svelte';
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
		let frameId: number | null = null;

		if (shouldAutoScroll && viewport) {
			// Use requestAnimationFrame to ensure DOM is updated
			frameId = requestAnimationFrame(() => {
				scrollToBottom(true);
			});
		}

		// Cancel pending frame on cleanup to prevent memory leak
		return () => {
			if (frameId !== null) {
				cancelAnimationFrame(frameId);
			}
		};
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
			<div class="flex h-full items-center justify-center px-4 py-10">
				<div class="w-full max-w-3xl rounded-2xl border border-border/70 bg-card/60 p-6 shadow-sm">
					<div class="grid gap-6 lg:grid-cols-[1.4fr_0.9fr]">
						<div class="space-y-4">
							<div class="space-y-3">
								<div class="inline-flex h-11 w-11 items-center justify-center rounded-xl border border-border/70 bg-background text-lg text-foreground/70">
									&lt;/&gt;
								</div>
								<div class="space-y-2">
									<h3 class="text-xl font-semibold tracking-tight text-foreground">
										Start a local vision workflow
									</h3>
									<p class="max-w-xl text-sm leading-6 text-muted-foreground">
										Describe the footage, image batch, or review task. Cloumask will build the
										plan, run the steps, and keep the approval checkpoints in one place.
									</p>
								</div>
							</div>

							<div class="flex flex-wrap gap-2 text-xs">
								<span class="rounded-full border border-border/70 bg-background px-3 py-1 text-foreground/80">
									Chat
								</span>
								<span class="rounded-full border border-border/70 bg-background px-3 py-1 text-foreground/80">
									Plan
								</span>
								<span class="rounded-full border border-border/70 bg-background px-3 py-1 text-foreground/80">
									Execute
								</span>
								<span class="rounded-full border border-border/70 bg-background px-3 py-1 text-foreground/80">
									Review
								</span>
							</div>

							<div class="rounded-xl border border-border/60 bg-background/80 p-4">
								<p class="text-[11px] font-semibold uppercase tracking-[0.22em] text-muted-foreground">
									Good first prompts
								</p>
								<div class="mt-3 space-y-2 text-sm text-foreground/85">
									<p class="rounded-lg border border-border/50 bg-card px-3 py-2">
										Plan an anonymization run for the latest loading bay footage.
									</p>
									<p class="rounded-lg border border-border/50 bg-card px-3 py-2">
										Review low-confidence people detections before exporting labels.
									</p>
									<p class="rounded-lg border border-border/50 bg-card px-3 py-2">
										Prepare a point-cloud detection workflow for the current warehouse scan.
									</p>
								</div>
							</div>
						</div>

						<div class="rounded-xl border border-border/60 bg-background/70 p-4">
							<p class="text-[11px] font-semibold uppercase tracking-[0.22em] text-muted-foreground">
								Before you send
							</p>
							<div class="mt-4 space-y-3 text-sm leading-6 text-muted-foreground">
								<p>Pick or create a project in the header so runs stay grouped.</p>
								<p>Use the message bar below to describe the job in plain language.</p>
								<p>Move through the workflow with the sidebar or keys 1-5.</p>
							</div>
							<div class="mt-5 rounded-lg border border-emerald-700/15 bg-emerald-500/8 px-3 py-2 text-xs text-emerald-900/75">
								Local-first setup. Status, plans, and review checkpoints stay visible as the run
								progresses.
							</div>
						</div>
					</div>
				</div>
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
