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
	import ProjectSelector from '$lib/components/Layout/ProjectSelector.svelte';
	import { getUIState } from '$lib/stores/ui.svelte';
	import MessageBubble from './MessageBubble.svelte';

	let { messages, isStreaming = false, class: className }: MessageListProps = $props();
	const ui = getUIState();
	const hasSelectedProject = $derived(ui.currentProject !== null);
	const selectedProjectName = $derived(ui.currentProject?.name?.trim() || 'Current project');

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
		const messageCount = messages.length;
		let frameId: number | null = null;

		if (viewport && messageCount === 0) {
			viewport.scrollTo({ top: 0, behavior: 'instant' });
			shouldAutoScroll = true;
			isAtBottom = true;
		} else if (shouldAutoScroll && viewport) {
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
			<div class="flex h-full items-center justify-center px-4 py-8 sm:px-6 xl:items-start xl:px-8 xl:py-12">
				<div
					class="w-full max-w-6xl rounded-[1.75rem] border border-border bg-card p-6 shadow-[0_24px_80px_-48px_rgba(12,59,31,0.42)] sm:p-8 xl:p-10"
					data-chat-empty-state
				>
					<div class="grid gap-6 xl:items-start xl:grid-cols-[minmax(0,1.35fr)_minmax(21rem,0.85fr)] xl:gap-8">
						<div class="space-y-4">
							<div class="space-y-3">
								<div class="inline-flex h-11 w-11 items-center justify-center rounded-xl border border-border/70 bg-background text-lg text-foreground/70">
									&lt;/&gt;
								</div>
								<div class="space-y-2">
									<h3 class="text-xl font-semibold tracking-tight text-foreground xl:text-[2rem]">
										{hasSelectedProject ? 'Ready to start the next run' : 'Start a local vision workflow'}
									</h3>
									<p class="max-w-2xl text-sm leading-6 text-foreground/82 xl:text-[15px]">
										{#if hasSelectedProject}
											{selectedProjectName} is already selected. Describe the footage, image batch,
											or review task and Cloumask will build the plan, run the steps, and keep the
											approval checkpoints in one place.
										{:else}
											Describe the footage, image batch, or review task. Cloumask will build the
											plan, run the steps, and keep the approval checkpoints in one place.
										{/if}
									</p>
								</div>
							</div>

							<div class="max-w-2xl rounded-xl border border-border bg-background p-4">
								<p class="text-[11px] font-semibold uppercase tracking-[0.18em] text-foreground/55">
									{hasSelectedProject ? 'Active project' : 'Choose project'}
								</p>
								{#if hasSelectedProject}
									<div class="mt-3 rounded-xl border border-border/80 bg-card px-4 py-3">
										<p class="text-sm font-semibold text-foreground">{selectedProjectName}</p>
										<p class="mt-1 text-sm leading-6 text-foreground/75">
											New chat runs, plans, and review checkpoints will stay grouped here. Switch
											projects from the header when you need a different workspace.
										</p>
									</div>
								{:else}
									<p class="mt-2 text-sm leading-6 text-foreground/82">
										Every run needs a project. Pick one here first so chat, plans, and review work stay grouped.
									</p>
									<ProjectSelector
										class="mt-3 w-full sm:max-w-sm"
										placeholder="Choose project to start..."
										triggerAriaLabel="Choose project to start chat"
									/>
								{/if}
							</div>

							<div class="flex flex-wrap gap-2 text-xs">
								<span class="rounded-full border border-border bg-background px-3 py-1 text-foreground/90">
									Chat
								</span>
								<span class="rounded-full border border-border bg-background px-3 py-1 text-foreground/90">
									Plan
								</span>
								<span class="rounded-full border border-border bg-background px-3 py-1 text-foreground/90">
									Execute
								</span>
								<span class="rounded-full border border-border bg-background px-3 py-1 text-foreground/90">
									Review
								</span>
							</div>

							<div class="rounded-xl border border-border bg-background p-4">
								<p class="text-[11px] font-semibold uppercase tracking-[0.22em] text-foreground/55">
									Good first prompts
								</p>
								<div class="mt-3 space-y-2 text-sm text-foreground/92">
									<p class="rounded-lg border border-border/80 bg-card px-3 py-2">
										Plan an anonymization run for the latest loading bay footage.
									</p>
									<p class="rounded-lg border border-border/80 bg-card px-3 py-2">
										Review low-confidence people detections before exporting labels.
									</p>
									<p class="rounded-lg border border-border/80 bg-card px-3 py-2">
										Prepare a point-cloud detection workflow for the current warehouse scan.
									</p>
								</div>
							</div>
						</div>

						<div class="rounded-xl border border-border bg-background p-4 xl:max-w-sm xl:justify-self-end xl:self-start">
							<p class="text-[11px] font-semibold uppercase tracking-[0.22em] text-foreground/55">
								{hasSelectedProject ? 'Ready when you are' : 'Before you send'}
							</p>
							<div class="mt-4 space-y-3 text-sm leading-6 text-foreground/78">
								{#if hasSelectedProject}
									<p>Use the message bar below to describe the job in plain language.</p>
									<p>Jump to Plan after the first draft appears, then return here to keep the run moving.</p>
								{:else}
									<p>Use the message bar below to describe the job in plain language.</p>
									<p>Move through the workflow with the sidebar or keys 1-5.</p>
								{/if}
							</div>
							<div class="mt-5 rounded-lg border border-emerald-700/25 bg-emerald-500/12 px-3 py-2 text-xs text-emerald-950/85">
								{#if hasSelectedProject}
									The current project stays attached to the thread until you switch workspaces.
								{:else}
									Local-first setup. Status, plans, and review checkpoints stay visible as the run
									progresses.
								{/if}
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
