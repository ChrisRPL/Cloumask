<script lang="ts" module>
	import type { ReviewItem } from '$lib/types/review';

	export interface ItemListProps {
		items: ReviewItem[];
		selectedItemId: string | null;
		onItemSelect?: (id: string) => void;
		onLoadMore?: () => void;
		hasMore?: boolean;
		isLoading?: boolean;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import ReviewListItem from './ReviewListItem.svelte';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Loader2 } from '@lucide/svelte';

	let {
		items,
		selectedItemId,
		onItemSelect,
		onLoadMore,
		hasMore = false,
		isLoading = false,
		class: className
	}: ItemListProps = $props();

	let listRef: HTMLDivElement | null = $state(null);

	// Keyboard navigation
	function handleKeydown(e: KeyboardEvent) {
		if (!items.length) return;

		const currentIndex = items.findIndex((item) => item.id === selectedItemId);

		if (e.key === 'ArrowDown' || e.key === 'j') {
			e.preventDefault();
			const nextIndex = Math.min(currentIndex + 1, items.length - 1);
			onItemSelect?.(items[nextIndex].id);
		} else if (e.key === 'ArrowUp' || e.key === 'k') {
			e.preventDefault();
			const prevIndex = Math.max(currentIndex - 1, 0);
			onItemSelect?.(items[prevIndex].id);
		}
	}
</script>

<div
	bind:this={listRef}
	class={cn(
		'flex flex-col h-full overflow-y-auto',
		'bg-background border-r border-border',
		className
	)}
	role="listbox"
	tabindex="0"
	onkeydown={handleKeydown}
>
	<!-- Item count header -->
	<div class="sticky top-0 z-10 px-3 py-2 bg-background/95 backdrop-blur border-b border-border">
		<span class="text-xs font-mono text-muted-foreground">
			{items.length} item{items.length !== 1 ? 's' : ''}
		</span>
	</div>

	<!-- Items -->
	<div class="flex-1 p-2 space-y-1">
		{#if items.length === 0}
			{#if isLoading}
				<div class="flex items-center justify-center h-32 text-sm text-muted-foreground font-mono">
					Loading...
				</div>
			{:else}
				<div class="flex h-full items-center justify-center p-3">
					<div class="w-full rounded-2xl border border-border/60 bg-card/35 p-5 font-mono">
						<div class="space-y-4">
							<div class="space-y-2">
								<div class="inline-flex items-center rounded-full border border-border/70 bg-background px-3 py-1 text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
									Review inbox
								</div>
								<p class="text-lg text-foreground">No items to review</p>
								<p class="text-sm leading-7 text-muted-foreground">
									Flagged detections and human-check checkpoints appear here after a run reaches
									review.
								</p>
							</div>

							<div class="grid gap-2 text-xs text-foreground/80">
								<div class="rounded-xl border border-border/60 bg-background/70 px-3 py-2">
									Low-confidence items
								</div>
								<div class="rounded-xl border border-border/60 bg-background/70 px-3 py-2">
									Manual edits before export
								</div>
								<div class="rounded-xl border border-border/60 bg-background/70 px-3 py-2">
									Batch approve or reject
								</div>
							</div>

							<div class="rounded-xl border border-emerald-700/15 bg-emerald-500/8 px-4 py-3 text-xs leading-6 text-emerald-900/75">
								Next: finish a run in Execute, then return here to triage the queue.
							</div>
						</div>
					</div>
				</div>
			{/if}
		{:else}
			{#each items as item (item.id)}
				<ReviewListItem
					{item}
					selected={item.id === selectedItemId}
					onClick={() => onItemSelect?.(item.id)}
				/>
			{/each}

			<!-- Load more button -->
			{#if hasMore}
				<div class="py-2">
					<Button
						variant="ghost"
						size="sm"
						onclick={onLoadMore}
						disabled={isLoading}
						class="w-full h-8 text-xs font-mono"
					>
						{#if isLoading}
							<Loader2 class="w-3 h-3 animate-spin mr-2" />
							Loading...
						{:else}
							Load more...
						{/if}
					</Button>
				</div>
			{/if}
		{/if}
	</div>
</div>
