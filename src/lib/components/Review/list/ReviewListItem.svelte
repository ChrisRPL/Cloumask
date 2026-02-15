<script lang="ts" module>
	import type { ReviewItem, ReviewStatus } from '$lib/types/review';

	export interface ReviewListItemProps {
		item: ReviewItem;
		selected?: boolean;
		class?: string;
		onClick?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';

	let { item, selected = false, class: className, onClick }: ReviewListItemProps = $props();

	const statusConfig = $derived.by(() => {
		const configs: Record<
			ReviewStatus,
			{ color: string; bg: string; label: string }
		> = {
			pending: { color: 'bg-amber-500', bg: 'bg-amber-500/10', label: 'Pending' },
			approved: { color: 'bg-green-500', bg: 'bg-green-500/10', label: 'Approved' },
			rejected: { color: 'bg-red-500', bg: 'bg-red-500/10', label: 'Rejected' },
			modified: { color: 'bg-blue-500', bg: 'bg-blue-500/10', label: 'Modified' }
		};
		return configs[item.status];
	});

	const minConfidence = $derived(
		item.annotations.length > 0
			? Math.min(...item.annotations.map((a) => a.confidence))
			: null
	);

	const avgConfidence = $derived(
		item.annotations.length > 0
			? item.annotations.reduce((sum, a) => sum + a.confidence, 0) / item.annotations.length
			: null
	);
</script>

<button
	type="button"
	class={cn(
		'group relative w-full text-left',
		'flex items-center gap-3 p-2',
		'rounded-md border border-transparent',
		'transition-all duration-150',
		'hover:bg-muted/30 hover:border-border',
		'focus:outline-none focus:ring-2 focus:ring-primary/50',
		selected && 'bg-muted/50 border-border ring-1 ring-primary/30',
		className
	)}
	onclick={onClick}
>
	<!-- Thumbnail -->
	<div class="relative w-24 h-16 flex-shrink-0 rounded overflow-hidden bg-muted/30">
		<img
			src={item.thumbnailUrl}
			alt={item.fileName}
			class="w-full h-full object-cover"
			loading="lazy"
		/>
		<!-- Annotation count badge -->
		{#if item.annotations.length > 0}
			<div
				class="absolute bottom-0.5 right-0.5 px-1 py-0.5 bg-background/90 rounded text-[10px] font-mono tabular-nums text-foreground"
			>
				{item.annotations.length}
			</div>
		{/if}
		<!-- Status dot -->
		<div
			class={cn(
				'absolute top-0.5 right-0.5 w-2 h-2 rounded-full ring-1 ring-background',
				statusConfig.color
			)}
		></div>
	</div>

	<!-- Info -->
	<div class="flex-1 min-w-0 space-y-0.5">
		<!-- File name -->
		<div class="flex items-center gap-2">
			<span class="text-xs font-mono truncate text-foreground">
				{item.fileName}
			</span>
			{#if item.flagged}
				<span class="text-amber-500 text-[10px]" title={item.flagReason}>!</span>
			{/if}
		</div>

		<!-- Status and confidence -->
		<div class="flex items-center gap-2 text-[10px] text-muted-foreground font-mono">
			<span
				class={cn(
					'inline-flex items-center gap-1 px-1.5 py-0.5 rounded-sm',
					statusConfig.bg
				)}
			>
				<span class={cn('w-1.5 h-1.5 rounded-full', statusConfig.color)}></span>
				{statusConfig.label}
			</span>

			{#if avgConfidence !== null}
				<span class="tabular-nums" title="Average confidence">
					{(avgConfidence * 100).toFixed(0)}%
				</span>
			{/if}
		</div>
	</div>

	<!-- Selection indicator -->
	{#if selected}
		<div class="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-6 bg-primary rounded-r"></div>
	{/if}
</button>
