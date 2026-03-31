<script lang="ts" module>
	export interface ActionBarProps {
		currentIndex: number;
		total: number;
		canPrev: boolean;
		canNext: boolean;
		isEditMode?: boolean;
		onPrev?: () => void;
		onNext?: () => void;
		onApprove?: () => void;
		onReject?: () => void;
		onToggleEdit?: () => void;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { ChevronLeft, ChevronRight, Check, X, Pencil } from '@lucide/svelte';

	let {
		currentIndex,
		total,
		canPrev,
		canNext,
		isEditMode = false,
		onPrev,
		onNext,
		onApprove,
		onReject,
		onToggleEdit,
		class: className
	}: ActionBarProps = $props();

	const hasCurrentItem = $derived(total > 0 && currentIndex >= 0);
	const canNavigateWithKeys = $derived(total > 1);
</script>

<div
	class={cn(
		'flex items-center justify-between gap-4 px-4 py-3',
		'border-t border-border bg-background',
		className
	)}
>
	{#if total > 0}
		<!-- Navigation -->
		<div class="flex items-center gap-2">
			<Button
				variant="outline"
				size="sm"
				onclick={onPrev}
				disabled={!canPrev}
				class="h-9 px-3 font-mono"
			>
				<ChevronLeft class="w-4 h-4 mr-1" />
				Prev
			</Button>
			<Button
				variant="outline"
				size="sm"
				onclick={onNext}
				disabled={!canNext}
				class="h-9 px-3 font-mono"
			>
				Next
				<ChevronRight class="w-4 h-4 ml-1" />
			</Button>
		</div>

		<!-- Main Actions -->
		<div class="flex items-center gap-2">
			<Button
				variant={isEditMode ? 'secondary' : 'outline'}
				size="sm"
				onclick={onToggleEdit}
				class="h-9 px-4 font-mono"
				disabled={!hasCurrentItem}
			>
				<Pencil class="w-4 h-4 mr-2" />
				Edit
				<kbd class="ml-2 px-1.5 py-0.5 text-[10px] bg-muted rounded">E</kbd>
			</Button>

			<Button
				variant="outline"
				size="sm"
				onclick={onReject}
				class="h-9 px-4 font-mono text-destructive hover:text-destructive hover:bg-destructive/10"
				disabled={!hasCurrentItem}
			>
				<X class="w-4 h-4 mr-2" />
				Reject
				<kbd class="ml-2 px-1.5 py-0.5 text-[10px] bg-muted rounded">R</kbd>
			</Button>

			<Button
				variant="default"
				size="sm"
				onclick={onApprove}
				class="h-9 px-4 font-mono"
				disabled={!hasCurrentItem}
			>
				<Check class="w-4 h-4 mr-2" />
				Approve
				<kbd class="ml-2 px-1.5 py-0.5 text-[10px] bg-primary-foreground/20 rounded">A</kbd>
			</Button>
		</div>
	{:else}
		<p class="text-xs font-mono text-muted-foreground">No review actions available</p>
	{/if}

	<!-- Progress indicator -->
	<div class="flex items-center gap-2 text-sm font-mono text-muted-foreground tabular-nums">
		<span>{currentIndex + 1}</span>
		<span>/</span>
		<span>{total}</span>
	</div>
</div>

<!-- Keyboard hints footer -->
{#if hasCurrentItem || canNavigateWithKeys}
	<div
		class={cn(
			'flex items-center justify-center gap-6 px-4 py-1.5',
			'border-t border-border bg-muted/30',
			'text-[10px] font-mono text-muted-foreground'
		)}
	>
		{#if hasCurrentItem}
			<span><kbd class="px-1 py-0.5 bg-muted rounded">A</kbd> Approve</span>
			<span><kbd class="px-1 py-0.5 bg-muted rounded">R</kbd> Reject</span>
			<span><kbd class="px-1 py-0.5 bg-muted rounded">E</kbd> Edit</span>
			<span><kbd class="px-1 py-0.5 bg-muted rounded">Del</kbd> Delete</span>
			<span><kbd class="px-1 py-0.5 bg-muted rounded">Ctrl+Z</kbd> Undo</span>
		{/if}
		{#if canNavigateWithKeys}
			<span><kbd class="px-1 py-0.5 bg-muted rounded">J/K</kbd> Navigate</span>
		{/if}
	</div>
{/if}
