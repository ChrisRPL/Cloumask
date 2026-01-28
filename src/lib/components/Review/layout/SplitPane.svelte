<script lang="ts" module>
	import type { Snippet } from 'svelte';

	export interface SplitPaneProps {
		leftWidth?: number;
		minLeftWidth?: number;
		maxLeftWidth?: number;
		left?: Snippet;
		right?: Snippet;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';

	let {
		leftWidth = 280,
		minLeftWidth = 200,
		maxLeftWidth = 400,
		left,
		right,
		class: className
	}: SplitPaneProps = $props();

	let currentWidth = $state(280);
	let isDragging = $state(false);
	let containerRef: HTMLDivElement | null = $state(null);

	// Sync with prop on mount
	$effect(() => {
		currentWidth = leftWidth;
	});

	function handleMouseDown(e: MouseEvent) {
		e.preventDefault();
		isDragging = true;
	}

	function handleMouseMove(e: MouseEvent) {
		if (!isDragging || !containerRef) return;

		const containerRect = containerRef.getBoundingClientRect();
		const newWidth = e.clientX - containerRect.left;
		currentWidth = Math.max(minLeftWidth, Math.min(maxLeftWidth, newWidth));
	}

	function handleMouseUp() {
		isDragging = false;
	}
</script>

<svelte:window onmousemove={handleMouseMove} onmouseup={handleMouseUp} />

<div
	bind:this={containerRef}
	class={cn('flex h-full overflow-hidden', isDragging && 'select-none cursor-col-resize', className)}
>
	<!-- Left pane -->
	<div class="flex-shrink-0 h-full overflow-hidden" style="width: {currentWidth}px">
		{#if left}
			{@render left()}
		{/if}
	</div>

	<!-- Resize handle -->
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions a11y_no_noninteractive_tabindex -->
	<div
		class={cn(
			'w-1 h-full cursor-col-resize flex-shrink-0',
			'bg-border hover:bg-primary/50 transition-colors',
			isDragging && 'bg-primary'
		)}
		role="separator"
		tabindex="0"
		aria-orientation="vertical"
		aria-valuenow={currentWidth}
		aria-valuemin={minLeftWidth}
		aria-valuemax={maxLeftWidth}
		onmousedown={handleMouseDown}
	></div>

	<!-- Right pane -->
	<div class="flex-1 h-full overflow-hidden min-w-0">
		{#if right}
			{@render right()}
		{/if}
	</div>
</div>
