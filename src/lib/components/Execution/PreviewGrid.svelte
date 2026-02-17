<script lang="ts" module>
	export interface PreviewGridProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { getExecutionState } from '$lib/stores/execution.svelte';
	import { getUIState } from '$lib/stores/ui.svelte';
	import { convertFileSrc } from '@tauri-apps/api/core';
	import { isTauri } from '$lib/utils/tauri';
	import { toLocalImageUrl } from '$lib/utils/local-image';
	import PreviewThumbnail from './PreviewThumbnail.svelte';
	import { ImageIcon, Loader2 } from '@lucide/svelte';

	let { class: className }: PreviewGridProps = $props();

	// Get execution state to show context-aware messages
	const execution = getExecutionState();
	const ui = getUIState();

	const isDesktopTauri = isTauri();
	const safeConvertFileSrc: ((path: string) => string) | null =
		typeof convertFileSrc === 'function' ? convertFileSrc : null;

	const previews = $derived.by(() =>
		execution.previews.map((preview) => ({
			...preview,
			thumbnailUrl:
				preview.assetType === 'pointcloud'
					? preview.thumbnailUrl
					: isDesktopTauri && safeConvertFileSrc
					? safeConvertFileSrc(preview.thumbnailUrl)
					: toLocalImageUrl(preview.thumbnailUrl)
		}))
	);

	// Derived states for better UX
	const isRunning = $derived(execution.isRunning);
	const isIdle = $derived(execution.status === 'idle');
	const hasProcessed = $derived(execution.stats.processed > 0 || previews.length > 0);

	function handlePreviewClick(preview: (typeof previews)[number]) {
		if (preview.assetType !== 'pointcloud') return;
		execution.setSelectedPointcloudPreview(preview);
		ui.setView('pointcloud');
	}
</script>

<div class={cn('flex flex-col h-full', className)}>
	<!-- Header -->
	<div class="px-4 py-2 border-b border-border flex items-center justify-between">
		<span class="text-xs font-mono text-muted-foreground">&gt; LIVE PREVIEW</span>
		<span class="text-xs font-mono text-muted-foreground/60 tabular-nums">
			{#if isRunning && hasProcessed}
				{execution.stats.processed} processed
			{:else if previews.length > 0}
				{previews.length} recent
			{:else}
				awaiting
			{/if}
		</span>
	</div>

	<!-- Grid -->
	<div class="flex-1 p-4 overflow-auto">
		{#if previews.length === 0}
			<!-- Empty state with context-aware placeholders -->
			<div class="grid grid-cols-3 gap-3">
				{#each Array(6) as _, i}
					<div
						class={cn(
							'aspect-video rounded-md border flex items-center justify-center',
							isRunning
								? 'bg-forest-dark/10 border-forest/30 animate-pulse'
								: 'bg-muted/20 border-border/50'
						)}
					>
						{#if isRunning && i === 0}
							<Loader2 class="h-6 w-6 text-forest-light animate-spin" />
						{:else}
							<ImageIcon class="h-6 w-6 text-muted-foreground/30" />
						{/if}
					</div>
				{/each}
			</div>
			<p class="text-center text-xs text-muted-foreground/60 font-mono mt-4">
				{#if isRunning}
					Processing items... previews will appear shortly
				{:else if isIdle}
					Start execution to see live previews
				{:else}
					Previews will appear as items are processed
				{/if}
			</p>
		{:else}
			<div class="grid grid-cols-3 gap-3">
				{#each previews as preview (preview.id)}
					<PreviewThumbnail {preview} onClick={() => handlePreviewClick(preview)} />
				{/each}
				<!-- Fill remaining slots with empty placeholders -->
				{#each Array(Math.max(0, 6 - previews.length)) as _}
					<div
						class="aspect-video bg-muted/20 rounded-md border border-border/50 flex items-center justify-center"
					>
						<ImageIcon class="h-6 w-6 text-muted-foreground/30" />
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>
