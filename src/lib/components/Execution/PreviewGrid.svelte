<script lang="ts" module>
	export interface PreviewGridProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import PreviewThumbnail, { type PreviewItem } from './PreviewThumbnail.svelte';
	import { ImageIcon } from '@lucide/svelte';

	let { class: className }: PreviewGridProps = $props();

	// Mock preview data (will be replaced with SSE updates)
	const mockPreviews: PreviewItem[] = $state([
		{
			id: '1',
			imagePath: '/images/sample-001.jpg',
			thumbnailUrl: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 180"%3E%3Crect fill="%23166534" width="320" height="180"/%3E%3Ctext fill="%23faf7f0" x="160" y="95" text-anchor="middle" font-family="monospace" font-size="14"%3Eimage-001.jpg%3C/text%3E%3C/svg%3E',
			annotations: [
				{ label: 'face', confidence: 0.95, bbox: { x: 0.2, y: 0.1, width: 0.3, height: 0.4 } },
			],
			status: 'processed',
		},
		{
			id: '2',
			imagePath: '/images/sample-002.jpg',
			thumbnailUrl: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 180"%3E%3Crect fill="%230c3b1f" width="320" height="180"/%3E%3Ctext fill="%23faf7f0" x="160" y="95" text-anchor="middle" font-family="monospace" font-size="14"%3Eimage-002.jpg%3C/text%3E%3C/svg%3E',
			annotations: [
				{ label: 'face', confidence: 0.87, bbox: { x: 0.4, y: 0.2, width: 0.25, height: 0.35 } },
				{ label: 'plate', confidence: 0.72, bbox: { x: 0.1, y: 0.6, width: 0.2, height: 0.1 } },
			],
			status: 'flagged',
		},
		{
			id: '3',
			imagePath: '/images/sample-003.jpg',
			thumbnailUrl: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 180"%3E%3Crect fill="%23166534" width="320" height="180"/%3E%3Ctext fill="%23faf7f0" x="160" y="95" text-anchor="middle" font-family="monospace" font-size="14"%3Eimage-003.jpg%3C/text%3E%3C/svg%3E',
			annotations: [],
			status: 'processed',
		},
	]);

	// In a real implementation, this would subscribe to SSE events
	// and maintain a rolling window of the 6 most recent previews
	const previews = $derived(mockPreviews.slice(0, 6));
</script>

<div class={cn('flex flex-col h-full', className)}>
	<!-- Header -->
	<div class="px-4 py-2 border-b border-border flex items-center justify-between">
		<span class="text-xs font-mono text-muted-foreground">&gt; LIVE PREVIEW</span>
		<span class="text-xs font-mono text-muted-foreground/60 tabular-nums">
			{previews.length} recent
		</span>
	</div>

	<!-- Grid -->
	<div class="flex-1 p-4 overflow-auto">
		{#if previews.length === 0}
			<!-- Empty state with skeleton placeholders -->
			<div class="grid grid-cols-3 gap-3">
				{#each Array(6) as _, i}
					<div
						class="aspect-video bg-muted/20 rounded-md border border-border/50 flex items-center justify-center"
					>
						<ImageIcon class="h-6 w-6 text-muted-foreground/30" />
					</div>
				{/each}
			</div>
			<p class="text-center text-xs text-muted-foreground/60 font-mono mt-4">
				Previews will appear as items are processed
			</p>
		{:else}
			<div class="grid grid-cols-3 gap-3">
				{#each previews as preview (preview.id)}
					<PreviewThumbnail {preview} />
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
