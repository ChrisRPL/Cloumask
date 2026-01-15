<script lang="ts" module>
	export interface Annotation {
		label: string;
		confidence: number;
		bbox: {
			x: number;
			y: number;
			width: number;
			height: number;
		};
	}

	export interface PreviewItem {
		id: string;
		imagePath: string;
		thumbnailUrl: string;
		annotations: Annotation[];
		status: 'processed' | 'flagged' | 'error';
	}

	export interface PreviewThumbnailProps {
		preview: PreviewItem;
		class?: string;
		onClick?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';

	let { preview, class: className, onClick }: PreviewThumbnailProps = $props();

	const statusDotClasses = $derived(() => {
		switch (preview.status) {
			case 'flagged':
				return 'bg-amber-500';
			case 'error':
				return 'bg-destructive';
			default:
				return 'bg-green-500';
		}
	});
</script>

<button
	type="button"
	class={cn(
		'relative aspect-video rounded-md overflow-hidden bg-muted/50 border border-border',
		'hover:border-primary/50 hover:scale-[1.02] transition-all cursor-pointer',
		'focus:outline-none focus:ring-2 focus:ring-primary/50',
		className
	)}
	onclick={onClick}
>
	<!-- Image -->
	<img
		src={preview.thumbnailUrl}
		alt={preview.imagePath}
		class="w-full h-full object-cover"
		loading="lazy"
	/>

	<!-- Bounding box overlays -->
	<svg class="absolute inset-0 w-full h-full pointer-events-none">
		{#each preview.annotations as ann}
			<rect
				x="{ann.bbox.x * 100}%"
				y="{ann.bbox.y * 100}%"
				width="{ann.bbox.width * 100}%"
				height="{ann.bbox.height * 100}%"
				class="fill-none stroke-forest-light stroke-2"
				rx="2"
			/>
		{/each}
	</svg>

	<!-- Status indicator dot -->
	<div
		class={cn(
			'absolute top-1 right-1 w-2 h-2 rounded-full ring-1 ring-background',
			statusDotClasses()
		)}
	></div>

	<!-- Annotation count badge -->
	{#if preview.annotations.length > 0}
		<div
			class="absolute bottom-1 left-1 px-1.5 py-0.5 bg-background/80 rounded text-xs font-mono tabular-nums"
		>
			{preview.annotations.length}
		</div>
	{/if}
</button>
