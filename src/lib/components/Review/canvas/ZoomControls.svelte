<script lang="ts" module>
	export interface ZoomControlsProps {
		zoom: number;
		minZoom?: number;
		maxZoom?: number;
		onZoomChange?: (zoom: number) => void;
		onFitToView?: () => void;
		onResetZoom?: () => void;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import { ZoomIn, ZoomOut, Maximize2, RotateCcw } from '@lucide/svelte';

	let {
		zoom,
		minZoom = 0.1,
		maxZoom = 5,
		onZoomChange,
		onFitToView,
		onResetZoom,
		class: className
	}: ZoomControlsProps = $props();

	const zoomPercentage = $derived(Math.round(zoom * 100));

	function handleZoomIn() {
		const newZoom = Math.min(zoom * 1.25, maxZoom);
		onZoomChange?.(newZoom);
	}

	function handleZoomOut() {
		const newZoom = Math.max(zoom / 1.25, minZoom);
		onZoomChange?.(newZoom);
	}

	function handleReset() {
		onResetZoom?.();
	}
</script>

<div
	class={cn(
		'flex items-center gap-1 p-1.5',
		'bg-background/95 backdrop-blur-sm',
		'border border-border rounded-lg shadow-sm',
		className
	)}
>
	<!-- Zoom Out -->
	<Tooltip.Provider>
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button
					variant="ghost"
					size="sm"
					onclick={handleZoomOut}
					disabled={zoom <= minZoom}
					class="h-8 w-8 p-0"
				>
					<ZoomOut class="w-4 h-4" />
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content side="top" class="font-mono text-xs">
				<p>
					Zoom Out
					<kbd class="ml-1 px-1 py-0.5 bg-muted rounded text-[10px]">-</kbd>
				</p>
			</Tooltip.Content>
		</Tooltip.Root>
	</Tooltip.Provider>

	<!-- Zoom Percentage Display -->
	<div
		class="min-w-[4rem] px-2 py-1 text-center text-xs font-mono tabular-nums text-muted-foreground"
	>
		{zoomPercentage}%
	</div>

	<!-- Zoom In -->
	<Tooltip.Provider>
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button
					variant="ghost"
					size="sm"
					onclick={handleZoomIn}
					disabled={zoom >= maxZoom}
					class="h-8 w-8 p-0"
				>
					<ZoomIn class="w-4 h-4" />
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content side="top" class="font-mono text-xs">
				<p>
					Zoom In
					<kbd class="ml-1 px-1 py-0.5 bg-muted rounded text-[10px]">+</kbd>
				</p>
			</Tooltip.Content>
		</Tooltip.Root>
	</Tooltip.Provider>

	<div class="w-px h-6 bg-border mx-1"></div>

	<!-- Fit to View -->
	<Tooltip.Provider>
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button variant="ghost" size="sm" onclick={onFitToView} class="h-8 w-8 p-0">
					<Maximize2 class="w-4 h-4" />
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content side="top" class="font-mono text-xs">
				<p>
					Fit to View
					<kbd class="ml-1 px-1 py-0.5 bg-muted rounded text-[10px]">0</kbd>
				</p>
			</Tooltip.Content>
		</Tooltip.Root>
	</Tooltip.Provider>

	<!-- Reset to 100% -->
	<Tooltip.Provider>
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button variant="ghost" size="sm" onclick={handleReset} class="h-8 w-8 p-0">
					<RotateCcw class="w-4 h-4" />
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content side="top" class="font-mono text-xs">
				<p>
					Reset to 100%
					<kbd class="ml-1 px-1 py-0.5 bg-muted rounded text-[10px]">1</kbd>
				</p>
			</Tooltip.Content>
		</Tooltip.Root>
	</Tooltip.Provider>
</div>
