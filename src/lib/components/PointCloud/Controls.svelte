<script lang="ts" module>
	export interface ControlsProps {
		class?: string;
		collapsed?: boolean;
		onToggle?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils';
	import { Label } from '$lib/components/ui/label';
	import { Switch } from '$lib/components/ui/switch';
	import { Separator } from '$lib/components/ui/separator';
	import { Settings2, ChevronRight, Eye, Palette, Camera } from '@lucide/svelte';
	import { getPointCloudState } from '$lib/stores/pointcloud.svelte';

	let { class: className, collapsed = true, onToggle }: ControlsProps = $props();

	const pcState = getPointCloudState();

	// Point size range
	const minPointSize = 0.5;
	const maxPointSize = 10;

	function handlePointSizeInput(e: Event) {
		const target = e.target as HTMLInputElement;
		const value = parseFloat(target.value);
		if (!isNaN(value)) {
			pcState.setPointSize(value);
		}
	}
</script>

<div
	class={cn(
		'absolute right-0 top-0 h-full z-10 transition-all duration-200',
		collapsed ? 'w-10' : 'w-64',
		className,
	)}
>
	<!-- Collapsed toggle button -->
	<button
		class={cn(
			'absolute top-4 h-10 flex items-center justify-center',
			'bg-card/95 backdrop-blur-sm border border-border rounded-l-md',
			'text-muted-foreground hover:text-foreground transition-colors',
			collapsed ? 'left-0 w-10' : '-left-6 w-6',
		)}
		onclick={onToggle}
	>
		{#if collapsed}
			<Settings2 class="h-4 w-4" />
		{:else}
			<ChevronRight class="h-4 w-4" />
		{/if}
	</button>

	<!-- Panel content -->
	{#if !collapsed}
		<div
			class="h-full bg-card/95 backdrop-blur-sm border-l border-border overflow-y-auto"
		>
			<!-- Header -->
			<div class="flex items-center gap-2 px-4 py-3 border-b border-border">
				<Settings2 class="h-4 w-4 text-muted-foreground" />
				<span class="text-sm font-mono font-medium">Controls</span>
			</div>

			<!-- Point Rendering Section -->
			<div class="p-4 space-y-4">
				<div class="flex items-center gap-2 text-xs font-mono text-muted-foreground">
					<Eye class="h-3.5 w-3.5" />
					<span>Point Rendering</span>
				</div>

				<!-- Point Size Slider -->
				<div class="space-y-2">
					<div class="flex items-center justify-between">
						<Label class="text-xs font-mono">Point Size</Label>
						<span class="text-xs font-mono text-muted-foreground">
							{pcState.pointSize.toFixed(1)}
						</span>
					</div>
					<input
						type="range"
						min={minPointSize}
						max={maxPointSize}
						step="0.5"
						value={pcState.pointSize}
						oninput={handlePointSizeInput}
						class="w-full h-1.5 bg-border rounded-full appearance-none cursor-pointer accent-primary"
					/>
					<div class="flex justify-between text-[10px] font-mono text-muted-foreground">
						<span>{minPointSize}</span>
						<span>{maxPointSize}</span>
					</div>
				</div>
			</div>

			<Separator />

			<!-- View Helpers Section -->
			<div class="p-4 space-y-4">
				<div class="flex items-center gap-2 text-xs font-mono text-muted-foreground">
					<Palette class="h-3.5 w-3.5" />
					<span>View Helpers</span>
				</div>

				<!-- Grid Toggle -->
				<div class="flex items-center justify-between">
					<Label class="text-xs font-mono">Show Grid</Label>
					<Switch
						checked={pcState.showGrid}
						onCheckedChange={(checked) => pcState.setShowGrid(checked)}
					/>
				</div>

				<!-- Axes Toggle -->
				<div class="flex items-center justify-between">
					<Label class="text-xs font-mono">Show Axes</Label>
					<Switch
						checked={pcState.showAxes}
						onCheckedChange={(checked) => pcState.setShowAxes(checked)}
					/>
				</div>

				<!-- Bounding Boxes Toggle -->
				<div class="flex items-center justify-between">
					<Label class="text-xs font-mono">Show Boxes</Label>
					<Switch
						checked={pcState.showBoundingBoxes}
						onCheckedChange={(checked) => pcState.setShowBoundingBoxes(checked)}
					/>
				</div>

				<!-- Label Toggle -->
				<div class="flex items-center justify-between">
					<Label class="text-xs font-mono">Show Labels</Label>
					<Switch
						checked={pcState.showLabels}
						onCheckedChange={(checked) => pcState.setShowLabels(checked)}
					/>
				</div>
			</div>

			<Separator />

			<!-- File Info Section (when file is loaded) -->
			{#if pcState.file}
				<div class="p-4 space-y-3">
					<div class="flex items-center gap-2 text-xs font-mono text-muted-foreground">
						<Camera class="h-3.5 w-3.5" />
						<span>File Info</span>
					</div>

					<div class="space-y-2 text-xs font-mono">
						<div class="flex justify-between">
							<span class="text-muted-foreground">Name</span>
							<span class="text-foreground truncate max-w-[120px]" title={pcState.file.name}>
								{pcState.file.name}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-muted-foreground">Format</span>
							<span class="text-foreground uppercase">{pcState.file.format}</span>
						</div>
						<div class="flex justify-between">
							<span class="text-muted-foreground">Points</span>
							<span class="text-foreground">{pcState.file.pointCount.toLocaleString()}</span>
						</div>
						<div class="flex justify-between">
							<span class="text-muted-foreground">Size</span>
							<span class="text-foreground">
								{(pcState.file.sizeBytes / 1024 / 1024).toFixed(1)} MB
							</span>
						</div>
					</div>
				</div>
			{/if}

			<!-- Keyboard Shortcuts -->
			<div class="p-4 space-y-3 border-t border-border">
				<div class="text-xs font-mono text-muted-foreground">Shortcuts</div>
				<div class="grid grid-cols-2 gap-1 text-[10px] font-mono">
					<span class="text-muted-foreground">1-3</span>
					<span>Navigate</span>
					<span class="text-muted-foreground">G/A/B</span>
					<span>Toggles</span>
					<span class="text-muted-foreground">C</span>
					<span>Color mode</span>
					<span class="text-muted-foreground">+/-</span>
					<span>Point size</span>
					<span class="text-muted-foreground">R</span>
					<span>Reset camera</span>
					<span class="text-muted-foreground">S</span>
					<span>Screenshot</span>
				</div>
			</div>
		</div>
	{/if}
</div>
