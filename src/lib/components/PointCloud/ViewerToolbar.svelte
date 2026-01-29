<script lang="ts" module>
	export interface ViewerToolbarProps {
		class?: string;
		onScreenshot?: () => void;
		onResetCamera?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils';
	import { Button } from '$lib/components/ui/button';
	import * as Select from '$lib/components/ui/select';
	import { Separator } from '$lib/components/ui/separator';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import {
		Orbit,
		Move,
		ZoomIn,
		Ruler,
		Camera,
		RotateCcw,
		Grid3X3,
		Axis3D,
		Box,
		Minus,
		Plus,
	} from '@lucide/svelte';
	import { getPointCloudState, type NavigationMode } from '$lib/stores/pointcloud.svelte';
	import type { ColorMode } from '$lib/utils/three';

	let { class: className, onScreenshot, onResetCamera }: ViewerToolbarProps = $props();

	const pcState = getPointCloudState();

	// Color mode options
	const colorModes: { value: ColorMode; label: string }[] = [
		{ value: 'height', label: 'Height' },
		{ value: 'intensity', label: 'Intensity' },
		{ value: 'rgb', label: 'RGB' },
		{ value: 'classification', label: 'Classification' },
		{ value: 'single', label: 'Single Color' },
	];

	// Navigation mode buttons
	const navModes: { mode: NavigationMode; icon: typeof Orbit; label: string; shortcut: string }[] = [
		{ mode: 'orbit', icon: Orbit, label: 'Orbit', shortcut: '1' },
		{ mode: 'pan', icon: Move, label: 'Pan', shortcut: '2' },
		{ mode: 'zoom', icon: ZoomIn, label: 'Zoom', shortcut: '3' },
		{ mode: 'measure', icon: Ruler, label: 'Measure', shortcut: 'M' },
	];

	function handleColorModeChange(value: string | undefined) {
		if (value) {
			pcState.setColorMode(value as ColorMode);
		}
	}

	function decreasePointSize() {
		pcState.setPointSize(pcState.pointSize - 0.5);
	}

	function increasePointSize() {
		pcState.setPointSize(pcState.pointSize + 0.5);
	}
</script>

<div
	class={cn(
		'flex items-center gap-2 p-2 bg-card/80 backdrop-blur-sm border-b border-border',
		className
	)}
>
	<!-- Navigation Mode Buttons -->
	<div class="flex items-center gap-1">
		{#each navModes as { mode, icon: Icon, label, shortcut }}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<Button
						variant={pcState.navigationMode === mode ? 'default' : 'ghost'}
						size="sm"
						onclick={() => pcState.setNavigationMode(mode)}
						class="h-8 w-8 p-0"
					>
						<Icon class="h-4 w-4" />
					</Button>
				</Tooltip.Trigger>
				<Tooltip.Content>
					<p class="font-mono text-xs">{label} [{shortcut}]</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/each}
	</div>

	<Separator orientation="vertical" class="h-6" />

	<!-- Color Mode Selector -->
	<div class="flex items-center gap-2">
		<span class="text-xs text-muted-foreground font-mono">Color:</span>
		<Select.Root type="single" value={pcState.colorMode} onValueChange={handleColorModeChange}>
			<Select.Trigger class="h-8 w-28 text-xs font-mono">
				{colorModes.find((c) => c.value === pcState.colorMode)?.label ?? 'Height'}
			</Select.Trigger>
			<Select.Content>
				{#each colorModes as { value, label }}
					<Select.Item {value} class="text-xs font-mono">{label}</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</div>

	<Separator orientation="vertical" class="h-6" />

	<!-- Point Size -->
	<div class="flex items-center gap-1">
		<span class="text-xs text-muted-foreground font-mono">Size:</span>
		<Button variant="ghost" size="sm" class="h-8 w-8 p-0" onclick={decreasePointSize}>
			<Minus class="h-3 w-3" />
		</Button>
		<span class="text-xs font-mono w-6 text-center">{pcState.pointSize.toFixed(1)}</span>
		<Button variant="ghost" size="sm" class="h-8 w-8 p-0" onclick={increasePointSize}>
			<Plus class="h-3 w-3" />
		</Button>
	</div>

	<Separator orientation="vertical" class="h-6" />

	<!-- Toggle Helpers -->
	<div class="flex items-center gap-1">
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button
					variant={pcState.showGrid ? 'default' : 'ghost'}
					size="sm"
					onclick={() => pcState.setShowGrid(!pcState.showGrid)}
					class="h-8 w-8 p-0"
				>
					<Grid3X3 class="h-4 w-4" />
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content>
				<p class="font-mono text-xs">Toggle Grid [G]</p>
			</Tooltip.Content>
		</Tooltip.Root>

		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button
					variant={pcState.showAxes ? 'default' : 'ghost'}
					size="sm"
					onclick={() => pcState.setShowAxes(!pcState.showAxes)}
					class="h-8 w-8 p-0"
				>
					<Axis3D class="h-4 w-4" />
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content>
				<p class="font-mono text-xs">Toggle Axes [A]</p>
			</Tooltip.Content>
		</Tooltip.Root>

		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button
					variant={pcState.showBoundingBoxes ? 'default' : 'ghost'}
					size="sm"
					onclick={() => pcState.setShowBoundingBoxes(!pcState.showBoundingBoxes)}
					class="h-8 w-8 p-0"
				>
					<Box class="h-4 w-4" />
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content>
				<p class="font-mono text-xs">Toggle Bounding Boxes [B]</p>
			</Tooltip.Content>
		</Tooltip.Root>
	</div>

	<!-- Spacer -->
	<div class="flex-1"></div>

	<!-- Actions -->
	<div class="flex items-center gap-1">
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button variant="ghost" size="sm" onclick={onResetCamera} class="h-8 w-8 p-0">
					<RotateCcw class="h-4 w-4" />
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content>
				<p class="font-mono text-xs">Reset Camera [R]</p>
			</Tooltip.Content>
		</Tooltip.Root>

		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button variant="ghost" size="sm" onclick={onScreenshot} class="h-8 w-8 p-0">
					<Camera class="h-4 w-4" />
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content>
				<p class="font-mono text-xs">Screenshot [S]</p>
			</Tooltip.Content>
		</Tooltip.Root>
	</div>
</div>
