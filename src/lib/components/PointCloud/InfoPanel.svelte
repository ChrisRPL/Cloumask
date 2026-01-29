<script lang="ts" module>
	export interface InfoPanelProps {
		class?: string;
		collapsed?: boolean;
		onToggle?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils';
	import { Button } from '$lib/components/ui/button';
	import { ChevronDown, ChevronUp, Target, MousePointer, Activity } from '@lucide/svelte';
	import { getPointCloudState } from '$lib/stores/pointcloud.svelte';

	let { class: className, collapsed = false, onToggle }: InfoPanelProps = $props();

	const pcState = getPointCloudState();

	// Format coordinate
	function formatCoord(val: number): string {
		return val.toFixed(1);
	}
</script>

<div
	class={cn(
		'bg-card/80 backdrop-blur-sm border-t border-border transition-all duration-200',
		collapsed ? 'h-8' : 'h-auto',
		className
	)}
>
	<!-- Toggle Header -->
	<button
		class="w-full h-8 px-3 flex items-center justify-between text-xs font-mono text-muted-foreground hover:text-foreground transition-colors"
		onclick={onToggle}
	>
		<span class="flex items-center gap-2">
			<Activity class="h-3.5 w-3.5" />
			Info Panel
		</span>
		{#if collapsed}
			<ChevronUp class="h-3.5 w-3.5" />
		{:else}
			<ChevronDown class="h-3.5 w-3.5" />
		{/if}
	</button>

	<!-- Content -->
	{#if !collapsed}
		<div class="px-3 pb-2 flex items-center gap-6 text-xs font-mono">
			<!-- Camera Position -->
			<div class="flex items-center gap-2">
				<Target class="h-3.5 w-3.5 text-muted-foreground" />
				<span class="text-muted-foreground">Camera:</span>
				<span class="text-foreground">
					({formatCoord(pcState.camera.position.x)}, {formatCoord(pcState.camera.position.y)}, {formatCoord(
						pcState.camera.position.z
					)})
				</span>
			</div>

			<!-- Target -->
			<div class="flex items-center gap-2">
				<span class="text-muted-foreground">Target:</span>
				<span class="text-foreground">
					({formatCoord(pcState.camera.target.x)}, {formatCoord(pcState.camera.target.y)}, {formatCoord(
						pcState.camera.target.z
					)})
				</span>
			</div>

			<!-- FPS -->
			<div class="flex items-center gap-2">
				<span class="text-muted-foreground">FPS:</span>
				<span class={cn('font-semibold', pcState.fps >= 30 ? 'text-primary' : 'text-destructive')}>
					{pcState.fps}
				</span>
			</div>

			<!-- Selection Info -->
			{#if pcState.selection}
				<div class="flex items-center gap-2 ml-auto border-l border-border pl-4">
					<MousePointer class="h-3.5 w-3.5 text-primary" />
					<span class="text-muted-foreground">Selection:</span>
					<span class="text-foreground">
						{pcState.selection.pointCount.toLocaleString()} points
					</span>
					{#if pcState.selection.className}
						<span class="text-muted-foreground">•</span>
						<span class="text-foreground">{pcState.selection.className}</span>
					{/if}
					{#if pcState.selection.confidence !== undefined}
						<span class="text-muted-foreground">•</span>
						<span class="text-foreground">{(pcState.selection.confidence * 100).toFixed(0)}%</span>
					{/if}
				</div>
			{/if}
		</div>
	{/if}
</div>
