<script lang="ts" module>
	export interface ViewerHeaderProps {
		class?: string;
		onLoad?: () => void;
		onExport?: () => void;
		onSettings?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils';
	import { isTauri } from '$lib/utils/tauri';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { CloudCog, Upload, Download, Settings } from '@lucide/svelte';
	import { getPointCloudState } from '$lib/stores/pointcloud.svelte';

	let { class: className, onLoad, onExport, onSettings }: ViewerHeaderProps = $props();

	const pcState = getPointCloudState();
	const isDesktopMode = isTauri() || import.meta.env.MODE === 'test';
	const showFileActions = $derived(isDesktopMode);

	// Format bytes to human readable
	function formatBytes(bytes: number): string {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
	}

	// Format point count
	function formatPoints(count: number): string {
		if (count >= 1_000_000) {
			return `${(count / 1_000_000).toFixed(1)}M`;
		}
		if (count >= 1_000) {
			return `${(count / 1_000).toFixed(1)}K`;
		}
		return count.toString();
	}
</script>

<div
	class={cn(
		'flex items-center justify-between gap-4 p-3 bg-card/80 backdrop-blur-sm border-b border-border',
		className
	)}
>
	<!-- Left: Title and File Info -->
	<div class="flex items-center gap-3">
		<div class="flex items-center gap-2">
			<CloudCog class="h-5 w-5 text-primary" />
			<h1 class="text-sm font-semibold font-mono">Point Cloud Viewer</h1>
		</div>

		{#if pcState.file}
			<div class="flex items-center gap-2 text-xs text-muted-foreground font-mono">
				<span class="text-foreground">{pcState.file.name}</span>
				<span>•</span>
				<span>{formatPoints(pcState.file.pointCount)} points</span>
				<span>•</span>
				<span>{formatBytes(pcState.file.sizeBytes)}</span>
			</div>
		{:else if pcState.isLoading}
			<div class="flex items-center gap-2">
				<Badge variant="secondary" class="text-xs font-mono">
					Loading... {pcState.loadProgress}%
				</Badge>
			</div>
		{:else}
			<div class="flex items-center gap-2 text-xs font-mono">
				<span class="text-muted-foreground">{isDesktopMode ? 'No file loaded' : 'Browser preview only'}</span>
			</div>
		{/if}
	</div>

	<!-- Right: Actions -->
	<div class="flex items-center gap-2">
		{#if showFileActions}
			<Button
				variant="outline"
				size="sm"
				onclick={onLoad}
				class="h-8 text-xs font-mono gap-1.5"
			>
				<Upload class="h-3.5 w-3.5" />
				Load
			</Button>
			<Button
				variant="outline"
				size="sm"
				onclick={onExport}
				disabled={!pcState.file && !pcState.isLoading}
				class="h-8 text-xs font-mono gap-1.5"
			>
				<Download class="h-3.5 w-3.5" />
				Export
			</Button>
		{/if}
		<Button
			variant="ghost"
			size="sm"
			onclick={onSettings}
			class="h-8 w-8 p-0"
			aria-label="Open settings"
		>
			<Settings class="h-4 w-4" />
		</Button>
	</div>
</div>
