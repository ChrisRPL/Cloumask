<script lang="ts" module>
	export interface PointCloudViewerProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils';
	import { onMount } from 'svelte';
	import { setPointCloudState, getPointCloudState } from '$lib/stores/pointcloud.svelte';
	import { resetCamera, type SceneContext } from '$lib/utils/three';
	import ViewerHeader from './ViewerHeader.svelte';
	import ViewerToolbar from './ViewerToolbar.svelte';
	import ThreeCanvas from './ThreeCanvas.svelte';
	import InfoPanel from './InfoPanel.svelte';

	let { class: className }: PointCloudViewerProps = $props();

	// Initialize pointcloud state in context
	const pcState = setPointCloudState();

	// Scene context reference
	let sceneContext: SceneContext | null = $state(null);

	// Info panel collapsed state
	let infoPanelCollapsed = $state(false);

	// Handle scene ready
	function handleSceneReady(ctx: SceneContext) {
		sceneContext = ctx;
	}

	// Handle FPS update
	function handleFpsUpdate(fps: number) {
		pcState.setFps(fps);
	}

	// Reset camera action
	function handleResetCamera() {
		if (sceneContext) {
			resetCamera(sceneContext.camera, sceneContext.controls);
		}
	}

	// Screenshot action
	function handleScreenshot() {
		if (sceneContext) {
			// Force render
			sceneContext.renderer.render(sceneContext.scene, sceneContext.camera);

			// Get data URL
			const dataUrl = sceneContext.renderer.domElement.toDataURL('image/png');

			// Create download link
			const link = document.createElement('a');
			link.download = `pointcloud-screenshot-${Date.now()}.png`;
			link.href = dataUrl;
			link.click();
		}
	}

	// Load file action (placeholder)
	function handleLoad() {
		// TODO: Integrate with Tauri file picker
		console.log('Load file - to be implemented');
	}

	// Export action (placeholder)
	function handleExport() {
		// TODO: Implement export
		console.log('Export - to be implemented');
	}

	// Settings action (placeholder)
	function handleSettings() {
		// TODO: Open settings modal
		console.log('Settings - to be implemented');
	}

	// Keyboard shortcuts
	function handleKeydown(event: KeyboardEvent) {
		// Ignore if typing
		if (
			event.target instanceof HTMLInputElement ||
			event.target instanceof HTMLTextAreaElement
		) {
			return;
		}

		switch (event.key.toLowerCase()) {
			case '1':
				pcState.setNavigationMode('orbit');
				break;
			case '2':
				pcState.setNavigationMode('pan');
				break;
			case '3':
				pcState.setNavigationMode('zoom');
				break;
			case 'm':
				pcState.setNavigationMode('measure');
				break;
			case 'g':
				pcState.setShowGrid(!pcState.showGrid);
				break;
			case 'a':
				pcState.setShowAxes(!pcState.showAxes);
				break;
			case 'b':
				pcState.setShowBoundingBoxes(!pcState.showBoundingBoxes);
				break;
			case 'r':
				handleResetCamera();
				break;
			case 's':
				if (!event.ctrlKey && !event.metaKey) {
					handleScreenshot();
				}
				break;
			case '+':
			case '=':
				pcState.setPointSize(pcState.pointSize + 0.5);
				break;
			case '-':
				pcState.setPointSize(pcState.pointSize - 0.5);
				break;
			case 'c':
				// Cycle color modes
				const modes: ('height' | 'intensity' | 'rgb' | 'classification')[] = [
					'height',
					'intensity',
					'rgb',
					'classification',
				];
				const currentIndex = modes.indexOf(pcState.colorMode as typeof modes[number]);
				const nextIndex = (currentIndex + 1) % modes.length;
				pcState.setColorMode(modes[nextIndex]);
				break;
		}
	}

	// Register keyboard listener
	onMount(() => {
		window.addEventListener('keydown', handleKeydown);
		return () => {
			window.removeEventListener('keydown', handleKeydown);
		};
	});
</script>

<div class={cn('flex flex-col h-full bg-background', className)}>
	<!-- Header -->
	<ViewerHeader onLoad={handleLoad} onExport={handleExport} onSettings={handleSettings} />

	<!-- Toolbar -->
	<ViewerToolbar onScreenshot={handleScreenshot} onResetCamera={handleResetCamera} />

	<!-- 3D Viewport -->
	<div class="flex-1 relative overflow-hidden">
		<ThreeCanvas onReady={handleSceneReady} onFpsUpdate={handleFpsUpdate} class="absolute inset-0" />

		<!-- Loading overlay -->
		{#if pcState.isLoading}
			<div class="absolute inset-0 bg-background/80 flex items-center justify-center">
				<div class="text-center">
					<div class="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></div>
					<p class="text-sm font-mono text-muted-foreground">
						Loading... {pcState.loadProgress}%
					</p>
				</div>
			</div>
		{/if}

		<!-- Error overlay -->
		{#if pcState.error}
			<div class="absolute inset-0 bg-background/80 flex items-center justify-center">
				<div class="text-center p-6 bg-card border border-destructive/20 rounded-lg max-w-md">
					<p class="text-sm font-mono text-destructive mb-2">Error loading point cloud</p>
					<p class="text-xs text-muted-foreground">{pcState.error}</p>
				</div>
			</div>
		{/if}
	</div>

	<!-- Info Panel -->
	<InfoPanel collapsed={infoPanelCollapsed} onToggle={() => (infoPanelCollapsed = !infoPanelCollapsed)} />
</div>
