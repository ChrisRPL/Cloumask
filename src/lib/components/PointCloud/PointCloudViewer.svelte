<script lang="ts" module>
	import type { BoundingBox3D } from '$lib/utils/three';

	export interface PointCloudViewerProps {
		class?: string;
		onPointClick?: (payload: { index: number; position: [number, number, number] }) => void;
		onBoxClick?: (payload: { box: BoundingBox3D }) => void;
		onCameraChange?: (payload: { position: [number, number, number]; target: [number, number, number] }) => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils';
	import { onMount } from 'svelte';
	import { setPointCloudState, getPointCloudState } from '$lib/stores/pointcloud.svelte';
	import { resetCamera, type SceneContext } from '$lib/utils/three';
	import { invoke } from '@tauri-apps/api/core';
	import { listen, type UnlistenFn } from '@tauri-apps/api/event';
	import { open, save } from '@tauri-apps/plugin-dialog';
	import type {
		PointCloudData,
		PointCloudMetadata,
		PointCloudChunk,
		Bounds3D,
		PointCloudFormat,
		ConversionOptions,
	} from '$lib/types/pointcloud';
	import { toFloat32Array, unpackColorsNormalized } from '$lib/types/pointcloud';
	import ViewerHeader from './ViewerHeader.svelte';
	import ViewerToolbar from './ViewerToolbar.svelte';
	import ThreeCanvas from './ThreeCanvas.svelte';
	import InfoPanel from './InfoPanel.svelte';
	import Controls from './Controls.svelte';
	import SettingsModal from './SettingsModal.svelte';

	let { class: className, onPointClick, onBoxClick, onCameraChange }: PointCloudViewerProps = $props();

	// Initialize pointcloud state in context
	const pcState = setPointCloudState();

	// Scene context reference
	let sceneContext: SceneContext | null = $state(null);

	// Panel collapsed states
	let infoPanelCollapsed = $state(false);
	let controlsCollapsed = $state(true);
	let settingsOpen = $state(false);

	// Point cloud data for ThreeCanvas
	let positions = $state<Float32Array | null>(null);
	let intensities = $state<Float32Array | null>(null);
	let colors = $state<Float32Array | null>(null);
	let classifications = $state<Uint8Array | null>(null);
	let bounds = $state<Bounds3D | null>(null);

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

	// Process loaded point cloud data
	function processPointCloudData(data: PointCloudData) {
		// Convert arrays to typed arrays for Three.js
		positions = toFloat32Array(data.positions);
		intensities = data.intensities ? toFloat32Array(data.intensities) : null;
		colors = data.colors ? unpackColorsNormalized(data.colors) : null;
		classifications = data.classifications ? new Uint8Array(data.classifications) : null;
		bounds = data.metadata.bounds;

		// Update store with file info
		pcState.setFile({
			name: data.metadata.path.split('/').pop() || 'unknown',
			path: data.metadata.path,
			format: data.metadata.format,
			pointCount: data.metadata.point_count,
			sizeBytes: data.metadata.file_size_bytes,
			bounds: bounds
				? {
						min: { x: bounds.min[0], y: bounds.min[1], z: bounds.min[2] },
						max: { x: bounds.max[0], y: bounds.max[1], z: bounds.max[2] },
					}
				: { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } },
		});
	}

	// Streaming threshold (points)
	const STREAMING_THRESHOLD = 1_000_000;

	// Streaming timeout (60 seconds)
	const STREAMING_TIMEOUT_MS = 60_000;

	// Load file via streaming for large files
	async function loadStreamedPointCloud(path: string, metadata: PointCloudMetadata): Promise<void> {
		const appendChunk = (target: number[], source: number[]) => {
			for (let i = 0; i < source.length; i++) {
				target.push(source[i]);
			}
		};

		return new Promise<void>((resolve, reject) => {
			// Accumulators for chunked data
			const allPositions: number[] = [];
			const allIntensities: number[] = [];
			const allColors: number[] = [];
			const allClassifications: number[] = [];
			let receivedChunks = 0;
			let totalChunks = 1;
			let isComplete = false;

			// Set up event listeners
			const unlisteners: UnlistenFn[] = [];

			// Cleanup helper
			const cleanup = () => {
				unlisteners.forEach((fn) => fn());
			};

			const finish = (error?: string) => {
				if (isComplete) return;
				isComplete = true;
				clearTimeout(timeoutId);
				cleanup();

				if (error) {
					pcState.setError(error);
					reject(new Error(error));
					return;
				}

				// All chunks received - process the complete data
				pcState.setLoadProgress(90);

				const data: PointCloudData = {
					metadata,
					positions: allPositions,
					intensities: allIntensities.length > 0 ? allIntensities : null,
					colors: allColors.length > 0 ? allColors : null,
					classifications: allClassifications.length > 0 ? allClassifications : null,
				};

				processPointCloudData(data);
				pcState.setLoadProgress(100);
				resolve();
			};

			// Timeout for streaming
			const timeoutId = setTimeout(() => {
				finish('Point cloud streaming timed out');
			}, STREAMING_TIMEOUT_MS);

			(async () => {
				try {
					const chunkListener = await listen<PointCloudChunk>('pointcloud:chunk', (event) => {
						const chunk = event.payload;
						totalChunks = chunk.total_chunks;
						receivedChunks++;

						// Accumulate data safely
						appendChunk(allPositions, chunk.positions);
						if (chunk.intensities) appendChunk(allIntensities, chunk.intensities);
						if (chunk.colors) appendChunk(allColors, chunk.colors);
						if (chunk.classifications) appendChunk(allClassifications, chunk.classifications);

						// Update progress
						const progress = 20 + Math.round((receivedChunks / totalChunks) * 60);
						pcState.setLoadProgress(progress);
					});
					unlisteners.push(chunkListener);

					const completeListener = await listen<PointCloudMetadata>('pointcloud:complete', () => {
						finish();
					});
					unlisteners.push(completeListener);

					const errorListener = await listen<string>('pointcloud:error', (event) => {
						finish(event.payload);
					});
					unlisteners.push(errorListener);

					// Start streaming - this returns immediately with metadata
					await invoke<PointCloudMetadata>('stream_pointcloud', {
						path,
						config: { chunk_size: 100000 },
					});
				} catch (e) {
					finish(e instanceof Error ? e.message : String(e));
				}
			})();
		});
	}

	// Load file action
	async function handleLoad() {
		try {
			// Open file picker
			const filePath = await open({
				multiple: false,
				filters: [
					{
						name: 'Point Cloud',
						extensions: ['las', 'laz', 'ply', 'pcd'],
					},
				],
			});

			// User cancelled
			if (!filePath || Array.isArray(filePath)) return;

			pcState.setLoading(true);
			pcState.setError(null);
			pcState.setLoadProgress(0);

			// Get metadata first to check file size
			const metadata = await invoke<PointCloudMetadata>('read_pointcloud_metadata', {
				path: filePath,
			});
			pcState.setLoadProgress(20);

			// Use streaming for large files
			if (metadata.point_count > STREAMING_THRESHOLD) {
				await loadStreamedPointCloud(filePath, metadata);
			} else {
				// Load directly for smaller files
				const data = await invoke<PointCloudData>('read_pointcloud', {
					path: filePath,
				});
				pcState.setLoadProgress(80);
				processPointCloudData(data);
				pcState.setLoadProgress(100);
			}
		} catch (e) {
			pcState.setError(e instanceof Error ? e.message : String(e));
		} finally {
			pcState.setLoading(false);
		}
	}

	// Export action
	function handleExport() {
		void (async () => {
			if (!pcState.file) return;

			try {
				const defaultName = pcState.file.name.replace(/\.[^/.]+$/, '');
				const outputPath = await save({
					defaultPath: `${defaultName}.ply`,
					filters: [
						{
							name: 'Point Cloud',
							extensions: ['ply', 'pcd', 'las', 'laz'],
						},
					],
				});

				if (!outputPath) return;

				const extension = outputPath.split('.').pop()?.toLowerCase();
				if (!extension || !['ply', 'pcd', 'las', 'laz'].includes(extension)) {
					throw new Error('Unsupported export format');
				}

				pcState.setLoading(true);
				pcState.setLoadProgress(0);

				const options: ConversionOptions = {
					target_format: extension as PointCloudFormat,
					preserve_intensity: true,
					preserve_rgb: true,
					preserve_classification: true,
					decimation: null,
				};

				await invoke<PointCloudMetadata>('convert_pointcloud', {
					input_path: pcState.file.path,
					output_path: outputPath,
					options,
				});

				pcState.setLoadProgress(100);
			} catch (e) {
				pcState.setError(e instanceof Error ? e.message : String(e));
			} finally {
				pcState.setLoading(false);
			}
		})();
	}

	// Settings action
	function handleSettings() {
		settingsOpen = true;
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
		<ThreeCanvas
			onReady={handleSceneReady}
			onFpsUpdate={handleFpsUpdate}
			onPointClick={onPointClick}
			onBoxClick={onBoxClick}
			onCameraChange={onCameraChange}
			{positions}
			{intensities}
			{colors}
			{classifications}
			{bounds}
			class="absolute inset-0"
		/>

		<!-- Controls Panel -->
		<Controls collapsed={controlsCollapsed} onToggle={() => (controlsCollapsed = !controlsCollapsed)} />

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

	<SettingsModal bind:open={settingsOpen} />
</div>
