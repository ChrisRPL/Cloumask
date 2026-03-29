<script lang="ts" module>
	import type { BoundingBox3D } from '$lib/utils/three';
	import type { PreviewItem, PointcloudPreviewAnnotation } from '$lib/types/execution';

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
	import * as THREE from 'three';
	import { isTauri } from '$lib/utils/tauri';
	import { Button } from '$lib/components/ui/button';
	import { setPointCloudState } from '$lib/stores/pointcloud.svelte';
	import { getExecutionState } from '$lib/stores/execution.svelte';
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
	const isDesktopTauri = isTauri() || import.meta.env.MODE === 'test';

	// Initialize pointcloud state in context
	const pcState = setPointCloudState();
	const execution = (() => {
		try {
			return getExecutionState();
		} catch {
			return null;
		}
	})();

	// Scene context reference
	let sceneContext: SceneContext | null = $state(null);
	let hydratedPreviewLoadKey = $state<string | null>(null);

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

	function toBoxColor(status: PointcloudPreviewAnnotation['status']): string {
		switch (status) {
			case 'accepted':
				return '#22c55e';
			case 'rejected':
				return '#ef4444';
			case 'edited':
				return '#38bdf8';
			default:
				return '#f59e0b';
		}
	}

	function previewToBoundingBoxes(preview: PreviewItem): BoundingBox3D[] {
		const annotations = preview.pointcloudAnnotations ?? [];
		return annotations.map((annotation) => {
			const [cx, cy, cz] = annotation.center;
			const [sx, sy, sz] = annotation.size;
			return {
				id: annotation.id,
				className: annotation.className,
				confidence: annotation.confidence,
				center: new THREE.Vector3(cx, cy, cz),
				size: new THREE.Vector3(sx, sy, sz),
				rotation: new THREE.Euler(0, 0, annotation.yaw),
				color: toBoxColor(annotation.status),
				visible: annotation.status !== 'rejected',
			};
		});
	}

	function getSelectedPointcloudPreview(): PreviewItem | null {
		if (!execution) return null;
		const preview = execution.selectedPointcloudPreview;
		if (!preview || preview.assetType !== 'pointcloud') return null;
		return preview;
	}

	function annotationStatusClasses(status: PointcloudPreviewAnnotation['status']): string {
		switch (status) {
			case 'accepted':
				return 'text-green-500 bg-green-500/10 border-green-500/40';
			case 'rejected':
				return 'text-red-500 bg-red-500/10 border-red-500/40';
			case 'edited':
				return 'text-sky-500 bg-sky-500/10 border-sky-500/40';
			default:
				return 'text-amber-500 bg-amber-500/10 border-amber-500/40';
		}
	}

	function updateAnnotationStatus(
		annotationId: string,
		status: PointcloudPreviewAnnotation['status']
	): void {
		const preview = getSelectedPointcloudPreview();
		if (!preview) return;
		execution?.setPointcloudAnnotationStatus(preview.id, annotationId, status);
	}

	function editAnnotationClass(annotationId: string, event: Event): void {
		const preview = getSelectedPointcloudPreview();
		if (!preview) return;
		const className = (event.target as HTMLInputElement).value.trim();
		if (!className) return;
		execution?.updatePointcloudAnnotation(preview.id, annotationId, {
			className,
			status: 'edited',
		});
	}

	function editAnnotationConfidence(annotationId: string, event: Event): void {
		const preview = getSelectedPointcloudPreview();
		if (!preview) return;
		const confidence = Number.parseFloat((event.target as HTMLInputElement).value);
		if (Number.isNaN(confidence)) return;
		execution?.updatePointcloudAnnotation(preview.id, annotationId, {
			confidence: Math.max(0, Math.min(1, confidence)),
			status: 'edited',
		});
	}

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

	async function loadPointCloudPath(filePath: string): Promise<void> {
		try {
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

	// Load file action
	async function handleLoad() {
		if (!isDesktopTauri) {
			pcState.setError('Point cloud file loading is available in desktop mode only.');
			return;
		}

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

		await loadPointCloudPath(filePath);
	}

	// Export action
	function handleExport() {
		void (async () => {
			if (!pcState.file) return;
			if (!isDesktopTauri) {
				pcState.setError('Point cloud export is available in desktop mode only.');
				return;
			}

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

	$effect(() => {
		if (!execution) return;
		const preview = execution.selectedPointcloudPreview;
		if (!preview || preview.assetType !== 'pointcloud') return;

		pcState.setBoundingBoxes(previewToBoundingBoxes(preview));

		const loadKey = `${preview.id}:${preview.imagePath}`;
		if (pcState.file?.path === preview.imagePath) {
			hydratedPreviewLoadKey = loadKey;
			return;
		}
		if (hydratedPreviewLoadKey === loadKey) return;
		hydratedPreviewLoadKey = loadKey;

		void loadPointCloudPath(preview.imagePath);
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

		{#if !pcState.file && !pcState.isLoading && !pcState.error}
			<div class="absolute inset-0 flex items-center justify-center p-6">
				<div
					class="w-full max-w-lg rounded-xl border border-border bg-card/92 p-6 text-center shadow-lg backdrop-blur-sm"
				>
					<p class="text-[11px] uppercase tracking-[0.24em] font-mono text-muted-foreground">
						Point Cloud
					</p>
					<h2 class="mt-3 text-lg font-mono text-foreground">
						{isDesktopTauri
							? 'Load a point cloud to start exploring'
							: 'Desktop mode required for file workflows'}
					</h2>
					<p class="mt-3 text-sm leading-6 text-muted-foreground">
						{isDesktopTauri
							? 'Open a local PCD, PLY, LAS, or LAZ file to inspect geometry, switch color modes, and export processed results.'
							: 'This web preview shows the shell only. Load and export stay in the desktop app.'}
					</p>
					<div class="mt-4 flex flex-wrap items-center justify-center gap-2 text-[10px] font-mono text-muted-foreground">
						<span class="rounded-full border border-border px-2.5 py-1">PCD</span>
						<span class="rounded-full border border-border px-2.5 py-1">PLY</span>
						<span class="rounded-full border border-border px-2.5 py-1">LAS</span>
						<span class="rounded-full border border-border px-2.5 py-1">LAZ</span>
					</div>
					{#if isDesktopTauri}
						<div class="mt-5 flex justify-center">
							<Button size="sm" class="font-mono" onclick={handleLoad}>
								Load point cloud
							</Button>
						</div>
					{/if}
				</div>
			</div>
		{/if}

		<!-- Annotation review panel -->
		{#if execution?.selectedPointcloudPreview?.assetType === 'pointcloud' && execution.selectedPointcloudPreview.pointcloudAnnotations && execution.selectedPointcloudPreview.pointcloudAnnotations.length > 0}
			<div
				class="absolute top-3 left-3 w-[26rem] max-h-[58%] overflow-y-auto rounded-md border border-border bg-card/90 backdrop-blur-sm p-3 space-y-2 shadow-lg"
			>
				<div class="flex items-center justify-between">
					<p class="text-xs uppercase tracking-wide font-mono text-muted-foreground">
						Annotations
					</p>
					<span class="text-xs font-mono text-foreground">
						{execution.selectedPointcloudPreview.pointcloudAnnotations.length}
					</span>
				</div>
				{#each execution.selectedPointcloudPreview.pointcloudAnnotations as annotation (annotation.id)}
					<div
						class={cn(
							'rounded-md border border-border/70 bg-background/60 p-2 space-y-2',
							pcState.selectedBoxId === annotation.id && 'border-primary/60'
						)}
					>
						<div class="flex items-center justify-between gap-2">
							<input
								type="text"
								value={annotation.className}
								aria-label={`Class ${annotation.id}`}
								class="h-7 w-36 rounded border border-border bg-background px-2 text-xs font-mono"
								onchange={(event) => editAnnotationClass(annotation.id, event)}
								onfocus={() => pcState.setSelectedBoxId(annotation.id)}
							/>
							<span
								class={cn(
									'inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wide font-mono',
									annotationStatusClasses(annotation.status)
								)}
							>
								{annotation.status}
							</span>
						</div>

						<div class="flex items-center justify-between gap-2">
							<span class="text-[10px] uppercase tracking-wide font-mono text-muted-foreground">
								Confidence
							</span>
							<input
								type="number"
								min="0"
								max="1"
								step="0.01"
								value={annotation.confidence.toFixed(2)}
								aria-label={`Confidence ${annotation.id}`}
								class="h-7 w-20 rounded border border-border bg-background px-2 text-xs font-mono"
								onchange={(event) => editAnnotationConfidence(annotation.id, event)}
								onfocus={() => pcState.setSelectedBoxId(annotation.id)}
							/>
						</div>

						<div class="flex items-center gap-1">
							<button
								type="button"
								aria-label={`Accept ${annotation.id}`}
								class={cn(
									'h-7 rounded border px-2 text-[10px] uppercase tracking-wide font-mono',
									annotation.status === 'accepted'
										? 'border-green-500/70 bg-green-500/15 text-green-500'
										: 'border-border text-muted-foreground hover:text-foreground'
								)}
								onclick={() => updateAnnotationStatus(annotation.id, 'accepted')}
							>
								Accept
							</button>
							<button
								type="button"
								aria-label={`Reject ${annotation.id}`}
								class={cn(
									'h-7 rounded border px-2 text-[10px] uppercase tracking-wide font-mono',
									annotation.status === 'rejected'
										? 'border-red-500/70 bg-red-500/15 text-red-500'
										: 'border-border text-muted-foreground hover:text-foreground'
								)}
								onclick={() => updateAnnotationStatus(annotation.id, 'rejected')}
							>
								Reject
							</button>
							<button
								type="button"
								aria-label={`Edit ${annotation.id}`}
								class={cn(
									'h-7 rounded border px-2 text-[10px] uppercase tracking-wide font-mono',
									annotation.status === 'edited'
										? 'border-sky-500/70 bg-sky-500/15 text-sky-500'
										: 'border-border text-muted-foreground hover:text-foreground'
								)}
								onclick={() => updateAnnotationStatus(annotation.id, 'edited')}
							>
								Mark edited
							</button>
						</div>
					</div>
				{/each}
			</div>
		{/if}

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
