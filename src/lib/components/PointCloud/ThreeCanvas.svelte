<script lang="ts" module>
	export interface ThreeCanvasProps {
		class?: string;
		onReady?: (context: SceneContext) => void;
		onFpsUpdate?: (fps: number) => void;
	}

	import type { SceneContext } from '$lib/utils/three';
</script>

<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import * as THREE from 'three';
	import { cn } from '$lib/utils';
	import {
		createScene,
		createHeightMaterial,
		createSimpleMaterial,
		updatePointSize,
		updateHeightRange,
		type ColorMode,
	} from '$lib/utils/three';
	import { getPointCloudState } from '$lib/stores/pointcloud.svelte';

	let { class: className, onReady, onFpsUpdate }: ThreeCanvasProps = $props();

	// Get state from context
	const pcState = getPointCloudState();

	// Canvas reference
	let canvasRef: HTMLCanvasElement | undefined = $state();

	// Scene context
	let sceneCtx: SceneContext | null = $state(null);

	// Point cloud mesh
	let pointsMesh: THREE.Points | null = null;

	// Demo point cloud data
	function createDemoPointCloud(): THREE.Points {
		const geometry = new THREE.BufferGeometry();
		const count = 100000;

		const positions = new Float32Array(count * 3);
		const colors = new Float32Array(count * 3);

		// Generate terrain-like point cloud
		for (let i = 0; i < count; i++) {
			const x = (Math.random() - 0.5) * 100;
			const z = (Math.random() - 0.5) * 100;

			// Create terrain using simplex-like noise
			const y =
				Math.sin(x * 0.05) * 5 +
				Math.cos(z * 0.05) * 5 +
				Math.sin(x * 0.1 + z * 0.1) * 2 +
				Math.random() * 2;

			positions[i * 3] = x;
			positions[i * 3 + 1] = y;
			positions[i * 3 + 2] = z;

			// Color based on height (forest green gradient)
			const t = (y + 10) / 25;
			colors[i * 3] = 0.086 + t * 0.4; // R
			colors[i * 3 + 1] = 0.329 + t * 0.5; // G
			colors[i * 3 + 2] = 0.2 + t * 0.3; // B
		}

		geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
		geometry.computeBoundingBox();

		// Create material based on current color mode
		const material = createHeightMaterial({
			colorMode: 'height',
			pointSize: pcState.pointSize,
			heightMin: -10,
			heightMax: 15,
		});

		const points = new THREE.Points(geometry, material);
		points.name = 'point-cloud';

		return points;
	}

	// Update helpers visibility
	$effect(() => {
		if (sceneCtx) {
			sceneCtx.helpers.grid.visible = pcState.showGrid;
			sceneCtx.helpers.axes.visible = pcState.showAxes;
		}
	});

	// Update point size
	$effect(() => {
		if (pointsMesh && pointsMesh.material) {
			const material = Array.isArray(pointsMesh.material) ? pointsMesh.material[0] : pointsMesh.material;
			if (material) {
				updatePointSize(material, pcState.pointSize);
			}
		}
	});

	// FPS tracking
	let frameCount = 0;
	let lastTime = performance.now();

	function trackFps() {
		frameCount++;
		const now = performance.now();

		if (now - lastTime >= 1000) {
			onFpsUpdate?.(frameCount);
			frameCount = 0;
			lastTime = now;
		}
	}

	// Initialize scene
	onMount(() => {
		if (!canvasRef) return;

		// Create scene - capture in local const for closure safety
		const ctx = createScene(canvasRef, {
			showGrid: pcState.showGrid,
			showAxes: pcState.showAxes,
		});
		sceneCtx = ctx;

		// Add demo point cloud
		pointsMesh = createDemoPointCloud();
		ctx.scene.add(pointsMesh);

		// Focus camera on point cloud
		if (pointsMesh.geometry.boundingBox) {
			const center = pointsMesh.geometry.boundingBox.getCenter(new THREE.Vector3());
			ctx.controls.target.copy(center);
			ctx.camera.position.set(center.x + 50, center.y + 30, center.z + 50);
		}

		// Set up render loop with FPS tracking
		// Use captured ctx to avoid stale references in animation callback
		ctx.renderer.setAnimationLoop(() => {
			trackFps();
			ctx.controls.update();
			ctx.renderer.render(ctx.scene, ctx.camera);
		});

		// Notify parent
		onReady?.(ctx);
	});

	// Cleanup - properly stop animation loop before disposal
	onDestroy(() => {
		if (sceneCtx) {
			sceneCtx.renderer.setAnimationLoop(null);
			sceneCtx.dispose();
		}
	});

	// Prevent context menu on right-click (for camera controls)
	function handleContextMenu(e: MouseEvent) {
		e.preventDefault();
	}
</script>

<canvas
	bind:this={canvasRef}
	class={cn('w-full h-full block', className)}
	oncontextmenu={handleContextMenu}
></canvas>
