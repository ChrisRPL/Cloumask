<script lang="ts" module>
	import type { SceneContext } from '$lib/utils/three';
	import type { Bounds3D } from '$lib/types/pointcloud';

	export interface ThreeCanvasProps {
		class?: string;
		onReady?: (context: SceneContext) => void;
		onFpsUpdate?: (fps: number) => void;
		/** Point positions as flat array [x,y,z,x,y,z,...] */
		positions?: Float32Array | null;
		/** Intensity values (normalized 0-1) */
		intensities?: Float32Array | null;
		/** RGB colors (normalized 0-1) [r,g,b,r,g,b,...] */
		colors?: Float32Array | null;
		/** Classification values (LAS-specific) */
		classifications?: Uint8Array | null;
		/** 3D bounding box */
		bounds?: Bounds3D | null;
	}
</script>

<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import * as THREE from 'three';
	import { cn } from '$lib/utils';
	import {
		createScene,
		createHeightMaterial,
		createIntensityMaterial,
		createRGBMaterial,
		createSimpleMaterial,
		updatePointSize,
		updateHeightRange,
		applyClassificationColors,
		focusOnBounds,
		createBoundingBoxes,
		updateBoxSelection,
		raycastBoundingBox,
		type ColorMode,
		type BoundingBox3D,
	} from '$lib/utils/three';
	import { getPointCloudState } from '$lib/stores/pointcloud.svelte';

	let {
		class: className,
		onReady,
		onFpsUpdate,
		positions = null,
		intensities = null,
		colors = null,
		classifications = null,
		bounds = null,
	}: ThreeCanvasProps = $props();

	// Track if we have real data or should show demo
	const hasData = $derived(positions !== null && positions.length > 0);

	// Get state from context
	const pcState = getPointCloudState();

	// Canvas reference
	let canvasRef: HTMLCanvasElement | undefined = $state();

	// Scene context
	let sceneCtx: SceneContext | null = $state(null);

	// Point cloud mesh
	let pointsMesh: THREE.Points | null = null;

	// Bounding boxes group
	let bboxGroup: THREE.Group | null = null;

	// Raycaster for box selection
	const raycaster = new THREE.Raycaster();
	const mouse = new THREE.Vector2();

	// Create material based on color mode and available data
	function createMaterialForMode(
		mode: ColorMode,
		geometry: THREE.BufferGeometry,
		heightMin: number,
		heightMax: number,
	): THREE.Material {
		const config = { colorMode: mode, pointSize: pcState.pointSize };

		switch (mode) {
			case 'height':
				return createHeightMaterial({
					...config,
					heightMin,
					heightMax,
				});
			case 'intensity':
				if (intensities) {
					// Add intensity attribute
					geometry.setAttribute('intensity', new THREE.BufferAttribute(intensities, 1));
					return createIntensityMaterial(config);
				}
				// Fallback to height if no intensity
				return createHeightMaterial({ ...config, heightMin, heightMax });
			case 'rgb':
				if (colors) {
					// Colors should already be set
					return createRGBMaterial(config);
				}
				// Fallback to height if no colors
				return createHeightMaterial({ ...config, heightMin, heightMax });
			case 'classification':
				if (classifications) {
					applyClassificationColors(geometry, classifications);
					return createSimpleMaterial({ ...config, colorMode: 'rgb' });
				}
				// Fallback to height if no classification
				return createHeightMaterial({ ...config, heightMin, heightMax });
			default:
				return createHeightMaterial({ ...config, heightMin, heightMax });
		}
	}

	// Create point cloud from loaded data
	function createPointCloudFromData(): THREE.Points {
		if (!positions) throw new Error('No positions data');

		const geometry = new THREE.BufferGeometry();

		// Set positions
		geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

		// Set colors if available
		if (colors) {
			geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
		}

		geometry.computeBoundingBox();

		// Calculate height range from bounds or geometry
		let heightMin = 0;
		let heightMax = 100;
		if (bounds) {
			heightMin = bounds.min[2];
			heightMax = bounds.max[2];
		} else if (geometry.boundingBox) {
			heightMin = geometry.boundingBox.min.z;
			heightMax = geometry.boundingBox.max.z;
		}

		// Create material based on current color mode
		const material = createMaterialForMode(pcState.colorMode, geometry, heightMin, heightMax);

		const points = new THREE.Points(geometry, material);
		points.name = 'point-cloud';

		return points;
	}

	// Demo point cloud data (shown when no data is loaded)
	function createDemoPointCloud(): THREE.Points {
		const geometry = new THREE.BufferGeometry();
		const count = 100000;

		const demoPositions = new Float32Array(count * 3);
		const demoColors = new Float32Array(count * 3);

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

			demoPositions[i * 3] = x;
			demoPositions[i * 3 + 1] = y;
			demoPositions[i * 3 + 2] = z;

			// Color based on height (forest green gradient)
			const t = (y + 10) / 25;
			demoColors[i * 3] = 0.086 + t * 0.4; // R
			demoColors[i * 3 + 1] = 0.329 + t * 0.5; // G
			demoColors[i * 3 + 2] = 0.2 + t * 0.3; // B
		}

		geometry.setAttribute('position', new THREE.BufferAttribute(demoPositions, 3));
		geometry.setAttribute('color', new THREE.BufferAttribute(demoColors, 3));
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

	// Remove existing point cloud from scene
	function removePointCloud() {
		if (pointsMesh && sceneCtx) {
			sceneCtx.scene.remove(pointsMesh);
			pointsMesh.geometry.dispose();
			if (Array.isArray(pointsMesh.material)) {
				pointsMesh.material.forEach((m) => m.dispose());
			} else {
				pointsMesh.material.dispose();
			}
			pointsMesh = null;
		}
	}

	// Remove existing bounding boxes from scene
	function removeBoundingBoxes() {
		if (bboxGroup && sceneCtx) {
			sceneCtx.scene.remove(bboxGroup);
			bboxGroup.traverse((obj) => {
				if (obj instanceof THREE.Mesh || obj instanceof THREE.LineSegments) {
					obj.geometry?.dispose();
					if (obj.material) {
						if (Array.isArray(obj.material)) {
							obj.material.forEach((m) => m.dispose());
						} else {
							(obj.material as THREE.Material).dispose();
						}
					}
				}
			});
			bboxGroup = null;
		}
	}

	// Handle click for bounding box selection
	function handleClick(event: MouseEvent) {
		if (!sceneCtx || !bboxGroup || !canvasRef) return;

		// Calculate mouse position in normalized device coordinates
		const rect = canvasRef.getBoundingClientRect();
		mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
		mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

		// Update raycaster
		raycaster.setFromCamera(mouse, sceneCtx.camera);

		// Check for intersection with bounding boxes
		const hitBox = raycastBoundingBox(raycaster, bboxGroup);

		if (hitBox) {
			pcState.setSelectedBoxId(hitBox.id);
		} else {
			pcState.setSelectedBoxId(null);
		}
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

	// React to data changes - create or update point cloud
	$effect(() => {
		if (!sceneCtx) return;

		// Remove existing point cloud
		removePointCloud();

		// Create new point cloud
		if (hasData && positions) {
			pointsMesh = createPointCloudFromData();
		} else {
			// Show demo if no data
			pointsMesh = createDemoPointCloud();
		}

		sceneCtx.scene.add(pointsMesh);

		// Focus camera on new point cloud
		if (bounds && hasData) {
			const box = new THREE.Box3(
				new THREE.Vector3(bounds.min[0], bounds.min[1], bounds.min[2]),
				new THREE.Vector3(bounds.max[0], bounds.max[1], bounds.max[2]),
			);
			focusOnBounds(sceneCtx.camera, sceneCtx.controls, box);
		} else if (pointsMesh.geometry.boundingBox) {
			const center = pointsMesh.geometry.boundingBox.getCenter(new THREE.Vector3());
			sceneCtx.controls.target.copy(center);
			sceneCtx.camera.position.set(center.x + 50, center.y + 30, center.z + 50);
			sceneCtx.controls.update();
		}
	});

	// React to color mode changes - update material
	$effect(() => {
		if (!pointsMesh || !sceneCtx) return;

		const geometry = pointsMesh.geometry;
		let heightMin = 0;
		let heightMax = 100;

		if (bounds) {
			heightMin = bounds.min[2];
			heightMax = bounds.max[2];
		} else if (geometry.boundingBox) {
			heightMin = geometry.boundingBox.min.z;
			heightMax = geometry.boundingBox.max.z;
		}

		// Dispose old material
		if (Array.isArray(pointsMesh.material)) {
			pointsMesh.material.forEach((m) => m.dispose());
		} else {
			pointsMesh.material.dispose();
		}

		// Create new material
		pointsMesh.material = createMaterialForMode(pcState.colorMode, geometry, heightMin, heightMax);
	});

	// React to bounding boxes changes
	$effect(() => {
		if (!sceneCtx) return;

		// Remove existing boxes
		removeBoundingBoxes();

		// Add new boxes if visible and available
		if (pcState.showBoundingBoxes && pcState.boundingBoxes.length > 0) {
			bboxGroup = createBoundingBoxes(
				pcState.boundingBoxes,
				pcState.selectedBoxId ?? undefined,
			);
			sceneCtx.scene.add(bboxGroup);
		}
	});

	// React to selected box changes - update selection styling
	$effect(() => {
		if (!bboxGroup) return;

		// Update selection state on all boxes
		bboxGroup.children.forEach((child) => {
			if (child instanceof THREE.Group && child.userData.boxId) {
				const isSelected = child.userData.boxId === pcState.selectedBoxId;
				updateBoxSelection(child, isSelected);
			}
		});
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

		// Point cloud will be created by the $effect when data changes
		// or with demo data if no data is provided

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
	onclick={handleClick}
></canvas>
