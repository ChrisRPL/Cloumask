/**
 * Three.js Scene Setup Utilities
 *
 * Provides initialization and lifecycle management for Three.js scenes
 * optimized for point cloud visualization with Cloumask's design system.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/** Scene context returned from createScene */
export interface SceneContext {
	scene: THREE.Scene;
	camera: THREE.PerspectiveCamera;
	renderer: THREE.WebGLRenderer;
	controls: OrbitControls;
	helpers: {
		grid: THREE.GridHelper;
		axes: THREE.AxesHelper;
	};
	dispose: () => void;
}

/** Configuration options for scene creation */
export interface SceneConfig {
	/** Background color (default: dark forest #0c3b1f) */
	backgroundColor?: number;
	/** Grid size (default: 100) */
	gridSize?: number;
	/** Grid divisions (default: 100) */
	gridDivisions?: number;
	/** Show grid helper (default: true) */
	showGrid?: boolean;
	/** Show axes helper (default: true) */
	showAxes?: boolean;
	/** Initial camera position */
	cameraPosition?: THREE.Vector3;
	/** Field of view (default: 60) */
	fov?: number;
	/** Near clipping plane (default: 0.1) */
	near?: number;
	/** Far clipping plane (default: 10000) */
	far?: number;
}

// Cloumask color palette for Three.js
export const CLOUMASK_COLORS = {
	// Dark mode (default for 3D viewer)
	background: 0x0c3b1f, // Dark forest
	gridMain: 0x166534, // Forest green
	gridSub: 0x14532d, // Darker green
	axes: 0x22c55e, // Light green
	accent: 0x22c55e, // Primary accent

	// Point cloud default colors
	pointDefault: 0x86efac, // Light mint
	pointHighlight: 0xfaf7f0, // Cream (selected)
	boundingBox: 0x22c55e, // Light green
	boundingBoxSelected: 0xfaf7f0, // Cream

	// Classification colors
	classification: {
		ground: 0x8b7355, // Brown
		building: 0x4a90a4, // Steel blue
		vegetation: 0x228b22, // Forest green
		vehicle: 0xff6b35, // Orange
		pedestrian: 0xffd700, // Gold
		unknown: 0x808080, // Gray
	},
} as const;

/**
 * Creates and initializes a Three.js scene for point cloud visualization
 *
 * @param canvas - HTML canvas element to render to
 * @param config - Scene configuration options
 * @returns SceneContext with scene, camera, renderer, controls, and dispose function
 */
export function createScene(canvas: HTMLCanvasElement, config: SceneConfig = {}): SceneContext {
	const {
		backgroundColor = CLOUMASK_COLORS.background,
		gridSize = 100,
		gridDivisions = 100,
		showGrid = true,
		showAxes = true,
		cameraPosition = new THREE.Vector3(50, 50, 50),
		fov = 60,
		near = 0.1,
		far = 10000,
	} = config;

	// Scene
	const scene = new THREE.Scene();
	scene.background = new THREE.Color(backgroundColor);

	// Camera
	const camera = new THREE.PerspectiveCamera(fov, canvas.clientWidth / canvas.clientHeight, near, far);
	camera.position.copy(cameraPosition);

	// Renderer
	const renderer = new THREE.WebGLRenderer({
		canvas,
		antialias: true,
		alpha: false,
		powerPreference: 'high-performance',
	});
	renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
	renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

	// Controls
	const controls = new OrbitControls(camera, canvas);
	controls.enableDamping = true;
	controls.dampingFactor = 0.05;
	controls.screenSpacePanning = true;
	controls.minDistance = 1;
	controls.maxDistance = 1000;
	controls.maxPolarAngle = Math.PI;

	// Lighting (ambient only for point clouds - they use vertex colors)
	const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
	scene.add(ambientLight);

	// Grid helper
	const grid = new THREE.GridHelper(gridSize, gridDivisions, CLOUMASK_COLORS.gridMain, CLOUMASK_COLORS.gridSub);
	grid.visible = showGrid;
	scene.add(grid);

	// Axes helper
	const axes = new THREE.AxesHelper(gridSize / 10);
	axes.visible = showAxes;
	scene.add(axes);

	// Disposed state flag
	let isDisposed = false;

	// Resize handler with DOM state verification
	const resizeObserver = new ResizeObserver(() => {
		if (isDisposed || !canvas.isConnected) return;
		const width = canvas.clientWidth;
		const height = canvas.clientHeight;

		if (width === 0 || height === 0) return;

		camera.aspect = width / height;
		camera.updateProjectionMatrix();
		renderer.setSize(width, height, false);
	});
	resizeObserver.observe(canvas);

	// Dispose function for cleanup
	const dispose = () => {
		isDisposed = true;

		// Stop animation loop if active
		renderer.setAnimationLoop(null);

		// Disconnect resize observer
		resizeObserver.disconnect();

		// Dispose controls
		controls.dispose();

		// Explicitly dispose grid and axes helpers
		grid.geometry?.dispose();
		if (grid.material) {
			if (Array.isArray(grid.material)) {
				grid.material.forEach((m) => m.dispose());
			} else {
				(grid.material as THREE.Material).dispose();
			}
		}

		axes.geometry?.dispose();
		if (axes.material) {
			if (Array.isArray(axes.material)) {
				axes.material.forEach((m) => m.dispose());
			} else {
				(axes.material as THREE.Material).dispose();
			}
		}

		// Dispose all other objects in scene
		scene.traverse((object) => {
			if (object instanceof THREE.Mesh || object instanceof THREE.Points || object instanceof THREE.Line) {
				object.geometry?.dispose();
				if (object.material) {
					if (Array.isArray(object.material)) {
						object.material.forEach((m) => m.dispose());
					} else {
						object.material.dispose();
					}
				}
			}
		});

		renderer.dispose();
	};

	return {
		scene,
		camera,
		renderer,
		controls,
		helpers: { grid, axes },
		dispose,
	};
}

/**
 * Focus camera on a bounding box
 */
export function focusOnBounds(
	camera: THREE.PerspectiveCamera,
	controls: OrbitControls,
	bounds: THREE.Box3,
	padding = 1.5
): void {
	const center = bounds.getCenter(new THREE.Vector3());
	const size = bounds.getSize(new THREE.Vector3());
	const maxDim = Math.max(size.x, size.y, size.z);
	const distance = maxDim * padding;

	camera.position.set(center.x + distance, center.y + distance * 0.5, center.z + distance);
	controls.target.copy(center);
	controls.update();
}

/**
 * Reset camera to default position
 */
export function resetCamera(camera: THREE.PerspectiveCamera, controls: OrbitControls, distance = 50): void {
	camera.position.set(distance, distance, distance);
	controls.target.set(0, 0, 0);
	controls.update();
}
