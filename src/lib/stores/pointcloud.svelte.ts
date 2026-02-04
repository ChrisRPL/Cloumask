/**
 * Point Cloud Viewer state management using Svelte 5 runes.
 *
 * Manages viewer state including camera, controls, point cloud data,
 * bounding boxes, and visualization settings.
 */

import { getContext, setContext } from 'svelte';
import type { ColorMode, BoundingBox3D } from '$lib/utils/three';

// ============================================================================
// Types
// ============================================================================

/** Navigation/camera mode */
export type NavigationMode = 'orbit' | 'pan' | 'zoom' | 'measure';

/** Measurement type */
export type MeasurementType = 'distance' | 'area' | 'angle';

/** Point cloud file information */
export interface PointCloudFile {
	name: string;
	path: string;
	format: string;
	pointCount: number;
	sizeBytes: number;
	bounds: {
		min: { x: number; y: number; z: number };
		max: { x: number; y: number; z: number };
	};
}

/** Camera information */
export interface CameraInfo {
	position: { x: number; y: number; z: number };
	target: { x: number; y: number; z: number };
	fov: number;
}

/** Selection information */
export interface SelectionInfo {
	pointCount: number;
	className?: string;
	confidence?: number;
	boxId?: string;
}

/** Point cloud state interface */
export interface PointCloudState {
	// File state
	readonly file: PointCloudFile | null;
	readonly isLoading: boolean;
	readonly loadProgress: number;
	readonly error: string | null;

	// Visualization settings
	readonly colorMode: ColorMode;
	readonly pointSize: number;
	readonly showGrid: boolean;
	readonly showAxes: boolean;
	readonly showBoundingBoxes: boolean;
	readonly showLabels: boolean;
	readonly backgroundColor: string;
	readonly lodEnabled: boolean;
	readonly lodPointBudget: number;

	// Navigation
	readonly navigationMode: NavigationMode;
	readonly measurementType: MeasurementType;

	// Camera
	readonly camera: CameraInfo;
	readonly fps: number;

	// Selection
	readonly selection: SelectionInfo | null;
	readonly selectedBoxId: string | null;

	// Bounding boxes
	readonly boundingBoxes: BoundingBox3D[];

	// Actions
	setFile(file: PointCloudFile | null): void;
	setLoading(loading: boolean): void;
	setLoadProgress(progress: number): void;
	setError(error: string | null): void;
	setColorMode(mode: ColorMode): void;
	setPointSize(size: number): void;
	setShowGrid(show: boolean): void;
	setShowAxes(show: boolean): void;
	setShowBoundingBoxes(show: boolean): void;
	setShowLabels(show: boolean): void;
	setBackgroundColor(color: string): void;
	setLodEnabled(enabled: boolean): void;
	setLodPointBudget(budget: number): void;
	setNavigationMode(mode: NavigationMode): void;
	setMeasurementType(type: MeasurementType): void;
	updateCamera(info: Partial<CameraInfo>): void;
	setFps(fps: number): void;
	setSelection(selection: SelectionInfo | null): void;
	setSelectedBoxId(id: string | null): void;
	setBoundingBoxes(boxes: BoundingBox3D[]): void;
	reset(): void;
}

// ============================================================================
// State Factory
// ============================================================================

const POINTCLOUD_STATE_KEY = Symbol('pointcloud-state');

/** Default camera position */
const DEFAULT_CAMERA: CameraInfo = {
	position: { x: 50, y: 50, z: 50 },
	target: { x: 0, y: 0, z: 0 },
	fov: 60,
};

/**
 * Creates point cloud viewer state using Svelte 5 runes.
 */
export function createPointCloudState(): PointCloudState {
	// File state
	let file = $state<PointCloudFile | null>(null);
	let isLoading = $state(false);
	let loadProgress = $state(0);
	let error = $state<string | null>(null);

	// Visualization settings
	let colorMode = $state<ColorMode>('height');
	let pointSize = $state(2);
	let showGrid = $state(true);
	let showAxes = $state(true);
	let showBoundingBoxes = $state(true);
	let showLabels = $state(true);
	let backgroundColor = $state('#0c3b1f');
	let lodEnabled = $state(true);
	let lodPointBudget = $state(500_000);

	// Navigation
	let navigationMode = $state<NavigationMode>('orbit');
	let measurementType = $state<MeasurementType>('distance');

	// Camera
	let camera = $state<CameraInfo>({ ...DEFAULT_CAMERA });
	let fps = $state(0);

	// Selection
	let selection = $state<SelectionInfo | null>(null);
	let selectedBoxId = $state<string | null>(null);

	// Bounding boxes
	let boundingBoxes = $state<BoundingBox3D[]>([]);

	return {
		// Getters
		get file() {
			return file;
		},
		get isLoading() {
			return isLoading;
		},
		get loadProgress() {
			return loadProgress;
		},
		get error() {
			return error;
		},
		get colorMode() {
			return colorMode;
		},
		get pointSize() {
			return pointSize;
		},
		get showGrid() {
			return showGrid;
		},
		get showAxes() {
			return showAxes;
		},
		get showBoundingBoxes() {
			return showBoundingBoxes;
		},
		get showLabels() {
			return showLabels;
		},
		get backgroundColor() {
			return backgroundColor;
		},
		get lodEnabled() {
			return lodEnabled;
		},
		get lodPointBudget() {
			return lodPointBudget;
		},
		get navigationMode() {
			return navigationMode;
		},
		get measurementType() {
			return measurementType;
		},
		get camera() {
			return camera;
		},
		get fps() {
			return fps;
		},
		get selection() {
			return selection;
		},
		get selectedBoxId() {
			return selectedBoxId;
		},
		get boundingBoxes() {
			return boundingBoxes;
		},

		// Actions
		setFile(f) {
			file = f;
		},
		setLoading(loading) {
			isLoading = loading;
		},
		setLoadProgress(progress) {
			loadProgress = Math.max(0, Math.min(100, progress));
		},
		setError(err) {
			error = err;
		},
		setColorMode(mode) {
			colorMode = mode;
		},
		setPointSize(size) {
			pointSize = Math.max(0.5, Math.min(10, size));
		},
		setShowGrid(show) {
			showGrid = show;
		},
		setShowAxes(show) {
			showAxes = show;
		},
		setShowBoundingBoxes(show) {
			showBoundingBoxes = show;
		},
		setShowLabels(show) {
			showLabels = show;
		},
		setBackgroundColor(color) {
			backgroundColor = color;
		},
		setLodEnabled(enabled) {
			lodEnabled = enabled;
		},
		setLodPointBudget(budget) {
			lodPointBudget = Math.max(50_000, Math.min(5_000_000, Math.round(budget)));
		},
		setNavigationMode(mode) {
			navigationMode = mode;
		},
		setMeasurementType(type) {
			measurementType = type;
		},
		updateCamera(info) {
			camera = { ...camera, ...info };
		},
		setFps(f) {
			fps = Math.round(f);
		},
		setSelection(sel) {
			selection = sel;
		},
		setSelectedBoxId(id) {
			selectedBoxId = id;
		},
		setBoundingBoxes(boxes) {
			boundingBoxes = boxes;
		},
		reset() {
			file = null;
			isLoading = false;
			loadProgress = 0;
			error = null;
			colorMode = 'height';
			pointSize = 2;
			showGrid = true;
			showAxes = true;
			showBoundingBoxes = true;
			showLabels = true;
			backgroundColor = '#0c3b1f';
			lodEnabled = true;
			lodPointBudget = 500_000;
			navigationMode = 'orbit';
			camera = { ...DEFAULT_CAMERA };
			fps = 0;
			selection = null;
			selectedBoxId = null;
			boundingBoxes = [];
		},
	};
}

// ============================================================================
// Context Helpers
// ============================================================================

/**
 * Initialize point cloud state and set it in Svelte context.
 */
export function setPointCloudState(): PointCloudState {
	const state = createPointCloudState();
	setContext(POINTCLOUD_STATE_KEY, state);
	return state;
}

/**
 * Get point cloud state from Svelte context.
 */
export function getPointCloudState(): PointCloudState {
	return getContext<PointCloudState>(POINTCLOUD_STATE_KEY);
}
