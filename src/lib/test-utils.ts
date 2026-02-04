import { vi } from 'vitest';
import * as THREE from 'three';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { SceneContext } from '$lib/utils/three';
import type { Bounds3D, PointCloudData, PointCloudMetadata } from '$lib/types/pointcloud';

export const mockInvoke = vi.fn();
export const mockListen = vi.fn(async () => () => {});
export const mockOpen = vi.fn();
export const mockSave = vi.fn();

export function createMockBounds(): Bounds3D {
	return {
		min: [0, 0, 0],
		max: [10, 10, 10],
	};
}

export function createMockMetadata(overrides: Partial<PointCloudMetadata> = {}): PointCloudMetadata {
	return {
		path: '/data/mock.pcd',
		format: 'pcd',
		point_count: 3,
		file_size_bytes: 2048,
		attributes: ['position'],
		bounds: createMockBounds(),
		has_intensity: false,
		has_rgb: false,
		has_classification: false,
		...overrides,
	};
}

export function createMockPointCloudData(
	overrides: Partial<PointCloudData> = {},
): PointCloudData {
	return {
		metadata: createMockMetadata(overrides.metadata),
		positions: [0, 0, 0, 5, 5, 5, 10, 10, 10],
		intensities: null,
		colors: null,
		classifications: null,
		...overrides,
	};
}

export function createMockSceneContext(): SceneContext {
	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera();
	const renderer = {
		setAnimationLoop: vi.fn(),
		render: vi.fn(),
		dispose: vi.fn(),
		domElement: document.createElement('canvas'),
	} as unknown as THREE.WebGLRenderer;
	const controls = {
		update: vi.fn(),
		dispose: vi.fn(),
		target: new THREE.Vector3(),
		addEventListener: vi.fn(),
		removeEventListener: vi.fn(),
	} as unknown as OrbitControls;
	const helpers = {
		grid: { visible: true },
		axes: { visible: true },
	} as SceneContext['helpers'];

	return {
		scene,
		camera,
		renderer,
		controls,
		helpers,
		dispose: vi.fn(),
	};
}
