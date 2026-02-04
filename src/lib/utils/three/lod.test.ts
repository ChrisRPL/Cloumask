import * as THREE from 'three';
import { PointCloudOctree } from '$lib/utils/three';

describe('PointCloudOctree', () => {
	it('builds and returns visible points', () => {
		const positions = new Float32Array([
			-1, -1, -1,
			1, -1, -1,
			-1, 1, -1,
			1, 1, -1,
			-1, -1, 1,
			1, -1, 1,
			-1, 1, 1,
			1, 1, 1,
		]);

		const octree = new PointCloudOctree({ maxDepth: 2, maxPointsPerNode: 2, decimationTarget: 2 });
		octree.build(positions, { min: [-2, -2, -2], max: [2, 2, 2] });

		const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
		camera.position.set(0, 0, 8);
		camera.lookAt(0, 0, 0);
		camera.updateMatrixWorld();

		const visible = octree.getVisiblePoints(camera, 100);
		expect(visible.pointCount).toBeGreaterThan(0);
		expect(visible.nodeCount).toBeGreaterThan(0);
	});

	it('invalidates to force an update', () => {
		const positions = new Float32Array([0, 0, 0, 1, 1, 1, -1, -1, -1]);
		const octree = new PointCloudOctree();
		octree.build(positions, { min: [-1, -1, -1], max: [1, 1, 1] });

		const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
		camera.position.set(0, 0, 6);
		camera.lookAt(0, 0, 0);
		camera.updateMatrixWorld();

		octree.getVisiblePoints(camera, 10);
		octree.invalidate();
		const visible = octree.getVisiblePoints(camera, 10);
		expect(visible.needsUpdate).toBe(true);
	});
});
