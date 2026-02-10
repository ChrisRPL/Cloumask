import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { createBoundingBoxes } from '$lib/utils/three';

describe('bounding box labels', () => {
	it('creates label sprites when enabled', () => {
		const boxes = [
			{
				id: 'box-1',
				className: 'Car',
				confidence: 0.92,
				center: new THREE.Vector3(0, 0, 0),
				size: new THREE.Vector3(2, 2, 2),
				rotation: new THREE.Euler(0, 0, 0),
			},
		];

		const group = createBoundingBoxes(boxes, undefined, {
			showClassName: true,
			showConfidence: true,
		});

		const boxGroup = group.children[0] as THREE.Group;
		const hasSprite = boxGroup.children.some((child) => child instanceof THREE.Sprite);
		expect(hasSprite).toBe(true);
	});

	it('omits label sprites when disabled', () => {
		const boxes = [
			{
				id: 'box-2',
				className: 'Pedestrian',
				confidence: 0.8,
				center: new THREE.Vector3(0, 0, 0),
				size: new THREE.Vector3(1, 2, 1),
				rotation: new THREE.Euler(0, 0, 0),
			},
		];

		const group = createBoundingBoxes(boxes, undefined, {
			showClassName: false,
			showConfidence: false,
		});

		const boxGroup = group.children[0] as THREE.Group;
		const hasSprite = boxGroup.children.some((child) => child instanceof THREE.Sprite);
		expect(hasSprite).toBe(false);
	});
});
