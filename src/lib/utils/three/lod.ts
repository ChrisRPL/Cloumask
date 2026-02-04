/**
 * Level of Detail (LOD) System for Point Cloud Rendering
 *
 * Uses an octree data structure to spatially partition point clouds,
 * enabling efficient frustum culling and screen-space LOD selection
 * for rendering 10M+ point clouds at 60fps.
 */

import * as THREE from 'three';

/** 3D bounds for an octree node */
export interface OctreeBounds {
	min: [number, number, number];
	max: [number, number, number];
}

/** An octree node containing point data */
export interface OctreeNode {
	/** Node bounds */
	bounds: OctreeBounds;
	/** Node depth level (0 = root) */
	level: number;
	/** Decimated points for this level [x,y,z,...] */
	positions: Float32Array;
	/** Optional colors for decimated points [r,g,b,...] */
	colors?: Float32Array;
	/** Optional intensity values */
	intensities?: Float32Array;
	/** Optional classification values */
	classifications?: Uint8Array;
	/** Child nodes (8 children for octree, or empty if leaf) */
	children: OctreeNode[];
	/** Total point count in this node and all children */
	pointCount: number;
}

/** Result of visible point collection */
export interface VisiblePointSet {
	/** Combined positions from visible nodes */
	positions: Float32Array;
	/** Combined colors (if available) */
	colors?: Float32Array;
	/** Combined intensities (if available) */
	intensities?: Float32Array;
	/** Combined classifications (if available) */
	classifications?: Uint8Array;
	/** Total visible point count */
	pointCount: number;
	/** Number of nodes used */
	nodeCount: number;
	/** Whether the geometry needs updating */
	needsUpdate: boolean;
}

/** Configuration for octree building */
export interface OctreeConfig {
	/** Maximum tree depth (default: 8) */
	maxDepth?: number;
	/** Maximum points per leaf node (default: 10000) */
	maxPointsPerNode?: number;
	/** Target points to keep when decimating (default: 5000) */
	decimationTarget?: number;
}

const DEFAULT_CONFIG: Required<OctreeConfig> = {
	maxDepth: 8,
	maxPointsPerNode: 10000,
	decimationTarget: 5000,
};

/**
 * Point Cloud Octree for LOD rendering
 */
export class PointCloudOctree {
	private root: OctreeNode | null = null;
	private config: Required<OctreeConfig>;
	private lastCameraPosition = new THREE.Vector3();
	private lastCameraTarget = new THREE.Vector3();
	private frustum = new THREE.Frustum();
	private projScreenMatrix = new THREE.Matrix4();

	constructor(config: OctreeConfig = {}) {
		this.config = { ...DEFAULT_CONFIG, ...config };
	}

	/**
	 * Build octree from point cloud data
	 */
	build(
		positions: Float32Array,
		bounds: OctreeBounds,
		colors?: Float32Array,
		intensities?: Float32Array,
		classifications?: Uint8Array,
	): void {
		this.root = this.buildNode(
			positions,
			bounds,
			0,
			colors,
			intensities,
			classifications,
		);
	}

	private buildNode(
		positions: Float32Array,
		bounds: OctreeBounds,
		level: number,
		colors?: Float32Array,
		intensities?: Float32Array,
		classifications?: Uint8Array,
	): OctreeNode {
		const pointCount = positions.length / 3;

		// If this is a leaf node (small enough or max depth reached)
		if (pointCount <= this.config.maxPointsPerNode || level >= this.config.maxDepth) {
			return {
				bounds,
				level,
				positions,
				colors,
				intensities,
				classifications,
				children: [],
				pointCount,
			};
		}

		// Subdivide into 8 children
		const childBounds = this.subdivide(bounds);
		const childData = this.partitionPoints(
			positions,
			bounds,
			childBounds,
			colors,
			intensities,
			classifications,
		);

		const children: OctreeNode[] = [];
		for (let i = 0; i < 8; i++) {
			if (childData[i].positions.length > 0) {
				children.push(
					this.buildNode(
						childData[i].positions,
						childBounds[i],
						level + 1,
						childData[i].colors,
						childData[i].intensities,
						childData[i].classifications,
					),
				);
			}
		}

		// Create decimated points for this level
		const decimated = this.decimate(
			positions,
			this.config.decimationTarget,
			colors,
			intensities,
			classifications,
		);

		return {
			bounds,
			level,
			positions: decimated.positions,
			colors: decimated.colors,
			intensities: decimated.intensities,
			classifications: decimated.classifications,
			children,
			pointCount,
		};
	}

	private subdivide(bounds: OctreeBounds): OctreeBounds[] {
		const mid = [
			(bounds.min[0] + bounds.max[0]) / 2,
			(bounds.min[1] + bounds.max[1]) / 2,
			(bounds.min[2] + bounds.max[2]) / 2,
		] as [number, number, number];

		// 8 child bounds (binary combinations of min/mid and mid/max)
		return [
			{ min: [bounds.min[0], bounds.min[1], bounds.min[2]], max: [mid[0], mid[1], mid[2]] },
			{ min: [mid[0], bounds.min[1], bounds.min[2]], max: [bounds.max[0], mid[1], mid[2]] },
			{ min: [bounds.min[0], mid[1], bounds.min[2]], max: [mid[0], bounds.max[1], mid[2]] },
			{ min: [mid[0], mid[1], bounds.min[2]], max: [bounds.max[0], bounds.max[1], mid[2]] },
			{ min: [bounds.min[0], bounds.min[1], mid[2]], max: [mid[0], mid[1], bounds.max[2]] },
			{ min: [mid[0], bounds.min[1], mid[2]], max: [bounds.max[0], mid[1], bounds.max[2]] },
			{ min: [bounds.min[0], mid[1], mid[2]], max: [mid[0], bounds.max[1], bounds.max[2]] },
			{ min: [mid[0], mid[1], mid[2]], max: [bounds.max[0], bounds.max[1], bounds.max[2]] },
		] as OctreeBounds[];
	}

	private partitionPoints(
		positions: Float32Array,
		parentBounds: OctreeBounds,
		childBounds: OctreeBounds[],
		colors?: Float32Array,
		intensities?: Float32Array,
		classifications?: Uint8Array,
	): Array<{
		positions: Float32Array;
		colors?: Float32Array;
		intensities?: Float32Array;
		classifications?: Uint8Array;
	}> {
		const pointCount = positions.length / 3;
		const childIndices: number[][] = Array.from({ length: 8 }, () => []);

		const mid = [
			(parentBounds.min[0] + parentBounds.max[0]) / 2,
			(parentBounds.min[1] + parentBounds.max[1]) / 2,
			(parentBounds.min[2] + parentBounds.max[2]) / 2,
		];

		// Assign each point to a child octant
		for (let i = 0; i < pointCount; i++) {
			const x = positions[i * 3];
			const y = positions[i * 3 + 1];
			const z = positions[i * 3 + 2];

			// Compute octant index (0-7)
			const ix = x >= mid[0] ? 1 : 0;
			const iy = y >= mid[1] ? 2 : 0;
			const iz = z >= mid[2] ? 4 : 0;
			const octant = ix | iy | iz;

			childIndices[octant].push(i);
		}

		// Extract data for each child
		return childIndices.map((indices) => {
			const count = indices.length;
			const childPositions = new Float32Array(count * 3);
			const childColors = colors ? new Float32Array(count * 3) : undefined;
			const childIntensities = intensities ? new Float32Array(count) : undefined;
			const childClassifications = classifications ? new Uint8Array(count) : undefined;

			for (let i = 0; i < count; i++) {
				const srcIdx = indices[i];
				childPositions[i * 3] = positions[srcIdx * 3];
				childPositions[i * 3 + 1] = positions[srcIdx * 3 + 1];
				childPositions[i * 3 + 2] = positions[srcIdx * 3 + 2];

				if (colors && childColors) {
					childColors[i * 3] = colors[srcIdx * 3];
					childColors[i * 3 + 1] = colors[srcIdx * 3 + 1];
					childColors[i * 3 + 2] = colors[srcIdx * 3 + 2];
				}
				if (intensities && childIntensities) {
					childIntensities[i] = intensities[srcIdx];
				}
				if (classifications && childClassifications) {
					childClassifications[i] = classifications[srcIdx];
				}
			}

			return {
				positions: childPositions,
				colors: childColors,
				intensities: childIntensities,
				classifications: childClassifications,
			};
		});
	}

	private decimate(
		positions: Float32Array,
		target: number,
		colors?: Float32Array,
		intensities?: Float32Array,
		classifications?: Uint8Array,
	): {
		positions: Float32Array;
		colors?: Float32Array;
		intensities?: Float32Array;
		classifications?: Uint8Array;
	} {
		const count = positions.length / 3;
		if (count <= target) {
			return { positions, colors, intensities, classifications };
		}

		// Uniform sampling - keep every Nth point
		const step = Math.ceil(count / target);
		const decimatedCount = Math.ceil(count / step);

		const decimatedPositions = new Float32Array(decimatedCount * 3);
		const decimatedColors = colors ? new Float32Array(decimatedCount * 3) : undefined;
		const decimatedIntensities = intensities ? new Float32Array(decimatedCount) : undefined;
		const decimatedClassifications = classifications ? new Uint8Array(decimatedCount) : undefined;

		let outIdx = 0;
		for (let i = 0; i < count && outIdx < decimatedCount; i += step) {
			decimatedPositions[outIdx * 3] = positions[i * 3];
			decimatedPositions[outIdx * 3 + 1] = positions[i * 3 + 1];
			decimatedPositions[outIdx * 3 + 2] = positions[i * 3 + 2];

			if (colors && decimatedColors) {
				decimatedColors[outIdx * 3] = colors[i * 3];
				decimatedColors[outIdx * 3 + 1] = colors[i * 3 + 1];
				decimatedColors[outIdx * 3 + 2] = colors[i * 3 + 2];
			}
			if (intensities && decimatedIntensities) {
				decimatedIntensities[outIdx] = intensities[i];
			}
			if (classifications && decimatedClassifications) {
				decimatedClassifications[outIdx] = classifications[i];
			}

			outIdx++;
		}

		return {
			positions: decimatedPositions,
			colors: decimatedColors,
			intensities: decimatedIntensities,
			classifications: decimatedClassifications,
		};
	}

	/**
	 * Get visible points based on camera frustum and LOD
	 */
	getVisiblePoints(
		camera: THREE.PerspectiveCamera,
		targetPointBudget: number = 500000,
	): VisiblePointSet {
		if (!this.root) {
			return {
				positions: new Float32Array(0),
				pointCount: 0,
				nodeCount: 0,
				needsUpdate: false,
			};
		}

		// Check if camera has moved significantly
		const cameraPos = camera.position;
		const needsUpdate =
			this.lastCameraPosition.distanceTo(cameraPos) > 0.1 ||
			!this.lastCameraTarget.equals(camera.getWorldDirection(new THREE.Vector3()));

		if (!needsUpdate) {
			return {
				positions: new Float32Array(0),
				pointCount: 0,
				nodeCount: 0,
				needsUpdate: false,
			};
		}

		this.lastCameraPosition.copy(cameraPos);
		camera.getWorldDirection(this.lastCameraTarget);

		// Update frustum
		this.projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
		this.frustum.setFromProjectionMatrix(this.projScreenMatrix);

		// Collect visible nodes
		const visibleNodes: OctreeNode[] = [];
		this.collectVisibleNodes(this.root, camera, visibleNodes, targetPointBudget);

		// Concatenate all visible point data
		const totalPoints = visibleNodes.reduce((sum, node) => sum + node.positions.length / 3, 0);
		const positions = new Float32Array(totalPoints * 3);
		const hasColors = visibleNodes.some((n) => n.colors);
		const hasIntensities = visibleNodes.some((n) => n.intensities);
		const hasClassifications = visibleNodes.some((n) => n.classifications);

		const colors = hasColors ? new Float32Array(totalPoints * 3) : undefined;
		const intensities = hasIntensities ? new Float32Array(totalPoints) : undefined;
		const classifications = hasClassifications ? new Uint8Array(totalPoints) : undefined;

		let offset = 0;
		for (const node of visibleNodes) {
			const count = node.positions.length / 3;
			positions.set(node.positions, offset * 3);

			if (colors && node.colors) {
				colors.set(node.colors, offset * 3);
			}
			if (intensities && node.intensities) {
				intensities.set(node.intensities, offset);
			}
			if (classifications && node.classifications) {
				classifications.set(node.classifications, offset);
			}

			offset += count;
		}

		return {
			positions,
			colors,
			intensities,
			classifications,
			pointCount: totalPoints,
			nodeCount: visibleNodes.length,
			needsUpdate: true,
		};
	}

	private collectVisibleNodes(
		node: OctreeNode,
		camera: THREE.PerspectiveCamera,
		result: OctreeNode[],
		budget: number,
	): void {
		// Check frustum intersection
		const box = new THREE.Box3(
			new THREE.Vector3(...node.bounds.min),
			new THREE.Vector3(...node.bounds.max),
		);

		if (!this.frustum.intersectsBox(box)) {
			return;
		}

		// Calculate screen-space size
		const screenSize = this.computeScreenSize(box, camera);

		// If node is small on screen or we're at budget, use this level
		if (node.children.length === 0 || screenSize < 100) {
			result.push(node);
			return;
		}

		// Check if we have budget for children
		const currentBudget = result.reduce((sum, n) => sum + n.positions.length / 3, 0);
		if (currentBudget + node.pointCount > budget) {
			// Use decimated version
			result.push(node);
			return;
		}

		// Recurse into children
		for (const child of node.children) {
			this.collectVisibleNodes(child, camera, result, budget);
		}
	}

	private computeScreenSize(box: THREE.Box3, camera: THREE.PerspectiveCamera): number {
		const center = box.getCenter(new THREE.Vector3());
		const size = box.getSize(new THREE.Vector3());
		const maxDim = Math.max(size.x, size.y, size.z);

		// Distance from camera to box center
		const distance = camera.position.distanceTo(center);

		// Approximate screen size based on perspective
		const fovRadians = (camera.fov * Math.PI) / 180;
		const screenHeight = 2 * distance * Math.tan(fovRadians / 2);
		const screenSize = (maxDim / screenHeight) * window.innerHeight;

		return screenSize;
	}

	/**
	 * Check if octree has been built
	 */
	isBuilt(): boolean {
		return this.root !== null;
	}

	/**
	 * Get total point count
	 */
	getTotalPointCount(): number {
		return this.root?.pointCount ?? 0;
	}

	/**
	 * Dispose of octree data and release memory
	 */
	dispose(): void {
		if (this.root) {
			this.disposeNode(this.root);
			this.root = null;
		}
	}

	private disposeNode(node: OctreeNode): void {
		// Recursively dispose children first
		for (const child of node.children) {
			this.disposeNode(child);
		}
		node.children = [];

		// Clear typed array references to help GC with large datasets
		(node as { positions: Float32Array | null }).positions = null;
		if (node.colors) (node as { colors: Float32Array | null }).colors = null;
		if (node.intensities) (node as { intensities: Float32Array | null }).intensities = null;
		if (node.classifications)
			(node as { classifications: Uint8Array | null }).classifications = null;
	}
}
