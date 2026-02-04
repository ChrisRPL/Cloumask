/**
 * Three.js 3D Bounding Box Utilities
 *
 * Utilities for creating and managing 3D bounding boxes
 * for object detection overlays in point cloud visualization.
 */

import * as THREE from 'three';
import { CLOUMASK_COLORS } from './setup';

/** 3D bounding box definition */
export interface BoundingBox3D {
	id: string;
	className: string;
	confidence: number;
	center: THREE.Vector3;
	size: THREE.Vector3;
	rotation: THREE.Euler;
	color?: string;
	visible?: boolean;
}

/** Label options for bounding boxes */
export interface LabelOptions {
	showClassName: boolean;
	showConfidence: boolean;
	fontSize: number;
	backgroundColor: string;
	textColor: string;
}

// Default edge material
const DEFAULT_BOX_COLOR = new THREE.Color(CLOUMASK_COLORS.boundingBox);
const SELECTED_BOX_COLOR = new THREE.Color(CLOUMASK_COLORS.boundingBoxSelected);

function resolveLabelOptions(options?: Partial<LabelOptions>): LabelOptions {
	return { ...DEFAULT_LABEL_OPTIONS, ...(options ?? {}) };
}

function buildLabelText(box: BoundingBox3D, options: LabelOptions): string {
	const parts: string[] = [];
	if (options.showClassName) {
		parts.push(box.className);
	}
	if (options.showConfidence) {
		parts.push(`${Math.round(box.confidence * 100)}%`);
	}
	return parts.join(' | ');
}

function createLabelSprite(text: string, options: LabelOptions): THREE.Sprite {
	const canvas = document.createElement('canvas');
	const ctx = canvas.getContext('2d');
	if (!ctx) {
		return new THREE.Sprite();
	}

	const fontStack = '"JetBrains Mono", "Fira Code", "SF Mono", monospace';
	const paddingX = options.fontSize * 0.7;
	const paddingY = options.fontSize * 0.5;
	const ratio = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

	ctx.font = `${options.fontSize}px ${fontStack}`;
	const metrics = ctx.measureText(text);
	const textWidth = metrics.width;
	const textHeight = options.fontSize * 1.3;

	canvas.width = Math.ceil((textWidth + paddingX * 2) * ratio);
	canvas.height = Math.ceil((textHeight + paddingY * 2) * ratio);
	ctx.scale(ratio, ratio);
	ctx.font = `${options.fontSize}px ${fontStack}`;

	// Background
	ctx.fillStyle = options.backgroundColor;
	const radius = Math.min(8, textHeight / 2);
	const width = textWidth + paddingX * 2;
	const height = textHeight + paddingY * 2;

	ctx.beginPath();
	ctx.moveTo(radius, 0);
	ctx.lineTo(width - radius, 0);
	ctx.quadraticCurveTo(width, 0, width, radius);
	ctx.lineTo(width, height - radius);
	ctx.quadraticCurveTo(width, height, width - radius, height);
	ctx.lineTo(radius, height);
	ctx.quadraticCurveTo(0, height, 0, height - radius);
	ctx.lineTo(0, radius);
	ctx.quadraticCurveTo(0, 0, radius, 0);
	ctx.closePath();
	ctx.fill();

	// Text
	ctx.fillStyle = options.textColor;
	ctx.textAlign = 'left';
	ctx.textBaseline = 'middle';
	ctx.fillText(text, paddingX, height / 2);

	const texture = new THREE.CanvasTexture(canvas);
	texture.needsUpdate = true;
	texture.minFilter = THREE.LinearFilter;
	texture.generateMipmaps = false;

	const material = new THREE.SpriteMaterial({
		map: texture,
		transparent: true,
		depthTest: false,
		depthWrite: false,
	});

	const sprite = new THREE.Sprite(material);
	sprite.userData = { labelAspect: canvas.width / canvas.height };

	return sprite;
}
/**
 * Creates a wireframe box mesh for a 3D bounding box
 */
export function createBoxWireframe(
	box: BoundingBox3D,
	selected = false
): THREE.LineSegments {
	// Create box geometry
	const geometry = new THREE.BoxGeometry(box.size.x, box.size.y, box.size.z);

	// Convert to edges geometry for wireframe
	const edgesGeometry = new THREE.EdgesGeometry(geometry);

	// Create material
	const color = selected ? SELECTED_BOX_COLOR : box.color ? new THREE.Color(box.color) : DEFAULT_BOX_COLOR;

	const material = new THREE.LineBasicMaterial({
		color,
		linewidth: selected ? 2 : 1,
		transparent: true,
		opacity: selected ? 1.0 : 0.8,
	});

	// Create line segments
	const wireframe = new THREE.LineSegments(edgesGeometry, material);

	// Position and rotate
	wireframe.position.copy(box.center);
	wireframe.rotation.copy(box.rotation);

	// Store metadata
	wireframe.userData = {
		boxId: box.id,
		className: box.className,
		confidence: box.confidence,
		isSelected: selected,
	};

	wireframe.name = `bbox-${box.id}`;

	return wireframe;
}

/**
 * Creates a filled (transparent) box mesh
 */
export function createBoxMesh(
	box: BoundingBox3D,
	selected = false
): THREE.Mesh {
	const geometry = new THREE.BoxGeometry(box.size.x, box.size.y, box.size.z);

	const color = selected ? SELECTED_BOX_COLOR : box.color ? new THREE.Color(box.color) : DEFAULT_BOX_COLOR;

	const material = new THREE.MeshBasicMaterial({
		color,
		transparent: true,
		opacity: selected ? 0.2 : 0.1,
		side: THREE.DoubleSide,
		depthWrite: false,
	});

	const mesh = new THREE.Mesh(geometry, material);
	mesh.position.copy(box.center);
	mesh.rotation.copy(box.rotation);

	mesh.userData = {
		boxId: box.id,
		className: box.className,
		confidence: box.confidence,
		isSelected: selected,
	};

	mesh.name = `bbox-fill-${box.id}`;

	return mesh;
}

/**
 * Creates a group containing both wireframe and filled box
 */
export function createBoxGroup(box: BoundingBox3D, selected = false): THREE.Group {
	const group = new THREE.Group();
	group.name = `bbox-group-${box.id}`;

	const wireframe = createBoxWireframe(box, selected);
	const mesh = createBoxMesh(box, selected);

	group.add(wireframe);
	group.add(mesh);

	group.userData = {
		boxId: box.id,
		className: box.className,
		confidence: box.confidence,
		box: box,
	};

	return group;
}

/**
 * Updates an existing box group's selection state
 */
export function updateBoxSelection(group: THREE.Group, selected: boolean): void {
	const box = group.userData.box as BoundingBox3D;
	if (!box) return;

	const color = selected ? SELECTED_BOX_COLOR : box.color ? new THREE.Color(box.color) : DEFAULT_BOX_COLOR;

	group.children.forEach((child) => {
		if (child instanceof THREE.LineSegments) {
			const material = child.material as THREE.LineBasicMaterial;
			material.color.copy(color);
			material.opacity = selected ? 1.0 : 0.8;
			child.userData.isSelected = selected;
		} else if (child instanceof THREE.Mesh) {
			const material = child.material as THREE.MeshBasicMaterial;
			material.color.copy(color);
			material.opacity = selected ? 0.2 : 0.1;
			child.userData.isSelected = selected;
		}
	});
}

/**
 * Classification color map for bounding boxes
 */
export const BOX_CLASS_COLORS: Record<string, number> = {
	vehicle: 0x22c55e, // Forest green
	car: 0x22c55e,
	truck: 0x16a34a,
	bus: 0x15803d,
	pedestrian: 0xfcd34d, // Yellow
	person: 0xfcd34d,
	cyclist: 0x60a5fa, // Blue
	bicycle: 0x60a5fa,
	motorcycle: 0x818cf8, // Purple
	building: 0x6b7280, // Gray
	vegetation: 0x166534, // Dark green
	unknown: 0x9ca3af, // Light gray
};

/**
 * Get color for a class name
 */
export function getClassColor(className: string): number {
	const normalizedName = className.toLowerCase();
	return BOX_CLASS_COLORS[normalizedName] ?? BOX_CLASS_COLORS.unknown;
}

/**
 * Creates all bounding boxes from an array of definitions
 */
export function createBoundingBoxes(boxes: BoundingBox3D[], selectedId?: string): THREE.Group {
	const container = new THREE.Group();
	container.name = 'bounding-boxes';

	boxes.forEach((box) => {
		if (box.visible === false) return;

		const isSelected = box.id === selectedId;
		const boxGroup = createBoxGroup(box, isSelected);
		container.add(boxGroup);
	});

	return container;
}

/**
 * Find a bounding box group by ID
 */
export function findBoxById(container: THREE.Group, id: string): THREE.Group | null {
	let found: THREE.Group | null = null;

	container.traverse((child) => {
		if (child instanceof THREE.Group && child.userData.boxId === id) {
			found = child;
		}
	});

	return found;
}

/**
 * Raycast to find bounding box at mouse position
 */
export function raycastBoundingBox(
	raycaster: THREE.Raycaster,
	container: THREE.Group
): BoundingBox3D | null {
	// Get all filled meshes (easier to click on)
	const meshes: THREE.Object3D[] = [];
	container.traverse((child) => {
		if (child instanceof THREE.Mesh && child.name.startsWith('bbox-fill-')) {
			meshes.push(child);
		}
	});

	const intersects = raycaster.intersectObjects(meshes, false);

	if (intersects.length > 0) {
		const hit = intersects[0].object;
		// Get parent group
		const group = hit.parent;
		if (group && group.userData.box) {
			return group.userData.box as BoundingBox3D;
		}
	}

	return null;
}
