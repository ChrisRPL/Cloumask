/**
 * Three.js Utilities for Point Cloud Visualization
 *
 * Re-exports all Three.js utilities for convenient importing.
 */

// Scene setup
export {
	createScene,
	focusOnBounds,
	resetCamera,
	CLOUMASK_COLORS,
	type SceneContext,
	type SceneConfig,
} from './setup';

// Materials
export {
	createHeightMaterial,
	createIntensityMaterial,
	createRGBMaterial,
	createSimpleMaterial,
	updatePointSize,
	updateHeightRange,
	applyClassificationColors,
	getClassificationColor,
	CLASSIFICATION_COLORS,
	type ColorMode,
	type PointMaterialConfig,
} from './materials';

// Bounding boxes
export {
	createBoxWireframe,
	createBoxMesh,
	createBoxGroup,
	createBoundingBoxes,
	updateBoxSelection,
	findBoxById,
	raycastBoundingBox,
	getClassColor,
	BOX_CLASS_COLORS,
	type BoundingBox3D,
	type LabelOptions,
} from './boxes';

// LOD system
export {
	PointCloudOctree,
	type OctreeBounds,
	type OctreeNode,
	type OctreeConfig,
	type VisiblePointSet,
} from './lod';
