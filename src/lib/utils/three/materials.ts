/**
 * Three.js Materials for Point Cloud Rendering
 *
 * Custom shader materials and standard materials optimized for
 * large point cloud datasets with multiple color modes.
 */

import * as THREE from 'three';
import { CLOUMASK_COLORS } from './setup';

/** Available color modes for point cloud visualization */
export type ColorMode = 'rgb' | 'intensity' | 'height' | 'classification' | 'single';

/** Point cloud material configuration */
export interface PointMaterialConfig {
	colorMode: ColorMode;
	pointSize: number;
	heightMin?: number;
	heightMax?: number;
	singleColor?: THREE.Color;
	opacity?: number;
}

/**
 * Vertex shader for height-based coloring
 * Uses rainbow gradient from blue (low) to red (high)
 */
const heightVertexShader = /* glsl */ `
  uniform float pointSize;
  uniform float heightMin;
  uniform float heightMax;

  varying vec3 vColor;

  vec3 heightToColor(float height) {
    float t = clamp((height - heightMin) / (heightMax - heightMin), 0.0, 1.0);

    // Forest-themed gradient: dark green -> light green -> cream
    vec3 a = vec3(0.055, 0.212, 0.118); // #0e3620 dark forest
    vec3 b = vec3(0.086, 0.329, 0.200); // #166534 forest green
    vec3 c = vec3(0.133, 0.773, 0.369); // #22c55e light green
    vec3 d = vec3(0.525, 0.937, 0.675); // #86efac mint
    vec3 e = vec3(0.980, 0.969, 0.941); // #faf7f0 cream

    if (t < 0.25) return mix(a, b, t * 4.0);
    if (t < 0.5) return mix(b, c, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(c, d, (t - 0.5) * 4.0);
    return mix(d, e, (t - 0.75) * 4.0);
  }

  void main() {
    vColor = heightToColor(position.z);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = pointSize * (300.0 / -mvPosition.z);
    gl_PointSize = clamp(gl_PointSize, 1.0, 50.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

/**
 * Fragment shader for circular points
 */
const circleFragmentShader = /* glsl */ `
  varying vec3 vColor;
  uniform float opacity;

  void main() {
    // Circular point shape
    vec2 center = gl_PointCoord - vec2(0.5);
    if (length(center) > 0.5) discard;

    // Soft edge
    float alpha = 1.0 - smoothstep(0.4, 0.5, length(center));
    gl_FragColor = vec4(vColor, opacity * alpha);
  }
`;

/**
 * Vertex shader for intensity-based coloring
 */
const intensityVertexShader = /* glsl */ `
  uniform float pointSize;
  attribute float intensity;

  varying vec3 vColor;

  void main() {
    // Intensity mapped to grayscale with forest tint
    float i = clamp(intensity, 0.0, 1.0);
    vColor = mix(vec3(0.055, 0.212, 0.118), vec3(0.980, 0.969, 0.941), i);

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = pointSize * (300.0 / -mvPosition.z);
    gl_PointSize = clamp(gl_PointSize, 1.0, 50.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

/**
 * Vertex shader for RGB coloring (uses vertex colors)
 */
const rgbVertexShader = /* glsl */ `
  uniform float pointSize;

  varying vec3 vColor;

  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = pointSize * (300.0 / -mvPosition.z);
    gl_PointSize = clamp(gl_PointSize, 1.0, 50.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

/**
 * Creates a shader material for height-based coloring
 */
export function createHeightMaterial(config: PointMaterialConfig): THREE.ShaderMaterial {
	return new THREE.ShaderMaterial({
		uniforms: {
			pointSize: { value: config.pointSize },
			heightMin: { value: config.heightMin ?? 0 },
			heightMax: { value: config.heightMax ?? 100 },
			opacity: { value: config.opacity ?? 1.0 },
		},
		vertexShader: heightVertexShader,
		fragmentShader: circleFragmentShader,
		transparent: config.opacity !== undefined && config.opacity < 1.0,
		depthWrite: true,
		depthTest: true,
	});
}

/**
 * Creates a shader material for intensity-based coloring
 */
export function createIntensityMaterial(config: PointMaterialConfig): THREE.ShaderMaterial {
	return new THREE.ShaderMaterial({
		uniforms: {
			pointSize: { value: config.pointSize },
			opacity: { value: config.opacity ?? 1.0 },
		},
		vertexShader: intensityVertexShader,
		fragmentShader: circleFragmentShader,
		transparent: config.opacity !== undefined && config.opacity < 1.0,
		depthWrite: true,
		depthTest: true,
	});
}

/**
 * Creates a shader material for RGB vertex colors
 */
export function createRGBMaterial(config: PointMaterialConfig): THREE.ShaderMaterial {
	return new THREE.ShaderMaterial({
		uniforms: {
			pointSize: { value: config.pointSize },
			opacity: { value: config.opacity ?? 1.0 },
		},
		vertexShader: rgbVertexShader,
		fragmentShader: circleFragmentShader,
		vertexColors: true,
		transparent: config.opacity !== undefined && config.opacity < 1.0,
		depthWrite: true,
		depthTest: true,
	});
}

/**
 * Creates a standard PointsMaterial for simple use cases
 */
export function createSimpleMaterial(config: PointMaterialConfig): THREE.PointsMaterial {
	return new THREE.PointsMaterial({
		size: config.pointSize,
		color: config.singleColor ?? new THREE.Color(CLOUMASK_COLORS.pointDefault),
		vertexColors: config.colorMode === 'rgb',
		sizeAttenuation: true,
		transparent: config.opacity !== undefined && config.opacity < 1.0,
		opacity: config.opacity ?? 1.0,
	});
}

/**
 * Classification color map
 */
export const CLASSIFICATION_COLORS: Record<number, THREE.Color> = {
	0: new THREE.Color(0x808080), // Unknown - Gray
	1: new THREE.Color(CLOUMASK_COLORS.classification.unknown), // Unclassified
	2: new THREE.Color(CLOUMASK_COLORS.classification.ground), // Ground
	3: new THREE.Color(0x228b22), // Low vegetation
	4: new THREE.Color(0x2e8b57), // Medium vegetation
	5: new THREE.Color(CLOUMASK_COLORS.classification.vegetation), // High vegetation
	6: new THREE.Color(CLOUMASK_COLORS.classification.building), // Building
	7: new THREE.Color(0xffd700), // Low point (noise)
	8: new THREE.Color(0x808080), // Reserved
	9: new THREE.Color(0x4169e1), // Water
	10: new THREE.Color(0x8b4513), // Rail
	11: new THREE.Color(0x696969), // Road surface
	12: new THREE.Color(0x808080), // Reserved
	13: new THREE.Color(0xffd700), // Wire guard
	14: new THREE.Color(0xff4500), // Wire conductor
	15: new THREE.Color(0x9370db), // Transmission tower
	16: new THREE.Color(0x4682b4), // Wire connector
	17: new THREE.Color(0x2f4f4f), // Bridge deck
	18: new THREE.Color(0x808080), // High noise
};

/**
 * Get color for a classification value
 */
export function getClassificationColor(classification: number): THREE.Color {
	return CLASSIFICATION_COLORS[classification] ?? CLASSIFICATION_COLORS[0];
}

/**
 * Apply classification colors to a geometry
 */
export function applyClassificationColors(
	geometry: THREE.BufferGeometry,
	classifications: Uint8Array
): void {
	const colors = new Float32Array(classifications.length * 3);

	for (let i = 0; i < classifications.length; i++) {
		const color = getClassificationColor(classifications[i]);
		colors[i * 3] = color.r;
		colors[i * 3 + 1] = color.g;
		colors[i * 3 + 2] = color.b;
	}

	geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

/**
 * Update point size on a material
 */
export function updatePointSize(material: THREE.Material, size: number): void {
	if (material instanceof THREE.ShaderMaterial) {
		material.uniforms.pointSize.value = size;
	} else if (material instanceof THREE.PointsMaterial) {
		material.size = size;
	}
}

/**
 * Update height range for height-based materials
 */
export function updateHeightRange(material: THREE.ShaderMaterial, min: number, max: number): void {
	if (material.uniforms.heightMin) {
		material.uniforms.heightMin.value = min;
	}
	if (material.uniforms.heightMax) {
		material.uniforms.heightMax.value = max;
	}
}
