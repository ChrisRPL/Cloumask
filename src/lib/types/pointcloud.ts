/**
 * TypeScript types for point cloud data structures.
 *
 * These types mirror the Rust structures in src-tauri/src/pointcloud/types.rs
 * for type-safe communication via Tauri IPC.
 */

/** Supported point cloud file formats */
export type PointCloudFormat = 'pcd' | 'ply' | 'las' | 'laz';

/** 3D axis-aligned bounding box */
export interface Bounds3D {
	min: [number, number, number];
	max: [number, number, number];
}

/** Point cloud metadata (fast to compute, no full file read for some formats) */
export interface PointCloudMetadata {
	/** File path */
	path: string;
	/** Detected format */
	format: PointCloudFormat;
	/** Total number of points */
	point_count: number;
	/** File size in bytes */
	file_size_bytes: number;
	/** Available attributes (e.g., ["position", "intensity", "rgb", "classification"]) */
	attributes: string[];
	/** 3D bounding box (may require scanning file for some formats) */
	bounds: Bounds3D | null;
	/** Whether file has intensity values */
	has_intensity: boolean;
	/** Whether file has RGB color */
	has_rgb: boolean;
	/** Whether file has classification (LAS-specific) */
	has_classification: boolean;
}

/** A chunk of point cloud data for streaming */
export interface PointCloudChunk {
	/** Chunk index (0-based) */
	chunk_index: number;
	/** Total number of chunks */
	total_chunks: number;
	/** Number of points in this chunk */
	point_count: number;
	/** Flat XYZ positions: [x0, y0, z0, x1, y1, z1, ...] */
	positions: number[];
	/** Optional intensity values (normalized 0-1) */
	intensities: number[] | null;
	/** Optional RGB colors as packed u32 (0xRRGGBB) */
	colors: number[] | null;
	/** Optional classification values (LAS-specific) */
	classifications: number[] | null;
}

/** Complete point cloud data (for small files loaded at once) */
export interface PointCloudData {
	/** Metadata about the point cloud */
	metadata: PointCloudMetadata;
	/** Flat XYZ positions: [x0, y0, z0, x1, y1, z1, ...] */
	positions: number[];
	/** Optional intensity values (normalized 0-1) */
	intensities: number[] | null;
	/** Optional RGB colors as packed u32 (0xRRGGBB) */
	colors: number[] | null;
	/** Optional classification values (LAS-specific) */
	classifications: number[] | null;
}

/** Decimation method for downsampling point clouds */
export type DecimationMethod =
	| { type: 'voxel_grid'; voxel_size: number }
	| { type: 'random'; keep_ratio: number }
	| { type: 'uniform'; step: number };

/** Options for format conversion */
export interface ConversionOptions {
	/** Target format */
	target_format: PointCloudFormat;
	/** Whether to preserve intensity values */
	preserve_intensity: boolean;
	/** Whether to preserve RGB colors */
	preserve_rgb: boolean;
	/** Whether to preserve classification */
	preserve_classification: boolean;
	/** Optional decimation to apply during conversion */
	decimation: DecimationMethod | null;
}

/** Configuration for streaming point cloud data */
export interface StreamConfig {
	/** Number of points per chunk (default: 100,000) */
	chunk_size: number;
}

/**
 * Helper to convert flat positions array to Float32Array.
 * Rust sends Vec<f32> as JSON number[], we need Float32Array for Three.js.
 */
export function toFloat32Array(arr: number[]): Float32Array {
	return new Float32Array(arr);
}

/**
 * Helper to convert packed u32 colors to Uint8Array RGB.
 * Colors come as 0xRRGGBB, we need [r, g, b, r, g, b, ...] for Three.js.
 */
export function unpackColors(packed: number[]): Uint8Array {
	const result = new Uint8Array(packed.length * 3);
	for (let i = 0; i < packed.length; i++) {
		const color = packed[i];
		result[i * 3] = (color >> 16) & 0xff; // R
		result[i * 3 + 1] = (color >> 8) & 0xff; // G
		result[i * 3 + 2] = color & 0xff; // B
	}
	return result;
}

/**
 * Helper to convert packed u32 colors to Float32Array normalized RGB.
 * Returns [r, g, b, ...] in 0-1 range for Three.js vertex colors.
 */
export function unpackColorsNormalized(packed: number[]): Float32Array {
	const result = new Float32Array(packed.length * 3);
	for (let i = 0; i < packed.length; i++) {
		const color = packed[i];
		result[i * 3] = ((color >> 16) & 0xff) / 255; // R
		result[i * 3 + 1] = ((color >> 8) & 0xff) / 255; // G
		result[i * 3 + 2] = (color & 0xff) / 255; // B
	}
	return result;
}
