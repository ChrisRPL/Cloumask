import type { StepType } from '$lib/types/pipeline';

/**
 * Infer a UI step type from backend tool name.
 */
export function inferStepType(toolName: string): StepType {
	const normalized = toolName.trim().toLowerCase();

	if (normalized === 'detect' || normalized === 'detect_3d') return 'detection';
	if (normalized === 'segment') return 'segmentation';
	if (normalized === 'anonymize' || normalized === 'anonymize_3d' || normalized === 'anonymize_pointcloud') {
		return 'anonymization';
	}
	if (normalized === 'export' || normalized === 'convert_format') return 'export';
	if (normalized === 'label_qa') return 'classification';
	if (
		normalized === 'scan_directory' ||
		normalized === 'find_duplicates' ||
		normalized === 'split_dataset' ||
		normalized === 'review' ||
		normalized === 'pointcloud_stats' ||
		normalized === 'process_pointcloud' ||
		normalized === 'project_3d_to_2d' ||
		normalized === 'extract_rosbag'
	) {
		return 'utility';
	}

	return 'custom';
}

const POINTCLOUD_TOOL_NAMES = new Set([
	'pointcloud_stats',
	'process_pointcloud',
	'detect_3d',
	'project_3d_to_2d',
	'anonymize_pointcloud',
	'extract_rosbag'
]);

/**
 * Returns true when a tool belongs to the pointcloud workflow family.
 */
export function isPointcloudToolName(toolName: string): boolean {
	return POINTCLOUD_TOOL_NAMES.has(toolName.trim().toLowerCase());
}

/**
 * Resolve a backend tool name for a newly added UI step.
 * Uses pointcloud variants when the current plan is pointcloud-oriented.
 */
export function defaultToolNameForStepType(
	stepType: StepType,
	options?: { preferPointcloud?: boolean }
): string {
	const preferPointcloud = options?.preferPointcloud ?? false;

	switch (stepType) {
		case 'detection':
			return preferPointcloud ? 'detect_3d' : 'detect';
		case 'segmentation':
			return 'segment';
		case 'anonymization':
			return preferPointcloud ? 'anonymize_pointcloud' : 'anonymize';
		case 'export':
			return 'export';
		case 'classification':
			return 'label_qa';
		case 'custom':
			return 'run_script';
		default:
			return stepType;
	}
}
