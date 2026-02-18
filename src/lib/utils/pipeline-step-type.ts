import type { StepType } from '$lib/types/pipeline';

/**
 * Infer a UI step type from backend tool name.
 */
export function inferStepType(toolName: string): StepType {
	const normalized = toolName.trim().toLowerCase();

	if (normalized === 'detect' || normalized === 'detect_3d') return 'detection';
	if (normalized === 'segment') return 'segmentation';
	if (normalized === 'anonymize' || normalized === 'anonymize_3d') return 'anonymization';
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
