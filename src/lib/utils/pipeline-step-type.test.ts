import { describe, expect, it } from 'vitest';
import { defaultToolNameForStepType, inferStepType, isPointcloudToolName } from './pipeline-step-type';

describe('inferStepType', () => {
	it('maps known tool names to specific UI step types', () => {
		expect(inferStepType('detect')).toBe('detection');
		expect(inferStepType('segment')).toBe('segmentation');
		expect(inferStepType('anonymize')).toBe('anonymization');
		expect(inferStepType('export')).toBe('export');
		expect(inferStepType('label_qa')).toBe('classification');
	});

	it('supports alternate tool names and falls back to custom', () => {
		expect(inferStepType('detect_3d')).toBe('detection');
		expect(inferStepType('anonymize_3d')).toBe('anonymization');
		expect(inferStepType('anonymize_pointcloud')).toBe('anonymization');
		expect(inferStepType('convert_format')).toBe('export');
		expect(inferStepType('scan_directory')).toBe('custom');
	});
});

describe('isPointcloudToolName', () => {
	it('detects pointcloud-oriented tool names', () => {
		expect(isPointcloudToolName('detect_3d')).toBe(true);
		expect(isPointcloudToolName('process_pointcloud')).toBe(true);
		expect(isPointcloudToolName('anonymize_pointcloud')).toBe(true);
		expect(isPointcloudToolName('detect')).toBe(false);
	});
});

describe('defaultToolNameForStepType', () => {
	it('maps step types to executable backend tools', () => {
		expect(defaultToolNameForStepType('detection')).toBe('detect');
		expect(defaultToolNameForStepType('anonymization')).toBe('anonymize');
		expect(defaultToolNameForStepType('classification')).toBe('label_qa');
		expect(defaultToolNameForStepType('custom')).toBe('run_script');
	});

	it('uses pointcloud variants when requested', () => {
		expect(defaultToolNameForStepType('detection', { preferPointcloud: true })).toBe('detect_3d');
		expect(defaultToolNameForStepType('anonymization', { preferPointcloud: true })).toBe(
			'anonymize_pointcloud'
		);
	});
});
