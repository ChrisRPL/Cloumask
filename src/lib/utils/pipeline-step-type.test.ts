import { describe, expect, it } from 'vitest';
import { inferStepType } from './pipeline-step-type';

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
		expect(inferStepType('convert_format')).toBe('export');
		expect(inferStepType('scan_directory')).toBe('custom');
	});
});
