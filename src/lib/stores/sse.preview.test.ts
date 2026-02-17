import { describe, expect, it } from 'vitest';
import type { ToolResultEventData } from '$lib/types/sse';
import { extractPreviewItems } from './sse.svelte';

function makeToolResultEventData(
	overrides: Partial<ToolResultEventData> = {}
): ToolResultEventData {
	return {
		tool_name: 'detect',
		step_index: 1,
		success: true,
		result: {},
		error: undefined,
		duration_seconds: 0,
		...overrides
	};
}

describe('extractPreviewItems', () => {
	it('uses preview_items annotations when provided by backend', () => {
		const eventData = makeToolResultEventData({
			result: {
				preview_items: [
					{
						image_path: '/tmp/img1.jpg',
						annotations: [
							{
								label: 'person',
								confidence: 0.95,
								bbox: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 }
							}
						]
					}
				]
			}
		});

		const previews = extractPreviewItems(eventData);

		expect(previews).toHaveLength(1);
		expect(previews[0].imagePath).toBe('/tmp/img1.jpg');
		expect(previews[0].annotations).toEqual([
			{
				label: 'person',
				confidence: 0.95,
				bbox: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 }
			}
		]);
	});

	it('derives annotations from result rows with center-format detections', () => {
		const eventData = makeToolResultEventData({
			result: {
				results: [
					{
						image_path: '/tmp/img2.jpg',
						detections: [
							{
								class_name: 'car',
								confidence: 0.9,
								bbox: { x: 0.5, y: 0.6, width: 0.2, height: 0.4 }
							}
						]
					}
				],
				sample_images: ['/tmp/img2.jpg']
			}
		});

		const previews = extractPreviewItems(eventData);

		expect(previews).toHaveLength(1);
		expect(previews[0].annotations).toHaveLength(1);
		expect(previews[0].annotations[0].label).toBe('car');
		expect(previews[0].annotations[0].confidence).toBe(0.9);
		expect(previews[0].annotations[0].bbox.x).toBeCloseTo(0.4, 6);
		expect(previews[0].annotations[0].bbox.y).toBeCloseTo(0.4, 6);
		expect(previews[0].annotations[0].bbox.width).toBeCloseTo(0.2, 6);
		expect(previews[0].annotations[0].bbox.height).toBeCloseTo(0.4, 6);
	});

	it('returns previews without annotations when only sample images are available', () => {
		const eventData = makeToolResultEventData({
			result: {
				sample_images: ['/tmp/img3.jpg']
			}
		});

		const previews = extractPreviewItems(eventData);

		expect(previews).toHaveLength(1);
		expect(previews[0].imagePath).toBe('/tmp/img3.jpg');
		expect(previews[0].annotations).toEqual([]);
	});
});
