import { describe, expect, it } from 'vitest';
import { createExecutionState } from './execution.svelte';
import type { PreviewItem } from '$lib/types/execution';

function makePointcloudPreview(): PreviewItem {
	return {
		id: 'preview-1',
		imagePath: '/tmp/sample.pcd',
		thumbnailUrl: '/tmp/sample.pcd',
		assetType: 'pointcloud',
		annotations: [],
		pointcloudAnnotations: [
			{
				id: 'ann-1',
				className: 'car',
				confidence: 0.75,
				center: [1, 2, 3],
				size: [4, 2, 1.5],
				yaw: 0,
				status: 'pending',
			},
		],
		status: 'flagged',
	};
}

describe('execution pointcloud annotation actions', () => {
	it('updates annotation status in previews and selected preview', () => {
		const state = createExecutionState();
		const preview = makePointcloudPreview();

		state.setPreviews([preview]);
		state.setSelectedPointcloudPreview(preview);
		state.setPointcloudAnnotationStatus(preview.id, 'ann-1', 'accepted');

		expect(state.previews[0].pointcloudAnnotations?.[0].status).toBe('accepted');
		expect(state.selectedPointcloudPreview?.pointcloudAnnotations?.[0].status).toBe('accepted');
	});

	it('marks preview as processed when all annotations are rejected', () => {
		const state = createExecutionState();
		const preview = makePointcloudPreview();

		state.setPreviews([preview]);
		state.setPointcloudAnnotationStatus(preview.id, 'ann-1', 'rejected');

		expect(state.previews[0].pointcloudAnnotations?.[0].status).toBe('rejected');
		expect(state.previews[0].status).toBe('processed');
	});

	it('edits annotation fields', () => {
		const state = createExecutionState();
		const preview = makePointcloudPreview();

		state.setPreviews([preview]);
		state.updatePointcloudAnnotation(preview.id, 'ann-1', {
			className: 'truck',
			confidence: 0.93,
			status: 'edited',
		});

		expect(state.previews[0].pointcloudAnnotations?.[0].className).toBe('truck');
		expect(state.previews[0].pointcloudAnnotations?.[0].confidence).toBe(0.93);
		expect(state.previews[0].pointcloudAnnotations?.[0].status).toBe('edited');
	});
});
