import { render, fireEvent, waitFor } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import PointCloudViewerTestHarness from './PointCloudViewerTestHarness.svelte';
import {
	mockInvoke,
	mockOpen,
	mockSave,
	createMockPointCloudData,
	createMockMetadata,
	createMockSceneContext,
} from '$lib/test-utils';

const mockSceneContext = createMockSceneContext();

vi.mock('$lib/utils/three', async () => {
	const actual = await vi.importActual<typeof import('$lib/utils/three')>('$lib/utils/three');
	return {
		...actual,
		createScene: vi.fn(() => mockSceneContext),
	};
});

describe('PointCloudViewer', () => {
	beforeEach(() => {
		mockInvoke.mockReset();
		mockOpen.mockReset();
		mockSave.mockReset();
	});

	it('opens settings modal from header action', async () => {
		const { getByLabelText, getByText } = render(PointCloudViewerTestHarness);
		await fireEvent.click(getByLabelText('Open settings'));
		expect(getByText('Viewer Settings')).toBeTruthy();
	});

	it('exports point cloud using convert command', async () => {
		mockOpen.mockResolvedValue('/data/mock.pcd');
		mockSave.mockResolvedValue('/data/output.ply');
		const metadata = createMockMetadata({ point_count: 3 });
		const data = createMockPointCloudData({ metadata });

		mockInvoke.mockImplementation((command: string) => {
			if (command === 'read_pointcloud_metadata') {
				return Promise.resolve(metadata);
			}
			if (command === 'read_pointcloud') {
				return Promise.resolve(data);
			}
			if (command === 'convert_pointcloud') {
				return Promise.resolve(metadata);
			}
			return Promise.reject(new Error(`Unknown command: ${command}`));
		});

		const { getByText } = render(PointCloudViewerTestHarness);

		await fireEvent.click(getByText('Load'));

		await waitFor(() => {
			expect(mockInvoke).toHaveBeenCalledWith('read_pointcloud_metadata', {
				path: '/data/mock.pcd',
			});
		});

		await waitFor(() => {
			const exportButton = getByText('Export');
			expect(exportButton).toBeTruthy();
		});

		await fireEvent.click(getByText('Export'));

		await waitFor(() => {
			expect(mockInvoke).toHaveBeenCalledWith('convert_pointcloud', {
				input_path: '/data/mock.pcd',
				output_path: '/data/output.ply',
				options: {
					target_format: 'ply',
					preserve_intensity: true,
					preserve_rgb: true,
					preserve_classification: true,
					decimation: null,
				},
			});
		});
	});

	it('auto-loads selected pointcloud preview from execution state', async () => {
		const metadata = createMockMetadata({
			path: '/data/detect-output.pcd',
			point_count: 3,
		});
		const data = createMockPointCloudData({ metadata });

		mockInvoke.mockImplementation((command: string, payload?: { path?: string }) => {
			if (command === 'read_pointcloud_metadata') {
				return Promise.resolve({
					...metadata,
					path: payload?.path ?? metadata.path,
				});
			}
			if (command === 'read_pointcloud') {
				return Promise.resolve({
					...data,
					metadata: {
						...metadata,
						path: payload?.path ?? metadata.path,
					},
				});
			}
			if (command === 'convert_pointcloud') {
				return Promise.resolve(metadata);
			}
			return Promise.reject(new Error(`Unknown command: ${command}`));
		});

		const preview = {
			id: 'detect_3d-0-/data/detect-output.pcd',
			imagePath: '/data/detect-output.pcd',
			thumbnailUrl: '/data/detect-output.pcd',
			assetType: 'pointcloud' as const,
			annotations: [],
			pointcloudAnnotations: [
				{
					id: 'det-0-car',
					className: 'car',
					confidence: 0.91,
					center: [1, 2, 3] as [number, number, number],
					size: [4, 2, 1] as [number, number, number],
					yaw: 0.25,
					status: 'pending' as const,
				},
			],
			status: 'flagged' as const,
		};

		render(PointCloudViewerTestHarness, {
			selectedPointcloudPreview: preview,
		});

		await waitFor(() => {
			expect(mockInvoke).toHaveBeenCalledWith('read_pointcloud_metadata', {
				path: '/data/detect-output.pcd',
			});
		});

		await waitFor(() => {
			expect(mockInvoke).toHaveBeenCalledWith('read_pointcloud', {
				path: '/data/detect-output.pcd',
			});
		});

		await waitFor(() => {
			expect(document.body.textContent).toContain('detect-output.pcd');
		});
	});
});
