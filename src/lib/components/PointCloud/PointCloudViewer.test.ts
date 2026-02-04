import { render, fireEvent, waitFor } from '@testing-library/svelte';
import { vi } from 'vitest';
import PointCloudViewer from './PointCloudViewer.svelte';
import { mockInvoke, mockOpen, mockSave, createMockPointCloudData, createMockMetadata } from '$lib/test-utils';

describe('PointCloudViewer', () => {
	beforeEach(() => {
		mockInvoke.mockReset();
		mockOpen.mockReset();
		mockSave.mockReset();
	});

	it('opens settings modal from header action', async () => {
		const { getByLabelText, getByText } = render(PointCloudViewer);
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

		const { getByText } = render(PointCloudViewer);

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
});
