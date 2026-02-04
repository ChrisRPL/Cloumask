import { render } from '@testing-library/svelte';
import { vi } from 'vitest';
import ThreeCanvas from './ThreeCanvas.svelte';
import { createMockSceneContext } from '$lib/test-utils';

const mockSceneContext = createMockSceneContext();

vi.mock('$lib/utils/three', async () => {
	const actual = await vi.importActual<typeof import('$lib/utils/three')>('$lib/utils/three');
	return {
		...actual,
		createScene: vi.fn(() => mockSceneContext),
	};
});

vi.mock('$lib/stores/pointcloud.svelte', () => {
	const state = {
		pointSize: 2,
		colorMode: 'height',
		showGrid: true,
		showAxes: true,
		showBoundingBoxes: false,
		showLabels: true,
		lodEnabled: false,
		lodPointBudget: 500000,
		backgroundColor: '#0c3b1f',
		boundingBoxes: [],
		selectedBoxId: null,
		setSelectedBoxId: vi.fn(),
		setSelection: vi.fn(),
		updateCamera: vi.fn(),
	};
	return { getPointCloudState: () => state };
});

describe('ThreeCanvas', () => {
	it('renders a canvas and initializes the scene', () => {
		const onReady = vi.fn();
		const { container } = render(ThreeCanvas, { props: { onReady } });
		expect(container.querySelector('canvas')).toBeTruthy();
		expect(onReady).toHaveBeenCalled();
		expect(mockSceneContext.renderer.setAnimationLoop).toHaveBeenCalled();
	});
});
