import { fireEvent, render, waitFor } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import AnnotationCanvas from './AnnotationCanvas.svelte';

const mockAnnotator = {
	setDrawingTool: vi.fn(),
	setAnnotations: vi.fn(),
	on: vi.fn(),
	setDrawingEnabled: vi.fn(),
	destroy: vi.fn(),
	cancelDrawing: vi.fn()
};

vi.mock('@annotorious/annotorious', () => ({
	createImageAnnotator: vi.fn(() => mockAnnotator)
}));

const FALLBACK_IMAGE =
	'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9fY0QAAAAASUVORK5CYII=';

describe('AnnotationCanvas', () => {
	it('uses fallback source and exits infinite loading when images fail', async () => {
		const { container, getByText, queryByText } = render(AnnotationCanvas, {
			props: {
				imageUrl: '/broken/source.png',
				fallbackImageUrl: FALLBACK_IMAGE,
				annotations: []
			}
		});

		const image = container.querySelector('img') as HTMLImageElement;
		expect(image).toBeTruthy();
		expect(image.getAttribute('src')).toBe('/broken/source.png');

		await fireEvent.error(image);
		await waitFor(() => {
			expect(image.getAttribute('src')).toBe(FALLBACK_IMAGE);
		});

		await fireEvent.error(image);
		await waitFor(() => {
			expect(getByText('Failed to load image')).toBeTruthy();
		});
		expect(queryByText('Loading image...')).toBeNull();
	});
});
