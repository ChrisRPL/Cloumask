import { vi } from 'vitest';
import { mockInvoke, mockListen, mockOpen, mockSave } from '$lib/test-utils';

class ResizeObserverMock {
	observe() {}
	unobserve() {}
	disconnect() {}
}

const mockCanvasContext = {
	canvas: document.createElement('canvas'),
	measureText: () => ({ width: 120 }),
	fillRect: vi.fn(),
	fillText: vi.fn(),
	beginPath: vi.fn(),
	moveTo: vi.fn(),
	lineTo: vi.fn(),
	quadraticCurveTo: vi.fn(),
	closePath: vi.fn(),
	fill: vi.fn(),
	scale: vi.fn(),
	setTransform: vi.fn(),
	textBaseline: '',
	font: '',
	fillStyle: '',
};

Object.defineProperty(globalThis, 'ResizeObserver', { value: ResizeObserverMock });
Object.defineProperty(window, 'matchMedia', {
	value: (query: string) => ({
		matches: false,
		media: query,
		onchange: null,
		addListener: vi.fn(),
		removeListener: vi.fn(),
		addEventListener: vi.fn(),
		removeEventListener: vi.fn(),
		dispatchEvent: vi.fn(),
	}),
});

HTMLCanvasElement.prototype.getContext = vi.fn((contextId: unknown) => {
	if (contextId === '2d') {
		return mockCanvasContext as unknown as CanvasRenderingContext2D;
	}
	return null;
}) as unknown as HTMLCanvasElement['getContext'];

vi.mock('@tauri-apps/api/core', () => ({
	invoke: mockInvoke,
}));

vi.mock('@tauri-apps/api/event', () => ({
	listen: mockListen,
}));

vi.mock('@tauri-apps/plugin-dialog', () => ({
	open: mockOpen,
	save: mockSave,
}));
