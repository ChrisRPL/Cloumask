import { vi } from 'vitest';
import { mockInvoke, mockListen, mockOpen, mockSave } from '$lib/test-utils';

class ResizeObserverMock {
	observe() {}
	unobserve() {}
	disconnect() {}
}

class EventSourceMock {
	url: string;
	readyState = 1;
	onopen: ((this: EventSource, ev: Event) => unknown) | null = null;
	onerror: ((this: EventSource, ev: Event) => unknown) | null = null;

	constructor(url: string) {
		this.url = url;
	}

	addEventListener() {}
	removeEventListener() {}
	close() {
		this.readyState = 2;
	}
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
Object.defineProperty(globalThis, 'EventSource', { value: EventSourceMock });
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

Object.defineProperty(navigator, 'clipboard', {
	value: {
		writeText: vi.fn(),
		readText: vi.fn(),
	},
});

HTMLCanvasElement.prototype.getContext = vi.fn((contextId: unknown) => {
	if (contextId === '2d') {
		return mockCanvasContext as unknown as CanvasRenderingContext2D;
	}
	return null;
}) as unknown as HTMLCanvasElement['getContext'];

Object.defineProperty(Element.prototype, 'scrollIntoView', {
	value: vi.fn(),
	writable: true,
});

vi.mock('@tauri-apps/api/core', () => ({
	invoke: mockInvoke,
	convertFileSrc: (path: string) => `tauri://localhost/${path}`,
}));

vi.mock('@tauri-apps/api/event', () => ({
	listen: mockListen,
}));

vi.mock('@tauri-apps/plugin-dialog', () => ({
	open: mockOpen,
	save: mockSave,
}));

vi.mock('@tauri-apps/plugin-shell', () => ({
	open: vi.fn(),
}));
