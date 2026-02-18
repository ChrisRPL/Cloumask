import { render, screen } from '@testing-library/svelte';
import { beforeEach, describe, expect, it } from 'vitest';
import { DEFAULT_SETTINGS } from './settings.svelte';
import SettingsStateProbe from './test-fixtures/SettingsStateProbe.svelte';

const STORAGE_KEY = 'cloumask:settings';

describe('settings theme defaults', () => {
	beforeEach(() => {
		localStorage.removeItem(STORAGE_KEY);
	});

	it('uses light theme in defaults', () => {
		expect(DEFAULT_SETTINGS.theme).toBe('light');
	});

	it('initializes light theme when storage is empty', () => {
		render(SettingsStateProbe);
		expect(screen.getByTestId('theme').textContent).toBe('light');
	});

	it('keeps explicit stored theme', () => {
		localStorage.setItem(STORAGE_KEY, JSON.stringify({ theme: 'dark' }));
		render(SettingsStateProbe);
		expect(screen.getByTestId('theme').textContent).toBe('dark');
	});
});
