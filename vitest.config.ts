import { defineConfig } from 'vitest/config';
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	test: {
		environment: 'jsdom',
		globals: true,
		setupFiles: ['./src/lib/test-utils/setup.ts'],
		include: ['src/**/*.test.ts'],
		deps: {
			inline: ['@sveltejs/kit'],
		},
	},
});
