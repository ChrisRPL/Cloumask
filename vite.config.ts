import { defineConfig } from 'vite';
import { sveltekit } from '@sveltejs/kit/vite';

const host = process.env.TAURI_DEV_HOST;

// https://vite.dev/config/
export default defineConfig({
	plugins: [sveltekit()],

	// Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
	clearScreen: false,

	// Environment variable prefixes to expose to the client
	envPrefix: ['VITE_', 'TAURI_'],

	server: {
		port: 5173,
		strictPort: true,
		host: host || false,
		hmr: host
			? {
					protocol: 'ws',
					host,
					port: 5174
				}
			: undefined,
		watch: {
			ignored: ['**/src-tauri/**']
		}
	},

	build: {
		// Tauri uses Chromium on Windows and WebKit on macOS and Linux
		target: process.env.TAURI_PLATFORM === 'windows' ? 'chrome105' : 'safari14',
		// Don't minify for debug builds
		minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,
		// Produce sourcemaps for debug builds
		sourcemap: !!process.env.TAURI_DEBUG
	}
});
