/**
 * Utility functions for Cloumask frontend.
 */

// Tauri IPC utilities
export {
	// System commands
	ping,
	echo,
	getAppInfo,
	// Sidecar commands
	getSidecarStatus,
	startSidecar,
	stopSidecar,
	restartSidecar,
	// Health commands
	checkHealth,
	checkReady,
	// Generic HTTP
	callSidecarGet,
	callSidecarPost,
	// Utilities
	isTauri,
	waitFor,
	waitForSidecar,
	waitForSidecarReady,
} from './tauri';

// Re-export cn from the existing utils.ts for convenience
export { cn } from '../utils';
