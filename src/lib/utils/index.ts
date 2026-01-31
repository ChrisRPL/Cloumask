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
	// Window controls
	getTauriWindow,
	minimizeWindow,
	toggleMaximize,
	isWindowMaximized,
	closeWindow,
	// Utilities
	isTauri,
	waitFor,
	waitForSidecar,
	waitForSidecarReady,
} from './tauri';

// Re-export cn from the existing utils.ts for convenience
export { cn } from '../utils';

// Storage utilities
export {
	createStorageAdapter,
	isStorageAvailable,
	getStorageItem,
	setStorageItem,
	removeStorageItem,
	STORAGE_KEYS,
} from './storage';
export type { StorageAdapter } from './storage';

// Keyboard utilities
export {
	SEQUENCE_TIMEOUT,
	getPlatform,
	parseCombo,
	formatCombo,
	formatComboString,
	formatSequence,
	matchesCombo,
	isInputElement,
	generateShortcutId,
	normalizeCombo,
	isSequenceStartCandidate,
	eventToCombo,
	comboToString,
	fuzzyScore,
} from './keyboard';
