/**
 * TypeScript types for Tauri IPC commands.
 *
 * These types match the Rust structs returned by Tauri commands.
 * Keep in sync with src-tauri/src/commands/*.rs
 */

// ============================================================================
// Sidecar Commands
// ============================================================================

/**
 * Status information for the Python sidecar process.
 * Returned by `sidecar_status` command.
 */
export interface SidecarStatus {
	/** Whether the sidecar process is currently running. */
	running: boolean;
	/** The base URL for the sidecar API (e.g., "http://127.0.0.1:8765"). */
	url: string;
	/** The port the sidecar is configured to use. */
	port: number;
}

// ============================================================================
// System Commands
// ============================================================================

/**
 * Application information.
 * Returned by `get_app_info` command.
 */
export interface AppInfo {
	/** Application name. */
	name: string;
	/** Application version from Cargo.toml. */
	version: string;
	/** Target platform (e.g., "macos", "windows", "linux"). */
	platform: string;
	/** Target architecture (e.g., "x86_64", "aarch64"). */
	arch: string;
	/** Whether running in debug mode. */
	debug: boolean;
}

// ============================================================================
// Command Type Helpers
// ============================================================================

/**
 * Type-safe command names for Tauri invoke.
 */
export type CommandName =
	// Sidecar commands
	| 'sidecar_status'
	| 'start_sidecar'
	| 'stop_sidecar'
	| 'restart_sidecar'
	// System commands
	| 'get_app_info'
	| 'ping'
	| 'echo'
	// Legacy
	| 'greet';

/**
 * Map command names to their return types.
 */
export interface CommandReturnTypes {
	sidecar_status: SidecarStatus;
	start_sidecar: void;
	stop_sidecar: void;
	restart_sidecar: void;
	get_app_info: AppInfo;
	ping: string;
	echo: string;
	greet: string;
}

/**
 * Map command names to their argument types.
 */
export interface CommandArgs {
	sidecar_status: Record<string, never>;
	start_sidecar: Record<string, never>;
	stop_sidecar: Record<string, never>;
	restart_sidecar: Record<string, never>;
	get_app_info: Record<string, never>;
	ping: Record<string, never>;
	echo: { message: string };
	greet: { name: string };
}
