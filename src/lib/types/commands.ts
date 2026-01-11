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
// Health Check Types
// ============================================================================

/**
 * Health check response from Python sidecar.
 * Returned by `check_health` command.
 */
export interface HealthResponse {
	/** Current health status: "healthy", "degraded", or "unhealthy". */
	status: 'healthy' | 'degraded' | 'unhealthy';
	/** Backend version. */
	version: string;
	/** ISO 8601 timestamp. */
	timestamp: string;
	/** Status of individual components. */
	components: Record<string, string>;
}

/**
 * Readiness check response from Python sidecar.
 * Returned by `check_ready` command.
 */
export interface ReadyResponse {
	/** Whether the sidecar is ready to accept requests. */
	ready: boolean;
	/** Individual readiness checks. */
	checks: Record<string, boolean>;
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
	// Sidecar lifecycle commands
	| 'sidecar_status'
	| 'start_sidecar'
	| 'stop_sidecar'
	| 'restart_sidecar'
	// Health check commands
	| 'check_health'
	| 'check_ready'
	// Generic sidecar HTTP commands
	| 'call_sidecar_get'
	| 'call_sidecar_post'
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
	// Sidecar lifecycle
	sidecar_status: SidecarStatus;
	start_sidecar: void;
	stop_sidecar: void;
	restart_sidecar: void;
	// Health check
	check_health: HealthResponse;
	check_ready: ReadyResponse;
	// Generic sidecar HTTP
	call_sidecar_get: unknown;
	call_sidecar_post: unknown;
	// System
	get_app_info: AppInfo;
	ping: string;
	echo: string;
	greet: string;
}

/**
 * Map command names to their argument types.
 */
export interface CommandArgs {
	// Sidecar lifecycle
	sidecar_status: Record<string, never>;
	start_sidecar: Record<string, never>;
	stop_sidecar: Record<string, never>;
	restart_sidecar: Record<string, never>;
	// Health check
	check_health: Record<string, never>;
	check_ready: Record<string, never>;
	// Generic sidecar HTTP
	call_sidecar_get: { endpoint: string };
	call_sidecar_post: { endpoint: string; body: unknown };
	// System
	get_app_info: Record<string, never>;
	ping: Record<string, never>;
	echo: { message: string };
	greet: { name: string };
}
