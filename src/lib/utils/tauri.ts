/**
 * Tauri IPC utilities with type-safe command invocation.
 *
 * Uses the existing types from commands.ts and maps to actual Rust command names.
 * Keep in sync with src-tauri/src/commands/*.rs
 */

import { invoke } from '@tauri-apps/api/core';
import type {
	CommandName,
	CommandReturnTypes,
	CommandArgs,
	AppInfo,
	HealthResponse,
	ReadyResponse,
	SidecarStatus,
	LLMStatus,
	LLMModelsResponse,
	LLMReadyResponse,
	IPCError,
} from '$lib/types/commands';

// ============================================================================
// Type-Safe Invoke Wrapper
// ============================================================================

/**
 * Type-safe wrapper around Tauri invoke.
 * Uses CommandReturnTypes and CommandArgs for compile-time safety.
 * Converts errors to user-friendly IPCError objects.
 */
async function invokeCommand<K extends CommandName>(
	command: K,
	args?: CommandArgs[K]
): Promise<CommandReturnTypes[K]> {
	try {
		return await invoke<CommandReturnTypes[K]>(command, args);
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		throw {
			command,
			message: `Command '${command}' failed`,
			details: message,
		} as IPCError;
	}
}

// ============================================================================
// System Commands
// ============================================================================

/** Test IPC connectivity with a simple ping. Returns "pong". */
export async function ping(): Promise<string> {
	return invokeCommand('ping');
}

/** Echo a message back (for testing). */
export async function echo(message: string): Promise<string> {
	return invokeCommand('echo', { message });
}

/** Get application information. */
export async function getAppInfo(): Promise<AppInfo> {
	return invokeCommand('get_app_info');
}

// ============================================================================
// Sidecar Commands
// ============================================================================

/**
 * Get the current status of the Python sidecar.
 * NOTE: Rust command is 'sidecar_status', not 'get_sidecar_status'
 */
export async function getSidecarStatus(): Promise<SidecarStatus> {
	return invokeCommand('sidecar_status');
}

/** Start the Python sidecar if not running. Non-blocking. */
export async function startSidecar(): Promise<void> {
	return invokeCommand('start_sidecar');
}

/** Stop the Python sidecar. */
export async function stopSidecar(): Promise<void> {
	return invokeCommand('stop_sidecar');
}

/** Restart the Python sidecar. Non-blocking. */
export async function restartSidecar(): Promise<void> {
	return invokeCommand('restart_sidecar');
}

// ============================================================================
// Health Check Commands
// ============================================================================

/** Check the health of the Python sidecar. */
export async function checkHealth(): Promise<HealthResponse> {
	return invokeCommand('check_health');
}

/** Check the readiness of the Python sidecar. */
export async function checkReady(): Promise<ReadyResponse> {
	return invokeCommand('check_ready');
}

// ============================================================================
// LLM Commands
// ============================================================================

/** Get the status of LLM service. */
export async function getLLMStatus(): Promise<LLMStatus> {
	return invokeCommand('get_llm_status');
}

/** List available LLM models. */
export async function listLLMModels(): Promise<LLMModelsResponse> {
	return invokeCommand('list_llm_models');
}

// ============================================================================
// Generic Sidecar HTTP Commands
// ============================================================================

/** Call a generic sidecar GET endpoint. */
export async function callSidecarGet<T = unknown>(endpoint: string): Promise<T> {
	return invokeCommand('call_sidecar_get', { endpoint }) as Promise<T>;
}

/** Call a generic sidecar POST endpoint. */
export async function callSidecarPost<T = unknown>(endpoint: string, body: unknown): Promise<T> {
	return invokeCommand('call_sidecar_post', { endpoint, body }) as Promise<T>;
}

// ============================================================================
// Window Control Commands (Tauri 2.0)
// ============================================================================

/**
 * Get the current Tauri window instance.
 * Returns null if not running in Tauri.
 */
export async function getTauriWindow() {
	if (!isTauri()) return null;
	const { getCurrentWindow } = await import('@tauri-apps/api/window');
	return getCurrentWindow();
}

/** Minimize the current window */
export async function minimizeWindow(): Promise<void> {
	const win = await getTauriWindow();
	if (win) await win.minimize();
}

/** Toggle maximize/restore for the current window */
export async function toggleMaximize(): Promise<void> {
	const win = await getTauriWindow();
	if (!win) return;

	const isMaximized = await win.isMaximized();
	if (isMaximized) {
		await win.unmaximize();
	} else {
		await win.maximize();
	}
}

/** Check if the current window is maximized */
export async function isWindowMaximized(): Promise<boolean> {
	const win = await getTauriWindow();
	if (!win) return false;
	return win.isMaximized();
}

/** Close the current window */
export async function closeWindow(): Promise<void> {
	const win = await getTauriWindow();
	if (win) await win.close();
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Check if we're running inside Tauri.
 * Useful for SSR compatibility and testing.
 */
export function isTauri(): boolean {
	return typeof window !== 'undefined' && '__TAURI__' in window;
}

/**
 * Wait for a condition with timeout.
 * @param condition - Async function that returns true when condition is met
 * @param options - timeout (ms) and polling interval (ms)
 * @returns true if condition met, false if timeout
 */
export async function waitFor(
	condition: () => Promise<boolean>,
	options: { timeout?: number; interval?: number } = {}
): Promise<boolean> {
	const { timeout = 10000, interval = 500 } = options;
	const start = Date.now();

	while (Date.now() - start < timeout) {
		try {
			if (await condition()) {
				return true;
			}
		} catch {
			// Condition threw, keep waiting
		}
		await new Promise((resolve) => setTimeout(resolve, interval));
	}

	return false;
}

/**
 * Wait for the sidecar to become healthy.
 * Polls checkHealth() until status === 'healthy' or timeout.
 */
export async function waitForSidecar(timeout = 10000): Promise<boolean> {
	return waitFor(
		async () => {
			const health = await checkHealth();
			return health.status === 'healthy';
		},
		{ timeout }
	);
}

/**
 * Wait for the sidecar to be ready (all checks passing).
 */
export async function waitForSidecarReady(timeout = 10000): Promise<boolean> {
	return waitFor(
		async () => {
			const ready = await checkReady();
			return ready.ready;
		},
		{ timeout }
	);
}

// ============================================================================
// Chat Thread API (Direct HTTP to Python Sidecar)
// ============================================================================

/** Base URL for the Python sidecar */
const SIDECAR_URL = 'http://127.0.0.1:8765';

/** Information about a chat thread */
export interface ThreadInfo {
	thread_id: string;
	created: boolean;
	awaiting_user: boolean;
	current_step: number;
	total_steps: number;
}

/** User decision for plan approval or checkpoint */
export type UserDecision = 'approve' | 'edit' | 'cancel' | 'retry';

/** Request to send a message to a chat thread */
export interface SendMessageRequest {
	content: string;
	decision?: UserDecision;
	plan_edits?: unknown[];
}

/** Response from sending a message */
export interface SendMessageResponse {
	status: string;
	thread_id: string;
	message_id?: string;
}

/**
 * Create a new chat thread.
 * Returns thread info with new thread_id.
 */
export async function createThread(): Promise<ThreadInfo> {
	const response = await fetch(`${SIDECAR_URL}/api/chat/thread/new`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' }
	});

	if (!response.ok) {
		throw new Error(`Failed to create thread: ${response.statusText}`);
	}

	return response.json();
}

/**
 * Get information about a chat thread.
 */
export async function getThreadInfo(threadId: string): Promise<ThreadInfo> {
	if (!threadId) {
		throw new Error('Invalid thread ID');
	}
	const response = await fetch(`${SIDECAR_URL}/api/chat/thread/${threadId}`);

	if (!response.ok) {
		throw new Error(`Failed to get thread: ${response.statusText}`);
	}

	return response.json();
}

/**
 * Send a message to a chat thread.
 * Triggers agent processing with SSE events streamed back.
 */
export async function sendMessage(
	threadId: string,
	request: SendMessageRequest
): Promise<SendMessageResponse> {
	if (!threadId) {
		throw new Error('Invalid thread ID');
	}
	const response = await fetch(`${SIDECAR_URL}/api/chat/send/${threadId}`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(request)
	});

	if (!response.ok) {
		throw new Error(`Failed to send message: ${response.statusText}`);
	}

	return response.json();
}

/**
 * Close a chat thread and cleanup resources.
 */
export async function closeThread(threadId: string): Promise<void> {
	if (!threadId) {
		throw new Error('Invalid thread ID');
	}
	const response = await fetch(`${SIDECAR_URL}/api/chat/thread/${threadId}`, {
		method: 'DELETE'
	});

	if (!response.ok) {
		throw new Error(`Failed to close thread: ${response.statusText}`);
	}
}

/**
 * Get the SSE stream URL for a thread.
 * Use with EventSource to receive real-time events.
 */
export function getStreamUrl(threadId: string): string {
	if (!threadId) {
		throw new Error('Invalid thread ID');
	}
	return `${SIDECAR_URL}/api/chat/stream/${threadId}`;
}

// ============================================================================
// LLM Readiness API (Direct HTTP to Python Sidecar)
// ============================================================================

/**
 * Check if LLM service is ready with the required model.
 * Does NOT automatically pull the model.
 */
export async function checkLLMReady(): Promise<LLMReadyResponse> {
	const response = await fetch(`${SIDECAR_URL}/llm/ensure-ready`);

	if (!response.ok) {
		throw new Error(`Failed to check LLM service: ${response.statusText}`);
	}

	return response.json();
}

/**
 * Ensure LLM service is ready, pulling the required model if needed.
 * WARNING: This may take several minutes for large models.
 */
export async function ensureLLMReady(): Promise<LLMReadyResponse> {
	const response = await fetch(`${SIDECAR_URL}/llm/ensure-ready`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' }
	});

	if (!response.ok) {
		throw new Error(`Failed to ensure LLM ready: ${response.statusText}`);
	}

	return response.json();
}

/**
 * Pull a specific model.
 * @param model - Model name (e.g., "qwen3:14b")
 */
export async function pullLLMModel(model: string): Promise<{ status: string; message: string }> {
	const response = await fetch(`${SIDECAR_URL}/llm/pull`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ model })
	});

	if (!response.ok) {
		throw new Error(`Failed to pull model: ${response.statusText}`);
	}

	return response.json();
}

/**
 * Wait for LLM service to be ready with the required model.
 * Polls checkLLMReady() until ready or timeout.
 */
export async function waitForLLMReady(timeout = 15000): Promise<LLMReadyResponse> {
	const start = Date.now();
	let lastResult: LLMReadyResponse | null = null;

	while (Date.now() - start < timeout) {
		try {
			lastResult = await checkLLMReady();
			if (lastResult.ready) {
				return lastResult;
			}
		} catch {
			// Keep trying
		}
		await new Promise((resolve) => setTimeout(resolve, 1000));
	}

	return lastResult ?? {
		ready: false,
		service_running: false,
		required_model: 'qwen3:14b',
		model_available: false,
		error: 'Timeout waiting for LLM service'
	};
}
