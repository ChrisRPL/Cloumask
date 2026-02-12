/**
 * SSE connection manager with auto-reconnection.
 *
 * Handles browser EventSource connection to the Python sidecar backend,
 * with exponential backoff for reconnection and heartbeat timeout detection.
 */

import type { SSEEvent, SSEEventType, ConnectionState, SSEConnectionInfo } from '$lib/types/sse';

// ============================================================================
// Configuration
// ============================================================================

export interface SSEManagerConfig {
	/** Base URL for the sidecar (default: http://127.0.0.1:8765) */
	baseUrl: string;
	/** Maximum reconnection attempts before giving up */
	maxReconnectAttempts: number;
	/** Initial delay between reconnection attempts in ms */
	initialReconnectDelayMs: number;
	/** Maximum delay between reconnection attempts in ms */
	maxReconnectDelayMs: number;
	/** Time to wait for heartbeat before reconnecting (30s interval + buffer) */
	heartbeatTimeoutMs: number;
}

/** Maximum reconnection attempts before giving up */
export const MAX_RECONNECT_ATTEMPTS = 10;

const DEFAULT_CONFIG: SSEManagerConfig = {
	baseUrl: 'http://127.0.0.1:8765',
	maxReconnectAttempts: MAX_RECONNECT_ATTEMPTS,
	initialReconnectDelayMs: 1000,
	maxReconnectDelayMs: 30000,
	heartbeatTimeoutMs: 60000 // 30s heartbeat interval + 30s buffer
};

// ============================================================================
// Types
// ============================================================================

export type SSEEventHandler = (event: SSEEvent) => void;
export type ConnectionStateHandler = (info: SSEConnectionInfo) => void;

// All SSE event types for listener registration
const ALL_EVENT_TYPES: SSEEventType[] = [
	'message',
	'thinking',
	'plan',
	'plan_approved',
	'tool_start',
	'tool_progress',
	'tool_result',
	'checkpoint',
	'await_input',
	'step_start',
	'step_complete',
	'complete',
	'error',
	'warning',
	'connected',
	'heartbeat'
];

// ============================================================================
// SSE Manager Class
// ============================================================================

/**
 * Manages SSE connection to the Python sidecar backend.
 *
 * Features:
 * - Auto-reconnection with exponential backoff
 * - Heartbeat timeout detection
 * - Event handler registration per type or wildcard
 * - Connection state tracking
 */
export class SSEManager {
	private config: SSEManagerConfig;
	private eventSource: EventSource | null = null;
	private threadId: string | null = null;
	private reconnectAttempts = 0;
	private reconnectTimeoutId: ReturnType<typeof setTimeout> | null = null;
	private heartbeatTimeoutId: ReturnType<typeof setTimeout> | null = null;
	private eventHandlers: Map<SSEEventType | '*', Set<SSEEventHandler>> = new Map();
	private stateHandlers: Set<ConnectionStateHandler> = new Set();
	private connectionState: ConnectionState = 'disconnected';
	private lastHeartbeat: string | null = null;
	private lastError: string | null = null;

	constructor(config: Partial<SSEManagerConfig> = {}) {
		this.config = { ...DEFAULT_CONFIG, ...config };
	}

	// -------------------------------------------------------------------------
	// Public API
	// -------------------------------------------------------------------------

	/**
	 * Connect to SSE stream for a thread.
	 * Closes any existing connection first.
	 */
	connect(threadId: string): void {
		if (this.eventSource) {
			this.disconnect();
		}

		this.threadId = threadId;
		this.reconnectAttempts = 0;
		this.attemptConnection();
	}

	/**
	 * Disconnect and cleanup all resources.
	 */
	disconnect(): void {
		this.clearTimeouts();

		if (this.eventSource) {
			this.eventSource.close();
			this.eventSource = null;
		}

		this.threadId = null;
		this.updateState('disconnected');
	}

	/**
	 * Register event handler for specific event type or all events ("*").
	 * Returns unsubscribe function.
	 */
	on(eventType: SSEEventType | '*', handler: SSEEventHandler): () => void {
		if (!this.eventHandlers.has(eventType)) {
			this.eventHandlers.set(eventType, new Set());
		}
		this.eventHandlers.get(eventType)!.add(handler);

		return () => {
			this.eventHandlers.get(eventType)?.delete(handler);
		};
	}

	/**
	 * Register one-time event handler.
	 */
	once(eventType: SSEEventType, handler: SSEEventHandler): () => void {
		const wrapper: SSEEventHandler = (event) => {
			unsubscribe();
			handler(event);
		};
		const unsubscribe = this.on(eventType, wrapper);
		return unsubscribe;
	}

	/**
	 * Remove all handlers for a specific event type.
	 */
	off(eventType: SSEEventType | '*'): void {
		this.eventHandlers.delete(eventType);
	}

	/**
	 * Register connection state change handler.
	 * Immediately called with current state.
	 * Supports multiple handlers (unlike single-handler pattern).
	 */
	onStateChange(handler: ConnectionStateHandler): () => void {
		this.stateHandlers.add(handler);
		// Immediately call with current state
		handler(this.getConnectionInfo());
		return () => {
			this.stateHandlers.delete(handler);
		};
	}

	/**
	 * Get current connection info.
	 */
	getConnectionInfo(): SSEConnectionInfo {
		return {
			state: this.connectionState,
			threadId: this.threadId,
			lastHeartbeat: this.lastHeartbeat,
			reconnectAttempts: this.reconnectAttempts,
			error: this.lastError
		};
	}

	/**
	 * Check if connected.
	 */
	isConnected(): boolean {
		return this.connectionState === 'connected';
	}

	/**
	 * Get the stream URL for a thread.
	 */
	getStreamUrl(threadId: string): string {
		return `${this.config.baseUrl}/api/chat/stream/${threadId}`;
	}

	// -------------------------------------------------------------------------
	// Private Methods
	// -------------------------------------------------------------------------

	private attemptConnection(): void {
		if (!this.threadId) return;

		this.updateState(this.reconnectAttempts === 0 ? 'connecting' : 'reconnecting');

		const url = this.getStreamUrl(this.threadId);

		try {
			this.eventSource = new EventSource(url);
			this.setupEventSourceHandlers();
		} catch (error) {
			this.handleConnectionError(error);
		}
	}

	private setupEventSourceHandlers(): void {
		if (!this.eventSource) return;

		this.eventSource.onopen = () => {
			// Connection opened, but wait for CONNECTED event to confirm
			this.resetHeartbeatTimeout();
		};

		this.eventSource.onerror = () => {
			// EventSource error - could be network issue or server closed
			this.handleConnectionError(new Error('EventSource connection error'));
		};

		// Register handlers for all event types
		for (const eventType of ALL_EVENT_TYPES) {
			this.eventSource.addEventListener(eventType, (e: MessageEvent) => {
				this.handleEvent(eventType, e);
			});
		}
	}

	private handleEvent(eventType: SSEEventType, e: MessageEvent): void {
		try {
			const rawData = typeof e.data === 'string' ? e.data.trim() : '';
			if (!rawData) {
				// Native EventSource "error" events do not include payload data.
				if (eventType === 'error') {
					return;
				}
				console.warn(`[SSE] Received empty payload for event: ${eventType}`);
				return;
			}

			const event: SSEEvent = JSON.parse(rawData);

			// Handle connection events
			if (eventType === 'connected') {
				this.updateState('connected');
				this.reconnectAttempts = 0;
				this.lastError = null;
			}

			// Reset heartbeat timeout on any event
			this.resetHeartbeatTimeout();

			if (eventType === 'heartbeat') {
				this.lastHeartbeat = event.timestamp;
			}

			// Dispatch to handlers
			this.dispatchEvent(event);
		} catch (error) {
			console.error(`[SSE] Failed to parse event: ${eventType}`, error);
		}
	}

	private dispatchEvent(event: SSEEvent): void {
		// Type-specific handlers
		const typeHandlers = this.eventHandlers.get(event.type as SSEEventType);
		if (typeHandlers) {
			for (const handler of typeHandlers) {
				try {
					handler(event);
				} catch (error) {
					console.error(`[SSE] Handler error for ${event.type}:`, error);
				}
			}
		}

		// Wildcard handlers
		const wildcardHandlers = this.eventHandlers.get('*');
		if (wildcardHandlers) {
			for (const handler of wildcardHandlers) {
				try {
					handler(event);
				} catch (error) {
					console.error(`[SSE] Wildcard handler error:`, error);
				}
			}
		}
	}

	private handleConnectionError(error: unknown): void {
		this.eventSource?.close();
		this.eventSource = null;

		// Extract error message
		const errorMessage = error instanceof Error ? error.message : 'Connection failed';
		this.lastError = errorMessage;

		// Check if we should retry
		if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
			this.scheduleReconnect();
		} else {
			this.updateState('error');
		}
	}

	private scheduleReconnect(): void {
		this.reconnectAttempts++;

		// Exponential backoff with jitter
		const delay = Math.min(
			this.config.initialReconnectDelayMs * Math.pow(2, this.reconnectAttempts - 1),
			this.config.maxReconnectDelayMs
		);
		const jitter = delay * 0.2 * Math.random();
		const finalDelay = delay + jitter;

		this.updateState('reconnecting');

		this.reconnectTimeoutId = setTimeout(() => {
			this.attemptConnection();
		}, finalDelay);
	}

	private resetHeartbeatTimeout(): void {
		if (this.heartbeatTimeoutId) {
			clearTimeout(this.heartbeatTimeoutId);
		}

		this.heartbeatTimeoutId = setTimeout(() => {
			// No heartbeat received, connection may be stale
			console.warn('[SSE] Heartbeat timeout, reconnecting...');
			this.handleConnectionError(new Error('Heartbeat timeout'));
		}, this.config.heartbeatTimeoutMs);
	}

	private clearTimeouts(): void {
		if (this.reconnectTimeoutId) {
			clearTimeout(this.reconnectTimeoutId);
			this.reconnectTimeoutId = null;
		}
		if (this.heartbeatTimeoutId) {
			clearTimeout(this.heartbeatTimeoutId);
			this.heartbeatTimeoutId = null;
		}
	}

	private updateState(state: ConnectionState): void {
		if (this.connectionState !== state) {
			this.connectionState = state;
			const info = this.getConnectionInfo();
			for (const handler of this.stateHandlers) {
				try {
					handler(info);
				} catch (error) {
					console.error('[SSE] State handler error:', error);
				}
			}
		}
	}
}

// ============================================================================
// Singleton Instance
// ============================================================================

let sseManagerInstance: SSEManager | null = null;

/**
 * Get the singleton SSE manager instance.
 * Creates one if it doesn't exist.
 */
export function getSSEManager(config?: Partial<SSEManagerConfig>): SSEManager {
	if (!sseManagerInstance) {
		sseManagerInstance = new SSEManager(config);
	}
	return sseManagerInstance;
}

/**
 * Reset the singleton instance (useful for testing).
 */
export function resetSSEManager(): void {
	if (sseManagerInstance) {
		sseManagerInstance.disconnect();
		sseManagerInstance = null;
	}
}
