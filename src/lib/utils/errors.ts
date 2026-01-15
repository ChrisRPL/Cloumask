/**
 * Error handling utilities.
 *
 * Provides user-friendly error messages and error type definitions
 * for consistent error handling across the application.
 */

// ============================================================================
// Error Types
// ============================================================================

/**
 * Base application error with code for categorization.
 */
export class AppError extends Error {
	constructor(
		message: string,
		public readonly code: string,
		public readonly recoverable: boolean = true,
		public readonly details?: unknown
	) {
		super(message);
		this.name = 'AppError';
	}
}

/**
 * Network-related errors (connection, timeout, etc.).
 */
export class NetworkError extends AppError {
	constructor(
		message: string,
		public readonly statusCode?: number
	) {
		super(message, 'NETWORK_ERROR', true);
		this.name = 'NetworkError';
	}
}

/**
 * Validation errors for user input.
 */
export class ValidationError extends AppError {
	constructor(
		message: string,
		public readonly field?: string
	) {
		super(message, 'VALIDATION_ERROR', true);
		this.name = 'ValidationError';
	}
}

/**
 * Tauri IPC errors.
 */
export class IPCError extends AppError {
	constructor(
		message: string,
		code: string = 'IPC_ERROR',
		details?: unknown
	) {
		super(message, code, true, details);
		this.name = 'IPCError';
	}
}

// ============================================================================
// Error Messages
// ============================================================================

/**
 * User-friendly error messages mapped by error code.
 */
export const ERROR_MESSAGES: Record<string, string> = {
	// Connection errors
	CONNECTION_FAILED: 'Unable to connect to backend. Is the application running?',
	CONNECTION_TIMEOUT: 'Connection timed out. Retrying...',
	CONNECTION_LOST: 'Lost connection to backend. Attempting to reconnect...',
	MAX_RETRIES: 'Connection failed after multiple attempts. Please restart the application.',
	HEARTBEAT_TIMEOUT: 'Connection appears stale. Reconnecting...',

	// Sidecar errors
	SIDECAR_NOT_RUNNING: 'Python backend is not running. Please restart the application.',
	SIDECAR_UNHEALTHY: 'Backend service is degraded. Some features may not work.',
	SIDECAR_SPAWN_FAILED: 'Failed to start backend service. Check the logs for details.',

	// Agent errors
	AGENT_ERROR: 'An error occurred while processing your request.',
	PLAN_INVALID: 'The generated plan contains errors. Please try again.',
	EXECUTION_FAILED: 'Pipeline execution failed. Check the error details.',

	// Thread errors
	THREAD_NOT_FOUND: 'Chat session not found. Starting a new session...',
	THREAD_EXPIRED: 'Your session has expired. Starting a new session...',
	THREAD_CREATE_FAILED: 'Failed to create chat session. Please try again.',

	// Network errors
	NETWORK_ERROR: 'A network error occurred. Please check your connection.',
	REQUEST_TIMEOUT: 'Request timed out. The operation is taking longer than expected.',
	SERVER_ERROR: 'Server error occurred. Please try again later.',

	// Generic
	UNKNOWN_ERROR: 'An unexpected error occurred.',
	VALIDATION_ERROR: 'Please check your input and try again.'
};

// ============================================================================
// Error Utilities
// ============================================================================

/**
 * Get user-friendly error message for an error.
 */
export function getUserFriendlyMessage(error: unknown, fallback?: string): string {
	// Handle our custom errors
	if (error instanceof AppError) {
		return ERROR_MESSAGES[error.code] || error.message;
	}

	// Handle standard errors with common messages
	if (error instanceof Error) {
		const message = error.message.toLowerCase();

		if (message.includes('timeout')) {
			return ERROR_MESSAGES.REQUEST_TIMEOUT;
		}
		if (message.includes('network') || message.includes('fetch')) {
			return ERROR_MESSAGES.NETWORK_ERROR;
		}
		if (message.includes('not found') || message.includes('404')) {
			return ERROR_MESSAGES.THREAD_NOT_FOUND;
		}

		return error.message;
	}

	// Handle string errors
	if (typeof error === 'string') {
		return error;
	}

	return fallback || ERROR_MESSAGES.UNKNOWN_ERROR;
}

/**
 * Get error code from an error object.
 */
export function getErrorCode(error: unknown): string {
	if (error instanceof AppError) {
		return error.code;
	}
	if (error instanceof Error) {
		return 'ERROR';
	}
	return 'UNKNOWN_ERROR';
}

/**
 * Check if an error is recoverable (can retry).
 */
export function isRecoverable(error: unknown): boolean {
	if (error instanceof AppError) {
		return error.recoverable;
	}

	// Network errors are usually recoverable
	if (error instanceof Error) {
		const message = error.message.toLowerCase();
		return (
			message.includes('timeout') ||
			message.includes('network') ||
			message.includes('connection')
		);
	}

	return true;
}

/**
 * Wrap an async function with error handling.
 */
export async function withErrorHandling<T>(
	fn: () => Promise<T>,
	errorHandler?: (error: unknown) => void
): Promise<T | null> {
	try {
		return await fn();
	} catch (error) {
		errorHandler?.(error);
		return null;
	}
}

/**
 * Create a retry wrapper for async operations.
 */
export async function withRetry<T>(
	fn: () => Promise<T>,
	options: {
		maxAttempts?: number;
		delayMs?: number;
		backoff?: boolean;
		onRetry?: (attempt: number, error: unknown) => void;
	} = {}
): Promise<T> {
	const { maxAttempts = 3, delayMs = 1000, backoff = true, onRetry } = options;

	let lastError: unknown;

	for (let attempt = 1; attempt <= maxAttempts; attempt++) {
		try {
			return await fn();
		} catch (error) {
			lastError = error;

			if (attempt < maxAttempts) {
				onRetry?.(attempt, error);
				const delay = backoff ? delayMs * Math.pow(2, attempt - 1) : delayMs;
				await new Promise((resolve) => setTimeout(resolve, delay));
			}
		}
	}

	throw lastError;
}
