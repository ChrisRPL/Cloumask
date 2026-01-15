/**
 * API client for script generation and management.
 *
 * Communicates with the Python sidecar's /api/scripts endpoints
 * for AI-powered script generation, validation, and persistence.
 */

import type {
	GenerateScriptRequest,
	GenerateScriptResponse,
	ValidateScriptRequest,
	ValidateScriptResponse,
	SaveScriptRequest,
	SaveScriptResponse,
	ListScriptsResponse,
	LoadScriptResponse
} from '$lib/types/scripts.js';

/** Base URL for the Python sidecar API */
const API_BASE = 'http://localhost:8765';

/**
 * Custom error for API failures.
 */
export class ScriptApiError extends Error {
	constructor(
		message: string,
		public status: number,
		public detail?: string
	) {
		super(message);
		this.name = 'ScriptApiError';
	}
}

/**
 * Make a request to the scripts API.
 */
async function apiRequest<T>(
	endpoint: string,
	options: RequestInit = {}
): Promise<T> {
	const url = `${API_BASE}/api/scripts${endpoint}`;

	const response = await fetch(url, {
		...options,
		headers: {
			'Content-Type': 'application/json',
			...options.headers
		}
	});

	if (!response.ok) {
		let detail: string | undefined;
		try {
			const errorData = await response.json();
			detail = errorData.detail || errorData.message;
		} catch {
			// Response wasn't JSON
		}
		throw new ScriptApiError(
			`API request failed: ${response.statusText}`,
			response.status,
			detail
		);
	}

	return response.json();
}

/**
 * Generate a Python script from natural language description.
 *
 * Uses AI (Ollama with coding models) to generate a script that
 * conforms to the Cloumask custom step interface.
 *
 * @param request - Generation request with prompt and optional context
 * @returns Generated script code and explanation
 *
 * @example
 * ```ts
 * const { script, explanation } = await generateScript({
 *   prompt: "Convert all images to grayscale"
 * });
 * ```
 */
export async function generateScript(
	request: GenerateScriptRequest
): Promise<GenerateScriptResponse> {
	return apiRequest<GenerateScriptResponse>('/generate', {
		method: 'POST',
		body: JSON.stringify(request)
	});
}

/**
 * Validate a script for syntax and interface compliance.
 *
 * Checks that the script:
 * - Has valid Python syntax
 * - Contains a process() function with correct signature
 * - Uses only allowed imports
 *
 * @param content - Script content to validate
 * @returns Validation result with errors and warnings
 *
 * @example
 * ```ts
 * const { valid, errors, warnings } = await validateScript(code);
 * if (!valid) {
 *   console.error('Validation errors:', errors);
 * }
 * ```
 */
export async function validateScript(
	content: string
): Promise<ValidateScriptResponse> {
	const request: ValidateScriptRequest = { content };
	return apiRequest<ValidateScriptResponse>('/validate', {
		method: 'POST',
		body: JSON.stringify(request)
	});
}

/**
 * Save a script to the user's scripts directory.
 *
 * Scripts are saved to ~/.cloumask/scripts/ for use in pipelines.
 *
 * @param request - Save request with name, content, and optional description
 * @returns Save result with path and creation status
 *
 * @example
 * ```ts
 * const { path, created } = await saveScript({
 *   name: "grayscale_converter",
 *   content: scriptCode,
 *   description: "Converts images to grayscale"
 * });
 * ```
 */
export async function saveScript(
	request: SaveScriptRequest
): Promise<SaveScriptResponse> {
	return apiRequest<SaveScriptResponse>('/save', {
		method: 'POST',
		body: JSON.stringify(request)
	});
}

/**
 * List all saved scripts.
 *
 * Returns scripts from ~/.cloumask/scripts/ sorted by modification time.
 *
 * @returns List of script metadata
 */
export async function listScripts(): Promise<ListScriptsResponse> {
	return apiRequest<ListScriptsResponse>('/list', {
		method: 'GET'
	});
}

/**
 * Load a script by name.
 *
 * @param name - Script name (with or without .py extension)
 * @returns Script content and metadata
 *
 * @example
 * ```ts
 * const { content } = await loadScript("grayscale_converter");
 * ```
 */
export async function loadScript(name: string): Promise<LoadScriptResponse> {
	return apiRequest<LoadScriptResponse>(`/${encodeURIComponent(name)}`, {
		method: 'GET'
	});
}

/**
 * Delete a script by name.
 *
 * @param name - Script name (with or without .py extension)
 * @returns True if deleted, false if not found
 */
export async function deleteScript(name: string): Promise<{ deleted: boolean }> {
	return apiRequest<{ deleted: boolean }>(`/${encodeURIComponent(name)}`, {
		method: 'DELETE'
	});
}

/**
 * Generate a script name from a prompt.
 *
 * Creates a safe filename by extracting key words from the prompt.
 *
 * @param prompt - Natural language prompt
 * @returns Suggested script name (without extension)
 */
export function suggestScriptName(prompt: string): string {
	// Extract meaningful words
	const words = prompt
		.toLowerCase()
		.replace(/[^a-z0-9\s]/g, '')
		.split(/\s+/)
		.filter((word) => word.length > 2)
		.filter(
			(word) =>
				!['the', 'and', 'for', 'with', 'that', 'this', 'from', 'into'].includes(word)
		)
		.slice(0, 3);

	if (words.length === 0) {
		return 'custom_script';
	}

	return words.join('_');
}
