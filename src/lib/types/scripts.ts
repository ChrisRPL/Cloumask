/**
 * Type definitions for AI-powered script generation.
 *
 * These types mirror the backend API schemas for script generation,
 * validation, and management.
 */

/**
 * Request to generate a new script from natural language.
 */
export interface GenerateScriptRequest {
	/** Natural language description of what the script should do */
	prompt: string;
	/** Optional: specific model to use (defaults to devstral-2:123b-cloud) */
	model?: string;
	/** Optional context for refinement */
	context?: {
		/** Existing code to refine */
		existing_code?: string;
		/** Whether this is a refinement request */
		refinement?: boolean;
	};
}

/**
 * Response from script generation.
 */
export interface GenerateScriptResponse {
	/** Generated Python script code */
	script: string;
	/** Model that was used for generation */
	model: string;
	/** Optional explanation of what the script does */
	explanation?: string;
}

/**
 * Request to validate a script.
 */
export interface ValidateScriptRequest {
	/** Script content to validate */
	content: string;
}

/**
 * Validation error or warning.
 */
export interface ValidationIssue {
	/** Error/warning message */
	message: string;
	/** Line number (if applicable) */
	line?: number;
	/** Column number (if applicable) */
	column?: number;
}

/**
 * Response from script validation.
 */
export interface ValidateScriptResponse {
	/** Whether the script is valid */
	valid: boolean;
	/** List of validation errors */
	errors: ValidationIssue[];
	/** List of validation warnings */
	warnings: ValidationIssue[];
}

/**
 * Request to save a script.
 */
export interface SaveScriptRequest {
	/** Script name (without .py extension) */
	name: string;
	/** Script content */
	content: string;
	/** Optional description */
	description?: string;
	/** Whether to overwrite existing file */
	overwrite?: boolean;
}

/**
 * Response from saving a script.
 */
export interface SaveScriptResponse {
	/** Full path where script was saved */
	path: string;
	/** Script name */
	name: string;
	/** Whether a new file was created (vs updated) */
	created: boolean;
}

/**
 * Script metadata for listing.
 */
export interface ScriptInfo {
	/** Script name (without extension) */
	name: string;
	/** Full path to script */
	path: string;
	/** Last modified timestamp */
	modified_at: number;
	/** File size in bytes */
	size_bytes: number;
}

/**
 * Response from listing scripts.
 */
export interface ListScriptsResponse {
	/** List of available scripts */
	scripts: ScriptInfo[];
}

/**
 * Response from loading a script.
 */
export interface LoadScriptResponse {
	/** Script name */
	name: string;
	/** Script content */
	content: string;
	/** Full path */
	path: string;
}

/**
 * State for the script builder UI.
 */
export interface ScriptBuilderState {
	/** Current prompt text */
	prompt: string;
	/** Generated/edited code */
	code: string;
	/** Whether generation is in progress */
	isGenerating: boolean;
	/** Whether validation is in progress */
	isValidating: boolean;
	/** Whether saving is in progress */
	isSaving: boolean;
	/** Validation result */
	validation: ValidateScriptResponse | null;
	/** Error message (if any) */
	error: string | null;
	/** Success message (if any) */
	success: string | null;
	/** Script name for saving */
	scriptName: string;
	/** Optional description */
	description: string;
}

/**
 * Default state for the script builder.
 */
export const DEFAULT_SCRIPT_BUILDER_STATE: ScriptBuilderState = {
	prompt: '',
	code: '',
	isGenerating: false,
	isValidating: false,
	isSaving: false,
	validation: null,
	error: null,
	success: null,
	scriptName: '',
	description: ''
};
