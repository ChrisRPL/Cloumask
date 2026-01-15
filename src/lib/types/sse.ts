/**
 * SSE event types matching backend/src/backend/api/streaming/events.py
 *
 * These types mirror the Python SSEEventType enum and data structures
 * to ensure type-safe event handling across the IPC boundary.
 */

// ============================================================================
// Event Types
// ============================================================================

/**
 * SSE event types matching Python SSEEventType enum.
 */
export type SSEEventType =
	// Agent state events
	| 'message'
	| 'thinking'
	| 'plan'
	| 'plan_approved'
	// Tool events
	| 'tool_start'
	| 'tool_progress'
	| 'tool_result'
	// Checkpoint events
	| 'checkpoint'
	| 'await_input'
	// Pipeline events
	| 'step_start'
	| 'step_complete'
	| 'complete'
	// Error events
	| 'error'
	| 'warning'
	// Connection events
	| 'connected'
	| 'heartbeat';

// ============================================================================
// Base Event Structure
// ============================================================================

/**
 * Base SSE event structure matching Python SSEEvent dataclass.
 */
export interface SSEEvent<T = unknown> {
	type: SSEEventType;
	timestamp: string;
	data: T;
}

// ============================================================================
// Event Data Payloads
// ============================================================================

/**
 * Data for MESSAGE events.
 * Mirrors Python MessageEventData.
 */
export interface MessageEventData {
	role: 'user' | 'assistant' | 'system';
	content: string;
	message_id?: string;
}

/**
 * Data for THINKING events.
 */
export interface ThinkingEventData {
	message: string;
}

/**
 * Plan step structure within PLAN events.
 */
export interface PlanStep {
	id: string;
	tool_name: string;
	description: string;
	parameters?: Record<string, unknown>;
}

/**
 * Data for PLAN events.
 * Mirrors Python PlanEventData.
 */
export interface PlanEventData {
	plan_id: string;
	steps: PlanStep[];
	total_steps: number;
}

/**
 * Data for TOOL_START events.
 * Mirrors Python ToolStartEventData.
 */
export interface ToolStartEventData {
	tool_name: string;
	step_index: number;
	parameters: Record<string, unknown>;
}

/**
 * Data for TOOL_PROGRESS events.
 * Mirrors Python ToolProgressEventData.
 */
export interface ToolProgressEventData {
	tool_name: string;
	step_index: number;
	current: number;
	total: number;
	message: string;
	percentage: number;
}

/**
 * Data for TOOL_RESULT events.
 * Mirrors Python ToolResultEventData.
 */
export interface ToolResultEventData {
	tool_name: string;
	step_index: number;
	success: boolean;
	result?: Record<string, unknown>;
	error?: string;
	duration_seconds: number;
}

/**
 * Data for CHECKPOINT events.
 * Mirrors Python CheckpointEventData.
 */
export interface CheckpointEventData {
	checkpoint_id: string;
	step_index: number;
	trigger_reason: string;
	progress_percent: number;
	quality_metrics: Record<string, unknown>;
	message: string;
}

/**
 * Input types for AWAIT_INPUT events.
 */
export type AwaitInputType = 'plan_approval' | 'checkpoint_approval' | 'clarification';

/**
 * Data for AWAIT_INPUT events.
 * Mirrors Python AwaitInputEventData.
 */
export interface AwaitInputEventData {
	input_type: AwaitInputType;
	prompt: string;
	options?: string[];
}

/**
 * Data for STEP_START and STEP_COMPLETE events.
 * Mirrors Python StepEventData.
 */
export interface StepEventData {
	step_index: number;
	step_id: string;
	tool_name: string;
	description: string;
	status: string;
}

/**
 * Data for PIPELINE_COMPLETE events (type: "complete").
 * Mirrors Python PipelineCompleteEventData.
 */
export interface PipelineCompleteEventData {
	pipeline_id: string;
	success: boolean;
	total_steps: number;
	completed_steps: number;
	failed_steps: number;
	duration_seconds: number;
	summary: string;
}

/**
 * Data for ERROR events.
 * Mirrors Python ErrorEventData.
 */
export interface ErrorEventData {
	error_code: string;
	message: string;
	details?: Record<string, unknown>;
	recoverable: boolean;
}

/**
 * Data for WARNING events.
 */
export interface WarningEventData {
	message: string;
	details?: Record<string, unknown>;
}

/**
 * Data for CONNECTED events.
 * Mirrors Python ConnectedEventData.
 */
export interface ConnectedEventData {
	thread_id: string;
	timestamp: string;
}

/**
 * Data for HEARTBEAT events.
 * Mirrors Python HeartbeatEventData.
 */
export interface HeartbeatEventData {
	sequence: number;
	timestamp: string;
}

// ============================================================================
// Connection State
// ============================================================================

/**
 * Connection state for the SSE manager.
 */
export type ConnectionState =
	| 'disconnected'
	| 'connecting'
	| 'connected'
	| 'reconnecting'
	| 'error';

/**
 * Information about the current SSE connection.
 */
export interface SSEConnectionInfo {
	state: ConnectionState;
	threadId: string | null;
	lastHeartbeat: string | null;
	reconnectAttempts: number;
	error: string | null;
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Type guard for MESSAGE events.
 */
export function isMessageEvent(event: SSEEvent): event is SSEEvent<MessageEventData> {
	return event.type === 'message';
}

/**
 * Type guard for PLAN events.
 */
export function isPlanEvent(event: SSEEvent): event is SSEEvent<PlanEventData> {
	return event.type === 'plan';
}

/**
 * Type guard for TOOL_PROGRESS events.
 */
export function isToolProgressEvent(event: SSEEvent): event is SSEEvent<ToolProgressEventData> {
	return event.type === 'tool_progress';
}

/**
 * Type guard for CHECKPOINT events.
 */
export function isCheckpointEvent(event: SSEEvent): event is SSEEvent<CheckpointEventData> {
	return event.type === 'checkpoint';
}

/**
 * Type guard for ERROR events.
 */
export function isErrorEvent(event: SSEEvent): event is SSEEvent<ErrorEventData> {
	return event.type === 'error';
}
