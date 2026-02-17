/**
 * SSE connection state management using Svelte 5 runes and context.
 *
 * Manages SSE connection lifecycle and routes events to appropriate stores.
 * This store acts as the central hub for real-time backend communication.
 */

import { getContext, setContext } from 'svelte';
import type {
	SSEConnectionInfo,
	ConnectionState,
	SSEEvent,
	MessageEventData,
	PlanEventData,
	ToolProgressEventData,
	ToolResultEventData,
	CheckpointEventData,
	AwaitInputEventData,
	StepEventData,
	PipelineCompleteEventData,
	ErrorEventData
} from '$lib/types/sse';
import { getSSEManager } from '$lib/utils/sse';
import { inferStepType } from '$lib/utils/pipeline-step-type';
import type { AgentState } from './agent.svelte';
import type { ExecutionState } from './execution.svelte';
import type { PipelineState, PipelineStep } from './pipeline.svelte';

// Re-export types
export type { SSEConnectionInfo, ConnectionState };

// ============================================================================
// Constants
// ============================================================================

const SSE_STATE_KEY = Symbol('sse-state');

// ============================================================================
// State Interface
// ============================================================================

export interface SSEState {
	readonly connectionInfo: SSEConnectionInfo;
	readonly isConnected: boolean;
	readonly isConnecting: boolean;
	readonly hasError: boolean;

	// Actions
	connect(threadId: string): void;
	disconnect(): void;
	reconnect(): void;

	// Store binding
	bindStores(stores: {
		agent: AgentState;
		execution: ExecutionState;
		pipeline: PipelineState;
	}): void;
}

// ============================================================================
// State Factory
// ============================================================================

/**
 * Creates SSE state using Svelte 5 runes.
 * Call this at the root layout to initialize the state.
 */
export function createSSEState(): SSEState {
	const manager = getSSEManager();

	// Reactive state
	let connectionInfo = $state<SSEConnectionInfo>(manager.getConnectionInfo());

	// Store references for event routing
	let boundStores: {
		agent: AgentState;
		execution: ExecutionState;
		pipeline: PipelineState;
	} | null = null;

	// Derived values
	const isConnected = $derived(connectionInfo.state === 'connected');
	const isConnecting = $derived(
		connectionInfo.state === 'connecting' || connectionInfo.state === 'reconnecting'
	);
	const hasError = $derived(connectionInfo.state === 'error');

	// Subscribe to state changes
	// $effect cleanup automatically handles unsubscription
	$effect(() => {
		const unsubscribe = manager.onStateChange((info) => {
			connectionInfo = info;
		});

		return unsubscribe;
	});

	// Setup event routing
	$effect(() => {
		const unsubscribe = manager.on('*', (event) => {
			if (boundStores) {
				routeEventToStores(event, boundStores);
			}
		});

		return unsubscribe;
	});

	return {
		// Getters
		get connectionInfo() {
			return connectionInfo;
		},
		get isConnected() {
			return isConnected;
		},
		get isConnecting() {
			return isConnecting;
		},
		get hasError() {
			return hasError;
		},

		// Actions
		connect(threadId: string) {
			manager.connect(threadId);
		},

		disconnect() {
			manager.disconnect();
		},

		reconnect() {
			if (connectionInfo.threadId) {
				manager.connect(connectionInfo.threadId);
			}
		},

		// Store binding
		bindStores(stores) {
			boundStores = stores;
		}
	};
}

// ============================================================================
// Event Routing
// ============================================================================

/**
 * Routes SSE events to appropriate stores.
 * Handles the snake_case to camelCase conversion for Python data.
 */
function routeEventToStores(
	event: SSEEvent,
	stores: {
		agent: AgentState;
		execution: ExecutionState;
		pipeline: PipelineState;
	}
): void {
	const { agent, execution, pipeline } = stores;

	switch (event.type) {
		case 'message': {
			const data = event.data as MessageEventData;
			if (data.content && data.role !== 'user') {
				agent.addMessage({
					role: data.role,
					content: data.content
				});

				// Some flows emit assistant messages without a terminal "complete" event.
				// Clear streaming/thinking state so the UI does not remain stuck.
				agent.setStreaming(false);
				if (agent.phase === 'understanding' || agent.phase === 'planning') {
					agent.setPhase('idle');
				}
			}
			break;
		}

			case 'thinking': {
				agent.setPhase('understanding');
				agent.setStreaming(true);
				// Could update a thinking message here if needed
			break;
		}

		case 'plan': {
			const data = event.data as PlanEventData;
			agent.setPhase('planning');
			agent.setStreaming(false);
			pipeline.setPipelineId(data.plan_id);

			// Convert Python snake_case to TypeScript camelCase
			const steps: PipelineStep[] = data.steps.map((step, index) => ({
				id: step.id,
				toolName: step.tool_name,
				type: inferStepType(step.tool_name),
				description: step.description,
				config: { params: step.parameters || {} },
				status: 'pending' as const,
				order: index
			}));

			pipeline.setSteps(steps);
			execution.updateProgress(0, data.total_steps);
			break;
		}

		case 'plan_approved': {
			agent.setPhase('executing');
			agent.setClarification(null);
			break;
		}

		case 'await_input': {
			const data = event.data as AwaitInputEventData;
			agent.setStreaming(false);
			agent.setClarification({
				id: crypto.randomUUID(),
				prompt: data.prompt,
				options: data.options,
				inputType: data.input_type
			});
			break;
		}

		case 'step_start': {
			const data = event.data as StepEventData;
			agent.setPhase('executing');
			execution.setStatus('running');
			execution.clearCheckpoint();
			execution.setCurrentStep(data.step_id);
			pipeline.updateStep(data.step_id, { status: 'running' });
			execution.updateProgress(data.step_index, pipeline.steps.length || execution.progress.total);
			break;
		}

		case 'tool_start': {
			agent.setStreaming(true);
			break;
		}

		case 'tool_progress': {
			const data = event.data as ToolProgressEventData;
			execution.updateProgress(data.current, data.total);
			break;
		}

		case 'tool_result': {
			const data = event.data as ToolResultEventData;
			agent.setStreaming(false);
			applyToolResultToExecution(data, execution);
			break;
		}

		case 'step_complete': {
			const data = event.data as StepEventData;
			const status = data.status === 'completed' ? 'completed' : 'failed';
			pipeline.updateStep(data.step_id, { status });
			execution.updateProgress(data.step_index + 1, pipeline.steps.length || execution.progress.total);
			break;
		}

		case 'checkpoint': {
			const data = event.data as CheckpointEventData;
			const quality = data.quality_metrics as Record<string, unknown>;
			agent.setPhase('checkpoint');
			execution.setCheckpoint({
				id: data.checkpoint_id,
				stepIndex: data.step_index,
				triggerReason: data.trigger_reason as
					| 'percentage'
					| 'quality_drop'
					| 'error_rate'
					| 'critical_step',
				progressPercent: data.progress_percent,
				qualityMetrics: {
					averageConfidence:
						numberFromResult(quality.average_confidence) ??
						numberFromResult(quality.averageConfidence) ??
						0,
					errorCount:
						numberFromResult(quality.error_count) ?? numberFromResult(quality.errorCount) ?? 0,
					totalProcessed:
						numberFromResult(quality.total_processed) ??
						numberFromResult(quality.totalProcessed) ??
						0,
					processingSpeed:
						numberFromResult(quality.processing_speed) ??
						numberFromResult(quality.processingSpeed) ??
						0
				},
				message: data.message,
				createdAt: event.timestamp
			});
			break;
		}

		case 'complete': {
			const data = event.data as PipelineCompleteEventData;
			agent.setPhase('complete');
			agent.setStreaming(false);
			agent.setClarification(null);
			execution.clearCheckpoint();
			execution.setStatus(data.success ? 'completed' : 'failed');
			execution.updateProgress(data.completed_steps, data.total_steps);
			break;
		}

		case 'error': {
			const data = event.data as ErrorEventData;
			agent.setError(data.message);
			if (!data.recoverable) {
				execution.setStatus('failed');
			}
			break;
		}

		case 'warning': {
			// Could show a toast notification here
			console.warn('[SSE] Warning:', (event.data as { message: string }).message);
			break;
		}

		case 'connected': {
			agent.setConnected(true);
			break;
		}

		case 'heartbeat': {
			// Heartbeat handled by SSE manager, nothing to do here
			break;
		}
	}
}

function applyToolResultToExecution(data: ToolResultEventData, execution: ExecutionState): void {
	if (!data.success) {
		execution.addError({
			stepId: `step-${data.step_index}`,
			message: data.error ?? 'Tool execution failed',
			recoverable: true
		});
		return;
	}

	const result = data.result ?? {};
	const filesProcessed = numberFromResult(result.files_processed) ?? numberFromResult(result.total_files);
	if (typeof filesProcessed === 'number' && filesProcessed > 0) {
		execution.updateStats({
			processed: Math.max(execution.stats.processed, filesProcessed)
		});
	}

	const detectedCount = numberFromResult(result.count);
	if (typeof detectedCount === 'number' && detectedCount > 0 && data.tool_name === 'detect') {
		execution.updateStats({
			detected: Math.max(execution.stats.detected, detectedCount)
		});
	}

	const faces = numberFromResult(result.faces_anonymized) ?? numberFromResult(result.faces_blurred) ?? 0;
	const plates =
		numberFromResult(result.plates_anonymized) ?? numberFromResult(result.plates_blurred) ?? 0;
	const anonymized = faces + plates;
	if (anonymized > 0) {
		execution.updateStats({
			flagged: Math.max(execution.stats.flagged, anonymized)
		});
	}

	const previews = extractPreviewItems(data);
	if (previews.length > 0) {
		execution.appendPreviews(previews);
	}
}

function numberFromResult(value: unknown): number | null {
	if (typeof value === 'number' && Number.isFinite(value)) return value;
	if (typeof value === 'string') {
		const parsed = Number(value);
		return Number.isFinite(parsed) ? parsed : null;
	}
	return null;
}

function clamp01(value: number): number {
	return Math.max(0, Math.min(1, value));
}

const IMAGE_EXTENSIONS = new Set([
	'.jpg',
	'.jpeg',
	'.png',
	'.bmp',
	'.tif',
	'.tiff',
	'.webp'
]);

function isImagePath(path: string): boolean {
	const lower = path.toLowerCase();
	for (const ext of IMAGE_EXTENSIONS) {
		if (lower.endsWith(ext)) return true;
	}
	return false;
}

function collectPreviewPaths(result: Record<string, unknown>): string[] {
	const paths: string[] = [];
	const maybePush = (value: unknown) => {
		if (typeof value !== 'string') return;
		if (!value.trim()) return;
		if (!isImagePath(value)) return;
		paths.push(value);
	};

	maybePush(result.image_path);
	maybePush(result.output_path);

	const imagePathKeys = ['sample_images', 'sample_files', 'output_files', 'output_images', 'image_paths'];
	for (const key of imagePathKeys) {
		const arr = result[key];
		if (!Array.isArray(arr)) continue;
		for (const item of arr) maybePush(item);
	}

	const resultRows = result.results;
	if (Array.isArray(resultRows)) {
		for (const row of resultRows) {
			if (typeof row !== 'object' || !row) continue;
			const candidate = (row as Record<string, unknown>).image_path;
			maybePush(candidate);
		}
	}

	const previewRows = result.preview_items;
	if (Array.isArray(previewRows)) {
		for (const row of previewRows) {
			if (typeof row !== 'object' || !row) continue;
			const candidate = (row as Record<string, unknown>).image_path;
			maybePush(candidate);
		}
	}

	return Array.from(new Set(paths));
}

type PreviewAnnotation = import('$lib/types/execution').PreviewAnnotation;
type PreviewItem = import('$lib/types/execution').PreviewItem;
type PreviewStatus = PreviewItem['status'];

function normalizeBBox(
	raw: unknown,
	options: { assumeCenter?: boolean } = {}
): PreviewAnnotation['bbox'] | null {
	if (typeof raw !== 'object' || raw === null) return null;
	const record = raw as Record<string, unknown>;

	const x1 = numberFromResult(record.x1);
	const y1 = numberFromResult(record.y1);
	const x2 = numberFromResult(record.x2);
	const y2 = numberFromResult(record.y2);
	if (x1 !== null && y1 !== null && x2 !== null && y2 !== null) {
		const x = clamp01(Math.min(x1, x2));
		const y = clamp01(Math.min(y1, y2));
		const width = clamp01(Math.abs(x2 - x1));
		const height = clamp01(Math.abs(y2 - y1));
		return {
			x,
			y,
			width: Math.min(width, 1 - x),
			height: Math.min(height, 1 - y)
		};
	}

	const width = clamp01(numberFromResult(record.width) ?? numberFromResult(record.w) ?? 0);
	const height = clamp01(numberFromResult(record.height) ?? numberFromResult(record.h) ?? 0);
	if (width <= 0 || height <= 0) return null;

	const hasCenterKeys = record.cx !== undefined || record.cy !== undefined;
	const assumeCenter = options.assumeCenter === true || hasCenterKeys;

	const rawX =
		numberFromResult(record.x) ?? numberFromResult(record.cx) ?? (assumeCenter ? 0.5 : 0.0);
	const rawY =
		numberFromResult(record.y) ?? numberFromResult(record.cy) ?? (assumeCenter ? 0.5 : 0.0);

	const x = assumeCenter ? rawX - width / 2 : rawX;
	const y = assumeCenter ? rawY - height / 2 : rawY;
	const clampedX = clamp01(x);
	const clampedY = clamp01(y);

	return {
		x: clampedX,
		y: clampedY,
		width: Math.min(width, 1 - clampedX),
		height: Math.min(height, 1 - clampedY)
	};
}

function normalizeAnnotation(raw: unknown, options: { assumeCenterBBox?: boolean } = {}): PreviewAnnotation | null {
	if (typeof raw !== 'object' || raw === null) return null;
	const record = raw as Record<string, unknown>;
	const label = (typeof record.label === 'string' ? record.label : record.class_name) ?? 'object';
	const confidence = clamp01(numberFromResult(record.confidence) ?? 1);
	const bbox = normalizeBBox(record.bbox ?? record, { assumeCenter: options.assumeCenterBBox });
	if (!bbox) return null;

	return {
		label: String(label),
		confidence,
		bbox
	};
}

function getPreviewStatus(result: Record<string, unknown>): PreviewStatus {
	const faces = numberFromResult(result.faces_anonymized) ?? numberFromResult(result.faces_blurred) ?? 0;
	const plates =
		numberFromResult(result.plates_anonymized) ?? numberFromResult(result.plates_blurred) ?? 0;
	return faces + plates > 0 ? 'flagged' : 'processed';
}

function previewItem(
	data: ToolResultEventData,
	imagePath: string,
	index: number,
	status: PreviewStatus,
	annotations: PreviewAnnotation[]
): PreviewItem {
	return {
		id: `${data.tool_name}-${data.step_index}-${index}-${imagePath}`,
		imagePath,
		thumbnailUrl: imagePath,
		annotations,
		status
	};
}

function extractAnnotationsFromRow(row: Record<string, unknown>): PreviewAnnotation[] {
	const annotations: PreviewAnnotation[] = [];

	const explicitAnnotations = row.annotations;
	if (Array.isArray(explicitAnnotations)) {
		for (const raw of explicitAnnotations) {
			const parsed = normalizeAnnotation(raw);
			if (parsed) annotations.push(parsed);
		}
	}

	if (annotations.length === 0 && Array.isArray(row.detections)) {
		for (const raw of row.detections) {
			const parsed = normalizeAnnotation(raw, { assumeCenterBBox: true });
			if (parsed) annotations.push(parsed);
		}
	}

	return annotations;
}

export function extractPreviewItems(data: ToolResultEventData): PreviewItem[] {
	const result = data.result ?? {};
	const status = getPreviewStatus(result);

	const previewRows = result.preview_items;
	if (Array.isArray(previewRows)) {
		const items: PreviewItem[] = [];
		for (const row of previewRows.slice(0, 6)) {
			if (typeof row !== 'object' || row === null) continue;
			const record = row as Record<string, unknown>;
			const imagePath = typeof record.image_path === 'string' ? record.image_path : null;
			if (!imagePath || !isImagePath(imagePath)) continue;
			const annotations = extractAnnotationsFromRow(record);
			items.push(previewItem(data, imagePath, items.length, status, annotations));
		}
		if (items.length > 0) return items;
	}

	const rowAnnotationsByPath = new Map<string, PreviewAnnotation[]>();
	const resultRows = result.results;
	if (Array.isArray(resultRows)) {
		for (const row of resultRows) {
			if (typeof row !== 'object' || row === null) continue;
			const record = row as Record<string, unknown>;
			const imagePath = typeof record.image_path === 'string' ? record.image_path : null;
			if (!imagePath) continue;
			rowAnnotationsByPath.set(imagePath, extractAnnotationsFromRow(record));
		}
	}

	const imagePaths = collectPreviewPaths(result);
	if (imagePaths.length === 0) return [];

	return imagePaths.slice(0, 6).map((imagePath, index) =>
		previewItem(data, imagePath, index, status, rowAnnotationsByPath.get(imagePath) ?? [])
	);
}

// ============================================================================
// Context Helpers
// ============================================================================

/**
 * Initialize SSE state and set it in Svelte context.
 * Call this in the root +layout.svelte.
 */
export function setSSEState(): SSEState {
	const state = createSSEState();
	setContext(SSE_STATE_KEY, state);
	return state;
}

/**
 * Get SSE state from Svelte context.
 * Call this in child components that need SSE state.
 */
export function getSSEState(): SSEState {
	return getContext<SSEState>(SSE_STATE_KEY);
}
