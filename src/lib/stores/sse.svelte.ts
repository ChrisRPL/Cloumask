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
	ThinkingEventData,
	PlanEventData,
	ToolProgressEventData,
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
			const data = event.data as ThinkingEventData;
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
			break;
		}

		case 'plan_approved': {
			agent.setPhase('executing');
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
			execution.setCurrentStep(data.step_id);
			pipeline.updateStep(data.step_id, { status: 'running' });
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
			agent.setStreaming(false);
			break;
		}

		case 'step_complete': {
			const data = event.data as StepEventData;
			const status = data.status === 'completed' ? 'completed' : 'failed';
			pipeline.updateStep(data.step_id, { status });
			execution.incrementProcessed();
			break;
		}

		case 'checkpoint': {
			const data = event.data as CheckpointEventData;
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
				qualityMetrics: data.quality_metrics as {
					averageConfidence: number;
					errorCount: number;
					totalProcessed: number;
					processingSpeed: number;
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
			execution.setStatus(data.success ? 'completed' : 'failed');
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
