<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface ChatPanelProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { onMount } from 'svelte';
	import { getAgentState } from '$lib/stores/agent.svelte';
	import { extractPreviewItems, getSSEState } from '$lib/stores/sse.svelte';
	import { getPipelineState } from '$lib/stores/pipeline.svelte';
	import { getExecutionState } from '$lib/stores/execution.svelte';
	import { getUIState } from '$lib/stores/ui.svelte';
	import {
		createThread,
		listThreads,
		getThreadState,
		sendMessage,
		closeThread,
		checkLLMReady,
		ensureLLMReady,
		pullLLMModelWithProgress,
		startSidecar,
		waitForSidecar,
		DEFAULT_REQUIRED_MODEL,
		isTauri
	} from '$lib/utils/tauri';
	import { inferStepType } from '$lib/utils/pipeline-step-type';
	import type { LLMPullProgressEvent } from '$lib/utils/tauri';
	import type {
		PersistedThreadCheckpoint,
		PersistedThreadCheckpointQualityMetrics,
		PersistedThreadState,
		ThreadSummary
	} from '$lib/utils/tauri';
	import type { UserDecision } from '$lib/types/agent';
	import type { LLMReadyResponse } from '$lib/types/commands';
	import type { MessageRole, ClarificationRequest } from '$lib/types/agent';
	import type { CheckpointInfo } from '$lib/types/execution';
	import type { ToolResultEventData } from '$lib/types/sse';

	import ChatHeader from './ChatHeader.svelte';
	import MessageList from './MessageList.svelte';
	import StreamingIndicator from './StreamingIndicator.svelte';
	import ClarificationForm from './ClarificationForm.svelte';
	import PlanPreviewCard from './PlanPreviewCard.svelte';
	import ChatInput from './ChatInput.svelte';

	let { class: className }: ChatPanelProps = $props();

	// Get stores
	const agent = getAgentState();
	const sse = getSSEState();
	const pipeline = getPipelineState();
	const execution = getExecutionState();
	const ui = getUIState();
	const isInTauri = isTauri();

	// Local state
	let inputValue = $state('');
	let isInitializing = $state(false);
	let initError = $state<string | null>(null);
	let llmStatus = $state<LLMReadyResponse | null>(null);
	let isCheckingLLM = $state(false);
	let isPullingModel = $state(false);
	let autoPullTriggered = $state(false);
	let modelPullProgress = $state<LLMPullProgressEvent | null>(null);
	let modelPullError = $state<string | null>(null);
	let isSending = $state(false);
	let isRecoveringSidecar = $state(false);
	let resumePreview = $state<string | null>(null);
	let resumedThreadStrip = $state<{ label: string; summary: string } | null>(null);

	// Derived state
	const llmNotReady = $derived(llmStatus !== null && !llmStatus.ready);
	const isTypingDisabled = $derived(isInitializing || isRecoveringSidecar);
	const isSendDisabled = $derived(isTypingDisabled || llmNotReady || isSending || !sse.isConnected);
	const showClarification = $derived(agent.pendingClarification !== null);
	const showPlanPreview = $derived(
		pipeline.steps.length > 0 &&
		['planning', 'awaiting_approval'].includes(agent.phase)
	);

	function normalizeConnectionError(error: unknown): string {
		const message = error instanceof Error ? error.message : String(error);
		if (/load failed|failed to fetch|network|connection|refused/i.test(message)) {
			return 'Cannot reach local AI backend yet. Retrying startup...';
		}
		return message;
	}

	async function recoverSidecarIfNeeded(): Promise<boolean> {
		if (!isInTauri || isRecoveringSidecar) return false;

		isRecoveringSidecar = true;
		try {
			await startSidecar();
			const sidecarHealthy = await waitForSidecar(20000);
			if (!sidecarHealthy) {
				initError = 'Local AI backend is still starting. Please wait a few seconds and retry.';
				return false;
			}
			return true;
		} catch (error) {
			console.error('[ChatPanel] Failed to recover sidecar:', error);
			initError = normalizeConnectionError(error);
			return false;
		} finally {
			isRecoveringSidecar = false;
		}
	}

	// Check LLM service readiness
	async function checkLLM() {
		isCheckingLLM = true;
		try {
			llmStatus = await checkLLMReady();
			if (llmStatus.ready) {
				modelPullProgress = null;
				modelPullError = null;
			}
		} catch (error) {
			console.error('[ChatPanel] Failed to check LLM service:', error);
			if (isInTauri) {
				const recovered = await recoverSidecarIfNeeded();
				if (recovered) {
					try {
						llmStatus = await checkLLMReady();
						return;
					} catch (retryError) {
						console.error('[ChatPanel] LLM readiness retry failed:', retryError);
					}
				}
			}
			llmStatus = {
				ready: false,
				service_running: false,
				required_model: DEFAULT_REQUIRED_MODEL,
				model_available: false,
				error: 'Failed to connect to backend'
			};
		} finally {
			isCheckingLLM = false;
		}
	}

	function formatBytes(bytes: number | null): string {
		if (bytes === null || bytes < 0) return '0 B';
		const units = ['B', 'KB', 'MB', 'GB', 'TB'];
		let size = bytes;
		let unitIndex = 0;
		while (size >= 1024 && unitIndex < units.length - 1) {
			size /= 1024;
			unitIndex += 1;
		}
		return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
	}

	function getEstimatedDownloadSize(modelName: string): string {
		const model = modelName.toLowerCase();
		if (model.includes(':14b')) return '~9 GB';
		if (model.includes(':8b')) return '~5 GB';
		if (model.includes(':7b')) return '~4 GB';
		if (model.includes(':4b') || model.includes(':3b')) return '~2 GB';
		return 'a few GB';
	}

	function normalizeMessageRole(role: unknown): MessageRole | null {
		if (role === 'user' || role === 'assistant' || role === 'system' || role === 'tool') {
			return role;
		}
		return null;
	}

	function normalizeStepStatus(
		status: unknown
	): 'pending' | 'running' | 'completed' | 'failed' | 'skipped' {
		if (
			status === 'pending' ||
			status === 'running' ||
			status === 'completed' ||
			status === 'failed' ||
			status === 'skipped'
		) {
			return status;
		}
		return 'pending';
	}

	function toFiniteNumber(value: unknown): number | null {
		return typeof value === 'number' && Number.isFinite(value) ? value : null;
	}

	function normalizeCheckpointTrigger(trigger: unknown): CheckpointInfo['triggerReason'] {
		if (
			trigger === 'percentage' ||
			trigger === 'quality_drop' ||
			trigger === 'error_rate' ||
			trigger === 'critical_step'
		) {
			return trigger;
		}
		return 'percentage';
	}

	function getCheckpointMetric(
		qualityMetrics: PersistedThreadCheckpointQualityMetrics | undefined,
		snakeCaseKey: keyof PersistedThreadCheckpointQualityMetrics,
		camelCaseKey: keyof PersistedThreadCheckpointQualityMetrics
	): number | null {
		return toFiniteNumber(qualityMetrics?.[snakeCaseKey]) ?? toFiniteNumber(qualityMetrics?.[camelCaseKey]);
	}

	function getLatestCheckpoint(checkpoints: PersistedThreadCheckpoint[] | undefined) {
		if (!Array.isArray(checkpoints)) return null;
		for (let index = checkpoints.length - 1; index >= 0; index -= 1) {
			const checkpoint = checkpoints[index];
			if (checkpoint && typeof checkpoint.id === 'string' && !checkpoint.resolved_at) {
				return checkpoint;
			}
		}
		return null;
	}

	function buildHydratedCheckpoint(
		state: PersistedThreadState,
		fallbackMessage: string | null
	): CheckpointInfo | null {
		const checkpoint = getLatestCheckpoint(state.checkpoints);
		if (!checkpoint) return null;

		const qualityMetrics = checkpoint.quality_metrics;
		return {
			id: checkpoint.id,
			stepIndex: toFiniteNumber(checkpoint.step_index) ?? 0,
			triggerReason: normalizeCheckpointTrigger(checkpoint.trigger_reason),
			progressPercent:
				toFiniteNumber(checkpoint.progress_percent) ??
				toFiniteNumber(state.metadata?.progress_percent) ??
				0,
			qualityMetrics: {
				averageConfidence:
					getCheckpointMetric(qualityMetrics, 'average_confidence', 'averageConfidence') ?? 0,
				errorCount: getCheckpointMetric(qualityMetrics, 'error_count', 'errorCount') ?? 0,
				totalProcessed:
					getCheckpointMetric(qualityMetrics, 'total_processed', 'totalProcessed') ?? 0,
				processingSpeed:
					getCheckpointMetric(qualityMetrics, 'processing_speed', 'processingSpeed') ?? 0
			},
			message:
				(typeof checkpoint.message === 'string' && checkpoint.message) ||
				fallbackMessage ||
				'Resume from the saved checkpoint when you are ready.',
			createdAt:
				(typeof checkpoint.created_at === 'string' && checkpoint.created_at) ||
				new Date().toISOString()
		};
	}

	function buildHydratedExecutionStats(
		state: PersistedThreadState,
		checkpoint: CheckpointInfo | null
	) {
		return {
			processed:
				checkpoint?.qualityMetrics.totalProcessed ||
				toFiniteNumber(state.metadata?.processed_files) ||
				0,
			detected: toFiniteNumber(state.metadata?.total_items) || 0,
			flagged: 0,
			errors: checkpoint?.qualityMetrics.errorCount ?? 0,
			startedAt: typeof state.metadata?.created_at === 'string' ? state.metadata.created_at : null,
			estimatedCompletion: null
		};
	}

	function getThreadResumePriority(thread: ThreadSummary): number {
		if (thread.awaiting_user) return 0;
		if (thread.total_steps > 0 && thread.current_step < thread.total_steps) return 1;
		return 2;
	}

	function selectThreadToResume(threads: ThreadSummary[]): ThreadSummary | null {
		let bestThread: ThreadSummary | null = null;
		let bestPriority = Number.POSITIVE_INFINITY;

		for (const thread of threads) {
			const priority = getThreadResumePriority(thread);
			if (priority > bestPriority) continue;
			if (priority === bestPriority && bestThread !== null) continue;

			// `listThreads()` already returns newest-first, so keep the first thread on priority ties.
			bestThread = thread;
			bestPriority = priority;
		}

		return bestThread;
	}

	function getThreadResumeStatus(thread: ThreadSummary): string {
		if (thread.awaiting_user) return 'awaiting review';
		if (thread.total_steps > 0 && thread.current_step >= thread.total_steps) return 'completed';
		if (thread.total_steps > 0) return 'in progress';
		return 'ready';
	}

	function getTrimmedThreadSummary(thread: ThreadSummary): string | null {
		const summary = thread.summary?.trim();
		return summary ? summary : null;
	}

	function getThreadResumeSummary(thread: ThreadSummary): string {
		const summary = getTrimmedThreadSummary(thread);
		if (summary) {
			return summary;
		}

		const completedSteps =
			thread.total_steps > 0 ? Math.max(0, Math.min(thread.current_step, thread.total_steps)) : 0;
		if (thread.total_steps > 0) {
			return `${getThreadResumeStatus(thread)}. Progress: ${completedSteps}/${thread.total_steps} steps.`;
		}
		return `${getThreadResumeStatus(thread)}.`;
	}

	function getPendingResumeSummary(thread: ThreadSummary): string {
		const summary = getThreadResumeSummary(thread);
		const progressMatch = /^(.*)\. Progress: (\d+)\/(\d+) steps\.$/.exec(summary);
		if (progressMatch) {
			return `${progressMatch[1]} (${progressMatch[2]}/${progressMatch[3]} steps)`;
		}
		return summary.endsWith('.') ? summary.slice(0, -1) : summary;
	}

	function getPendingResumeLabel(thread: ThreadSummary): string {
		const title = thread.title?.trim();
		if (title && title !== thread.thread_id) {
			return `${title} (${thread.thread_id})`;
		}
		return thread.thread_id;
	}

	function buildResumedThreadMessage(thread: ThreadSummary): string {
		return `Resumed backend thread ${thread.thread_id}. Status: ${getThreadResumeSummary(thread)}`;
	}

	function buildPendingResumeMessage(thread: ThreadSummary): string {
		return `Resuming ${getPendingResumeLabel(thread)}: ${getPendingResumeSummary(thread)}`;
	}

	function buildResumedThreadStrip(thread: ThreadSummary): { label: string; summary: string } {
		return {
			label: getPendingResumeLabel(thread),
			summary: getThreadResumeSummary(thread)
		};
	}

	function dismissResumedThreadStrip() {
		resumedThreadStrip = null;
	}

	function hasSystemMessage(
		messages: PersistedThreadState['messages'],
		content: string
	): boolean {
		if (!Array.isArray(messages)) return false;
		return messages.some(
			(message) => normalizeMessageRole(message.role) === 'system' && message.content === content
		);
	}

	function buildResumeClarification(awaitingUser: boolean, planApproved: boolean): ClarificationRequest | null {
		if (!awaitingUser) return null;
		if (planApproved) {
			return {
				id: crypto.randomUUID(),
				prompt: 'Resume from the saved checkpoint when you are ready.',
				inputType: 'checkpoint_approval'
			};
		}

		return {
			id: crypto.randomUUID(),
			prompt: 'Review the saved plan and choose how to continue.',
			inputType: 'plan_approval'
		};
	}

	function buildPersistedToolResult(
		toolName: string,
		stepIndex: number,
		result: Record<string, unknown>,
		success: boolean
	): ToolResultEventData {
		return {
			tool_name: toolName,
			step_index: stepIndex,
			success,
			result,
			error: typeof result.error === 'string' ? result.error : undefined,
			duration_seconds: 0
		};
	}

	function replayPersistedExecutionResults(
		steps: Array<{ id: string; toolName: string; status: string }>,
		state: PersistedThreadState
	) {
		const executionResults = state.execution_results;
		if (!executionResults) return;

		for (const [index, step] of steps.entries()) {
			const rawResult = executionResults[step.id];
			if (typeof rawResult !== 'object' || rawResult === null) continue;
			const result = rawResult as Record<string, unknown>;

			if (step.status === 'failed') {
				execution.addError({
					stepId: step.id,
					message: typeof result.error === 'string' ? result.error : 'Tool execution failed',
					recoverable: true
				});
				continue;
			}

			const filesProcessed = toFiniteNumber(result.files_processed) ?? toFiniteNumber(result.total_files);
			if (filesProcessed !== null && filesProcessed > 0) {
				execution.updateStats({
					processed: Math.max(execution.stats.processed, filesProcessed)
				});
			}

			const detectedCount = toFiniteNumber(result.count);
			if (
				detectedCount !== null &&
				detectedCount > 0 &&
				(step.toolName === 'detect' || step.toolName === 'detect_3d')
			) {
				execution.updateStats({
					detected: Math.max(execution.stats.detected, detectedCount)
				});
			}

			const faces =
				toFiniteNumber(result.faces_anonymized) ?? toFiniteNumber(result.faces_blurred) ?? 0;
			const plates =
				toFiniteNumber(result.plates_anonymized) ?? toFiniteNumber(result.plates_blurred) ?? 0;
			const anonymized = faces + plates;
			if (anonymized > 0) {
				execution.updateStats({
					flagged: Math.max(execution.stats.flagged, anonymized)
				});
			}

			const previews = extractPreviewItems(
				buildPersistedToolResult(step.toolName, index, result, true)
			);
			if (previews.length > 0) {
				execution.appendPreviews(previews);
			}
		}
	}

	function hydratePersistedThread(
		threadId: string,
		state: Awaited<ReturnType<typeof getThreadState>>,
		threadSummary: ThreadSummary | null = null
	) {
		const messages = Array.isArray(state.messages) ? state.messages : [];
		const plan = Array.isArray(state.plan) ? state.plan : [];
		const currentStep =
			typeof state.current_step === 'number' && Number.isFinite(state.current_step)
				? state.current_step
				: 0;
		const awaitingUser = state.awaiting_user === true;
		const planApproved = state.plan_approved === true;
		const pipelineId =
			typeof state.metadata?.pipeline_id === 'string' ? state.metadata.pipeline_id : null;

		agent.startNewConversation();
		agent.setThreadId(threadId);
		pipeline.reset();
		execution.reset();

		for (const message of messages) {
			const role = normalizeMessageRole(message.role);
			if (!role || typeof message.content !== 'string') continue;
			const created = agent.addMessage({
				role,
				content: message.content
			});
			if (typeof message.timestamp === 'string' && message.timestamp) {
				agent.updateMessage(created.id, { timestamp: message.timestamp });
			}
		}
		const resumeMessage = threadSummary ? buildResumedThreadMessage(threadSummary) : null;
		resumedThreadStrip = threadSummary ? buildResumedThreadStrip(threadSummary) : null;
		if (resumeMessage && !hasSystemMessage(messages, resumeMessage)) {
			agent.addMessage({
				role: 'system',
				content: resumeMessage
			});
		}

		const steps = plan
			.filter(
				(step): step is NonNullable<typeof plan>[number] =>
					typeof step?.id === 'string' &&
					typeof step?.tool_name === 'string' &&
					typeof step?.description === 'string'
			)
			.map((step, index) => ({
				id: step.id,
				toolName: step.tool_name,
				type: inferStepType(step.tool_name),
				description: step.description,
				config: { params: step.parameters ?? {} },
				status: normalizeStepStatus(step.status),
				order: index,
				result: step.result ?? undefined,
				error: typeof step.error === 'string' ? step.error : undefined,
				startedAt: typeof step.started_at === 'string' ? step.started_at : undefined,
				completedAt: typeof step.completed_at === 'string' ? step.completed_at : undefined
			}));
		const completedStepCount = steps.filter((step) => step.status === 'completed').length;
		const failedStepCount = steps.filter((step) => step.status === 'failed').length;
		const effectiveCurrentStep =
			steps.length > 0 && currentStep >= steps.length && completedStepCount < steps.length
				? completedStepCount
				: steps.length > 0
					? Math.max(0, Math.min(Math.max(currentStep, completedStepCount), steps.length))
					: Math.max(0, currentStep);
		const latestAssistantMessage =
			[...messages]
				.reverse()
				.find((message) => message.role === 'assistant' && typeof message.content === 'string')
				?.content ?? null;
		const hydratedCheckpoint = buildHydratedCheckpoint(state, latestAssistantMessage);
		const boundedProgressCurrent =
			steps.length > 0 ? effectiveCurrentStep : 0;
		const hydratedStepIndex =
			steps.length > 0 ? Math.max(0, Math.min(effectiveCurrentStep, steps.length - 1)) : -1;

		pipeline.setPipelineId(pipelineId);
		pipeline.setSteps(steps);
		execution.updateProgress(boundedProgressCurrent, steps.length);
		execution.updateStats(buildHydratedExecutionStats(state, hydratedCheckpoint));
		execution.setCurrentStep(hydratedStepIndex >= 0 ? steps[hydratedStepIndex]?.id ?? null : null);
		execution.setCheckpoint(hydratedCheckpoint);
		if (!hydratedCheckpoint) {
			replayPersistedExecutionResults(steps, state);
		}

		const clarification = buildResumeClarification(awaitingUser, planApproved);
		if (clarification) {
			agent.setClarification(clarification);
			if (planApproved) {
				agent.setPhase('checkpoint');
			}
		} else if (hydratedCheckpoint) {
			agent.setPhase('checkpoint');
		} else if (planApproved && failedStepCount > 0) {
			agent.setPhase('complete');
			execution.setStatus('failed');
		} else if (
			planApproved &&
			steps.length > 0 &&
			effectiveCurrentStep >= steps.length &&
			completedStepCount >= steps.length
		) {
			agent.setPhase('complete');
			execution.setStatus('completed');
		} else if (planApproved && steps.length > 0) {
			agent.setPhase('executing');
			execution.setStatus('running');
		} else if (steps.length > 0) {
			agent.setPhase('planning');
		} else {
			agent.setPhase('idle');
		}
	}

	// Pull the required model
	async function handlePullModel() {
		if (!llmStatus || isPullingModel) return;

		isPullingModel = true;
		modelPullError = null;
		try {
			const modelName = llmStatus.required_model;
			modelPullProgress = {
				model: modelName,
				status: 'Starting model download',
				digest: null,
				totalBytes: null,
				completedBytes: null,
				progressPercent: 0,
				raw: null
			};
			await pullLLMModelWithProgress(modelName, (event) => {
				modelPullProgress = event;
			});

			const refreshed = await checkLLMReady();
			llmStatus = refreshed;

			// Fallback readiness check in case the stream ends before the model is fully indexed.
			if (!refreshed.ready) {
				const ensured = await ensureLLMReady();
				llmStatus = ensured;
			}

			// Allow re-attempt if backend reports still not ready.
			if (!llmStatus.ready) {
				autoPullTriggered = false;
			}
		} catch (error) {
			console.error('[ChatPanel] Failed to download model:', error);
			modelPullError = error instanceof Error ? error.message : 'Failed to download model';
			autoPullTriggered = false;
		} finally {
			isPullingModel = false;
		}
	}

	// Auto-download model when service is up but model is missing.
	$effect(() => {
		if (
			isInTauri &&
			llmStatus &&
			llmStatus.service_running &&
			!llmStatus.model_available &&
			!isPullingModel &&
			!autoPullTriggered
		) {
			autoPullTriggered = true;
			void handlePullModel();
		}
	});

	// Initialize thread and SSE connection
	async function initializeChat() {
		// Guard against concurrent initialization
		if (isInitializing) return;
		// Already initialized and connected
		if (agent.threadId && sse.isConnected) return;

		isInitializing = true;
		initError = null;
		resumePreview = null;
		resumedThreadStrip = null;

		try {
			if (isInTauri) {
				await recoverSidecarIfNeeded();
			}

			let threadId = agent.threadId;

			// Prefer the latest resumable backend thread before creating a new one.
			if (!threadId) {
				const existingThreads = await listThreads(20).catch(() => []);
				const threadToResume = selectThreadToResume(existingThreads);
				threadId = threadToResume?.thread_id ?? null;
				if (threadId && threadToResume) {
					resumePreview = buildPendingResumeMessage(threadToResume);
					const persistedState = await getThreadState(threadId).catch(() => null);
					if (persistedState) {
						hydratePersistedThread(threadId, persistedState, threadToResume);
					}
				}
			}

			if (!threadId) {
				const thread = await createThread();
				threadId = thread.thread_id;
				agent.setThreadId(threadId);
			} else {
				agent.setThreadId(threadId);
			}

			// Connect SSE using the local threadId to avoid race condition
			sse.connect(threadId);
		} catch (error) {
			console.error('[ChatPanel] Failed to initialize:', error);
			initError = normalizeConnectionError(error);
		} finally {
			resumePreview = null;
			isInitializing = false;
		}
	}

	async function handleRecoverConnection() {
		await recoverSidecarIfNeeded();
		await initializeChat();
		await checkLLM();
	}

	// Send user message
	async function handleSend(content: string) {
		if (!content.trim() || !agent.threadId || isSending) return;
		if (!sse.isConnected) {
			agent.setError('Waiting for local AI backend to reconnect. Please try again in a moment.');
			return;
		}

		isSending = true;
		resumedThreadStrip = null;

		// Add user message to store
		agent.addMessage({
			role: 'user',
			content: content.trim()
		});

		// Clear input
		inputValue = '';

		// Send to backend
		try {
			await sendMessage(agent.threadId, { content: content.trim() });
		} catch (error) {
			console.error('[ChatPanel] Failed to send message:', error);
			agent.setError(error instanceof Error ? error.message : 'Failed to send message');
		} finally {
			isSending = false;
		}
	}

	// Handle clarification response
	async function handleClarificationSubmit(response: { decision: UserDecision; selected?: string[] }) {
		if (!agent.threadId || !agent.pendingClarification) return;
		const clarification = agent.pendingClarification;

		if (clarification.inputType === 'plan_approval' && response.decision === 'approve') {
			// Ensure execution UI leaves any stale complete/idle state immediately.
			execution.start();
			execution.setCurrentStep(null);
		}

		// Clear clarification
		agent.setClarification(null);

		// Send decision to backend
		try {
			const content = response.selected?.join(', ') || response.decision;
			await sendMessage(agent.threadId, {
				content,
				decision: response.decision
			});
		} catch (error) {
			console.error('[ChatPanel] Failed to submit clarification:', error);
			agent.setError(error instanceof Error ? error.message : 'Failed to submit response');
		}
	}

	// Handle clarification cancel
	function handleClarificationCancel() {
		agent.setClarification(null);
		// Optionally send cancel to backend
		if (agent.threadId) {
			sendMessage(agent.threadId, {
				content: 'Cancelled',
				decision: 'cancel'
			}).catch(console.error);
		}
	}

	// Clear conversation
	async function handleClear() {
		const previousThreadId = agent.threadId;
		resumedThreadStrip = null;
		agent.startNewConversation();
		pipeline.clearPipeline();
		if (previousThreadId) {
			try {
				await closeThread(previousThreadId);
			} catch (error) {
				console.warn('[ChatPanel] Failed to close previous thread:', error);
			}
		}
		// Reconnect with new thread
		await initializeChat();
	}

	// Navigate to plan editor
	function handleViewPlan() {
		ui.setView('plan');
	}

	// Export conversation (placeholder)
	function handleExport() {
		// Convert messages to markdown
		const markdown = agent.messages
			.map((msg) => {
				const prefix = msg.role === 'user' ? '**You:**' : '**Agent:**';
				return `${prefix}\n${msg.content}\n`;
			})
			.join('\n---\n\n');

		// Copy to clipboard
		navigator.clipboard.writeText(markdown).catch(console.error);
	}

	// Initialize on mount
	onMount(() => {
		const handleWindowKeydown = (event: KeyboardEvent) => {
			if (event.key === 'Escape' && resumedThreadStrip) {
				dismissResumedThreadStrip();
			}
		};

		// Check LLM service readiness first
		checkLLM();
		// Initialize chat connection
		initializeChat();
		window.addEventListener('keydown', handleWindowKeydown);

		// Cleanup on unmount - don't disconnect, just let SSE continue
		return () => {
			window.removeEventListener('keydown', handleWindowKeydown);
			// SSE connection persists across view switches
		};
	});

	// Recover automatically if chat is disconnected after startup.
	$effect(() => {
		if (!isInTauri) return;
		if (!agent.threadId || sse.isConnected || isInitializing || isRecoveringSidecar) return;

		const retryTimeout = setTimeout(() => {
			void initializeChat();
			void checkLLM();
		}, 2500);

		return () => clearTimeout(retryTimeout);
	});

	$effect(() => {
		if (!isInTauri || !initError || isInitializing || isRecoveringSidecar) return;

		const retryTimeout = setTimeout(() => {
			void handleRecoverConnection();
		}, 4000);

		return () => clearTimeout(retryTimeout);
	});
</script>

<div class={cn('flex flex-col h-full bg-background', className)}>
	<!-- Header -->
	<ChatHeader
		phase={agent.phase}
		isConnected={sse.isConnected}
		onClear={handleClear}
		onExport={handleExport}
	/>

	{#if isInitializing && resumePreview}
		<div class="px-4 py-2 mx-4 mt-2 rounded border border-border/60 bg-muted/20 text-xs font-mono text-muted-foreground">
			{resumePreview}
		</div>
	{/if}

	{#if !isInitializing && resumedThreadStrip}
		<div class="px-4 py-2 mx-4 mt-2 rounded border border-border/60 bg-card/40 text-xs text-muted-foreground flex items-center justify-between gap-3">
			<div>
				<span class="font-medium text-foreground/90">Resumed:</span>
				{' '}
				{resumedThreadStrip.label}
				{' '}
				<span class="text-muted-foreground">• {resumedThreadStrip.summary}</span>
			</div>
			<button
				type="button"
				class="shrink-0 rounded px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
				aria-label="Dismiss resumed thread summary"
				onclick={dismissResumedThreadStrip}
			>
				Dismiss
			</button>
		</div>
	{/if}

	<!-- Messages -->
	<MessageList
		messages={agent.messages}
		isStreaming={agent.isStreaming}
		class="flex-1"
	/>

	<!-- Streaming indicator -->
	{#if agent.isStreaming}
		<StreamingIndicator phase={agent.phase} />
	{/if}

	<!-- Plan preview (when planning/awaiting approval) -->
	{#if showPlanPreview}
		<div class="px-4 pb-2">
			<PlanPreviewCard
				steps={pipeline.sortedSteps}
				onViewPlan={handleViewPlan}
			/>
		</div>
	{/if}

	<!-- Clarification form (when pending) -->
	{#if showClarification && agent.pendingClarification}
		<div class="px-4 pb-2">
			<ClarificationForm
				clarification={agent.pendingClarification}
				disabled={!sse.isConnected}
				onSubmit={handleClarificationSubmit}
				onCancel={handleClarificationCancel}
			/>
		</div>
	{/if}

	<!-- Error display -->
	{#if agent.lastError}
		<div class="px-4 py-2 mx-4 mb-2 rounded bg-destructive/10 border border-destructive/20 text-destructive text-sm">
			{agent.lastError}
		</div>
	{/if}

	<!-- Connection error -->
	{#if initError}
		<div class="px-4 py-2 mx-4 mb-2 rounded bg-amber-500/10 border border-amber-500/20 text-amber-600 text-sm flex items-center justify-between">
			<span>{initError}</span>
			<button
				class="text-xs underline"
				onclick={handleRecoverConnection}
			>
				Retry
			</button>
		</div>
	{/if}

	<!-- AI service not ready warning -->
	{#if llmNotReady && llmStatus}
		<div class="px-4 py-3 mx-4 mb-2 rounded bg-amber-500/10 border border-amber-500/20 text-amber-700 text-sm">
			<div class="flex items-start gap-3">
				<div class="flex-1">
					{#if !llmStatus.service_running}
						<p class="font-medium">AI service is starting...</p>
						<p class="text-xs mt-1 text-amber-600">
							The language model is initializing. This may take a moment on first launch.
						</p>
					{:else if !llmStatus.model_available}
						<p class="font-medium">Downloading AI model...</p>
						<p class="text-xs mt-1 text-amber-600">
							{#if modelPullProgress?.progressPercent !== null}
								{Math.round(modelPullProgress?.progressPercent ?? 0)}%
								({formatBytes(modelPullProgress?.completedBytes ?? null)} / {formatBytes(modelPullProgress?.totalBytes ?? null)})
							{:else if modelPullProgress?.status}
								{modelPullProgress.status}
							{:else}
								First-time setup requires downloading the AI model ({getEstimatedDownloadSize(llmStatus.required_model)}).
							{/if}
						</p>
						{#if modelPullError}
							<p class="text-xs mt-1 text-destructive">{modelPullError}</p>
						{/if}
						{#if modelPullProgress?.progressPercent !== null}
							<div class="mt-2 h-2 w-full max-w-md bg-amber-200/60 rounded-full overflow-hidden">
								<div
									class="h-full bg-forest transition-all duration-300"
									style={`width: ${modelPullProgress?.progressPercent ?? 0}%`}
								></div>
							</div>
						{/if}
					{:else}
						<p class="font-medium">AI service not ready</p>
						<p class="text-xs mt-1 text-amber-600">{llmStatus.error}</p>
					{/if}
				</div>
				<div class="flex gap-2">
					{#if llmStatus.service_running && !llmStatus.model_available}
						<button
							class="px-3 py-1 text-xs rounded bg-forest text-white hover:bg-forest-dark disabled:opacity-50"
							onclick={handlePullModel}
							disabled={isPullingModel}
						>
							{isPullingModel ? 'Downloading...' : 'Download model'}
						</button>
					{/if}
					<button
						class="px-2 py-1 text-xs underline"
						onclick={handleRecoverConnection}
						disabled={isCheckingLLM || isPullingModel}
					>
						{isCheckingLLM ? 'Checking...' : 'Refresh'}
					</button>
				</div>
			</div>
		</div>
	{/if}

	<!-- Input -->
	<ChatInput
		value={inputValue}
		disabled={isTypingDisabled}
		disableSend={isSendDisabled}
		placeholder={
			isInitializing || isRecoveringSidecar
				? 'Connecting...'
				: sse.isConnected
					? 'Type a message...'
					: 'Reconnecting to local AI service...'
		}
		onSend={handleSend}
		onValueChange={(v) => (inputValue = v)}
	/>
</div>
