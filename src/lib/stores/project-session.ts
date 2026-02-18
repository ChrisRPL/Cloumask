/**
 * Project-scoped session persistence for frontend runtime stores.
 *
 * Keeps isolated snapshots for each project so switching projects restores the
 * correct conversations/plans/execution/review state without stale bleed.
 */

import type {
	AgentState,
	AgentPhase,
	ClarificationRequest,
	Message
} from '$lib/types/agent';
import type {
	PipelineState,
	PipelineStep
} from '$lib/types/pipeline';
import type {
	ExecutionState,
	ExecutionStatus,
	ExecutionProgress,
	ExecutionStats,
	ExecutionError,
	CheckpointInfo,
	PreviewItem
} from '$lib/types/execution';
import type {
	ReviewState,
	ReviewItem,
	ReviewFilters
} from '$lib/types/review';

export const PROJECT_SESSION_STORAGE_KEY = 'cloumask:session:projects:v1';
export const DEFAULT_PROJECT_SESSION_KEY = '__default__';

export interface ProjectSessionStores {
	agent: AgentState;
	pipeline: PipelineState;
	execution: ExecutionState;
	review: ReviewState;
}

export interface AgentSessionSnapshot {
	threadId: string | null;
	messages: Message[];
	phase: AgentPhase;
	pendingClarification: ClarificationRequest | null;
	lastError: string | null;
}

export interface PipelineSessionSnapshot {
	steps: PipelineStep[];
	isEditing: boolean;
	selectedStepId: string | null;
	isDirty: boolean;
	pipelineId: string | null;
}

export interface ExecutionSessionSnapshot {
	status: ExecutionStatus;
	progress: ExecutionProgress;
	stats: ExecutionStats;
	previews: PreviewItem[];
	errors: ExecutionError[];
	currentStepId: string | null;
	checkpoint: CheckpointInfo | null;
}

export interface ReviewSessionSnapshot {
	items: ReviewItem[];
	selectedIds: string[];
	filters: ReviewFilters;
	currentItemId: string | null;
}

export interface ProjectSessionSnapshot {
	agent: AgentSessionSnapshot;
	pipeline: PipelineSessionSnapshot;
	execution: ExecutionSessionSnapshot;
	review: ReviewSessionSnapshot;
}

type SessionMap = Record<string, ProjectSessionSnapshot>;

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null;
}

function isProjectSessionSnapshot(value: unknown): value is ProjectSessionSnapshot {
	if (!isRecord(value)) return false;
	return (
		isRecord(value.agent) &&
		isRecord(value.pipeline) &&
		isRecord(value.execution) &&
		isRecord(value.review)
	);
}

function readSessionMap(): SessionMap {
	if (typeof window === 'undefined') return {};
	try {
		const raw = localStorage.getItem(PROJECT_SESSION_STORAGE_KEY);
		if (!raw) return {};
		const parsed = JSON.parse(raw);
		if (!isRecord(parsed)) return {};

		const sessions: SessionMap = {};
		for (const [key, value] of Object.entries(parsed)) {
			if (isProjectSessionSnapshot(value)) {
				sessions[key] = value;
			}
		}
		return sessions;
	} catch {
		return {};
	}
}

function writeSessionMap(map: SessionMap): void {
	if (typeof window === 'undefined') return;
	localStorage.setItem(PROJECT_SESSION_STORAGE_KEY, JSON.stringify(map));
}

function cloneSnapshot(snapshot: ProjectSessionSnapshot): ProjectSessionSnapshot {
	return JSON.parse(JSON.stringify(snapshot)) as ProjectSessionSnapshot;
}

function resetStores(stores: ProjectSessionStores): void {
	stores.agent.reset();
	stores.pipeline.reset();
	stores.execution.reset();
	stores.review.reset();
}

function applyAgentSnapshot(snapshot: AgentSessionSnapshot, stores: ProjectSessionStores): void {
	stores.agent.setThreadId(snapshot.threadId);

	for (const message of snapshot.messages) {
		const created = stores.agent.addMessage({
			role: message.role,
			content: message.content,
			toolCalls: message.toolCalls,
			toolCallId: message.toolCallId,
			isStreaming: message.isStreaming
		});

		stores.agent.updateMessage(created.id, {
			id: message.id,
			timestamp: message.timestamp
		});
	}

	stores.agent.setClarification(snapshot.pendingClarification);
	stores.agent.setError(snapshot.lastError);
	if (snapshot.phase !== 'error') {
		stores.agent.setPhase(snapshot.phase);
	} else if (!snapshot.lastError) {
		stores.agent.setPhase('error');
	}
	stores.agent.setStreaming(false);
	stores.agent.setConnected(false);
}

function applyPipelineSnapshot(snapshot: PipelineSessionSnapshot, stores: ProjectSessionStores): void {
	stores.pipeline.setSteps(snapshot.steps);
	stores.pipeline.setPipelineId(snapshot.pipelineId);
	stores.pipeline.setEditing(snapshot.isEditing);
	stores.pipeline.selectStep(snapshot.selectedStepId);

	// setSteps marks clean; this recreates dirty flag when needed.
	if (snapshot.isDirty && snapshot.steps.length > 0) {
		stores.pipeline.updateStep(snapshot.steps[0].id, {});
	}
}

function applyExecutionSnapshot(snapshot: ExecutionSessionSnapshot, stores: ProjectSessionStores): void {
	stores.execution.setStatus(snapshot.status);
	stores.execution.updateProgress(snapshot.progress.current, snapshot.progress.total);
	stores.execution.updateStats(snapshot.stats);
	stores.execution.setPreviews(snapshot.previews);
	stores.execution.setCurrentStep(snapshot.currentStepId);

	stores.execution.clearErrors();
	for (const error of snapshot.errors) {
		stores.execution.addError({
			stepId: error.stepId,
			message: error.message,
			recoverable: error.recoverable
		});
	}
	// addError updates error stats; restore explicit stats afterward.
	stores.execution.updateStats(snapshot.stats);

	if (snapshot.checkpoint) {
		stores.execution.setCheckpoint(snapshot.checkpoint);
	} else {
		stores.execution.clearCheckpoint();
	}
	stores.execution.setStatus(snapshot.status);
}

function applyReviewSnapshot(snapshot: ReviewSessionSnapshot, stores: ProjectSessionStores): void {
	stores.review.loadItems(snapshot.items);
	stores.review.resetFilters();
	stores.review.setFilter('status', snapshot.filters.status);
	stores.review.setFilter('label', snapshot.filters.label);
	stores.review.setFilter('minConfidence', snapshot.filters.minConfidence);
	stores.review.setFilter('maxConfidence', snapshot.filters.maxConfidence);
	stores.review.setFilter('searchQuery', snapshot.filters.searchQuery);

	stores.review.clearSelection();
	const itemIds = new Set(snapshot.items.map((item) => item.id));
	for (const selectedId of snapshot.selectedIds) {
		if (itemIds.has(selectedId)) {
			stores.review.selectItem(selectedId);
		}
	}

	if (snapshot.currentItemId && itemIds.has(snapshot.currentItemId)) {
		stores.review.setCurrentItem(snapshot.currentItemId);
	} else if (snapshot.items.length === 0) {
		stores.review.setCurrentItem(null);
	}

	stores.review.setLoading(false);
}

export function toProjectSessionKey(projectId: string | null | undefined): string {
	const trimmed = projectId?.trim();
	return trimmed ? trimmed : DEFAULT_PROJECT_SESSION_KEY;
}

export function captureProjectSession(stores: ProjectSessionStores): ProjectSessionSnapshot {
	return cloneSnapshot({
		agent: {
			threadId: stores.agent.threadId,
			messages: stores.agent.messages,
			phase: stores.agent.phase,
			pendingClarification: stores.agent.pendingClarification,
			lastError: stores.agent.lastError
		},
		pipeline: {
			steps: stores.pipeline.steps,
			isEditing: stores.pipeline.isEditing,
			selectedStepId: stores.pipeline.selectedStepId,
			isDirty: stores.pipeline.isDirty,
			pipelineId: stores.pipeline.pipelineId
		},
		execution: {
			status: stores.execution.status,
			progress: stores.execution.progress,
			stats: stores.execution.stats,
			previews: stores.execution.previews,
			errors: stores.execution.errors,
			currentStepId: stores.execution.currentStepId,
			checkpoint: stores.execution.checkpoint
		},
		review: {
			items: stores.review.items,
			selectedIds: Array.from(stores.review.selectedIds),
			filters: stores.review.filters,
			currentItemId: stores.review.currentItemId
		}
	});
}

export function saveProjectSession(
	projectId: string | null | undefined,
	stores: ProjectSessionStores
): void {
	if (typeof window === 'undefined') return;
	const map = readSessionMap();
	map[toProjectSessionKey(projectId)] = captureProjectSession(stores);
	writeSessionMap(map);
}

export function loadProjectSession(
	projectId: string | null | undefined
): ProjectSessionSnapshot | null {
	const map = readSessionMap();
	return map[toProjectSessionKey(projectId)] ?? null;
}

export function applyProjectSession(
	snapshot: ProjectSessionSnapshot | null,
	stores: ProjectSessionStores
): void {
	resetStores(stores);
	if (!snapshot) return;
	applyAgentSnapshot(snapshot.agent, stores);
	applyPipelineSnapshot(snapshot.pipeline, stores);
	applyExecutionSnapshot(snapshot.execution, stores);
	applyReviewSnapshot(snapshot.review, stores);
}

export function restoreProjectSession(
	projectId: string | null | undefined,
	stores: ProjectSessionStores
): void {
	applyProjectSession(loadProjectSession(projectId), stores);
}
