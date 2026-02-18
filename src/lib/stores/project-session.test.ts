import { beforeEach, describe, expect, it } from 'vitest';
import { createAgentState } from './agent.svelte';
import { createPipelineState } from './pipeline.svelte';
import { createExecutionState } from './execution.svelte';
import { createReviewState } from './review.svelte';
import {
	DEFAULT_PROJECT_SESSION_KEY,
	PROJECT_SESSION_STORAGE_KEY,
	type ProjectSessionStores,
	restoreProjectSession,
	saveProjectSession,
	toProjectSessionKey
} from './project-session';

function createStores(): ProjectSessionStores {
	return {
		agent: createAgentState(),
		pipeline: createPipelineState(),
		execution: createExecutionState(),
		review: createReviewState()
	};
}

function seedStores(stores: ProjectSessionStores, suffix: string): void {
	stores.agent.reset();
	stores.pipeline.reset();
	stores.execution.reset();
	stores.review.reset();

	stores.agent.setThreadId(`thread-${suffix}`);
	const message = stores.agent.addMessage({
		role: 'user',
		content: `hello-${suffix}`
	});
	stores.agent.updateMessage(message.id, {
		id: `message-${suffix}`,
		timestamp: '2026-01-02T03:04:05.000Z'
	});
	stores.agent.setPhase('planning');

	const stepId = `step-${suffix}`;
	stores.pipeline.setSteps([
		{
			id: stepId,
			toolName: 'detect',
			type: 'detection',
			description: `Detect ${suffix}`,
			config: { params: { mode: suffix } },
			status: 'pending',
			order: 0
		}
	]);
	stores.pipeline.setPipelineId(`pipeline-${suffix}`);
	stores.pipeline.selectStep(stepId);
	stores.pipeline.setEditing(true);
	stores.pipeline.updateStep(stepId, { description: `Detect ${suffix} updated` });

	stores.execution.setStatus('running');
	stores.execution.updateProgress(2, 5);
	stores.execution.updateStats({
		processed: 10,
		detected: 4,
		flagged: 2,
		errors: 0,
		startedAt: '2026-01-02T03:04:05.000Z',
		estimatedCompletion: '2026-01-02T03:10:05.000Z'
	});
	stores.execution.setPreviews([
		{
			id: `preview-${suffix}`,
			imagePath: `/tmp/${suffix}.png`,
			thumbnailUrl: `/tmp/${suffix}-thumb.png`,
			annotations: [],
			status: 'processed'
		}
	]);
	stores.execution.addError({
		stepId,
		message: `error-${suffix}`,
		recoverable: true
	});
	stores.execution.updateStats({ errors: 1 });
	stores.execution.setCurrentStep(stepId);

	stores.review.loadItems([
		{
			id: `review-${suffix}`,
			filePath: `/tmp/review-${suffix}.png`,
			fileName: `review-${suffix}.png`,
			dimensions: { width: 640, height: 480 },
			thumbnailUrl: `/thumb/review-${suffix}.png`,
			annotations: [],
			originalAnnotations: [],
			status: 'pending',
			flagged: false
		}
	]);
	stores.review.setFilter('searchQuery', suffix);
	stores.review.selectItem(`review-${suffix}`);
	stores.review.setCurrentItem(`review-${suffix}`);
}

describe('project session persistence', () => {
	beforeEach(() => {
		localStorage.clear();
	});

	it('restores conversations/plans/execution/review state on reopen', () => {
		const firstSession = createStores();
		seedStores(firstSession, 'alpha');
		saveProjectSession('project-alpha', firstSession);

		const reopened = createStores();
		restoreProjectSession('project-alpha', reopened);

		expect(reopened.agent.threadId).toBe('thread-alpha');
		expect(reopened.agent.messages).toHaveLength(1);
		expect(reopened.agent.messages[0].content).toBe('hello-alpha');
		expect(reopened.agent.phase).toBe('planning');

		expect(reopened.pipeline.pipelineId).toBe('pipeline-alpha');
		expect(reopened.pipeline.steps).toHaveLength(1);
		expect(reopened.pipeline.selectedStepId).toBe('step-alpha');
		expect(reopened.pipeline.isEditing).toBe(true);

		expect(reopened.execution.status).toBe('running');
		expect(reopened.execution.progress.current).toBe(2);
		expect(reopened.execution.progress.total).toBe(5);
		expect(reopened.execution.currentStepId).toBe('step-alpha');
		expect(reopened.execution.errors).toHaveLength(1);
		expect(reopened.execution.previews).toHaveLength(1);

		expect(reopened.review.items).toHaveLength(1);
		expect(reopened.review.currentItemId).toBe('review-alpha');
		expect(reopened.review.selectedIds.has('review-alpha')).toBe(true);
		expect(reopened.review.filters.searchQuery).toBe('alpha');
	});

	it('keeps session state isolated per project and clears unknown project state', () => {
		const stores = createStores();

		seedStores(stores, 'alpha');
		saveProjectSession('project-alpha', stores);

		seedStores(stores, 'beta');
		saveProjectSession('project-beta', stores);

		restoreProjectSession('project-alpha', stores);
		expect(stores.agent.threadId).toBe('thread-alpha');
		expect(stores.pipeline.pipelineId).toBe('pipeline-alpha');
		expect(stores.review.currentItemId).toBe('review-alpha');
		expect(stores.review.filters.searchQuery).toBe('alpha');

		restoreProjectSession('missing-project', stores);
		expect(stores.agent.messages).toHaveLength(0);
		expect(stores.pipeline.steps).toHaveLength(0);
		expect(stores.execution.status).toBe('idle');
		expect(stores.review.items).toHaveLength(0);
	});

	it('normalizes empty project id to default key', () => {
		expect(toProjectSessionKey(undefined)).toBe(DEFAULT_PROJECT_SESSION_KEY);
		expect(toProjectSessionKey(null)).toBe(DEFAULT_PROJECT_SESSION_KEY);
		expect(toProjectSessionKey('')).toBe(DEFAULT_PROJECT_SESSION_KEY);
		expect(toProjectSessionKey('  ')).toBe(DEFAULT_PROJECT_SESSION_KEY);
	});

	it('stores sessions in localStorage map keyed by project', () => {
		const alphaStores = createStores();
		seedStores(alphaStores, 'alpha');
		saveProjectSession('project-alpha', alphaStores);

		const betaStores = createStores();
		seedStores(betaStores, 'beta');
		saveProjectSession('project-beta', betaStores);

		const raw = localStorage.getItem(PROJECT_SESSION_STORAGE_KEY);
		expect(raw).toBeTruthy();
		const parsed = JSON.parse(raw ?? '{}') as Record<string, unknown>;
		expect(parsed['project-alpha']).toBeTruthy();
		expect(parsed['project-beta']).toBeTruthy();
	});
});
