<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface PlanEditorProps {
		class?: string;
		onStartExecution?: () => void;
		onCancel?: () => void;
	}
</script>

<script lang="ts">
	import { getPipelineState } from '$lib/stores/pipeline.svelte';
	import { getAgentState } from '$lib/stores/agent.svelte';
	import { getUIState } from '$lib/stores/ui.svelte';
	import { getKeyboardState } from '$lib/stores/keyboard.svelte';
	import { sendMessage } from '$lib/utils/tauri';
	import type { StepType, StepConfig as StepConfigType } from '$lib/types/pipeline';

	import PlanHeader from './PlanHeader.svelte';
	import StepList from './StepList.svelte';
	import StepConfigPanel from './StepConfig.svelte';
	import AddStepButton from './AddStepButton.svelte';
	import { estimatePipelineTime, formatDuration, getDefaultConfig } from './constants';

	let {
		class: className,
		onStartExecution,
		onCancel,
	}: PlanEditorProps = $props();

	// Get stores from context
	const pipeline = getPipelineState();
	const agent = getAgentState();
	const ui = getUIState();
	const keyboard = getKeyboardState();

	// Local state
	let configPanelStepId = $state<string | null>(null);

	// Derived state
	const isAwaitingApproval = $derived(agent.phase === 'awaiting_approval');
	const canStart = $derived(pipeline.stepCount > 0 && !agent.isBusy);
	const configPanelOpen = $derived(configPanelStepId !== null);
	const selectedStep = $derived(
		configPanelStepId ? pipeline.steps.find((s) => s.id === configPanelStepId) : null
	);

	// Estimate pipeline time (assume 100 items, no GPU for now)
	const estimatedTime = $derived.by(() => {
		if (pipeline.stepCount === 0) return undefined;
		const ms = estimatePipelineTime(pipeline.sortedSteps, 100, false);
		return formatDuration(ms);
	});

	// Handle start execution
	async function handleStart() {
		if (!agent.threadId || !canStart) return;

		try {
			// Send approval with any edits
			await sendMessage(agent.threadId, {
				content: 'Approved',
				decision: 'approve',
			});

			// Transition to execution view
			ui.setView('execute');
			onStartExecution?.();
		} catch (error) {
			console.error('[PlanEditor] Failed to start execution:', error);
			agent.setError(error instanceof Error ? error.message : 'Failed to start execution');
		}
	}

	// Handle cancel
	function handleCancel() {
		// Send cancel to backend if connected
		if (agent.threadId) {
			sendMessage(agent.threadId, {
				content: 'Cancelled',
				decision: 'cancel',
			}).catch(console.error);
		}

		// Clear pipeline and go back to chat
		pipeline.clearPipeline();
		agent.setClarification(null);
		ui.setView('chat');
		onCancel?.();
	}

	// Handle edit toggle
	function handleToggleEdit() {
		const wasEditing = pipeline.isEditing;
		pipeline.setEditing(!wasEditing);
		// Close config panel when exiting edit mode
		if (wasEditing) {
			configPanelStepId = null;
		}
	}

	// Handle reset changes
	function handleReset() {
		// TODO: Implement reset to original plan
		pipeline.markClean();
	}

	// Handle step reorder
	function handleReorder(fromIndex: number, toIndex: number) {
		pipeline.moveStep(fromIndex, toIndex);
	}

	// Handle step selection
	function handleSelectStep(id: string) {
		pipeline.selectStep(id);
	}

	// Handle open config
	function handleOpenConfig(id: string) {
		configPanelStepId = id;
		pipeline.selectStep(id);
	}

	// Handle close config
	function handleCloseConfig() {
		configPanelStepId = null;
	}

	// Handle toggle step enabled
	function handleToggleEnabled(id: string) {
		const step = pipeline.steps.find((s) => s.id === id);
		if (step) {
			pipeline.updateStep(id, {
				status: step.status === 'skipped' ? 'pending' : 'skipped',
			});
		}
	}

	// Handle delete step
	function handleDeleteStep(id: string) {
		pipeline.removeStep(id);
		if (configPanelStepId === id) {
			configPanelStepId = null;
		}
	}

	// Handle config update
	function handleConfigUpdate(updates: Partial<StepConfigType>) {
		if (!configPanelStepId) return;
		const step = pipeline.steps.find((s) => s.id === configPanelStepId);
		if (step) {
			pipeline.updateStep(configPanelStepId, {
				config: { ...step.config, ...updates },
			});
		}
	}

	// Handle add step
	function handleAddStep(type: StepType) {
		const defaultConfig = getDefaultConfig(type);
		pipeline.addStep({
			toolName: type,
			type,
			description: `${type.charAt(0).toUpperCase() + type.slice(1)} step`,
			config: {
				model: defaultConfig.model as string | undefined,
				confidence: defaultConfig.confidence as number | undefined,
				params: defaultConfig,
			},
			critical: false,
		});
	}

	// ============================================================================
	// Keyboard Shortcuts (registered with keyboard store for scope awareness)
	// ============================================================================

	$effect(() => {
		const unregisterFns: (() => void)[] = [];

		// Escape - close config panel or exit edit mode
		unregisterFns.push(
			(() => {
				const id = keyboard.register({
					combo: 'escape',
					action: () => {
						if (configPanelOpen) {
							configPanelStepId = null;
						} else if (pipeline.isEditing) {
							pipeline.setEditing(false);
						}
					},
					scope: 'plan',
					description: 'Close panel / Exit edit mode',
					category: 'Plan Editor',
					priority: 10, // Lower than global escape
				});
				return () => keyboard.unregister(id);
			})()
		);

		// Enter - start execution if awaiting approval
		unregisterFns.push(
			(() => {
				const id = keyboard.register({
					combo: 'enter',
					action: () => {
						if (isAwaitingApproval && canStart) {
							handleStart();
						}
					},
					scope: 'plan',
					description: 'Start execution',
					category: 'Plan Editor',
				});
				return () => keyboard.unregister(id);
			})()
		);

		// Note: 'e' for toggle edit, 'j/k' for navigation, and 'space' for toggle step
		// are registered globally in +layout.svelte

		return () => {
			for (const fn of unregisterFns) fn();
		};
	});
</script>

<div class={cn('flex flex-col h-full bg-background', className)}>
	<!-- Header -->
	<PlanHeader
		stepCount={pipeline.stepCount}
		enabledCount={pipeline.enabledSteps.length}
		estimatedTime={estimatedTime}
		isEditing={pipeline.isEditing}
		{isAwaitingApproval}
		{canStart}
		isDirty={pipeline.isDirty}
		onToggleEdit={handleToggleEdit}
		onStart={handleStart}
		onCancel={handleCancel}
		onReset={handleReset}
	/>

	<!-- Main content area -->
	<div class="flex-1 overflow-hidden flex">
		<!-- Step list -->
		<div class="flex-1 overflow-auto">
			<StepList
				steps={pipeline.sortedSteps}
				isEditing={pipeline.isEditing}
				selectedStepId={pipeline.selectedStepId}
				onReorder={handleReorder}
				onSelectStep={handleSelectStep}
				onOpenConfig={handleOpenConfig}
				onToggleEnabled={handleToggleEnabled}
				onDeleteStep={handleDeleteStep}
				class="py-2"
			/>

			<!-- Add step button (edit mode only) -->
			{#if pipeline.isEditing}
				<AddStepButton onAddStep={handleAddStep} />
			{/if}
		</div>

		<!-- Config panel (slide-out) -->
		{#if configPanelOpen && selectedStep}
			<StepConfigPanel
				step={selectedStep}
				onUpdate={handleConfigUpdate}
				onClose={handleCloseConfig}
			/>
		{/if}
	</div>

	<!-- Keyboard hints -->
	{#if isAwaitingApproval && !configPanelOpen}
		<div class="px-4 py-2 border-t border-border text-xs text-muted-foreground/60 font-mono flex gap-4">
			<span><kbd class="px-1 py-0.5 bg-muted rounded">Enter</kbd> Start</span>
			<span><kbd class="px-1 py-0.5 bg-muted rounded">E</kbd> Edit</span>
			<span><kbd class="px-1 py-0.5 bg-muted rounded">Esc</kbd> Close</span>
		</div>
	{/if}
</div>
