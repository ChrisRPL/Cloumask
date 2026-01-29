<script lang="ts" module>
	export interface ExecutionViewProps {
		class?: string;
		onComplete?: () => void;
		onCancel?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { getExecutionState } from '$lib/stores/execution.svelte';
	import { getPipelineState } from '$lib/stores/pipeline.svelte';
	import { getAgentState } from '$lib/stores/agent.svelte';
	import { getUIState } from '$lib/stores/ui.svelte';
	import { getKeyboardState } from '$lib/stores/keyboard.svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import ExecutionHeader from './ExecutionHeader.svelte';
	import ProgressSection from './ProgressSection.svelte';
	import CheckpointBanner from './CheckpointBanner.svelte';
	import PreviewGrid from './PreviewGrid.svelte';
	import StatsPanel from './StatsPanel.svelte';
	import ErrorLog from './ErrorLog.svelte';
	import { KEYBOARD_SHORTCUTS } from './constants';

	let { class: className, onComplete, onCancel }: ExecutionViewProps = $props();

	// Get stores from context
	const execution = getExecutionState();
	const pipeline = getPipelineState();
	const agent = getAgentState();
	const ui = getUIState();
	const keyboard = getKeyboardState();

	// Local state
	let showErrorLog = $state(false);
	let confirmCancelOpen = $state(false);
	let userDismissedErrors = $state(false);

	// Derived state
	const currentStep = $derived(
		pipeline.steps.find((s) => s.id === execution.currentStepId) ?? null
	);
	const currentStepIndex = $derived(
		currentStep ? pipeline.steps.findIndex((s) => s.id === currentStep.id) + 1 : 0
	);
	const canPause = $derived(execution.isRunning);
	const canResume = $derived(execution.isPaused || execution.status === 'checkpoint');
	const isCheckpoint = $derived(execution.status === 'checkpoint');

	// ============================================================================
	// Keyboard Shortcuts (registered with keyboard store for scope awareness)
	// ============================================================================

	$effect(() => {
		const unregisterFns: (() => void)[] = [];

		// Escape - show cancel confirmation
		unregisterFns.push(
			(() => {
				const id = keyboard.register({
					combo: 'escape',
					action: () => {
						if (execution.isRunning || execution.isPaused) {
							confirmCancelOpen = true;
						}
					},
					scope: 'execution',
					description: 'Cancel execution',
					category: 'Execution',
					priority: 10, // Lower than global escape
				});
				return () => keyboard.unregister(id);
			})()
		);

		// E - toggle error log
		unregisterFns.push(
			(() => {
				const id = keyboard.register({
					combo: 'e',
					action: () => {
						showErrorLog = !showErrorLog;
					},
					scope: 'execution',
					description: 'Toggle error log',
					category: 'Execution',
				});
				return () => keyboard.unregister(id);
			})()
		);

		// Note: Space (pause/resume), Enter (continue), and R (review)
		// are registered globally in +layout.svelte

		return () => {
			for (const fn of unregisterFns) fn();
		};
	});

	// Actions
	function handlePause() {
		execution.pause();
	}

	function handleResume() {
		execution.resume();
	}

	function handleCancel() {
		confirmCancelOpen = true;
	}

	function confirmCancel() {
		execution.cancel();
		confirmCancelOpen = false;
		onCancel?.();
	}

	function handleContinue() {
		execution.clearCheckpoint();
	}

	function handleReview() {
		ui.setView('review');
	}

	function handleAbort() {
		execution.cancel();
	}

	function toggleErrorLog() {
		showErrorLog = !showErrorLog;
		if (!showErrorLog) {
			userDismissedErrors = true;
		}
	}

	// Auto-show error log when new errors occur (unless user dismissed)
	$effect(() => {
		if (execution.hasErrors && !showErrorLog && !userDismissedErrors) {
			showErrorLog = true;
		}
	});

	// Reset dismissal flag when errors are cleared
	$effect(() => {
		if (!execution.hasErrors) {
			userDismissedErrors = false;
		}
	});

	// Notify completion
	$effect(() => {
		if (execution.status === 'completed') {
			onComplete?.();
		}
	});
</script>

<div class={cn('flex flex-col h-full bg-background', className)}>
	<!-- Header with controls -->
	<ExecutionHeader
		status={execution.status}
		currentStepTitle={currentStep?.description ?? 'Initializing...'}
		currentStepIndex={currentStepIndex}
		totalSteps={pipeline.steps.length}
		{canPause}
		{canResume}
		onPause={handlePause}
		onResume={handleResume}
		onCancel={handleCancel}
	/>

	<!-- Checkpoint banner (inline, pushes content down) -->
	{#if execution.checkpoint}
		<CheckpointBanner
			checkpoint={execution.checkpoint}
			onContinue={handleContinue}
			onReview={handleReview}
			onAbort={handleAbort}
		/>
	{/if}

	<!-- Progress section -->
	<ProgressSection class="px-4 py-3 border-b border-border" />

	<!-- Main content: Preview grid + Stats panel -->
	<div class="flex-1 overflow-hidden flex">
		<PreviewGrid class="flex-1" />
		<StatsPanel class="w-80 border-l border-border" />
	</div>

	<!-- Error log (collapsible, at bottom) -->
	{#if execution.hasErrors}
		<ErrorLog
			errors={execution.errors}
			isExpanded={showErrorLog}
			onToggle={toggleErrorLog}
			onClear={() => execution.clearErrors()}
		/>
	{/if}

	<!-- Keyboard hints footer -->
	<div
		class="px-4 py-2 border-t border-border text-xs text-muted-foreground/60 font-mono flex gap-4"
	>
		{#each KEYBOARD_SHORTCUTS as shortcut}
			<span>
				<kbd class="px-1 py-0.5 bg-muted/30 rounded text-muted-foreground">{shortcut.key}</kbd>
				{shortcut.action}
			</span>
		{/each}
	</div>
</div>

<!-- Cancel confirmation dialog -->
<Dialog.Root bind:open={confirmCancelOpen}>
	<Dialog.Content class="sm:max-w-md">
		<Dialog.Header>
			<Dialog.Title class="font-mono">Cancel Execution?</Dialog.Title>
			<Dialog.Description>
				This will stop the current pipeline execution. Progress will be lost.
			</Dialog.Description>
		</Dialog.Header>
		<Dialog.Footer class="flex gap-2">
			<Button variant="ghost" onclick={() => (confirmCancelOpen = false)}>Keep Running</Button>
			<Button variant="destructive" onclick={confirmCancel}>Cancel Execution</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
