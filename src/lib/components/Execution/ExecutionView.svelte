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

	// Local state
	let showErrorLog = $state(false);
	let confirmCancelOpen = $state(false);

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

	// Keyboard handler
	function handleKeydown(event: KeyboardEvent) {
		// Ignore if typing in an input
		if (
			event.target instanceof HTMLInputElement ||
			event.target instanceof HTMLTextAreaElement
		) {
			return;
		}

		switch (event.code) {
			case 'Space':
				event.preventDefault();
				if (canPause) {
					execution.pause();
				} else if (canResume) {
					execution.resume();
				}
				break;
			case 'Escape':
				event.preventDefault();
				if (execution.isRunning || execution.isPaused) {
					confirmCancelOpen = true;
				}
				break;
			case 'KeyR':
				event.preventDefault();
				ui.setView('review');
				break;
			case 'Enter':
				event.preventDefault();
				if (isCheckpoint) {
					handleContinue();
				}
				break;
			case 'KeyE':
				event.preventDefault();
				showErrorLog = !showErrorLog;
				break;
		}
	}

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

	// Set up keyboard listener
	$effect(() => {
		if (typeof window === 'undefined') return;
		window.addEventListener('keydown', handleKeydown);
		return () => window.removeEventListener('keydown', handleKeydown);
	});

	// Auto-show error log when errors occur
	$effect(() => {
		if (execution.hasErrors && !showErrorLog) {
			showErrorLog = true;
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
			onToggle={() => (showErrorLog = !showErrorLog)}
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
