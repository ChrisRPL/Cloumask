<script lang="ts" module>
	export interface ExecutionViewProps {
		class?: string;
		onComplete?: () => void;
		onCancel?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { untrack } from 'svelte';
	import { getExecutionState } from '$lib/stores/execution.svelte';
	import { getPipelineState } from '$lib/stores/pipeline.svelte';
	import { getAgentState } from '$lib/stores/agent.svelte';
	import { getUIState } from '$lib/stores/ui.svelte';
	import { getKeyboardState } from '$lib/stores/keyboard.svelte';
	import { sendMessage } from '$lib/utils/tauri';
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
	const showExecutionWorkspace = $derived.by(() => {
		if (!pipeline.steps.length) return false;
		if (execution.previews.length > 0) return true;
		if (execution.hasErrors) return true;
		if (execution.isRunning || execution.isPaused) return true;
		return ['checkpoint', 'completed', 'failed', 'cancelled'].includes(execution.status);
	});
	const visibleKeyboardShortcuts = $derived.by(() => {
		const visibleKeys = new Set<string>(['R']);

		if (execution.isRunning || execution.isPaused) {
			visibleKeys.add('Space');
			visibleKeys.add('Esc');
		}

		if (execution.checkpoint && agent.threadId) {
			visibleKeys.add('Enter');
		}

		if (execution.hasErrors) {
			visibleKeys.add('E');
		}

		return KEYBOARD_SHORTCUTS.filter((shortcut) => visibleKeys.has(shortcut.key));
	});

	// ============================================================================
	// Keyboard Shortcuts (registered with keyboard store for scope awareness)
	// ============================================================================

	// NOTE: We use untrack() to prevent this effect from tracking registeredShortcuts reads
	// inside keyboard.register(). Without untrack, register() reads the shortcuts map,
	// which would cause an infinite loop (effect_update_depth_exceeded error).
	$effect(() => {
		const unregisterFns: (() => void)[] = [];

		untrack(() => {
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
		}); // end untrack

		// Note: Space (pause/resume), Enter (continue), and R (review)
		// are registered globally in +layout.svelte

		return () => {
			untrack(() => {
				for (const fn of unregisterFns) fn();
			});
		};
	});

	// Actions
	function handlePause() {
		execution.pause();
	}

	function handleResume() {
		if (execution.status === 'checkpoint') {
			handleContinue();
			return;
		}
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

	async function handleContinue() {
		if (!execution.checkpoint || !agent.threadId) {
			execution.clearCheckpoint();
			return;
		}
		try {
			await sendMessage(agent.threadId, {
				content: 'continue',
				decision: 'approve'
			});
			execution.clearCheckpoint();
		} catch (error) {
			console.error('[ExecutionView] Failed to continue checkpoint:', error);
			execution.addError({
				stepId: execution.currentStepId ?? 'checkpoint',
				message: 'Failed to continue checkpoint',
				recoverable: true
			});
		}
	}

	function handleReview() {
		ui.setView('review');
	}

	async function handleAbort() {
		if (execution.checkpoint && agent.threadId) {
			try {
				await sendMessage(agent.threadId, {
					content: 'cancel',
					decision: 'cancel'
				});
			} catch (error) {
				console.error('[ExecutionView] Failed to abort checkpoint:', error);
			}
		}
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
	{#if showExecutionWorkspace}
		<div class="flex flex-1 overflow-hidden">
			<PreviewGrid class="flex-1" />
			<StatsPanel class="w-80 border-l border-border" />
		</div>
	{:else}
		<div class="flex flex-1 items-center justify-center px-6 py-12">
			<div class="w-full max-w-4xl rounded-2xl border border-border/70 bg-card/35 p-6 shadow-sm">
				<div class="grid gap-6 lg:grid-cols-[1.3fr_0.95fr]">
					<div class="space-y-5 font-mono">
						<div class="space-y-3">
							<div class="inline-flex items-center rounded-full border border-border/70 bg-background px-3 py-1 text-[11px] uppercase tracking-[0.24em] text-muted-foreground">
								Execution workspace
							</div>
							<div class="space-y-2">
								<p class="text-2xl text-foreground">No live execution yet</p>
								<p class="max-w-2xl text-sm leading-7 text-muted-foreground">
									Start a pipeline in Chat or Plan. When the run begins, this view fills with
									recent previews, progress, counts, and agent commentary in one place.
								</p>
							</div>
						</div>

						<div class="grid gap-3 sm:grid-cols-3">
							<div class="rounded-xl border border-border/60 bg-background/70 p-4">
								<p class="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
									Previews
								</p>
								<p class="mt-3 text-sm leading-6 text-foreground/80">
									Recent frames and detections land here as the pipeline moves.
								</p>
							</div>
							<div class="rounded-xl border border-border/60 bg-background/70 p-4">
								<p class="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
									Stats
								</p>
								<p class="mt-3 text-sm leading-6 text-foreground/80">
									Processed totals, detections, flags, and errors stay visible.
								</p>
							</div>
							<div class="rounded-xl border border-border/60 bg-background/70 p-4">
								<p class="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
									Agent output
								</p>
								<p class="mt-3 text-sm leading-6 text-foreground/80">
									Checkpoints and finish notes show up beside the run context.
								</p>
							</div>
						</div>
					</div>

					<div class="rounded-xl border border-border/60 bg-background/70 p-5 font-mono">
						<p class="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
							Good next steps
						</p>
						<div class="mt-4 space-y-3 text-sm leading-7 text-foreground/80">
							<p>1. Describe the job in Chat and approve the plan.</p>
							<p>2. Return here to watch previews and progress.</p>
							<p>3. Jump to Review when the run is ready for triage.</p>
						</div>
						<div class="mt-5 rounded-xl border border-emerald-700/15 bg-emerald-500/8 px-4 py-3 text-xs leading-6 text-emerald-900/75">
							Shortcut path: <span class="font-semibold text-foreground/85">1</span> chat,
							<span class="font-semibold text-foreground/85"> 2</span> plan,
							<span class="font-semibold text-foreground/85"> 3</span> execute,
							<span class="font-semibold text-foreground/85"> R</span> review.
						</div>
					</div>
				</div>
			</div>
		</div>
	{/if}

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
		{#each visibleKeyboardShortcuts as shortcut}
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
