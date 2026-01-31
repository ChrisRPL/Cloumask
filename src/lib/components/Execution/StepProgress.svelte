<script lang="ts" module>
	import type { PipelineStep } from '$lib/types/pipeline';

	export interface StepProgressProps {
		steps: PipelineStep[];
		currentStepId: string | null;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Check, AlertCircle, Loader2, Circle } from '@lucide/svelte';
	import { STEP_STATUS_CLASSES, STEP_LINE_CLASSES } from './constants';

	let { steps, currentStepId, class: className }: StepProgressProps = $props();

	function getStepDotClass(step: PipelineStep, isCurrent: boolean) {
		if (isCurrent && step.status === 'running') {
			return STEP_STATUS_CLASSES.running;
		}
		return STEP_STATUS_CLASSES[step.status] ?? STEP_STATUS_CLASSES.pending;
	}

	function getLineClass(prevStep: PipelineStep) {
		return STEP_LINE_CLASSES[prevStep.status] ?? STEP_LINE_CLASSES.pending;
	}
</script>

<div class={cn('flex items-center gap-1', className)}>
	{#each steps as step, i (step.id)}
		<!-- Connecting line (except for first step) -->
		{#if i > 0}
			<div class={cn('h-0.5 flex-1 min-w-4 transition-colors', getLineClass(steps[i - 1]))}></div>
		{/if}

		<!-- Step indicator -->
		<div
			class={cn(
				'relative flex items-center justify-center w-6 h-6 rounded-full transition-all',
				getStepDotClass(step, step.id === currentStepId)
			)}
			title={`${step.description} (${step.status})`}
		>
			{#if step.status === 'completed'}
				<Check class="h-3 w-3 text-primary-foreground" />
			{:else if step.status === 'running'}
				<Loader2 class="h-3 w-3 text-primary-foreground animate-spin" />
			{:else if step.status === 'failed'}
				<AlertCircle class="h-3 w-3 text-destructive-foreground" />
			{:else}
				<Circle class="h-2 w-2 text-muted-foreground" />
			{/if}
		</div>
	{/each}

	{#if steps.length === 0}
		<span class="text-xs text-muted-foreground font-mono">No steps defined</span>
	{/if}
</div>
