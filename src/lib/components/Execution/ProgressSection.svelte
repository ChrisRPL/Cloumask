<script lang="ts" module>
	export interface ProgressSectionProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { getExecutionState } from '$lib/stores/execution.svelte';
	import { getPipelineState } from '$lib/stores/pipeline.svelte';
	import StepProgress from './StepProgress.svelte';
	import ProgressBar from './ProgressBar.svelte';
	import TimeDisplay from './TimeDisplay.svelte';

	let { class: className }: ProgressSectionProps = $props();

	const execution = getExecutionState();
	const pipeline = getPipelineState();
</script>

<div class={cn('space-y-3', className)}>
	<!-- Step progress indicators -->
	<StepProgress steps={pipeline.steps} currentStepId={execution.currentStepId} />

	<!-- Progress bar + time display -->
	<div class="flex items-center gap-4">
		<ProgressBar
			current={execution.progress.current}
			total={execution.progress.total}
			percentage={execution.progress.percentage}
			status={execution.status}
			class="flex-1"
		/>
		<TimeDisplay
			startedAt={execution.stats.startedAt}
			estimatedCompletion={execution.stats.estimatedCompletion}
		/>
	</div>
</div>
