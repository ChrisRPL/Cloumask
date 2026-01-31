<script lang="ts" module>
	import type { ExecutionStatus } from '$lib/types/execution';

	export interface ProgressBarProps {
		current: number;
		total: number;
		percentage: number;
		status: ExecutionStatus;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Progress } from '$lib/components/ui/progress';

	let { current, total, percentage, status, class: className }: ProgressBarProps = $props();

	// Color based on status
	const progressColor = $derived.by(() => {
		switch (status) {
			case 'running':
				return 'bg-primary';
			case 'paused':
				return 'bg-amber-500';
			case 'checkpoint':
				return 'bg-amber-400';
			case 'failed':
				return 'bg-destructive';
			case 'completed':
				return 'bg-green-600';
			default:
				return 'bg-primary';
		}
	});
</script>

<div class={cn('flex items-center gap-3', className)}>
	<!-- Progress bar -->
	<div class="flex-1 relative">
		<Progress value={percentage} max={100} class="h-2" />
		<!-- Custom colored fill overlay -->
		<div
			class={cn(
				'absolute inset-0 h-2 rounded-full transition-all duration-300',
				progressColor
			)}
			style="width: {percentage}%"
		></div>
	</div>

	<!-- Percentage display -->
	<span class="text-sm font-mono tabular-nums text-muted-foreground min-w-[4ch] text-right">
		{percentage}%
	</span>

	<!-- Items counter -->
	<span class="text-xs font-mono tabular-nums text-muted-foreground/60">
		{current}/{total}
	</span>
</div>
