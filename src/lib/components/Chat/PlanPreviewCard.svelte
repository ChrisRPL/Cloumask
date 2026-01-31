<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { PipelineStep } from '$lib/types/pipeline';

	export interface PlanPreviewCardProps {
		steps: PipelineStep[];
		class?: string;
		onViewPlan?: () => void;
	}
</script>

<script lang="ts">
	import { ChevronRight, Circle, CheckCircle2, XCircle, Loader2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	let {
		steps,
		class: className,
		onViewPlan
	}: PlanPreviewCardProps = $props();

	// Get status icon component
	function getStatusIcon(status: string) {
		switch (status) {
			case 'completed':
				return CheckCircle2;
			case 'failed':
				return XCircle;
			case 'running':
				return Loader2;
			default:
				return Circle;
		}
	}

	// Get status color
	function getStatusColor(status: string): string {
		switch (status) {
			case 'completed':
				return 'text-forest-light';
			case 'failed':
				return 'text-destructive';
			case 'running':
				return 'text-forest-light animate-spin';
			default:
				return 'text-muted-foreground/40';
		}
	}

	// Summary stats
	const totalSteps = $derived(steps.length);
	const completedSteps = $derived(steps.filter((s) => s.status === 'completed').length);
	const hasRunning = $derived(steps.some((s) => s.status === 'running'));
</script>

{#if steps.length > 0}
	<div
		class={cn(
			'rounded-lg border border-border bg-card/30 overflow-hidden',
			className
		)}
	>
		<!-- Header -->
		<div class="flex items-center justify-between px-3 py-2 border-b border-border/50 bg-muted/20">
			<div class="flex items-center gap-2">
				<span class="text-xs font-medium text-muted-foreground uppercase tracking-wide">
					Pipeline
				</span>
				<span class="text-xs text-muted-foreground/60 tabular-nums">
					{completedSteps}/{totalSteps}
				</span>
			</div>
			{#if onViewPlan}
				<Button
					variant="ghost"
					size="sm"
					class="h-6 px-2 text-xs gap-1"
					onclick={onViewPlan}
				>
					View
					<ChevronRight class="h-3 w-3" />
				</Button>
			{/if}
		</div>

		<!-- Steps list (compact) -->
		<div class="p-2 space-y-1 max-h-40 overflow-y-auto">
			{#each steps.slice(0, 5) as step, index (step.id)}
				{@const StatusIcon = getStatusIcon(step.status)}
				<div class="flex items-center gap-2 text-xs">
					<span class="w-4 h-4 shrink-0 flex items-center justify-center">
						<StatusIcon class={cn('h-3 w-3', getStatusColor(step.status))} />
					</span>
					<span class="text-muted-foreground/60 tabular-nums w-4">{index + 1}.</span>
					<span class="truncate text-foreground/80">
						{step.description || step.toolName}
					</span>
				</div>
			{/each}

			{#if steps.length > 5}
				<div class="text-xs text-muted-foreground/60 pl-6">
					+{steps.length - 5} more steps
				</div>
			{/if}
		</div>
	</div>
{/if}
