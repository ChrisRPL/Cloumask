<script lang="ts" module>
	import type { CheckpointInfo } from '$lib/types/execution';

	export interface CheckpointBannerProps {
		checkpoint: CheckpointInfo;
		class?: string;
		onContinue?: () => void;
		onReview?: () => void;
		onAbort?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button';
	import { AlertTriangle, Play, Eye, X } from '@lucide/svelte';
	import { CHECKPOINT_TRIGGERS } from './constants';

	let { checkpoint, class: className, onContinue, onReview, onAbort }: CheckpointBannerProps =
		$props();

	const triggerLabel = $derived(
		CHECKPOINT_TRIGGERS[checkpoint.triggerReason] ?? 'Checkpoint triggered'
	);
</script>

<div
	class={cn(
		'px-4 py-3 bg-amber-500/10 border-b border-amber-500/30 animate-fade-in',
		className
	)}
>
	<div class="flex items-start justify-between gap-4">
		<!-- Left side: Alert info -->
		<div class="flex items-start gap-3">
			<AlertTriangle class="h-5 w-5 text-amber-500 mt-0.5 shrink-0" />
			<div class="space-y-1">
				<!-- Terminal-style header -->
				<div class="flex items-center gap-2">
					<span class="font-mono font-semibold text-amber-500">[CHECKPOINT]</span>
					<span class="text-sm font-mono text-foreground">{triggerLabel}</span>
				</div>

				<!-- Message -->
				<p class="text-sm text-muted-foreground font-mono">{checkpoint.message}</p>

				<!-- Quality metrics -->
				<div class="flex gap-4 text-xs font-mono text-muted-foreground/80">
					<span>
						Progress: <span class="text-foreground tabular-nums">{checkpoint.progressPercent}%</span>
					</span>
					<span>
						Confidence: <span class="text-foreground tabular-nums">
							{Math.round(checkpoint.qualityMetrics.averageConfidence * 100)}%
						</span>
					</span>
					<span>
						Processed: <span class="text-foreground tabular-nums">
							{checkpoint.qualityMetrics.totalProcessed}
						</span>
					</span>
					{#if checkpoint.qualityMetrics.errorCount > 0}
						<span>
							Errors: <span class="text-destructive tabular-nums">
								{checkpoint.qualityMetrics.errorCount}
							</span>
						</span>
					{/if}
				</div>
			</div>
		</div>

		<!-- Right side: Actions -->
		<div class="flex items-center gap-2 shrink-0">
			<Button variant="ghost" size="sm" onclick={onAbort} class="text-muted-foreground">
				<X class="h-4 w-4 mr-1" />
				Abort
			</Button>
			<Button variant="outline" size="sm" onclick={onReview}>
				<Eye class="h-4 w-4 mr-1" />
				Review
			</Button>
			<Button variant="default" size="sm" onclick={onContinue} title="Continue (Enter)">
				<Play class="h-4 w-4 mr-1" />
				Continue
			</Button>
		</div>
	</div>
</div>
