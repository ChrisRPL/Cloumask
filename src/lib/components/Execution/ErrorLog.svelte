<script lang="ts" module>
	import type { ExecutionError } from '$lib/types/execution';

	export interface ErrorLogProps {
		errors: ExecutionError[];
		isExpanded?: boolean;
		class?: string;
		onToggle?: () => void;
		onClear?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { ChevronDown, Trash2 } from '@lucide/svelte';

	let {
		errors,
		isExpanded = false,
		class: className,
		onToggle,
		onClear,
	}: ErrorLogProps = $props();

	function formatTime(timestamp: string): string {
		const date = new Date(timestamp);
		return date.toLocaleTimeString('en-US', {
			hour: '2-digit',
			minute: '2-digit',
			second: '2-digit',
			hour12: false,
		});
	}
</script>

<div class={cn('border-t border-destructive/20 bg-destructive/5', className)}>
	<!-- Header (clickable to toggle) -->
	<button
		type="button"
		class="flex items-center justify-between w-full px-4 py-2 hover:bg-destructive/10 transition-colors"
		onclick={onToggle}
	>
		<span class="text-sm font-mono text-destructive">
			[ERRORS] {errors.length} issue{errors.length !== 1 ? 's' : ''}
		</span>
		<div class="flex items-center gap-2">
			{#if isExpanded}
				<Button
					variant="ghost"
					size="icon-sm"
					onclick={(e) => {
						e.stopPropagation();
						onClear?.();
					}}
					title="Clear all errors"
					class="h-6 w-6 text-muted-foreground hover:text-destructive"
				>
					<Trash2 class="h-3 w-3" />
				</Button>
			{/if}
			<ChevronDown
				class={cn('h-4 w-4 text-muted-foreground transition-transform', isExpanded && 'rotate-180')}
			/>
		</div>
	</button>

	<!-- Error list (collapsible) -->
	{#if isExpanded}
		<ScrollArea class="max-h-40">
			<div class="px-4 pb-3 space-y-1">
				{#each errors as error, i (i)}
					<div
						class={cn(
							'flex items-start gap-2 text-xs font-mono py-1 px-2 rounded',
							error.recoverable ? 'bg-amber-500/10' : 'bg-destructive/10'
						)}
					>
						<span class="text-muted-foreground shrink-0 tabular-nums">
							[{formatTime(error.timestamp)}]
						</span>
						<span class="text-muted-foreground shrink-0">
							{error.stepId}:
						</span>
						<span class={error.recoverable ? 'text-amber-600' : 'text-destructive'}>
							{error.message}
						</span>
						{#if error.recoverable}
							<span class="text-muted-foreground/60 shrink-0">(recoverable)</span>
						{/if}
					</div>
				{/each}
			</div>
		</ScrollArea>
	{/if}
</div>
