<script lang="ts" module>
	import type { ExecutionStatus } from '$lib/types/execution';

	export interface ExecutionHeaderProps {
		status: ExecutionStatus;
		currentStepTitle: string;
		currentStepIndex: number;
		totalSteps: number;
		canPause: boolean;
		canResume: boolean;
		class?: string;
		onPause?: () => void;
		onResume?: () => void;
		onCancel?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button';
	import { Pause, Play, Square } from '@lucide/svelte';
	import { STATUS_DISPLAY } from './constants';

	let {
		status,
		currentStepTitle,
		currentStepIndex,
		totalSteps,
		canPause,
		canResume,
		class: className,
		onPause,
		onResume,
		onCancel,
	}: ExecutionHeaderProps = $props();

	const statusConfig = $derived(STATUS_DISPLAY[status]);
</script>

<header
	class={cn(
		'flex items-center justify-between px-4 py-3 border-b border-border bg-muted/20',
		className
	)}
>
	<!-- Left side: Status and step info -->
	<div class="flex items-center gap-4">
		<!-- Status badge (terminal style) -->
		<span class={cn('font-mono font-medium', statusConfig.color)}>
			&lt;{statusConfig.label}&gt;
		</span>

		<!-- Step counter -->
		{#if totalSteps > 0}
			<span class="text-muted-foreground tabular-nums text-sm font-mono">
				Step {currentStepIndex}/{totalSteps}
			</span>
		{/if}

		<!-- Current step title -->
		<span class="text-foreground font-mono text-sm truncate max-w-md">
			{currentStepTitle}
		</span>
	</div>

	<!-- Right side: Controls -->
	<div class="flex items-center gap-2">
		<!-- Pause/Resume button -->
		{#if canPause}
			<Button
				variant="ghost"
				size="sm"
				onclick={onPause}
				title="Pause execution (Space)"
			>
				<Pause class="h-4 w-4 mr-1" />
				Pause
			</Button>
		{:else if canResume}
			<Button
				variant="default"
				size="sm"
				onclick={onResume}
				title="Resume execution (Space)"
			>
				<Play class="h-4 w-4 mr-1" />
				Resume
			</Button>
		{/if}

		<!-- Cancel button -->
		<Button
			variant="ghost"
			size="sm"
			onclick={onCancel}
			title="Cancel execution (Esc)"
			class="text-muted-foreground hover:text-destructive"
		>
			<Square class="h-4 w-4 mr-1" />
			Cancel
		</Button>
	</div>
</header>
