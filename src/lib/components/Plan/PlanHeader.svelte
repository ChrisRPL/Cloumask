<script lang="ts" module>
	export interface PlanHeaderProps {
		stepCount: number;
		enabledCount: number;
		estimatedTime?: string;
		isEditing: boolean;
		isAwaitingApproval: boolean;
		canStart: boolean;
		isDirty: boolean;
		class?: string;
		onToggleEdit?: () => void;
		onStart?: () => void;
		onCancel?: () => void;
		onReset?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button';
	import { Play, Pencil, Check, X, RotateCcw } from '@lucide/svelte';

	let {
		stepCount,
		enabledCount,
		estimatedTime,
		isEditing,
		isAwaitingApproval,
		canStart,
		isDirty,
		class: className,
		onToggleEdit,
		onStart,
		onCancel,
		onReset,
	}: PlanHeaderProps = $props();
</script>

<header
	class={cn(
		'flex items-center justify-between px-4 py-3 border-b border-border bg-muted/20',
		className
	)}
>
	<!-- Left side: Title and stats -->
	<div class="flex items-center gap-4">
		<span class="text-forest-light font-mono font-medium"># PIPELINE</span>
		<span class="text-muted-foreground tabular-nums text-sm font-mono">
			{enabledCount}/{stepCount} steps
		</span>
		{#if estimatedTime}
			<span class="text-muted-foreground/60 text-sm font-mono">
				~{estimatedTime}
			</span>
		{/if}
		{#if isDirty}
			<span class="text-xs text-amber-500 font-mono">[modified]</span>
		{/if}
	</div>

	<!-- Right side: Actions -->
	<div class="flex items-center gap-2">
		<!-- Reset button (when dirty) -->
		{#if isDirty && !isEditing}
			<Button
				variant="ghost"
				size="sm"
				onclick={onReset}
				title="Reset changes"
			>
				<RotateCcw class="h-4 w-4 mr-1" />
				Reset
			</Button>
		{/if}

		<!-- Edit toggle -->
		<Button
			variant={isEditing ? 'default' : 'ghost'}
			size="sm"
			onclick={onToggleEdit}
			title={isEditing ? 'Done editing' : 'Edit pipeline'}
		>
			{#if isEditing}
				<Check class="h-4 w-4 mr-1" />
				Done
			{:else}
				<Pencil class="h-4 w-4 mr-1" />
				Edit
			{/if}
		</Button>

		<!-- Cancel button -->
		<Button
			variant="ghost"
			size="sm"
			onclick={onCancel}
			title="Cancel and return to chat"
		>
			<X class="h-4 w-4 mr-1" />
			Cancel
		</Button>

		<!-- Start button (prominent when awaiting approval) -->
		{#if isAwaitingApproval}
			<Button
				variant="default"
				size="sm"
				onclick={onStart}
				disabled={!canStart}
				class="bg-primary hover:bg-primary/90"
				title="Start execution (Enter)"
			>
				<Play class="h-4 w-4 mr-1" />
				Start
			</Button>
		{/if}
	</div>
</header>
