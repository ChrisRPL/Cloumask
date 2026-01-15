<script lang="ts" module>
	import type { PipelineStep } from '$lib/types/pipeline';

	export interface StepListProps {
		steps: PipelineStep[];
		isEditing?: boolean;
		selectedStepId?: string | null;
		class?: string;
		onReorder?: (fromIndex: number, toIndex: number) => void;
		onSelectStep?: (id: string) => void;
		onOpenConfig?: (id: string) => void;
		onToggleEnabled?: (id: string) => void;
		onDeleteStep?: (id: string) => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { draggable, droppable, type DragDropState } from '@thisux/sveltednd';
	import StepListItem from './StepListItem.svelte';

	let {
		steps,
		isEditing = false,
		selectedStepId = null,
		class: className,
		onReorder,
		onSelectStep,
		onOpenConfig,
		onToggleEnabled,
		onDeleteStep,
	}: StepListProps = $props();

	// Track drag state
	let draggingStepId = $state<string | null>(null);
	let dragOverIndex = $state<number | null>(null);

	// Handle drag start
	function handleDragStart(step: PipelineStep) {
		draggingStepId = step.id;
	}

	// Handle drag end
	function handleDragEnd() {
		draggingStepId = null;
		dragOverIndex = null;
	}

	// Handle drag over
	function handleDragOver(index: number) {
		dragOverIndex = index;
	}

	// Handle drag leave
	function handleDragLeave() {
		dragOverIndex = null;
	}

	// Handle drop
	function handleDrop(state: DragDropState<PipelineStep>) {
		const { draggedItem, targetContainer } = state;
		if (!targetContainer || !draggedItem) return;

		const fromIndex = steps.findIndex((s) => s.id === draggedItem.id);
		const toIndex = parseInt(targetContainer);

		if (fromIndex !== -1 && !isNaN(toIndex) && fromIndex !== toIndex) {
			onReorder?.(fromIndex, toIndex);
		}

		draggingStepId = null;
		dragOverIndex = null;
	}
</script>

<div class={cn('space-y-1', className)}>
	{#if steps.length === 0}
		<div class="flex flex-col items-center justify-center py-12 text-muted-foreground">
			<p class="text-sm">No steps in pipeline</p>
			<p class="text-xs mt-1">Steps will appear here when the agent creates a plan</p>
		</div>
	{:else}
		{#each steps as step, index (step.id)}
			<div
				class="relative"
				use:droppable={{
					container: String(index),
					callbacks: {
						onDrop: handleDrop,
						onDragEnter: () => handleDragOver(index),
						onDragLeave: handleDragLeave,
					},
				}}
			>
				<!-- Drop indicator line -->
				{#if dragOverIndex === index && draggingStepId !== step.id}
					<div class="absolute -top-0.5 left-0 right-0 h-0.5 bg-primary rounded-full z-10"></div>
				{/if}

				<div
					use:draggable={{
						container: String(index),
						dragData: step,
						disabled: !isEditing,
						callbacks: {
							onDragStart: () => handleDragStart(step),
							onDragEnd: handleDragEnd,
						},
					}}
				>
					<StepListItem
						{step}
						{index}
						{isEditing}
						isSelected={selectedStepId === step.id}
						isDragging={draggingStepId === step.id}
						onSelect={() => onSelectStep?.(step.id)}
						onOpenConfig={() => onOpenConfig?.(step.id)}
						onToggleEnabled={() => onToggleEnabled?.(step.id)}
						onDelete={() => onDeleteStep?.(step.id)}
					/>
				</div>
			</div>
		{/each}

		<!-- Drop zone at the end -->
		{#if isEditing && draggingStepId}
			<div
				class="relative h-8"
				use:droppable={{
					container: String(steps.length),
					callbacks: {
						onDrop: handleDrop,
						onDragEnter: () => handleDragOver(steps.length),
						onDragLeave: handleDragLeave,
					},
				}}
			>
				{#if dragOverIndex === steps.length}
					<div class="absolute top-0 left-0 right-0 h-0.5 bg-primary rounded-full"></div>
				{/if}
			</div>
		{/if}
	{/if}
</div>

<style>
	:global(.svelte-dnd-dragging) {
		opacity: 0.6;
		cursor: grabbing;
	}
</style>
