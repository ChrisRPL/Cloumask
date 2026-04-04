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
		<div class="flex items-center justify-center px-6 py-12">
			<div class="w-full max-w-5xl rounded-2xl border border-border/70 bg-card/35 p-6 shadow-sm">
				<div class="grid gap-6 lg:grid-cols-[1.2fr_0.92fr]">
					<div class="space-y-5 font-mono">
						<div class="space-y-3">
							<div class="inline-flex items-center rounded-full border border-border/70 bg-background px-3 py-1 text-[11px] uppercase tracking-[0.24em] text-muted-foreground">
								Pipeline workspace
							</div>
							<div class="space-y-2">
								<p class="text-2xl text-foreground">No steps in pipeline</p>
								<p class="max-w-2xl text-sm leading-7 text-muted-foreground">
									Describe the job in Chat to generate a plan, then come back here to review the
									steps.
								</p>
							</div>
						</div>

						<div class="grid gap-3 sm:grid-cols-3">
							<div class="rounded-xl border border-border/60 bg-background/70 p-4">
								<p class="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
									Draft
								</p>
								<p class="mt-3 text-sm leading-6 text-foreground/80">
									Chat turns the current project brief into a step-by-step pipeline here.
								</p>
							</div>
							<div class="rounded-xl border border-border/60 bg-background/70 p-4">
								<p class="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
									Tune
								</p>
								<p class="mt-3 text-sm leading-6 text-foreground/80">
									Adjust step order, enable or skip tools, and refine configs before the run.
								</p>
							</div>
							<div class="rounded-xl border border-border/60 bg-background/70 p-4">
								<p class="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
									Approve
								</p>
								<p class="mt-3 text-sm leading-6 text-foreground/80">
									Send the checked plan to Execute only when the workflow looks right.
								</p>
							</div>
						</div>
					</div>

					<div class="rounded-xl border border-border/60 bg-background/70 p-5 font-mono">
						<p class="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
							What happens here
						</p>
						<div class="mt-4 space-y-3 text-sm leading-7 text-foreground/80">
							<p>1. Start in Chat with the job goal and current project context.</p>
							<p>2. Return here to inspect the generated steps before execution.</p>
							<p>3. Use Execute after the pipeline is approved and ready to run.</p>
						</div>
						<div class="mt-5 rounded-xl border border-emerald-700/15 bg-emerald-500/8 px-4 py-3 text-xs leading-6 text-emerald-900/75">
							Shortcut path: 1 chat, 2 plan, 3 execute.
						</div>
					</div>
				</div>
			</div>
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
						interactive: ['button', 'input', '[role="checkbox"]', '[data-no-drag]'],
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
