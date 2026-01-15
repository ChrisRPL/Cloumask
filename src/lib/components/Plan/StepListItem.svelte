<script lang="ts" module>
	import type { PipelineStep } from '$lib/types/pipeline';

	export interface StepListItemProps {
		step: PipelineStep;
		index: number;
		isEditing?: boolean;
		isSelected?: boolean;
		isDragging?: boolean;
		class?: string;
		onOpenConfig?: () => void;
		onToggleEnabled?: () => void;
		onDelete?: () => void;
		onSelect?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import {
		GripVertical,
		Settings,
		Trash2,
		Search,
		Scissors,
		EyeOff,
		Download,
		Tag,
		Wand2,
	} from '@lucide/svelte';
	import { STEP_TYPE_CONFIGS, STATUS_LABELS, STATUS_COLORS } from './constants';

	let {
		step,
		index,
		isEditing = false,
		isSelected = false,
		isDragging = false,
		class: className,
		onOpenConfig,
		onToggleEnabled,
		onDelete,
		onSelect,
	}: StepListItemProps = $props();

	// Icon mapping
	const iconMap = {
		Search,
		Scissors,
		EyeOff,
		Download,
		Tag,
		Wand2,
	} as const;

	// Get step type config
	const typeConfig = $derived(STEP_TYPE_CONFIGS[step.type]);
	const StepIcon = $derived(iconMap[typeConfig?.icon as keyof typeof iconMap] ?? Search);

	// Format config summary
	const configSummary = $derived(() => {
		const parts: string[] = [];
		if (step.config.model) {
			parts.push(String(step.config.model));
		}
		if (step.config.confidence !== undefined) {
			parts.push(`${Math.round(Number(step.config.confidence) * 100)}%`);
		}
		return parts.join(' @ ');
	});

	// Status display
	const statusLabel = $derived(STATUS_LABELS[step.status] ?? `[${step.status}]`);
	const statusColor = $derived(STATUS_COLORS[step.status] ?? 'text-muted-foreground');

	// Is step enabled (not skipped)
	const isEnabled = $derived(step.status !== 'skipped');

	function handleCheckChange(checked: boolean | 'indeterminate') {
		if (checked !== 'indeterminate') {
			onToggleEnabled?.();
		}
	}
</script>

<div
	class={cn(
		'group flex items-center gap-3 px-3 py-2.5 rounded-md',
		'border border-transparent transition-all duration-150',
		isSelected && 'border-border bg-muted/30',
		isDragging && 'opacity-50 scale-[1.02] shadow-lg',
		!isEnabled && 'opacity-50',
		!isDragging && !isSelected && 'hover:bg-muted/20',
		className
	)}
	onclick={onSelect}
	onkeydown={(e) => e.key === 'Enter' && onSelect?.()}
	role="button"
	tabindex="0"
>
	<!-- Drag handle -->
	{#if isEditing}
		<div class="cursor-grab text-muted-foreground/40 hover:text-muted-foreground active:cursor-grabbing">
			<GripVertical class="h-4 w-4" />
		</div>
	{/if}

	<!-- Enable/disable checkbox -->
	<Checkbox
		checked={isEnabled}
		onCheckedChange={handleCheckChange}
		disabled={!isEditing}
		class="shrink-0"
	/>

	<!-- Step number + type icon -->
	<div class="flex items-center gap-2 min-w-0 shrink-0">
		<span class="text-muted-foreground/60 tabular-nums w-5 text-right text-sm">
			{index + 1}.
		</span>
		<StepIcon class="h-4 w-4 text-forest-light shrink-0" />
	</div>

	<!-- Step name and config summary -->
	<div class="flex-1 min-w-0">
		<span class="text-sm truncate block font-medium">
			{step.description || step.toolName}
		</span>
		{#if configSummary()}
			<span class="text-xs text-muted-foreground/60 truncate block font-mono">
				{configSummary()}
			</span>
		{/if}
	</div>

	<!-- Status indicator -->
	<span class={cn('text-xs font-mono shrink-0', statusColor)}>
		{statusLabel}
	</span>

	<!-- Config button -->
	<button
		class={cn(
			'p-1 rounded transition-opacity text-muted-foreground hover:text-foreground',
			'opacity-0 group-hover:opacity-100 focus:opacity-100'
		)}
		onclick={(e) => {
			e.stopPropagation();
			onOpenConfig?.();
		}}
		title="Configure step"
	>
		<Settings class="h-4 w-4" />
	</button>

	<!-- Delete button (edit mode only) -->
	{#if isEditing}
		<button
			class={cn(
				'p-1 rounded transition-opacity text-muted-foreground hover:text-destructive',
				'opacity-0 group-hover:opacity-100 focus:opacity-100'
			)}
			onclick={(e) => {
				e.stopPropagation();
				onDelete?.();
			}}
			title="Delete step"
		>
			<Trash2 class="h-4 w-4" />
		</button>
	{/if}
</div>
