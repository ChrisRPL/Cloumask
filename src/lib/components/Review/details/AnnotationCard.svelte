<script lang="ts" module>
	export interface AnnotationCardProps {
		annotation: import('$lib/types/review').Annotation;
		isSelected?: boolean;
		availableLabels?: string[];
		onSelect?: () => void;
		onLabelChange?: (label: string) => void;
		onDelete?: () => void;
		onVisibilityToggle?: () => void;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import * as Select from '$lib/components/ui/select/index.js';
	import { Eye, EyeOff, Trash2, Square, Pentagon, Image } from '@lucide/svelte';
	import type { Annotation } from '$lib/types/review';

	let {
		annotation,
		isSelected = false,
		availableLabels = ['person', 'face', 'license_plate', 'vehicle', 'sign', 'other'],
		onSelect,
		onLabelChange,
		onDelete,
		onVisibilityToggle,
		class: className
	}: AnnotationCardProps = $props();

	const typeIcon = $derived.by(() => {
		switch (annotation.type) {
			case 'bbox':
				return Square;
			case 'polygon':
				return Pentagon;
			case 'mask':
				return Image;
			default:
				return Square;
		}
	});

	const confidenceColor = $derived.by(() => {
		if (annotation.confidence >= 0.8) return 'text-green-500';
		if (annotation.confidence >= 0.5) return 'text-amber-500';
		return 'text-red-500';
	});

	function formatBbox(ann: Annotation): string {
		if (!ann.bbox) return '';
		const b = ann.bbox;
		return `x:${(b.x * 100).toFixed(1)}% y:${(b.y * 100).toFixed(1)}% w:${(b.width * 100).toFixed(1)}% h:${(b.height * 100).toFixed(1)}%`;
	}

	function formatPolygon(ann: Annotation): string {
		if (!ann.polygon) return '';
		return `${ann.polygon.length} points`;
	}
</script>

<div
	class={cn(
		'group relative p-3 rounded-lg border transition-all cursor-pointer',
		'hover:border-primary/50 hover:bg-muted/50',
		isSelected ? 'border-primary bg-primary/5 ring-1 ring-primary/20' : 'border-border bg-card',
		!annotation.visible && 'opacity-50',
		className
	)}
	role="button"
	tabindex="0"
	onclick={onSelect}
	onkeydown={(e) => e.key === 'Enter' && onSelect?.()}
>
	<!-- Header Row -->
	<div class="flex items-center justify-between gap-2 mb-2">
		<div class="flex items-center gap-2 min-w-0">
			<!-- Type Icon with Color -->
			{#if typeIcon === Square}
				<div
					class="flex-shrink-0 w-6 h-6 rounded flex items-center justify-center"
					style="background-color: {annotation.color}20"
				>
					<Square class="w-3.5 h-3.5" style="color: {annotation.color}" />
				</div>
			{:else if typeIcon === Pentagon}
				<div
					class="flex-shrink-0 w-6 h-6 rounded flex items-center justify-center"
					style="background-color: {annotation.color}20"
				>
					<Pentagon class="w-3.5 h-3.5" style="color: {annotation.color}" />
				</div>
			{:else}
				<div
					class="flex-shrink-0 w-6 h-6 rounded flex items-center justify-center"
					style="background-color: {annotation.color}20"
				>
					<Image class="w-3.5 h-3.5" style="color: {annotation.color}" />
				</div>
			{/if}

			<!-- Label Selector -->
			<Select.Root
				type="single"
				value={annotation.label}
				onValueChange={(v) => v && onLabelChange?.(v)}
			>
				<Select.Trigger class="h-7 min-w-[100px] text-xs font-mono border-0 bg-transparent p-0">
					{annotation.label}
				</Select.Trigger>
				<Select.Content class="font-mono">
					{#each availableLabels as label (label)}
						<Select.Item value={label} class="text-xs">
							{label}
						</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
		</div>

		<!-- Confidence Badge -->
		<div class={cn('text-[10px] font-mono tabular-nums', confidenceColor)}>
			{(annotation.confidence * 100).toFixed(0)}%
		</div>
	</div>

	<!-- Coordinates Row -->
	<div class="text-[10px] font-mono text-muted-foreground mb-2 truncate">
		{#if annotation.type === 'bbox'}
			{formatBbox(annotation)}
		{:else if annotation.type === 'polygon'}
			{formatPolygon(annotation)}
		{:else if annotation.type === 'mask'}
			mask
		{/if}
	</div>

	<!-- Action Buttons (visible on hover/select) -->
	<div
		class={cn(
			'flex items-center gap-1 transition-opacity',
			isSelected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
		)}
	>
		<Button
			variant="ghost"
			size="sm"
			onclick={(e: MouseEvent) => {
				e.stopPropagation();
				onVisibilityToggle?.();
			}}
			class="h-6 w-6 p-0"
		>
			{#if annotation.visible}
				<Eye class="w-3.5 h-3.5" />
			{:else}
				<EyeOff class="w-3.5 h-3.5" />
			{/if}
		</Button>

		<Button
			variant="ghost"
			size="sm"
			onclick={(e: MouseEvent) => {
				e.stopPropagation();
				onDelete?.();
			}}
			class="h-6 w-6 p-0 text-destructive hover:text-destructive hover:bg-destructive/10"
		>
			<Trash2 class="w-3.5 h-3.5" />
		</Button>
	</div>

	<!-- ID Badge (debug/dev) -->
	<div class="absolute top-1 right-1 text-[8px] font-mono text-muted-foreground/50 opacity-0 group-hover:opacity-100">
		{annotation.id.slice(0, 8)}
	</div>
</div>
