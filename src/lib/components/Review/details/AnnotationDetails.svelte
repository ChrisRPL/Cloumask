<script lang="ts" module>
	export interface AnnotationDetailsProps {
		annotations: import('$lib/types/review').Annotation[];
		selectedAnnotationId?: string | null;
		availableLabels?: string[];
		onAnnotationSelect?: (id: string) => void;
		onAnnotationLabelChange?: (id: string, label: string) => void;
		onAnnotationDelete?: (id: string) => void;
		onAnnotationVisibilityToggle?: (id: string) => void;
		onAddAnnotation?: () => void;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Plus, Layers } from '@lucide/svelte';
	import AnnotationCard from './AnnotationCard.svelte';

	let {
		annotations,
		selectedAnnotationId = null,
		availableLabels = ['person', 'face', 'license_plate', 'vehicle', 'sign', 'other'],
		onAnnotationSelect,
		onAnnotationLabelChange,
		onAnnotationDelete,
		onAnnotationVisibilityToggle,
		onAddAnnotation,
		class: className
	}: AnnotationDetailsProps = $props();

	const annotationCount = $derived(annotations.length);
	const visibleCount = $derived(annotations.filter((a) => a.visible).length);
</script>

<div
	class={cn(
		'flex flex-col h-full',
		'border-t border-border bg-background',
		className
	)}
>
	<!-- Header -->
	<div class="flex items-center justify-between px-4 py-2 border-b border-border">
		<div class="flex items-center gap-2">
			<Layers class="w-4 h-4 text-muted-foreground" />
			<span class="text-sm font-mono font-medium">Annotations</span>
			<span class="text-xs font-mono text-muted-foreground tabular-nums">
				{visibleCount}/{annotationCount}
			</span>
		</div>

		<Button
			variant="outline"
			size="sm"
			onclick={onAddAnnotation}
			class="h-7 px-2 text-xs font-mono"
		>
			<Plus class="w-3.5 h-3.5 mr-1" />
			Add
		</Button>
	</div>

	<!-- Annotation List -->
	<div class="flex-1 overflow-y-auto p-2">
		{#if annotations.length === 0}
			<div class="flex flex-col items-center justify-center h-full text-center p-4">
				<Layers class="w-8 h-8 text-muted-foreground/30 mb-2" />
				<p class="text-sm font-mono text-muted-foreground">No annotations</p>
				<p class="text-xs text-muted-foreground/70 mt-1">
					Click "Add" or use the canvas tools to create annotations
				</p>
			</div>
		{:else}
			<div class="flex flex-col gap-2">
				{#each annotations as annotation (annotation.id)}
					<AnnotationCard
						{annotation}
						isSelected={selectedAnnotationId === annotation.id}
						{availableLabels}
						onSelect={() => onAnnotationSelect?.(annotation.id)}
						onLabelChange={(label) => onAnnotationLabelChange?.(annotation.id, label)}
						onDelete={() => onAnnotationDelete?.(annotation.id)}
						onVisibilityToggle={() => onAnnotationVisibilityToggle?.(annotation.id)}
					/>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Footer Stats -->
	<div
		class={cn(
			'flex items-center justify-between px-4 py-1.5',
			'border-t border-border bg-muted/30',
			'text-[10px] font-mono text-muted-foreground'
		)}
	>
		<span>
			{#if selectedAnnotationId}
				Selected: {selectedAnnotationId.slice(0, 8)}...
			{:else}
				No selection
			{/if}
		</span>
		<span>
			<kbd class="px-1 py-0.5 bg-muted rounded">Del</kbd> Delete
		</span>
	</div>
</div>
