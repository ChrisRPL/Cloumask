<script lang="ts" module>
	import type { PipelineStep, StepConfig as StepConfigType } from '$lib/types/pipeline';

	export interface StepConfigProps {
		step: PipelineStep;
		class?: string;
		onUpdate?: (updates: Partial<StepConfigType>) => void;
		onClose?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button';
	import { X } from '@lucide/svelte';
	import { STEP_TYPE_CONFIGS, getDefaultConfig } from './constants';
	import ConfigField from './ConfigField.svelte';

	let {
		step,
		class: className,
		onUpdate,
		onClose,
	}: StepConfigProps = $props();

	// Get config schema for this step type
	const typeConfig = $derived(STEP_TYPE_CONFIGS[step.type]);
	const schema = $derived(typeConfig?.configSchema ?? []);

	// Local config state for editing
	let localConfig = $state<Record<string, unknown>>({});

	// Initialize local config from step
	$effect(() => {
		localConfig = {
			...getDefaultConfig(step.type),
			...step.config.params,
			model: step.config.model,
			confidence: step.config.confidence,
		};
	});

	// Track if config has been modified
	const isDirty = $derived.by(() => {
		const original = {
			...getDefaultConfig(step.type),
			...step.config.params,
			model: step.config.model,
			confidence: step.config.confidence,
		};
		return JSON.stringify(localConfig) !== JSON.stringify(original);
	});

	// Handle field value change
	function handleFieldChange(key: string, value: unknown) {
		localConfig = { ...localConfig, [key]: value };
	}

	// Reset to defaults
	function handleReset() {
		localConfig = getDefaultConfig(step.type);
	}

	// Apply changes
	function handleApply() {
		// Extract standard fields from local config
		const { model, confidence, ...params } = localConfig;

		onUpdate?.({
			model: model as string | undefined,
			confidence: confidence as number | undefined,
			params,
		});

		onClose?.();
	}
</script>

<aside
	class={cn(
		'w-80 border-l border-border bg-card/50 flex flex-col',
		'animate-in slide-in-from-right duration-200',
		className
	)}
>
	<!-- Header -->
	<header class="flex items-center justify-between px-4 py-3 border-b border-border">
		<div class="flex items-center gap-2 min-w-0">
			<span class="text-forest-light font-mono">$</span>
			<span class="font-medium truncate">{step.toolName}</span>
		</div>
		<button
			onclick={onClose}
			class="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
			title="Close (Esc)"
		>
			<X class="h-4 w-4" />
		</button>
	</header>

	<!-- Config form -->
	<div class="flex-1 overflow-auto p-4 space-y-4">
		{#if schema.length > 0}
			{#each schema as field (field.key)}
				<ConfigField
					{field}
					value={localConfig[field.key]}
					onValueChange={(v) => handleFieldChange(field.key, v)}
				/>
			{/each}
		{:else}
			<p class="text-sm text-muted-foreground text-center py-8">
				No configuration options available for this step type.
			</p>
		{/if}
	</div>

	<!-- Actions -->
	<footer class="flex items-center justify-between px-4 py-3 border-t border-border">
		<Button variant="ghost" size="sm" onclick={handleReset}>
			Reset
		</Button>
		<div class="flex items-center gap-2">
			<Button variant="outline" size="sm" onclick={onClose}>
				Cancel
			</Button>
			<Button
				variant="default"
				size="sm"
				onclick={handleApply}
				disabled={!isDirty}
			>
				Apply
			</Button>
		</div>
	</footer>
</aside>
