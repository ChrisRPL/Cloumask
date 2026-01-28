<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface BehaviorInput {
		name: string;
		types: string[];
		description: string;
		required?: boolean;
	}

	export interface BehaviorOutput {
		name: string;
		types: string[];
		description: string;
	}

	export interface ResourceUsage {
		cpu: 'low' | 'medium' | 'high';
		memory: 'low' | 'medium' | 'high';
		gpu: boolean;
	}

	export interface ScriptBehavior {
		inputs: BehaviorInput[];
		outputs: BehaviorOutput[];
		operations: string[];
		warnings: string[];
		estimated_time: string | null;
		resource_usage: ResourceUsage | null;
	}

	export interface BehaviorCardProps {
		behavior: ScriptBehavior;
		showCode?: boolean;
		code?: string;
		class?: string;
	}
</script>

<script lang="ts">
	import {
		FileInput,
		FileOutput,
		Zap,
		AlertTriangle,
		Clock,
		Cpu,
		HardDrive,
		Monitor,
		ChevronDown,
		ChevronRight,
	} from '@lucide/svelte';

	let { behavior, showCode = false, code = '', class: className }: BehaviorCardProps = $props();

	let isCodeExpanded = $state(false);

	function getResourceColor(level: 'low' | 'medium' | 'high'): string {
		switch (level) {
			case 'low':
				return 'text-green-500';
			case 'medium':
				return 'text-amber-500';
			case 'high':
				return 'text-red-500';
		}
	}
</script>

<div class={cn('rounded-lg border border-border bg-card p-4 space-y-4', className)}>
	<!-- Inputs Section -->
	{#if behavior.inputs.length > 0}
		<section>
			<h4 class="text-sm font-medium flex items-center gap-2 mb-2">
				<FileInput class="h-4 w-4 text-blue-500" />
				Inputs
			</h4>
			<div class="space-y-1">
				{#each behavior.inputs as input}
					<div class="pl-6 text-sm">
						<span class="font-medium">{input.name}</span>
						<span class="text-muted-foreground ml-1">({input.types.join(', ')})</span>
						{#if !input.required}
							<span class="text-xs text-muted-foreground ml-1">(optional)</span>
						{/if}
						<p class="text-xs text-muted-foreground">{input.description}</p>
					</div>
				{/each}
			</div>
		</section>
	{/if}

	<!-- Operations Section -->
	{#if behavior.operations.length > 0}
		<section>
			<h4 class="text-sm font-medium flex items-center gap-2 mb-2">
				<Zap class="h-4 w-4 text-amber-500" />
				Operations
			</h4>
			<ul class="pl-6 text-sm space-y-1">
				{#each behavior.operations as operation}
					<li class="flex items-center gap-2">
						<span class="text-muted-foreground">→</span>
						<span>{operation}</span>
					</li>
				{/each}
			</ul>
		</section>
	{/if}

	<!-- Outputs Section -->
	{#if behavior.outputs.length > 0}
		<section>
			<h4 class="text-sm font-medium flex items-center gap-2 mb-2">
				<FileOutput class="h-4 w-4 text-green-500" />
				Outputs
			</h4>
			<div class="space-y-1">
				{#each behavior.outputs as output}
					<div class="pl-6 text-sm">
						<span class="font-medium">{output.name}</span>
						<span class="text-muted-foreground ml-1">({output.types.join(', ')})</span>
						<p class="text-xs text-muted-foreground">{output.description}</p>
					</div>
				{/each}
			</div>
		</section>
	{/if}

	<!-- Resource & Time Info -->
	{#if behavior.estimated_time || behavior.resource_usage}
		<section class="flex flex-wrap gap-4 pt-2 border-t border-border">
			{#if behavior.estimated_time}
				<div class="flex items-center gap-1.5 text-xs text-muted-foreground">
					<Clock class="h-3.5 w-3.5" />
					<span>{behavior.estimated_time}</span>
				</div>
			{/if}

			{#if behavior.resource_usage}
				<div class="flex items-center gap-1.5 text-xs">
					<Cpu class={cn('h-3.5 w-3.5', getResourceColor(behavior.resource_usage.cpu))} />
					<span class="text-muted-foreground">CPU: {behavior.resource_usage.cpu}</span>
				</div>

				<div class="flex items-center gap-1.5 text-xs">
					<HardDrive class={cn('h-3.5 w-3.5', getResourceColor(behavior.resource_usage.memory))} />
					<span class="text-muted-foreground">Memory: {behavior.resource_usage.memory}</span>
				</div>

				{#if behavior.resource_usage.gpu}
					<div class="flex items-center gap-1.5 text-xs text-purple-500">
						<Monitor class="h-3.5 w-3.5" />
						<span>GPU</span>
					</div>
				{/if}
			{/if}
		</section>
	{/if}

	<!-- Warnings Section -->
	{#if behavior.warnings.length > 0}
		<section class="pt-2 border-t border-border">
			<div class="space-y-1">
				{#each behavior.warnings as warning}
					<div class="flex items-start gap-2 text-amber-600 text-sm">
						<AlertTriangle class="h-4 w-4 flex-shrink-0 mt-0.5" />
						<span>{warning}</span>
					</div>
				{/each}
			</div>
		</section>
	{/if}

	<!-- Optional Code View (collapsed by default) -->
	{#if showCode && code}
		<section class="pt-2 border-t border-border">
			<button
				type="button"
				class="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
				onclick={() => (isCodeExpanded = !isCodeExpanded)}
			>
				{#if isCodeExpanded}
					<ChevronDown class="h-3.5 w-3.5" />
				{:else}
					<ChevronRight class="h-3.5 w-3.5" />
				{/if}
				<span>View generated code</span>
			</button>

			{#if isCodeExpanded}
				<div class="mt-2">
					<pre
						class="p-3 rounded bg-muted/50 text-xs font-mono overflow-x-auto max-h-64 overflow-y-auto">{code}</pre>
				</div>
			{/if}
		</section>
	{/if}
</div>
