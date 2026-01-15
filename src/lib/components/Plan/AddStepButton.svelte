<script lang="ts" module>
	import type { StepType } from '$lib/types/pipeline';

	export interface AddStepButtonProps {
		class?: string;
		onAddStep?: (type: StepType) => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import {
		Plus,
		Search,
		Scissors,
		EyeOff,
		Download,
		Tag,
		Wand2,
	} from '@lucide/svelte';
	import { STEP_TYPE_CONFIGS } from './constants';

	let {
		class: className,
		onAddStep,
	}: AddStepButtonProps = $props();

	// Icon mapping
	const iconMap = {
		Search,
		Scissors,
		EyeOff,
		Download,
		Tag,
		Wand2,
	} as const;

	// Step types available to add
	const stepTypes = Object.values(STEP_TYPE_CONFIGS);

	function handleSelect(type: StepType) {
		onAddStep?.(type);
	}
</script>

<div class={cn('px-3 py-2', className)}>
	<DropdownMenu.Root>
		<DropdownMenu.Trigger>
			{#snippet child({ props })}
				<Button
					{...props}
					variant="outline"
					size="sm"
					class="w-full border-dashed text-muted-foreground hover:text-foreground"
				>
					<Plus class="h-4 w-4 mr-2" />
					Add Step
				</Button>
			{/snippet}
		</DropdownMenu.Trigger>
		<DropdownMenu.Content align="start" class="w-56">
			<DropdownMenu.Label class="font-mono text-xs text-muted-foreground">
				Step Types
			</DropdownMenu.Label>
			<DropdownMenu.Separator />
			{#each stepTypes as config}
				{@const Icon = iconMap[config.icon as keyof typeof iconMap] ?? Search}
				<DropdownMenu.Item
					onclick={() => handleSelect(config.type)}
					class="flex items-center gap-3 cursor-pointer"
				>
					<Icon class="h-4 w-4 text-forest-light" />
					<div class="flex-1">
						<span class="font-medium">{config.label}</span>
						<span class="text-xs text-muted-foreground ml-2 font-mono">
							{config.prefix}
						</span>
					</div>
				</DropdownMenu.Item>
			{/each}
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
