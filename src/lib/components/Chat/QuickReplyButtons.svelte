<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface QuickReplyButtonsProps {
		options: string[];
		disabled?: boolean;
		class?: string;
		onSelect: (option: string) => void;
	}
</script>

<script lang="ts">
	import { Button } from '$lib/components/ui/button';

	let {
		options,
		disabled = false,
		class: className,
		onSelect
	}: QuickReplyButtonsProps = $props();
</script>

{#if options.length > 0}
	<div
		class={cn(
			'flex flex-wrap gap-2 px-3 py-2',
			className
		)}
		role="group"
		aria-label="Quick reply options"
	>
		{#each options as option (option)}
			<Button
				variant="outline"
				size="sm"
				class="text-xs h-7 px-3"
				{disabled}
				onclick={() => onSelect(option)}
			>
				{option}
			</Button>
		{/each}
	</div>
{/if}
