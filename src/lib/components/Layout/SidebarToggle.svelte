<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface SidebarToggleProps {
		expanded?: boolean;
		class?: string;
		onclick?: () => void;
	}
</script>

<script lang="ts">
	import Icon from '$lib/components/ui/icons.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	let { expanded = true, class: className, onclick }: SidebarToggleProps = $props();
</script>

<Tooltip.Root>
	<Tooltip.Trigger
		class={cn(
			'flex items-center justify-center w-full h-10 rounded-md',
			'transition-colors duration-150',
			'hover:bg-sidebar-accent text-sidebar-foreground/70',
			className
		)}
		{onclick}
	>
		{#if expanded}
			<Icon name="panel-left-close" class="size-5" />
		{:else}
			<Icon name="panel-left" class="size-5" />
		{/if}
	</Tooltip.Trigger>
	<Tooltip.Content side="right" sideOffset={8}>
		<p class="flex items-center gap-2">
			{expanded ? 'Collapse' : 'Expand'} sidebar
			<kbd class="ml-auto text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded font-mono">
				Ctrl+B
			</kbd>
		</p>
	</Tooltip.Content>
</Tooltip.Root>
