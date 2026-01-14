<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { ViewConfig } from '$lib/types/ui';

	export interface NavItemProps {
		view: ViewConfig;
		active?: boolean;
		collapsed?: boolean;
		class?: string;
		onclick?: () => void;
	}
</script>

<script lang="ts">
	import Icon from '$lib/components/ui/icons.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	let {
		view,
		active = false,
		collapsed = false,
		class: className,
		onclick,
	}: NavItemProps = $props();

	// Map view icon names to icon component names
	const iconNameMap: Record<string, 'message-square' | 'clipboard-list' | 'play' | 'check-circle' | 'box' | 'settings'> = {
		MessageSquare: 'message-square',
		ClipboardList: 'clipboard-list',
		Play: 'play',
		CheckCircle: 'check-circle',
		Box: 'box',
		Settings: 'settings',
	};

	const iconName = $derived(iconNameMap[view.icon] ?? 'message-square');
</script>

{#if collapsed}
	<Tooltip.Root>
		<Tooltip.Trigger
			class={cn(
				'flex items-center justify-center w-full h-10 rounded-md',
				'transition-colors duration-150',
				'hover:bg-sidebar-accent',
				active && 'bg-sidebar-accent text-sidebar-accent-foreground',
				!active && 'text-sidebar-foreground/70',
				className
			)}
			{onclick}
		>
			<Icon name={iconName} class="size-5" />
		</Tooltip.Trigger>
		<Tooltip.Content side="right" sideOffset={8}>
			<p class="flex items-center gap-2">
				{view.label}
				<kbd
					class="ml-auto text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded font-mono"
				>
					{view.shortcut}
				</kbd>
			</p>
		</Tooltip.Content>
	</Tooltip.Root>
{:else}
	<button
		type="button"
		{onclick}
		class={cn(
			'flex items-center gap-3 w-full h-10 px-3 rounded-md',
			'transition-colors duration-150',
			'hover:bg-sidebar-accent',
			active && 'bg-sidebar-accent text-sidebar-accent-foreground',
			!active && 'text-sidebar-foreground/70',
			className
		)}
		aria-current={active ? 'page' : undefined}
	>
		<Icon name={iconName} class="size-5 shrink-0" />
		<span class="flex-1 text-left truncate text-sm">{view.label}</span>
		<kbd class="text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded font-mono">
			{view.shortcut}
		</kbd>
	</button>
{/if}
