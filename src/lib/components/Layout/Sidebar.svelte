<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface SidebarProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { getUIState, VIEWS } from '$lib/stores/ui';
	import { Separator } from '$lib/components/ui/separator';
	import NavItem from './NavItem.svelte';
	import SidebarToggle from './SidebarToggle.svelte';

	let { class: className }: SidebarProps = $props();

	const ui = getUIState();

	// Separate main views from settings
	const mainViews = VIEWS.filter((v) => v.id !== 'settings');
	const settingsView = VIEWS.find((v) => v.id === 'settings');
</script>

<aside
	class={cn(
		'h-full bg-sidebar border-r border-sidebar-border',
		'flex flex-col',
		'transition-[width] duration-200 ease-out',
		ui.sidebarExpanded ? 'w-64' : 'w-16',
		className
	)}
>
	<!-- Toggle Button -->
	<div class="p-2">
		<SidebarToggle expanded={ui.sidebarExpanded} onclick={() => ui.toggleSidebar()} />
	</div>

	<Separator class="mx-2" />

	<!-- Main Navigation -->
	<nav class="flex-1 p-2 space-y-1">
		{#each mainViews as view (view.id)}
			<NavItem
				{view}
				active={ui.currentView === view.id}
				collapsed={!ui.sidebarExpanded}
				onclick={() => ui.setView(view.id)}
			/>
		{/each}
	</nav>

	<Separator class="mx-2" />

	<!-- Settings at bottom -->
	{#if settingsView}
		<div class="p-2">
			<NavItem
				view={settingsView}
				active={ui.currentView === 'settings'}
				collapsed={!ui.sidebarExpanded}
				onclick={() => ui.setView('settings')}
			/>
		</div>
	{/if}
</aside>
