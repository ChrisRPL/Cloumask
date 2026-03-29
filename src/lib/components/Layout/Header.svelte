<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface HeaderProps {
		class?: string;
	}
</script>

<script lang="ts">
	import Logo from './Logo.svelte';
	import ProjectSelector from './ProjectSelector.svelte';
	import { ConnectionStatus } from '$lib/components/Status';

	let { class: className }: HeaderProps = $props();
</script>

<header
	class={cn(
		'h-12 shrink-0 flex min-w-0 items-center justify-between gap-3 px-3 sm:px-4',
		'bg-card border-b border-border',
		className
	)}
	data-tauri-drag-region
>
	<div class="flex min-w-0 items-center gap-3">
		<Logo size="sm" />
	</div>

	<div class="flex shrink-0 items-center gap-2 sm:gap-3">
		<ConnectionStatus compact />
		<ProjectSelector class="w-52 sm:w-56 lg:w-60" />
	</div>
</header>
