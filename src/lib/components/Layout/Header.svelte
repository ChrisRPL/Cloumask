<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface HeaderProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { isTauri } from '$lib/utils/tauri';
	import Logo from './Logo.svelte';
	import ProjectSelector from './ProjectSelector.svelte';
	import WindowControls from './WindowControls.svelte';

	let { class: className }: HeaderProps = $props();

	const isInTauri = isTauri();
</script>

<header
	class={cn(
		'h-12 shrink-0 flex items-center justify-between px-4',
		'bg-card border-b border-border',
		className
	)}
	data-tauri-drag-region
>
	<div class="flex items-center gap-4">
		{#if isInTauri}
			<WindowControls />
		{/if}
		<Logo size="sm" showText />
	</div>

	<ProjectSelector class="max-w-[200px]" />
</header>
