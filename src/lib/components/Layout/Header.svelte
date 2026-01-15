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
		'h-12 shrink-0 flex items-center justify-between px-4',
		'bg-card border-b border-border',
		className
	)}
	data-tauri-drag-region
>
	<div class="flex items-center gap-4">
		<Logo size="sm" />
	</div>

	<div class="flex items-center gap-3">
		<ConnectionStatus compact />
		<ProjectSelector class="max-w-[200px]" />
	</div>
</header>
