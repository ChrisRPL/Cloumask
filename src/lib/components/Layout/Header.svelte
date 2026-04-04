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
		'h-12 shrink-0 flex min-w-0 items-center justify-between gap-3 overflow-hidden px-3 sm:px-4',
		'bg-card border-b border-border',
		className
	)}
	data-tauri-drag-region
>
	<div class="flex shrink-0 items-center gap-3">
		<Logo size="sm" />
	</div>

	<div class="flex min-w-0 flex-1 items-center justify-end gap-2 sm:gap-3">
		<ProjectSelector
			class="min-w-[11rem] flex-1 max-w-[16rem] sm:min-w-[13rem] sm:max-w-[19rem] lg:max-w-[22rem] xl:max-w-[24rem]"
			placeholder="Select project"
			emptyTriggerAriaLabel="Select project"
		/>
		<div class="shrink-0">
			<ConnectionStatus compact />
		</div>
	</div>
</header>
