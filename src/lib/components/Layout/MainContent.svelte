<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { Snippet } from 'svelte';

	export interface MainContentProps {
		children: Snippet;
		class?: string;
	}
</script>

<script lang="ts">
	let { children, class: className }: MainContentProps = $props();
</script>

<main class={cn('min-w-0 flex-1 overflow-auto bg-background', className)}>
	<div class="h-full min-w-0 animate-fade-in">
		{@render children()}
	</div>
</main>
