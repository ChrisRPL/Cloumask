<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { ViewConfig } from '$lib/types/ui';

	export interface ViewPlaceholderProps {
		view: ViewConfig;
		class?: string;
	}
</script>

<script lang="ts">
	import Icon from '$lib/components/ui/icons.svelte';

	let { view, class: className }: ViewPlaceholderProps = $props();

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

<div
	class={cn(
		'flex flex-col items-center justify-center h-full text-center p-8',
		className
	)}
>
	<div class="p-4 rounded-full bg-muted mb-4">
		<Icon name={iconName} class="size-12 text-muted-foreground" />
	</div>
	<h2 class="text-2xl font-semibold text-foreground mb-2">{view.label}</h2>
	<p class="text-muted-foreground max-w-md">
		Coming soon. This view is part of the Cloumask UI development roadmap.
	</p>
	<p class="mt-4 text-xs text-muted-foreground/60 font-mono">
		Press <kbd class="bg-muted px-1.5 py-0.5 rounded">{view.shortcut}</kbd> to navigate here
	</p>
</div>
