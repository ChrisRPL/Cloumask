<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { ViewConfig } from '$lib/types/ui';

	export interface ViewPlaceholderProps {
		view: ViewConfig;
		class?: string;
	}
</script>

<script lang="ts">
	import {
		MessageSquare,
		ClipboardList,
		Play,
		CheckCircle,
		Box,
		Settings,
	} from 'lucide-svelte';

	let { view, class: className }: ViewPlaceholderProps = $props();

	// Map icon names to components
	const iconMap: Record<string, typeof MessageSquare> = {
		MessageSquare,
		ClipboardList,
		Play,
		CheckCircle,
		Box,
		Settings,
	};

	const IconComponent = $derived(iconMap[view.icon] ?? MessageSquare);
</script>

<div
	class={cn(
		'flex flex-col items-center justify-center h-full text-center p-8',
		className
	)}
>
	<div class="p-4 rounded-full bg-muted mb-4">
		<IconComponent class="size-12 text-muted-foreground" />
	</div>
	<h2 class="text-2xl font-semibold text-foreground mb-2">{view.label}</h2>
	<p class="text-muted-foreground max-w-md">
		Coming soon. This view is part of the Cloumask UI development roadmap.
	</p>
	<p class="mt-4 text-xs text-muted-foreground/60 font-mono">
		Press <kbd class="bg-muted px-1.5 py-0.5 rounded">{view.shortcut}</kbd> to navigate here
	</p>
</div>
