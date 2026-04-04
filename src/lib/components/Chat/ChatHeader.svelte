<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { AgentPhase } from '$lib/types/agent';

	export interface ChatHeaderProps {
		phase: AgentPhase;
		isConnected: boolean;
		class?: string;
		onClear: () => void;
		onExport?: () => void;
	}

	// Phase display labels
	const phaseLabels: Record<AgentPhase, string> = {
		idle: 'ready',
		understanding: 'thinking',
		planning: 'planning',
		awaiting_approval: 'awaiting',
		executing: 'running',
		checkpoint: 'paused',
		complete: 'done',
		error: 'error'
	};

	// Phase badge colors
	function getPhaseColor(phase: AgentPhase): string {
		switch (phase) {
			case 'understanding':
			case 'planning':
			case 'executing':
				return 'bg-forest-light/20 text-forest-light border-forest-light/30';
			case 'awaiting_approval':
			case 'checkpoint':
				return 'bg-amber-500/20 text-amber-600 border-amber-500/30';
			case 'complete':
				return 'bg-forest/20 text-forest border-forest/30';
			case 'error':
				return 'bg-destructive/20 text-destructive border-destructive/30';
			default:
				return 'bg-muted text-muted-foreground border-border';
		}
	}
</script>

<script lang="ts">
	import { Trash2, Download } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';

	let {
		phase,
		class: className,
		onClear,
		onExport
	}: ChatHeaderProps = $props();

	const phaseLabel = $derived(phaseLabels[phase]);
	const phaseColor = $derived(getPhaseColor(phase));
	const isActive = $derived(['understanding', 'planning', 'executing'].includes(phase));
</script>

<header
	class={cn(
		'flex items-center justify-between px-4 py-3 border-b border-border bg-background/50',
		className
	)}
>
	<!-- Left: Title and status -->
	<div class="flex items-center gap-3">
		<h2 class="text-sm font-medium">Chat</h2>

		<!-- Phase badge -->
		{#if phase !== 'idle'}
			<Badge
				variant="outline"
				class={cn('text-xs h-5 px-2 gap-1', phaseColor)}
			>
				{#if isActive}
					<span class="h-1.5 w-1.5 rounded-full bg-current animate-pulse"></span>
				{/if}
				{phaseLabel}
			</Badge>
		{/if}
	</div>

	<!-- Right: Actions -->
	<div class="flex items-center gap-1">
		{#if onExport}
			<Button
				variant="ghost"
				size="icon"
				class="h-8 w-8"
				onclick={onExport}
				title="Export conversation"
			>
				<Download class="h-4 w-4" />
				<span class="sr-only">Export</span>
			</Button>
		{/if}

		<Button
			variant="ghost"
			size="icon"
			class="h-8 w-8 text-muted-foreground hover:text-destructive"
			onclick={onClear}
			title="Clear conversation"
		>
			<Trash2 class="h-4 w-4" />
			<span class="sr-only">Clear</span>
		</Button>
	</div>
</header>
