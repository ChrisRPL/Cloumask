<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { AgentPhase } from '$lib/types/agent';

	export interface StreamingIndicatorProps {
		phase: AgentPhase;
		class?: string;
	}

	// Phase labels for terminal-style display
	const phaseLabels: Record<AgentPhase, string> = {
		idle: '',
		understanding: 'processing',
		planning: 'planning',
		awaiting_approval: 'awaiting input',
		executing: 'executing',
		checkpoint: 'checkpoint',
		complete: 'done',
		error: 'error'
	};
</script>

<script lang="ts">
	let { phase, class: className }: StreamingIndicatorProps = $props();

	const label = $derived(phaseLabels[phase] || '');
	const showIndicator = $derived(
		['understanding', 'planning', 'executing'].includes(phase)
	);
</script>

{#if showIndicator && label}
	<div
		class={cn(
			'flex items-center gap-2 px-3 py-1.5',
			'text-xs text-muted-foreground font-medium tracking-wide',
			className
		)}
	>
		<span class="text-forest-light">&lt;</span>
		<span>{label}</span>
		<span class="inline-flex gap-0.5">
			<span class="animate-pulse">.</span>
			<span class="animate-pulse" style="animation-delay: 150ms">.</span>
			<span class="animate-pulse" style="animation-delay: 300ms">.</span>
		</span>
		<span class="animate-blink text-forest-light">_</span>
	</div>
{/if}

<style>
	@keyframes blink {
		0%,
		50% {
			opacity: 1;
		}
		51%,
		100% {
			opacity: 0;
		}
	}

	.animate-blink {
		animation: blink 1s step-end infinite;
	}
</style>
