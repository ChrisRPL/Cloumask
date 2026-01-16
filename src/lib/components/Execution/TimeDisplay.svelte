<script lang="ts" module>
	export interface TimeDisplayProps {
		startedAt: string | null;
		estimatedCompletion: string | null;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Clock } from '@lucide/svelte';

	let { startedAt, estimatedCompletion, class: className }: TimeDisplayProps = $props();

	// Elapsed time state
	let elapsed = $state(0);

	// Format duration from milliseconds
	function formatDuration(ms: number): string {
		if (ms < 0) return '0:00';
		const totalSeconds = Math.floor(ms / 1000);
		const hours = Math.floor(totalSeconds / 3600);
		const minutes = Math.floor((totalSeconds % 3600) / 60);
		const seconds = totalSeconds % 60;

		if (hours > 0) {
			return `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
		}
		return `${minutes}:${seconds.toString().padStart(2, '0')}`;
	}

	// Calculate remaining time
	const remaining = $derived.by(() => {
		if (!estimatedCompletion) return null;
		const estimated = new Date(estimatedCompletion).getTime();
		const now = Date.now();
		return Math.max(0, estimated - now);
	});

	// Update elapsed time every second
	$effect(() => {
		if (!startedAt) {
			elapsed = 0;
			return;
		}

		const startTime = new Date(startedAt).getTime();
		elapsed = Date.now() - startTime;

		const interval = setInterval(() => {
			elapsed = Date.now() - startTime;
		}, 1000);

		return () => clearInterval(interval);
	});
</script>

<div class={cn('flex items-center gap-2 text-xs font-mono text-muted-foreground', className)}>
	<Clock class="h-3 w-3" />
	<span class="tabular-nums">{formatDuration(elapsed)}</span>
	{#if remaining !== null && remaining > 0}
		<span class="text-muted-foreground/60">
			~{formatDuration(remaining)} left
		</span>
	{/if}
</div>
