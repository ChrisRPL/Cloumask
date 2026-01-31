<script lang="ts" module>
	export interface StatsPanelProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { getExecutionState } from '$lib/stores/execution.svelte';
	import { FileCheck, Search, Flag, AlertCircle } from '@lucide/svelte';
	import StatCard from './StatCard.svelte';
	import CommentaryStream from './CommentaryStream.svelte';

	let { class: className }: StatsPanelProps = $props();

	const execution = getExecutionState();
</script>

<div class={cn('flex flex-col h-full bg-muted/10', className)}>
	<!-- Stats grid -->
	<div class="grid grid-cols-2 gap-2 p-3 border-b border-border">
		<StatCard label="Processed" value={execution.stats.processed} icon={FileCheck} />
		<StatCard label="Detected" value={execution.stats.detected} icon={Search} />
		<StatCard
			label="Flagged"
			value={execution.stats.flagged}
			icon={Flag}
			variant="warning"
		/>
		<StatCard
			label="Errors"
			value={execution.stats.errors}
			icon={AlertCircle}
			variant="destructive"
		/>
	</div>

	<!-- Commentary stream (takes remaining space) -->
	<CommentaryStream class="flex-1 overflow-hidden" />
</div>
