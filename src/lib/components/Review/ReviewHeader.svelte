<script lang="ts" module>
	export interface ReviewHeaderProps {
		total: number;
		pending: number;
		approved: number;
		rejected: number;
		currentIndex: number;
		onApproveAll?: () => void;
		onRejectAll?: () => void;
		onDone?: () => void;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { CheckCircle2, XCircle, ArrowLeft } from '@lucide/svelte';

	let {
		total,
		pending,
		approved,
		rejected,
		currentIndex,
		onApproveAll,
		onRejectAll,
		onDone,
		class: className
	}: ReviewHeaderProps = $props();

	let showApproveDialog = $state(false);
	let showRejectDialog = $state(false);

	const progress = $derived(total > 0 ? ((total - pending) / total) * 100 : 0);
	const canApproveAll = $derived(pending > 0);

	function handleApproveAll() {
		if (!canApproveAll) return;
		showApproveDialog = false;
		onApproveAll?.();
	}

	function handleRejectAll() {
		showRejectDialog = false;
		onRejectAll?.();
	}
</script>

<div
	class={cn(
		'flex items-center justify-between gap-4 px-4 py-3',
		'border-b border-border bg-background',
		className
	)}
>
	<!-- Left: Title and progress -->
	<div class="flex items-center gap-4">
		<div class="flex items-center gap-2">
			<CheckCircle2 class="w-5 h-5 text-primary" />
			<h1 class="text-lg font-semibold font-mono">Review Queue</h1>
		</div>

		<!-- Progress bar -->
		<div class="hidden sm:flex items-center gap-3">
			<div class="w-32 h-1.5 bg-muted rounded-full overflow-hidden">
				<div
					class="h-full bg-primary transition-all duration-300"
					style="width: {progress}%"
				></div>
			</div>
			<span class="text-xs font-mono text-muted-foreground tabular-nums">
				{currentIndex + 1} / {total}
			</span>
		</div>
	</div>

	<!-- Center: Stats -->
	<div class="hidden md:flex items-center gap-4 text-xs font-mono">
		<span class="flex items-center gap-1.5">
			<span class="w-2 h-2 rounded-full bg-amber-500"></span>
			<span class="text-muted-foreground">Pending:</span>
			<span class="tabular-nums">{pending}</span>
		</span>
		<span class="flex items-center gap-1.5">
			<span class="w-2 h-2 rounded-full bg-green-500"></span>
			<span class="text-muted-foreground">Approved:</span>
			<span class="tabular-nums">{approved}</span>
		</span>
		<span class="flex items-center gap-1.5">
			<span class="w-2 h-2 rounded-full bg-red-500"></span>
			<span class="text-muted-foreground">Rejected:</span>
			<span class="tabular-nums">{rejected}</span>
		</span>
	</div>

	<!-- Right: Actions -->
	<div class="flex items-center gap-2">
		<!-- Approve All -->
		{#if canApproveAll}
			<Dialog.Root bind:open={showApproveDialog}>
				<Dialog.Trigger>
					<Button variant="outline" size="sm" class="h-8 px-3 text-xs font-mono">
						Approve All
					</Button>
				</Dialog.Trigger>
				<Dialog.Content class="max-w-sm">
					<Dialog.Header>
						<Dialog.Title class="font-mono">Approve All Pending</Dialog.Title>
						<Dialog.Description class="font-mono text-sm">
							This will approve all {pending} pending items. This action can be undone.
						</Dialog.Description>
					</Dialog.Header>
					<Dialog.Footer>
						<Button variant="outline" size="sm" onclick={() => (showApproveDialog = false)}>
							Cancel
						</Button>
						<Button size="sm" onclick={handleApproveAll}>
							<CheckCircle2 class="w-4 h-4 mr-1" />
							Approve All
						</Button>
					</Dialog.Footer>
				</Dialog.Content>
			</Dialog.Root>
		{:else}
			<Button variant="outline" size="sm" class="h-8 px-3 text-xs font-mono" disabled>
				Approve All
			</Button>
		{/if}

		<!-- Done button -->
		<Button variant="default" size="sm" onclick={onDone} class="h-8 px-4 text-xs font-mono">
			<ArrowLeft class="w-3.5 h-3.5 mr-1.5" />
			Done
		</Button>
	</div>
</div>
