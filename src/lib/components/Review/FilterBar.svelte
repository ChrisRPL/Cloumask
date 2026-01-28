<script lang="ts" module>
	import type { ReviewStatus } from '$lib/types/review';

	export type FilterStatus = ReviewStatus | 'all';

	export interface FilterBarProps {
		statusFilter?: FilterStatus;
		searchQuery?: string;
		minConfidence?: number;
		maxConfidence?: number;
		onStatusChange?: (status: FilterStatus) => void;
		onConfidenceChange?: (min: number, max: number) => void;
		onSearchChange?: (query: string) => void;
		onReset?: () => void;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import * as Select from '$lib/components/ui/select/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Search, X } from '@lucide/svelte';

	let {
		statusFilter = 'all',
		searchQuery = '',
		minConfidence = 0,
		maxConfidence = 1,
		onStatusChange,
		onConfidenceChange,
		onSearchChange,
		onReset,
		class: className
	}: FilterBarProps = $props();

	let searchValue = $state('');
	let searchTimeout: ReturnType<typeof setTimeout> | null = null;

	// Sync search value with prop
	$effect(() => {
		searchValue = searchQuery;
	});

	const statusOptions: { value: ReviewStatus | 'all'; label: string; color: string }[] = [
		{ value: 'all', label: 'All', color: 'bg-muted-foreground' },
		{ value: 'pending', label: 'Pending', color: 'bg-amber-500' },
		{ value: 'approved', label: 'Approved', color: 'bg-green-500' },
		{ value: 'rejected', label: 'Rejected', color: 'bg-red-500' },
		{ value: 'modified', label: 'Modified', color: 'bg-blue-500' }
	];

	const currentStatusOption = $derived(
		statusOptions.find((opt) => opt.value === statusFilter) ?? statusOptions[0]
	);

	function handleSearchInput(e: Event) {
		const target = e.target as HTMLInputElement;
		searchValue = target.value;

		// Debounce search
		if (searchTimeout) clearTimeout(searchTimeout);
		searchTimeout = setTimeout(() => {
			onSearchChange?.(searchValue);
		}, 300);
	}

	function clearSearch() {
		searchValue = '';
		onSearchChange?.('');
	}

	function handleStatusSelect(value: string | undefined) {
		if (value) {
			onStatusChange?.(value as ReviewStatus | 'all');
		}
	}

	const hasActiveFilters = $derived(
		statusFilter !== 'all' ||
			minConfidence > 0 ||
			maxConfidence < 1 ||
			searchQuery !== ''
	);
</script>

<div
	class={cn(
		'flex flex-wrap items-center gap-3 p-3',
		'border-b border-border bg-background/50',
		'font-mono text-sm',
		className
	)}
>
	<!-- Status Filter -->
	<div class="flex items-center gap-2">
		<span class="text-xs text-muted-foreground uppercase tracking-wider">Status</span>
		<Select.Root type="single" value={statusFilter} onValueChange={handleStatusSelect}>
			<Select.Trigger
				class={cn(
					'h-8 min-w-[120px] px-3 font-mono text-xs',
					'border-border bg-background',
					'hover:bg-muted/30'
				)}
			>
				<span class="flex items-center gap-2">
					<span class={cn('w-2 h-2 rounded-full', currentStatusOption.color)}></span>
					{currentStatusOption.label}
				</span>
			</Select.Trigger>
			<Select.Content class="font-mono">
				{#each statusOptions as option}
					<Select.Item value={option.value} class="text-xs">
						<span class="flex items-center gap-2">
							<span class={cn('w-2 h-2 rounded-full', option.color)}></span>
							{option.label}
						</span>
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</div>

	<!-- Confidence Filter -->
	<div class="flex items-center gap-2">
		<span class="text-xs text-muted-foreground uppercase tracking-wider">Confidence</span>
		<div
			class={cn(
				'flex items-center gap-1 h-8 px-3 rounded-md border border-border bg-background',
				'text-xs tabular-nums'
			)}
		>
			<span class="text-muted-foreground">{'>'}</span>
			<span>{(minConfidence * 100).toFixed(0)}%</span>
		</div>
	</div>

	<!-- Search Input -->
	<div class="flex-1 min-w-[200px] max-w-[300px]">
		<div class="relative">
			<Search
				class="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground"
			/>
			<Input
				type="text"
				placeholder="Search files..."
				value={searchValue}
				oninput={handleSearchInput}
				class={cn(
					'h-8 pl-8 pr-8 font-mono text-xs',
					'bg-background border-border',
					'placeholder:text-muted-foreground/50'
				)}
			/>
			{#if searchValue}
				<button
					type="button"
					onclick={clearSearch}
					class="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
				>
					<X class="w-3.5 h-3.5" />
				</button>
			{/if}
		</div>
	</div>

	<!-- Spacer -->
	<div class="flex-1"></div>

	<!-- Reset Button -->
	{#if hasActiveFilters}
		<Button
			variant="ghost"
			size="sm"
			onclick={onReset}
			class="h-8 px-3 text-xs font-mono text-muted-foreground hover:text-foreground"
		>
			Reset filters
		</Button>
	{/if}

	<!-- Terminal-style filter summary -->
	<div
		class={cn(
			'hidden md:flex items-center gap-1.5',
			'px-2 py-1 rounded bg-muted/30',
			'text-[10px] text-muted-foreground font-mono'
		)}
	>
		<span class="text-primary">$</span>
		<span>
			--status={statusFilter}
			{#if minConfidence > 0}--min-conf={minConfidence}{/if}
			{#if searchQuery}--query="{searchQuery}"{/if}
		</span>
	</div>
</div>
