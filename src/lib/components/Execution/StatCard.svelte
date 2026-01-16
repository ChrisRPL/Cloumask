<script lang="ts" module>
	import type { Component } from 'svelte';

	export interface StatCardProps {
		label: string;
		value: number;
		icon: Component;
		variant?: 'default' | 'warning' | 'destructive';
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';

	let { label, value, icon: Icon, variant = 'default', class: className }: StatCardProps = $props();

	const variantClasses = $derived.by(() => {
		if (value === 0) return '';
		switch (variant) {
			case 'destructive':
				return 'bg-destructive/10 border-destructive/20';
			case 'warning':
				return 'bg-amber-500/10 border-amber-500/20';
			default:
				return '';
		}
	});

	const valueClasses = $derived.by(() => {
		if (value === 0) return 'text-foreground';
		switch (variant) {
			case 'destructive':
				return 'text-destructive';
			case 'warning':
				return 'text-amber-500';
			default:
				return 'text-foreground';
		}
	});
</script>

<div
	class={cn(
		'flex items-center gap-3 p-3 rounded-md bg-muted/30 border border-transparent transition-colors',
		variantClasses,
		className
	)}
>
	<div class="shrink-0">
		<Icon class="h-4 w-4 text-muted-foreground" />
	</div>
	<div class="flex-1 min-w-0">
		<div class={cn('text-xl font-mono tabular-nums font-semibold', valueClasses)}>
			{value.toLocaleString()}
		</div>
		<div class="text-xs text-muted-foreground font-mono truncate">{label}</div>
	</div>
</div>
