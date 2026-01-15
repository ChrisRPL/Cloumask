<script lang="ts" module>
	import type { ConfigFieldSchema } from '$lib/types/plan-editor';

	export interface ConfigFieldProps {
		field: ConfigFieldSchema;
		value: unknown;
		disabled?: boolean;
		class?: string;
		onValueChange?: (value: unknown) => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Label } from '$lib/components/ui/label';
	import { Input } from '$lib/components/ui/input';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Select from '$lib/components/ui/select';

	let {
		field,
		value,
		disabled = false,
		class: className,
		onValueChange,
	}: ConfigFieldProps = $props();

	// Handle select change
	function handleSelectChange(newValue: string | undefined) {
		if (newValue) {
			onValueChange?.(newValue);
		}
	}

	// Handle checkbox change
	function handleCheckboxChange(checked: boolean | 'indeterminate') {
		if (checked !== 'indeterminate') {
			onValueChange?.(checked);
		}
	}

	// Handle slider/number change
	function handleRangeChange(event: Event) {
		const target = event.target as HTMLInputElement;
		onValueChange?.(parseFloat(target.value));
	}

	// Handle text change
	function handleTextChange(event: Event) {
		const target = event.target as HTMLInputElement;
		onValueChange?.(target.value);
	}

	// Get selected option for select fields
	const selectedOption = $derived(
		field.type === 'select' && field.options
			? field.options.find((opt) => opt.value === value)
			: undefined
	);
</script>

<div class={cn('space-y-2', className)}>
	{#if field.type === 'select' && field.options}
		<!-- Select dropdown -->
		<Label class="text-xs text-muted-foreground">{field.label}</Label>
		<Select.Root
			type="single"
			value={String(value ?? '')}
			onValueChange={handleSelectChange}
			disabled={disabled}
		>
			<Select.Trigger class="w-full font-mono text-sm">
				{selectedOption?.label ?? 'Select...'}
			</Select.Trigger>
			<Select.Content>
				{#each field.options as option}
					<Select.Item value={option.value} label={option.label}>
						{option.label}
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	{:else if field.type === 'slider' || (field.type === 'number' && field.min !== undefined && field.max !== undefined)}
		<!-- Slider with value display -->
		<div class="flex items-center justify-between">
			<Label class="text-xs text-muted-foreground">{field.label}</Label>
			<span class="text-xs font-mono tabular-nums text-muted-foreground">
				{typeof value === 'number' ? value.toFixed(field.step && field.step < 1 ? 2 : 0) : value}
			</span>
		</div>
		<input
			type="range"
			min={field.min}
			max={field.max}
			step={field.step}
			value={typeof value === 'number' ? value : field.default}
			oninput={handleRangeChange}
			{disabled}
			class={cn(
				'w-full h-2 rounded-full appearance-none cursor-pointer',
				'bg-muted accent-primary',
				'[&::-webkit-slider-thumb]:appearance-none',
				'[&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4',
				'[&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary',
				'[&::-webkit-slider-thumb]:cursor-pointer',
				'[&::-webkit-slider-thumb]:transition-transform',
				'[&::-webkit-slider-thumb]:hover:scale-110',
				disabled && 'opacity-50 cursor-not-allowed'
			)}
		/>
	{:else if field.type === 'number'}
		<!-- Number input -->
		<Label class="text-xs text-muted-foreground">{field.label}</Label>
		<Input
			type="number"
			min={field.min}
			max={field.max}
			step={field.step}
			value={String(value ?? field.default)}
			oninput={handleRangeChange}
			{disabled}
			class="font-mono text-sm"
		/>
	{:else if field.type === 'checkbox'}
		<!-- Checkbox -->
		<div class="flex items-center gap-3">
			<Checkbox
				checked={Boolean(value)}
				onCheckedChange={handleCheckboxChange}
				{disabled}
			/>
			<Label class="text-sm cursor-pointer">{field.label}</Label>
		</div>
	{:else}
		<!-- Text input (default) -->
		<Label class="text-xs text-muted-foreground">{field.label}</Label>
		<Input
			type="text"
			value={String(value ?? field.default ?? '')}
			oninput={handleTextChange}
			{disabled}
			placeholder={field.description}
			class="font-mono text-sm"
		/>
	{/if}

	<!-- Description tooltip -->
	{#if field.description && field.type !== 'checkbox'}
		<p class="text-xs text-muted-foreground/60">{field.description}</p>
	{/if}
</div>
