<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { ClarificationRequest, UserDecision } from '$lib/types/agent';

	export interface ClarificationFormProps {
		clarification: ClarificationRequest;
		disabled?: boolean;
		class?: string;
		onSubmit: (response: { decision: UserDecision; selected?: string[] }) => void;
		onCancel: () => void;
	}
</script>

<script lang="ts">
	import { Check, X, Pencil, Play, Square } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Label } from '$lib/components/ui/label';

	let {
		clarification,
		disabled = false,
		class: className,
		onSubmit,
		onCancel
	}: ClarificationFormProps = $props();

	// State for option selection
	let selectedOptions = $state<string[]>([]);

	// Determine form type
	const isPlanApproval = $derived(clarification.inputType === 'plan_approval');
	const isCheckpointApproval = $derived(clarification.inputType === 'checkpoint_approval');
	const isClarification = $derived(clarification.inputType === 'clarification');

	// Toggle option selection
	function toggleOption(option: string) {
		if (selectedOptions.includes(option)) {
			selectedOptions = selectedOptions.filter((o) => o !== option);
		} else {
			selectedOptions = [...selectedOptions, option];
		}
	}

	// Handle approve action
	function handleApprove() {
		if (isClarification && clarification.options?.length) {
			onSubmit({ decision: 'approve', selected: selectedOptions });
		} else {
			onSubmit({ decision: 'approve' });
		}
	}

	// Handle edit action (for plan approval)
	function handleEdit() {
		onSubmit({ decision: 'edit' });
	}

	// Handle cancel/stop action
	function handleCancel() {
		onCancel();
	}
</script>

<div
	class={cn(
		'rounded-lg border border-border bg-card/50 p-4',
		'space-y-4',
		className
	)}
	role="form"
	aria-label="Agent clarification request"
>
	<!-- Prompt -->
	<div class="text-sm text-foreground">
		{clarification.prompt}
	</div>

	<!-- Options (for clarification type) -->
	{#if isClarification && clarification.options?.length}
		<div class="space-y-2">
			{#each clarification.options as option (option)}
				<div class="flex items-center gap-3">
					<Checkbox
						id={`option-${option}`}
						checked={selectedOptions.includes(option)}
						onCheckedChange={() => toggleOption(option)}
						{disabled}
					/>
					<Label
						for={`option-${option}`}
						class="text-sm cursor-pointer"
					>
						{option}
					</Label>
				</div>
			{/each}
		</div>
	{/if}

	<!-- Action buttons -->
	<div class="flex items-center gap-2 pt-2">
		{#if isPlanApproval}
			<!-- Plan approval: Approve / Edit / Cancel -->
			<Button
				variant="default"
				size="sm"
				{disabled}
				onclick={handleApprove}
				class="gap-1.5"
			>
				<Check class="h-3.5 w-3.5" />
				Approve
			</Button>
			<Button
				variant="outline"
				size="sm"
				{disabled}
				onclick={handleEdit}
				class="gap-1.5"
			>
				<Pencil class="h-3.5 w-3.5" />
				Edit
			</Button>
			<Button
				variant="ghost"
				size="sm"
				{disabled}
				onclick={handleCancel}
				class="gap-1.5 text-muted-foreground"
			>
				<X class="h-3.5 w-3.5" />
				Cancel
			</Button>
		{:else if isCheckpointApproval}
			<!-- Checkpoint approval: Continue / Stop -->
			<Button
				variant="default"
				size="sm"
				{disabled}
				onclick={handleApprove}
				class="gap-1.5"
			>
				<Play class="h-3.5 w-3.5" />
				Continue
			</Button>
			<Button
				variant="outline"
				size="sm"
				{disabled}
				onclick={handleCancel}
				class="gap-1.5"
			>
				<Square class="h-3.5 w-3.5" />
				Stop
			</Button>
		{:else}
			<!-- Clarification: Submit / Cancel -->
			<Button
				variant="default"
				size="sm"
				disabled={disabled || (clarification.options?.length ? selectedOptions.length === 0 : false)}
				onclick={handleApprove}
				class="gap-1.5"
			>
				<Check class="h-3.5 w-3.5" />
				Submit
			</Button>
			<Button
				variant="ghost"
				size="sm"
				{disabled}
				onclick={handleCancel}
				class="gap-1.5 text-muted-foreground"
			>
				<X class="h-3.5 w-3.5" />
				Cancel
			</Button>
		{/if}
	</div>
</div>
