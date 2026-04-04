<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface ChatInputProps {
		value: string;
		disabled?: boolean;
		disableSend?: boolean;
		placeholder?: string;
		class?: string;
		onSend: (message: string) => void;
		onValueChange: (value: string) => void;
	}
</script>

<script lang="ts">
	import { CornerDownLeft } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	let {
		value,
		disabled = false,
		disableSend = false,
		placeholder = 'Type a message...',
		class: className,
		onSend,
		onValueChange
	}: ChatInputProps = $props();

	let textareaRef: HTMLTextAreaElement | null = $state(null);

	// Auto-resize textarea
	function autoResize() {
		if (!textareaRef) return;
		textareaRef.style.height = 'auto';
		const newHeight = Math.min(textareaRef.scrollHeight, 120); // max 5 rows approx
		textareaRef.style.height = `${newHeight}px`;
	}

	// Handle input change
	function handleInput(event: Event) {
		const target = event.target as HTMLTextAreaElement;
		onValueChange(target.value);
		autoResize();
	}

	// Handle key events
	function handleKeydown(event: KeyboardEvent) {
		// Enter to send (without shift)
		if (event.key === 'Enter' && !event.shiftKey) {
			if (event.repeat) return;
			event.preventDefault();
			if (value.trim() && !disableSend) {
				onSend(value.trim());
			}
		}
	}

	// Submit via button
	function handleSubmit() {
		if (value.trim() && !disableSend) {
			onSend(value.trim());
		}
	}

	// Reset height when value is cleared
	$effect(() => {
		if (!value && textareaRef) {
			textareaRef.style.height = 'auto';
		}
	});
</script>

<div
	class={cn(
		'flex items-end gap-2 p-3 border-t border-border bg-background/50',
		className
	)}
>
	<!-- Input container -->
	<div
		class={cn(
			'flex-1 relative rounded-lg border border-border bg-background',
			'focus-within:ring-1 focus-within:ring-ring focus-within:border-ring',
			disabled && 'opacity-50'
		)}
	>
		<textarea
			bind:this={textareaRef}
			{value}
			{placeholder}
			{disabled}
			rows="1"
			class={cn(
				'w-full px-3 py-2 bg-transparent resize-none',
				'text-sm leading-relaxed placeholder:text-muted-foreground',
				'focus:outline-none',
				'disabled:cursor-not-allowed'
			)}
			oninput={handleInput}
			onkeydown={handleKeydown}
			aria-label="Message input"
		></textarea>
	</div>

	<!-- Send button -->
	<Button
		variant="default"
		size="icon"
		class="h-9 w-9 shrink-0"
		disabled={disableSend || !value.trim()}
		onclick={handleSubmit}
	>
		<CornerDownLeft class="h-4 w-4" />
		<span class="sr-only">Send message</span>
	</Button>
</div>

<!-- Keyboard hint -->
{#if !disabled && !disableSend}
	<div class="px-3 pb-2 text-xs text-muted-foreground/60 flex justify-between">
		<span>Enter to send, Shift+Enter for newline</span>
		{#if value.length > 0}
			<span class="tabular-nums">{value.length}</span>
		{/if}
	</div>
{/if}
