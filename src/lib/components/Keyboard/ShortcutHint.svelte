<script lang="ts" module>
	export interface ShortcutHintProps {
		/** Key combination to display (e.g., 'ctrl+k', '?', 'escape') */
		combo: string;
		/** Optional class override */
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils';
	import { formatComboString } from '$lib/utils/keyboard';
	import { getKeyboardState } from '$lib/stores/keyboard.svelte';

	let { combo, class: className }: ShortcutHintProps = $props();

	const keyboard = getKeyboardState();

	// Format the combo for the current platform
	const displayCombo = $derived(formatComboString(combo, keyboard.platform));
</script>

<kbd
	class={cn(
		'inline-flex items-center justify-center gap-0.5',
		'px-1.5 py-0.5 min-w-[1.25rem]',
		'bg-muted border border-border rounded',
		'text-xs font-mono text-muted-foreground',
		'select-none',
		className
	)}
>
	{displayCombo}
</kbd>
