<script lang="ts" module>
	export interface FocusTrapProps {
		/** Whether the focus trap is active */
		active?: boolean;
		/** Callback when Escape is pressed */
		onEscape?: () => void;
		/** Element to return focus to when trap is deactivated */
		returnFocusTo?: HTMLElement | null;
	}
</script>

<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		active = true,
		onEscape,
		returnFocusTo,
		children,
	}: FocusTrapProps & { children: Snippet } = $props();

	let containerRef = $state<HTMLDivElement | null>(null);
	let previouslyFocused = $state<HTMLElement | null>(null);

	// Get all focusable elements within the container
	function getFocusableElements(): HTMLElement[] {
		if (!containerRef) return [];

		const selector = [
			'a[href]',
			'button:not([disabled])',
			'textarea:not([disabled])',
			'input:not([disabled])',
			'select:not([disabled])',
			'[tabindex]:not([tabindex="-1"])',
		].join(', ');

		return Array.from(containerRef.querySelectorAll<HTMLElement>(selector));
	}

	// Handle keydown for Tab and Escape
	function handleKeydown(event: KeyboardEvent) {
		if (!active) return;

		if (event.key === 'Escape' && onEscape) {
			event.preventDefault();
			onEscape();
			return;
		}

		if (event.key === 'Tab') {
			const focusable = getFocusableElements();
			if (focusable.length === 0) {
				event.preventDefault();
				return;
			}

			const first = focusable[0];
			const last = focusable[focusable.length - 1];

			if (event.shiftKey && document.activeElement === first) {
				event.preventDefault();
				last.focus();
			} else if (!event.shiftKey && document.activeElement === last) {
				event.preventDefault();
				first.focus();
			}
		}
	}

	// Store and restore focus
	$effect(() => {
		if (active && containerRef) {
			// Store currently focused element
			previouslyFocused = (returnFocusTo ?? document.activeElement) as HTMLElement;

			// Focus first focusable element
			const focusable = getFocusableElements();
			if (focusable.length > 0) {
				focusable[0].focus();
			} else {
				containerRef.focus();
			}
		}

		return () => {
			// Return focus when deactivated (only if element still exists in DOM)
			if (previouslyFocused && previouslyFocused.isConnected && previouslyFocused.focus) {
				previouslyFocused.focus();
			}
		};
	});
</script>

<div
	bind:this={containerRef}
	onkeydown={handleKeydown}
	tabindex="-1"
	class="outline-none"
	role="group"
	aria-label="Focus trap container"
>
	{@render children()}
</div>
