<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface WindowControlsProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { isTauri, minimizeWindow, toggleMaximize, closeWindow } from '$lib/utils/tauri';

	let { class: className }: WindowControlsProps = $props();

	const isInTauri = isTauri();
</script>

{#if isInTauri}
	<div class={cn('flex items-center gap-2', className)}>
		<!-- Close (red) -->
		<button
			type="button"
			onclick={closeWindow}
			class="size-3 rounded-full bg-[#ff5f57] hover:bg-[#ff5f57]/80 transition-colors focus:outline-none focus:ring-1 focus:ring-offset-1 focus:ring-[#ff5f57]"
			aria-label="Close window"
		></button>
		<!-- Minimize (yellow) -->
		<button
			type="button"
			onclick={minimizeWindow}
			class="size-3 rounded-full bg-[#febc2e] hover:bg-[#febc2e]/80 transition-colors focus:outline-none focus:ring-1 focus:ring-offset-1 focus:ring-[#febc2e]"
			aria-label="Minimize window"
		></button>
		<!-- Maximize (green) -->
		<button
			type="button"
			onclick={toggleMaximize}
			class="size-3 rounded-full bg-[#28c840] hover:bg-[#28c840]/80 transition-colors focus:outline-none focus:ring-1 focus:ring-offset-1 focus:ring-[#28c840]"
			aria-label="Maximize window"
		></button>
	</div>
{/if}
