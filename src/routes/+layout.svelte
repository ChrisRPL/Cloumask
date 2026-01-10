<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();
</script>

{@render children()}

<style>
	:global(html, body) {
		margin: 0;
		padding: 0;
		min-height: 100vh;
		font-family: Inter, Avenir, Helvetica, Arial, sans-serif;
		font-size: 16px;
		line-height: 1.5;
		font-weight: 400;
		color: #0f0f0f;
		background-color: #f6f6f6;
		font-synthesis: none;
		text-rendering: optimizeLegibility;
		-webkit-font-smoothing: antialiased;
		-moz-osx-font-smoothing: grayscale;
	}

	@media (prefers-color-scheme: dark) {
		:global(html, body) {
			color: #f6f6f6;
			background-color: #1a1a1a;
		}
	}
</style>
