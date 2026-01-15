<script lang="ts">
	import '../app.css';
	import type { Snippet } from 'svelte';
	import { setUIState, VIEWS } from '$lib/stores/ui.svelte';
	import { setSettingsState } from '$lib/stores/settings.svelte';
	import { setAgentState } from '$lib/stores/agent.svelte';
	import { setPipelineState } from '$lib/stores/pipeline.svelte';
	import { setExecutionState } from '$lib/stores/execution.svelte';
	import { setReviewState } from '$lib/stores/review.svelte';
	import { setSSEState } from '$lib/stores/sse.svelte';
	import { Header, Sidebar, MainContent } from '$lib/components/Layout';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();

	// Initialize all stores at root level (context is set by the function call)
	const ui = setUIState();
	setSettingsState();
	const agent = setAgentState();
	const pipeline = setPipelineState();
	const execution = setExecutionState();
	setReviewState();

	// Initialize SSE state and bind stores for event routing
	const sse = setSSEState();
	sse.bindStores({ agent, execution, pipeline });

	// Keyboard shortcuts handler
	function handleKeydown(event: KeyboardEvent) {
		const target = event.target as HTMLElement;
		// Ignore if typing in input/textarea
		if (
			target.tagName === 'INPUT' ||
			target.tagName === 'TEXTAREA' ||
			target.isContentEditable
		) {
			return;
		}

		// Ctrl/Cmd + B: Toggle sidebar
		if ((event.ctrlKey || event.metaKey) && event.key === 'b') {
			event.preventDefault();
			ui.toggleSidebar();
			return;
		}

		// Number keys 1-5 and comma for views (without modifiers)
		if (!event.ctrlKey && !event.metaKey && !event.altKey) {
			const view = VIEWS.find((v) => v.shortcutKey === event.key);
			if (view) {
				event.preventDefault();
				ui.setView(view.id);
			}
		}
	}

	// Setup keyboard listener
	$effect(() => {
		if (typeof window === 'undefined') return;
		window.addEventListener('keydown', handleKeydown);
		return () => window.removeEventListener('keydown', handleKeydown);
	});
</script>

<Tooltip.Provider>
	<div class="h-screen flex flex-col bg-background overflow-hidden">
		<Header />
		<div class="flex-1 flex overflow-hidden">
			<Sidebar />
			<MainContent>
				{@render children()}
			</MainContent>
		</div>
	</div>
</Tooltip.Provider>
