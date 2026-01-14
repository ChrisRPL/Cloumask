<script lang="ts">
	import '../app.css';
	import type { Snippet } from 'svelte';
	import { setUIState, VIEWS, type ViewId } from '$lib/stores/ui';
	import { Header, Sidebar, MainContent } from '$lib/components/Layout';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();

	// Initialize UI state at root level
	const ui = setUIState();

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
		if (typeof window !== 'undefined') {
			window.addEventListener('keydown', handleKeydown);
			return () => window.removeEventListener('keydown', handleKeydown);
		}
	});
</script>

<div class="h-screen flex flex-col bg-background overflow-hidden">
	<Header />
	<div class="flex-1 flex overflow-hidden">
		<Sidebar />
		<MainContent>
			{@render children()}
		</MainContent>
	</div>
</div>
