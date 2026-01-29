<script lang="ts">
	import '../app.css';
	import type { Snippet } from 'svelte';
	import { setUIState, VIEWS, type ViewId } from '$lib/stores/ui.svelte';
	import { setSettingsState } from '$lib/stores/settings.svelte';
	import { setAgentState } from '$lib/stores/agent.svelte';
	import { setPipelineState } from '$lib/stores/pipeline.svelte';
	import { setExecutionState } from '$lib/stores/execution.svelte';
	import { setReviewState } from '$lib/stores/review.svelte';
	import { setSetupState } from '$lib/stores/setup.svelte';
	import { setSSEState } from '$lib/stores/sse.svelte';
	import { setKeyboardState, type KeyboardScope } from '$lib/stores/keyboard.svelte';
	import { Header, Sidebar, MainContent } from '$lib/components/Layout';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { CommandPalette, KeyboardShortcutsOverlay } from '$lib/components/Keyboard';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();

	// Initialize all stores at root level (context is set by the function call)
	const ui = setUIState();
	setSettingsState();
	setSetupState();
	const agent = setAgentState();
	const pipeline = setPipelineState();
	const execution = setExecutionState();
	const review = setReviewState();

	// Initialize keyboard state
	const keyboard = setKeyboardState();

	// Initialize SSE state and bind stores for event routing
	const sse = setSSEState();
	sse.bindStores({ agent, execution, pipeline });

	// Map view IDs to keyboard scopes
	const viewToScope: Record<ViewId, KeyboardScope> = {
		chat: 'chat',
		plan: 'plan',
		execute: 'execution',
		review: 'review',
		pointcloud: 'pointcloud',
		settings: 'global',
	};

	// Auto-update keyboard scope when view changes
	$effect(() => {
		const scope = viewToScope[ui.currentView] ?? 'global';
		keyboard.setScope(scope);
	});

	// Register global shortcuts
	$effect(() => {
		const unregisterFns: (() => void)[] = [];

		// View switching shortcuts (1-5, comma)
		for (const view of VIEWS) {
			const id = keyboard.register({
				combo: view.shortcutKey,
				action: () => ui.setView(view.id),
				scope: 'global',
				description: `Switch to ${view.label} view`,
				category: 'Navigation',
			});
			unregisterFns.push(() => keyboard.unregister(id));
		}

		// Sidebar toggle (Ctrl+B)
		const sidebarId = keyboard.register({
			combo: 'ctrl+b',
			action: () => ui.toggleSidebar(),
			scope: 'global',
			description: 'Toggle sidebar',
			category: 'Navigation',
		});
		unregisterFns.push(() => keyboard.unregister(sidebarId));

		// Command palette (Ctrl+K)
		const paletteId = keyboard.register({
			combo: 'ctrl+k',
			action: () => keyboard.toggleCommandPalette(),
			scope: 'global',
			description: 'Open command palette',
			category: 'Navigation',
			priority: 100, // High priority
		});
		unregisterFns.push(() => keyboard.unregister(paletteId));

		// Help overlay (?)
		const helpId = keyboard.register({
			combo: '?',
			action: () => keyboard.toggleHelpOverlay(),
			scope: 'global',
			description: 'Show keyboard shortcuts',
			category: 'Help',
		});
		unregisterFns.push(() => keyboard.unregister(helpId));

		// Escape (close overlays)
		const escapeId = keyboard.register({
			combo: 'escape',
			action: () => {
				if (keyboard.isCommandPaletteOpen) {
					keyboard.closeCommandPalette();
				} else if (keyboard.isHelpOverlayOpen) {
					keyboard.closeHelpOverlay();
				}
			},
			scope: 'global',
			description: 'Close overlay / Cancel',
			category: 'Navigation',
			priority: 50,
		});
		unregisterFns.push(() => keyboard.unregister(escapeId));

		// ====================================================================
		// Review View Shortcuts
		// ====================================================================

		// j/↓ - Next item
		const reviewNextId = keyboard.register({
			combo: ['j', 'arrowdown'],
			action: () => review.nextItem(),
			scope: 'review',
			description: 'Next item',
			category: 'Review',
		});
		unregisterFns.push(() => keyboard.unregister(reviewNextId));

		// k/↑ - Previous item
		const reviewPrevId = keyboard.register({
			combo: ['k', 'arrowup'],
			action: () => review.previousItem(),
			scope: 'review',
			description: 'Previous item',
			category: 'Review',
		});
		unregisterFns.push(() => keyboard.unregister(reviewPrevId));

		// Note: Approve (a) and Reject (r) are handled by ReviewQueue component
		// because they need undo/redo support via the command pattern

		// ====================================================================
		// Execution View Shortcuts
		// ====================================================================

		// Space - Pause/Resume
		const execToggleId = keyboard.register({
			combo: 'space',
			action: () => {
				if (execution.isPaused) {
					execution.resume();
				} else if (execution.isRunning) {
					execution.pause();
				}
			},
			scope: 'execution',
			description: 'Pause / Resume',
			category: 'Execution',
		});
		unregisterFns.push(() => keyboard.unregister(execToggleId));

		// Enter - Continue at checkpoint
		const execContinueId = keyboard.register({
			combo: 'enter',
			action: () => {
				if (execution.checkpoint) {
					execution.resume();
				}
			},
			scope: 'execution',
			description: 'Continue at checkpoint',
			category: 'Execution',
		});
		unregisterFns.push(() => keyboard.unregister(execContinueId));

		// r - Open review (navigate to review view)
		const execReviewId = keyboard.register({
			combo: 'r',
			action: () => ui.setView('review'),
			scope: 'execution',
			description: 'Open review queue',
			category: 'Execution',
		});
		unregisterFns.push(() => keyboard.unregister(execReviewId));

		// ====================================================================
		// Plan View Shortcuts
		// ====================================================================

		// e - Toggle edit mode
		const planEditId = keyboard.register({
			combo: 'e',
			action: () => pipeline.setEditing(!pipeline.isEditing),
			scope: 'plan',
			description: 'Toggle edit mode',
			category: 'Plan Editor',
		});
		unregisterFns.push(() => keyboard.unregister(planEditId));

		// j/↓ - Navigate to next step
		const planNextId = keyboard.register({
			combo: ['j', 'arrowdown'],
			action: () => {
				const steps = pipeline.sortedSteps;
				const currentIdx = steps.findIndex((s) => s.id === pipeline.selectedStepId);
				if (currentIdx < steps.length - 1) {
					pipeline.selectStep(steps[currentIdx + 1].id);
				} else if (currentIdx === -1 && steps.length > 0) {
					pipeline.selectStep(steps[0].id);
				}
			},
			scope: 'plan',
			description: 'Next step',
			category: 'Plan Editor',
		});
		unregisterFns.push(() => keyboard.unregister(planNextId));

		// k/↑ - Navigate to previous step
		const planPrevId = keyboard.register({
			combo: ['k', 'arrowup'],
			action: () => {
				const steps = pipeline.sortedSteps;
				const currentIdx = steps.findIndex((s) => s.id === pipeline.selectedStepId);
				if (currentIdx > 0) {
					pipeline.selectStep(steps[currentIdx - 1].id);
				}
			},
			scope: 'plan',
			description: 'Previous step',
			category: 'Plan Editor',
		});
		unregisterFns.push(() => keyboard.unregister(planPrevId));

		// Space - Toggle step enabled/skipped
		const planToggleId = keyboard.register({
			combo: 'space',
			action: () => {
				const stepId = pipeline.selectedStepId;
				if (!stepId) return;
				const step = pipeline.steps.find((s) => s.id === stepId);
				if (step) {
					pipeline.updateStep(stepId, {
						status: step.status === 'skipped' ? 'pending' : 'skipped',
					});
				}
			},
			scope: 'plan',
			description: 'Toggle step enabled',
			category: 'Plan Editor',
		});
		unregisterFns.push(() => keyboard.unregister(planToggleId));

		return () => {
			for (const fn of unregisterFns) fn();
		};
	});

	// Central keyboard event listener
	$effect(() => {
		if (typeof window === 'undefined') return;

		function handleKeydown(event: KeyboardEvent) {
			keyboard.handleKeyEvent(event);
		}

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

	<!-- Keyboard overlays (rendered at root for proper stacking) -->
	<CommandPalette />
	<KeyboardShortcutsOverlay />
</Tooltip.Provider>
