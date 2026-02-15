<script lang="ts">
	import '../app.css';
	import type { Snippet } from 'svelte';
	import { untrack } from 'svelte';
	import { setUIState, VIEWS, type ViewId } from '$lib/stores/ui.svelte';
	import { setSettingsState, getSettingsState } from '$lib/stores/settings.svelte';
	import { setAgentState } from '$lib/stores/agent.svelte';
	import { setPipelineState } from '$lib/stores/pipeline.svelte';
	import { setExecutionState } from '$lib/stores/execution.svelte';
	import { setReviewState } from '$lib/stores/review.svelte';
	import { setSetupState } from '$lib/stores/setup.svelte';
	import { setSSEState } from '$lib/stores/sse.svelte';
	import { setKeyboardState, type KeyboardScope } from '$lib/stores/keyboard.svelte';
	import { sendMessage } from '$lib/utils/tauri';
	import { Header, Sidebar, MainContent } from '$lib/components/Layout';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { CommandPalette, KeyboardShortcutsOverlay } from '$lib/components/Keyboard';

	interface Props {
		children: Snippet;
	}

	let { children }: Props = $props();

	// Initialize all stores at root level (context is set by the function call)
	const ui = setUIState();
	const settings = setSettingsState();
	setSetupState();
	const agent = setAgentState();
	const pipeline = setPipelineState();
	const execution = setExecutionState();
	const review = setReviewState();

	// Apply theme class to html element based on settings
	$effect(() => {
		if (typeof window === 'undefined') return;
		const html = document.documentElement;
		const theme = settings.settings.theme;

		function applyTheme(isDark: boolean) {
			if (isDark) {
				html.classList.add('dark');
			} else {
				html.classList.remove('dark');
			}
		}

		if (theme === 'system') {
			const mq = window.matchMedia('(prefers-color-scheme: dark)');
			applyTheme(mq.matches);
			const handler = (e: MediaQueryListEvent) => applyTheme(e.matches);
			mq.addEventListener('change', handler);
			return () => mq.removeEventListener('change', handler);
		} else {
			applyTheme(theme === 'dark');
		}
	});

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
	// NOTE: We read ui.currentView (tracked) but use untrack for setScope
	// because setScope reads and writes scopeStack internally, which would
	// cause an infinite loop if tracked.
	$effect(() => {
		const scope = viewToScope[ui.currentView] ?? 'global';
		untrack(() => {
			keyboard.setScope(scope);
		});
	});

	// Register global shortcuts
	// NOTE: We use untrack() to prevent this effect from tracking registeredShortcuts reads
	// inside keyboard.register(). Without untrack, register() reads the shortcuts map,
	// which would make this effect re-run every time a shortcut is registered, causing
	// an infinite loop (effect_update_depth_exceeded error).
	$effect(() => {
		const unregisterFns: (() => void)[] = [];

		untrack(() => {
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

			// j - Next item
			const reviewNextJId = keyboard.register({
				combo: 'j',
				action: () => review.nextItem(),
				scope: 'review',
				description: 'Next item',
				category: 'Review',
			});
			unregisterFns.push(() => keyboard.unregister(reviewNextJId));

			// ↓ - Next item (same action as j)
			const reviewNextArrowId = keyboard.register({
				combo: 'arrowdown',
				action: () => review.nextItem(),
				scope: 'review',
				description: 'Next item',
				category: 'Review',
			});
			unregisterFns.push(() => keyboard.unregister(reviewNextArrowId));

			// k - Previous item
			const reviewPrevKId = keyboard.register({
				combo: 'k',
				action: () => review.previousItem(),
				scope: 'review',
				description: 'Previous item',
				category: 'Review',
			});
			unregisterFns.push(() => keyboard.unregister(reviewPrevKId));

			// ↑ - Previous item (same action as k)
			const reviewPrevArrowId = keyboard.register({
				combo: 'arrowup',
				action: () => review.previousItem(),
				scope: 'review',
				description: 'Previous item',
				category: 'Review',
			});
			unregisterFns.push(() => keyboard.unregister(reviewPrevArrowId));

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
				action: async () => {
					if (execution.checkpoint && agent.threadId) {
						try {
							await sendMessage(agent.threadId, {
								content: 'continue',
								decision: 'approve',
							});
							execution.clearCheckpoint();
						} catch (error) {
							console.error('[Keyboard] Failed to continue checkpoint:', error);
						}
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

			// Helper function for plan navigation
			function navigateToNextStep() {
				const steps = pipeline.sortedSteps;
				const currentIdx = steps.findIndex((s) => s.id === pipeline.selectedStepId);
				if (currentIdx < steps.length - 1) {
					pipeline.selectStep(steps[currentIdx + 1].id);
				} else if (currentIdx === -1 && steps.length > 0) {
					pipeline.selectStep(steps[0].id);
				}
			}

			function navigateToPrevStep() {
				const steps = pipeline.sortedSteps;
				const currentIdx = steps.findIndex((s) => s.id === pipeline.selectedStepId);
				if (currentIdx > 0) {
					pipeline.selectStep(steps[currentIdx - 1].id);
				}
			}

			// j - Navigate to next step
			const planNextJId = keyboard.register({
				combo: 'j',
				action: navigateToNextStep,
				scope: 'plan',
				description: 'Next step',
				category: 'Plan Editor',
			});
			unregisterFns.push(() => keyboard.unregister(planNextJId));

			// ↓ - Navigate to next step (same action as j)
			const planNextArrowId = keyboard.register({
				combo: 'arrowdown',
				action: navigateToNextStep,
				scope: 'plan',
				description: 'Next step',
				category: 'Plan Editor',
			});
			unregisterFns.push(() => keyboard.unregister(planNextArrowId));

			// k - Navigate to previous step
			const planPrevKId = keyboard.register({
				combo: 'k',
				action: navigateToPrevStep,
				scope: 'plan',
				description: 'Previous step',
				category: 'Plan Editor',
			});
			unregisterFns.push(() => keyboard.unregister(planPrevKId));

			// ↑ - Navigate to previous step (same action as k)
			const planPrevArrowId = keyboard.register({
				combo: 'arrowup',
				action: navigateToPrevStep,
				scope: 'plan',
				description: 'Previous step',
				category: 'Plan Editor',
			});
			unregisterFns.push(() => keyboard.unregister(planPrevArrowId));

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
		}); // end untrack

		return () => {
			untrack(() => {
				for (const fn of unregisterFns) fn();
			});
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
