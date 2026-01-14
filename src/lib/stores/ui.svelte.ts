/**
 * UI state management using Svelte 5 runes and context.
 *
 * Provides centralized state for sidebar, view navigation, and projects.
 * Uses localStorage for persistence across sessions.
 */

import { getContext, setContext } from 'svelte';
import type { ViewId, ViewConfig, Project, UIState } from '$lib/types/ui';

// Re-export types for convenience
export type { ViewId, ViewConfig, Project, UIState };

// ============================================================================
// Constants
// ============================================================================

export const SIDEBAR_COLLAPSED_WIDTH = 64; // w-16
export const SIDEBAR_EXPANDED_WIDTH = 256; // w-64
export const STORAGE_KEY_SIDEBAR = 'cloumask:sidebar:expanded';
export const STORAGE_KEY_VIEW = 'cloumask:view:current';

/** View configurations for navigation */
export const VIEWS: ViewConfig[] = [
	{ id: 'chat', label: 'Chat', icon: 'MessageSquare', shortcut: '1', shortcutKey: '1' },
	{ id: 'plan', label: 'Plan', icon: 'ClipboardList', shortcut: '2', shortcutKey: '2' },
	{ id: 'execute', label: 'Execute', icon: 'Play', shortcut: '3', shortcutKey: '3' },
	{ id: 'review', label: 'Review', icon: 'CheckCircle', shortcut: '4', shortcutKey: '4' },
	{ id: 'pointcloud', label: 'Point Cloud', icon: 'Box', shortcut: '5', shortcutKey: '5' },
	{ id: 'settings', label: 'Settings', icon: 'Settings', shortcut: ',', shortcutKey: ',' },
];

// ============================================================================
// State Factory
// ============================================================================

const UI_STATE_KEY = Symbol('ui-state');

/**
 * Creates UI state using Svelte 5 runes.
 * Call this at the root layout to initialize the state.
 */
export function createUIState(): UIState {
	// Load persisted sidebar state
	const getInitialSidebarExpanded = (): boolean => {
		if (typeof window === 'undefined') return true;
		const stored = localStorage.getItem(STORAGE_KEY_SIDEBAR);
		return stored !== 'false';
	};

	// Load persisted view
	const getInitialView = (): ViewId => {
		if (typeof window === 'undefined') return 'chat';
		const stored = localStorage.getItem(STORAGE_KEY_VIEW) as ViewId | null;
		return stored && VIEWS.some((v) => v.id === stored) ? stored : 'chat';
	};

	// Reactive state using Svelte 5 runes
	let sidebarExpanded = $state(getInitialSidebarExpanded());
	let currentView = $state<ViewId>(getInitialView());
	let currentProject = $state<Project | null>(null);
	let recentProjects = $state<Project[]>([]);

	// Derived sidebar width
	const sidebarWidth = $derived(sidebarExpanded ? SIDEBAR_EXPANDED_WIDTH : SIDEBAR_COLLAPSED_WIDTH);

	// Persist sidebar state on change
	$effect(() => {
		if (typeof window !== 'undefined') {
			localStorage.setItem(STORAGE_KEY_SIDEBAR, String(sidebarExpanded));
		}
	});

	// Persist view on change
	$effect(() => {
		if (typeof window !== 'undefined') {
			localStorage.setItem(STORAGE_KEY_VIEW, currentView);
		}
	});

	return {
		// Getters (reactive via closure)
		get sidebarExpanded() {
			return sidebarExpanded;
		},
		get currentView() {
			return currentView;
		},
		get currentProject() {
			return currentProject;
		},
		get recentProjects() {
			return recentProjects;
		},
		get sidebarWidth() {
			return sidebarWidth;
		},

		// Actions
		toggleSidebar() {
			sidebarExpanded = !sidebarExpanded;
		},
		setSidebarExpanded(value: boolean) {
			sidebarExpanded = value;
		},
		setView(view: ViewId) {
			currentView = view;
		},
		setProject(project: Project | null) {
			currentProject = project;
		},
		addRecentProject(project: Project) {
			recentProjects = [project, ...recentProjects.filter((p) => p.id !== project.id)].slice(0, 10);
		},
	};
}

// ============================================================================
// Context Helpers
// ============================================================================

/**
 * Initialize UI state and set it in Svelte context.
 * Call this in the root +layout.svelte.
 */
export function setUIState(): UIState {
	const state = createUIState();
	setContext(UI_STATE_KEY, state);
	return state;
}

/**
 * Get UI state from Svelte context.
 * Call this in child components that need UI state.
 */
export function getUIState(): UIState {
	return getContext<UIState>(UI_STATE_KEY);
}
