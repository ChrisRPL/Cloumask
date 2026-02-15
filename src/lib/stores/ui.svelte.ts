/**
 * UI state management using Svelte 5 runes and context.
 *
 * Provides centralized state for sidebar, view navigation, and projects.
 * Uses localStorage for persistence across sessions.
 */

import { getContext, setContext } from "svelte";
import type { ViewId, ViewConfig, Project, UIState } from "$lib/types/ui";

// Re-export types for convenience
export type { ViewId, ViewConfig, Project, UIState };

// ============================================================================
// Constants
// ============================================================================

export const SIDEBAR_COLLAPSED_WIDTH = 64; // w-16
export const SIDEBAR_EXPANDED_WIDTH = 256; // w-64
export const STORAGE_KEY_SIDEBAR = "cloumask:sidebar:expanded";
export const STORAGE_KEY_VIEW = "cloumask:view:current";
export const STORAGE_KEY_CURRENT_PROJECT = "cloumask:project:current";
export const STORAGE_KEY_RECENT_PROJECTS = "cloumask:project:recent";

/** View configurations for navigation */
export const VIEWS: ViewConfig[] = [
  {
    id: "chat",
    label: "Chat",
    icon: "MessageSquare",
    shortcut: "1",
    shortcutKey: "1",
  },
  {
    id: "plan",
    label: "Plan",
    icon: "ClipboardList",
    shortcut: "2",
    shortcutKey: "2",
  },
  {
    id: "execute",
    label: "Execute",
    icon: "Play",
    shortcut: "3",
    shortcutKey: "3",
  },
  {
    id: "review",
    label: "Review",
    icon: "CheckCircle",
    shortcut: "4",
    shortcutKey: "4",
  },
  {
    id: "pointcloud",
    label: "Point Cloud",
    icon: "Box",
    shortcut: "5",
    shortcutKey: "5",
  },
  {
    id: "settings",
    label: "Settings",
    icon: "Settings",
    shortcut: ",",
    shortcutKey: ",",
  },
];

// ============================================================================
// State Factory
// ============================================================================

const UI_STATE_KEY = Symbol("ui-state");

/**
 * Creates UI state using Svelte 5 runes.
 * Call this at the root layout to initialize the state.
 */
export function createUIState(): UIState {
  // Load persisted sidebar state
  const getInitialSidebarExpanded = (): boolean => {
    if (typeof window === "undefined") return true;
    const stored = localStorage.getItem(STORAGE_KEY_SIDEBAR);
    if (stored !== null) return stored !== "false";
    return !window.matchMedia("(max-width: 768px)").matches;
  };

  // Load persisted view
  const getInitialView = (): ViewId => {
    if (typeof window === "undefined") return "chat";
    const stored = localStorage.getItem(STORAGE_KEY_VIEW) as ViewId | null;
    return stored && VIEWS.some((v) => v.id === stored) ? stored : "chat";
  };

  const parseProject = (value: unknown): Project | null => {
    if (!value || typeof value !== "object") return null;
    const candidate = value as Partial<Project> & { lastOpened?: string | Date };
    if (
      typeof candidate.id !== "string" ||
      typeof candidate.name !== "string" ||
      typeof candidate.path !== "string"
    ) {
      return null;
    }
    return {
      id: candidate.id,
      name: candidate.name,
      path: candidate.path,
      lastOpened:
        candidate.lastOpened
          ? new Date(candidate.lastOpened)
          : undefined,
    };
  };

  const getInitialCurrentProject = (): Project | null => {
    if (typeof window === "undefined") return null;
    try {
      const raw = localStorage.getItem(STORAGE_KEY_CURRENT_PROJECT);
      if (!raw) return null;
      return parseProject(JSON.parse(raw));
    } catch {
      return null;
    }
  };

  const getInitialRecentProjects = (): Project[] => {
    if (typeof window === "undefined") return [];
    try {
      const raw = localStorage.getItem(STORAGE_KEY_RECENT_PROJECTS);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed
        .map((project) => parseProject(project))
        .filter((project): project is Project => project !== null)
        .slice(0, 10);
    } catch {
      return [];
    }
  };

  // Reactive state using Svelte 5 runes
  let sidebarExpanded = $state(getInitialSidebarExpanded());
  let currentView = $state<ViewId>(getInitialView());
  let currentProject = $state<Project | null>(getInitialCurrentProject());
  let recentProjects = $state<Project[]>(getInitialRecentProjects());

  // Derived sidebar width
  const sidebarWidth = $derived(
    sidebarExpanded ? SIDEBAR_EXPANDED_WIDTH : SIDEBAR_COLLAPSED_WIDTH,
  );

  // Persist sidebar state on change
  $effect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(STORAGE_KEY_SIDEBAR, String(sidebarExpanded));
    }
  });

  // Persist view on change
  $effect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(STORAGE_KEY_VIEW, currentView);
    }
  });

  // Persist current project and recents
  $effect(() => {
    if (typeof window === "undefined") return;
    if (currentProject) {
      localStorage.setItem(STORAGE_KEY_CURRENT_PROJECT, JSON.stringify(currentProject));
    } else {
      localStorage.removeItem(STORAGE_KEY_CURRENT_PROJECT);
    }
  });

  $effect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(STORAGE_KEY_RECENT_PROJECTS, JSON.stringify(recentProjects));
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
      if (project) {
        recentProjects = [
          project,
          ...recentProjects.filter((p) => p.id !== project.id),
        ].slice(0, 10);
      }
    },
    addRecentProject(project: Project) {
      recentProjects = [
        project,
        ...recentProjects.filter((p) => p.id !== project.id),
      ].slice(0, 10);
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
