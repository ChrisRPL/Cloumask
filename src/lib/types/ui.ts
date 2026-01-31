/**
 * UI type definitions for Cloumask layout and navigation.
 */

/** Available application views */
export type ViewId =
  | "chat"
  | "plan"
  | "execute"
  | "review"
  | "pointcloud"
  | "settings";

/** View configuration for navigation */
export interface ViewConfig {
  id: ViewId;
  label: string;
  icon: string;
  shortcut: string;
  shortcutKey: string;
}

/** Project representation */
export interface Project {
  id: string;
  name: string;
  path: string;
  lastOpened?: Date;
}

/** Sidebar state */
export interface SidebarState {
  expanded: boolean;
  width: number;
}

/** UI state type (for context typing) */
export interface UIState {
  readonly sidebarExpanded: boolean;
  readonly currentView: ViewId;
  readonly currentProject: Project | null;
  readonly recentProjects: Project[];
  readonly sidebarWidth: number;
  toggleSidebar(): void;
  setSidebarExpanded(value: boolean): void;
  setView(view: ViewId): void;
  setProject(project: Project | null): void;
  addRecentProject(project: Project): void;
}
