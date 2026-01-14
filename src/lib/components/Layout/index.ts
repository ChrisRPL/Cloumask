/**
 * Layout components for Cloumask app shell.
 */

export { default as Header } from './Header.svelte';
export { default as Logo } from './Logo.svelte';
export { default as MainContent } from './MainContent.svelte';
export { default as NavItem } from './NavItem.svelte';
export { default as ProjectSelector } from './ProjectSelector.svelte';
export { default as Sidebar } from './Sidebar.svelte';
export { default as SidebarToggle } from './SidebarToggle.svelte';
export { default as WindowControls } from './WindowControls.svelte';
export { default as ViewPlaceholder } from './ViewPlaceholder.svelte';

// Re-export types
export type { HeaderProps } from './Header.svelte';
export type { LogoProps, LogoSize } from './Logo.svelte';
export type { MainContentProps } from './MainContent.svelte';
export type { NavItemProps } from './NavItem.svelte';
export type { ProjectSelectorProps } from './ProjectSelector.svelte';
export type { SidebarProps } from './Sidebar.svelte';
export type { SidebarToggleProps } from './SidebarToggle.svelte';
export type { WindowControlsProps } from './WindowControls.svelte';
export type { ViewPlaceholderProps } from './ViewPlaceholder.svelte';
