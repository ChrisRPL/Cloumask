import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

// Type helpers for shadcn-svelte component props
export type WithElementRef<T, E extends HTMLElement = HTMLElement> = T & {
	ref?: E | null;
};

// Remove 'children' prop from type (Svelte 5 snippet pattern)
export type WithoutChildren<T> = T extends { children?: infer _ } ? Omit<T, "children"> : T;

// Remove 'child' prop from type (bits-ui pattern)
export type WithoutChild<T> = T extends { child?: infer _ } ? Omit<T, "child"> : T;

// Remove both 'children' and 'child' props from type
export type WithoutChildrenOrChild<T> = WithoutChildren<WithoutChild<T>>;
