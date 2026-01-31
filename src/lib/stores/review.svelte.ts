/**
 * Review state management using Svelte 5 runes and context.
 *
 * Provides centralized state for the review queue including item management,
 * filtering, multi-selection, batch operations, and annotation editing.
 */

import { getContext, setContext } from "svelte";
import type {
  ReviewState,
  ReviewItem,
  ReviewFilters,
  ReviewStatus,
  Annotation,
  AnnotationType,
  BoundingBox,
  Point,
  ImageDimensions,
  CanvasTransform,
  BoxInteractionMode,
  ResizeHandle,
} from "$lib/types/review";

// Re-export types for convenience
export type {
  ReviewState,
  ReviewItem,
  ReviewFilters,
  ReviewStatus,
  Annotation,
  AnnotationType,
  BoundingBox,
  Point,
  ImageDimensions,
  CanvasTransform,
  BoxInteractionMode,
  ResizeHandle,
};

// ============================================================================
// Constants
// ============================================================================

const REVIEW_STATE_KEY = Symbol("review-state");

const DEFAULT_FILTERS: ReviewFilters = {
  status: "all",
  label: null,
  minConfidence: 0,
  maxConfidence: 1,
  searchQuery: "",
};

// ============================================================================
// Helpers
// ============================================================================

function generateId(): string {
  return crypto.randomUUID();
}

function now(): string {
  return new Date().toISOString();
}

// ============================================================================
// State Factory
// ============================================================================

/**
 * Creates review state using Svelte 5 runes.
 * Call this at the root layout to initialize the state.
 */
export function createReviewState(): ReviewState {
  // Reactive state
  let items = $state<ReviewItem[]>([]);
  let selectedIds = $state<Set<string>>(new Set());
  let filters = $state<ReviewFilters>({ ...DEFAULT_FILTERS });
  let isLoading = $state(false);
  let currentItemId = $state<string | null>(null);

  // Derived values
  const filteredItems = $derived.by(() => {
    return items.filter((item) => {
      // Status filter
      if (filters.status !== "all" && item.status !== filters.status) {
        return false;
      }

      // Label filter
      if (
        filters.label &&
        !item.annotations.some((a) => a.label === filters.label)
      ) {
        return false;
      }

      // Search filter
      if (filters.searchQuery) {
        const query = filters.searchQuery.toLowerCase();
        if (!item.fileName.toLowerCase().includes(query)) {
          return false;
        }
      }

      // Confidence range filter
      if (item.annotations.length > 0) {
        const maxConf = Math.max(...item.annotations.map((a) => a.confidence));
        if (
          maxConf < filters.minConfidence ||
          maxConf > filters.maxConfidence
        ) {
          return false;
        }
      }

      return true;
    });
  });

  const selectedCount = $derived(selectedIds.size);
  const pendingCount = $derived(
    items.filter((i) => i.status === "pending").length,
  );
  const approvedCount = $derived(
    items.filter((i) => i.status === "approved").length,
  );
  const rejectedCount = $derived(
    items.filter((i) => i.status === "rejected").length,
  );
  const currentItem = $derived(
    items.find((i) => i.id === currentItemId) ?? null,
  );
  const hasSelection = $derived(selectedIds.size > 0);

  return {
    // Getters
    get items() {
      return items;
    },
    get selectedIds() {
      return selectedIds;
    },
    get filters() {
      return filters;
    },
    get isLoading() {
      return isLoading;
    },
    get currentItemId() {
      return currentItemId;
    },

    // Derived getters
    get filteredItems() {
      return filteredItems;
    },
    get selectedCount() {
      return selectedCount;
    },
    get pendingCount() {
      return pendingCount;
    },
    get approvedCount() {
      return approvedCount;
    },
    get rejectedCount() {
      return rejectedCount;
    },
    get currentItem() {
      return currentItem;
    },
    get hasSelection() {
      return hasSelection;
    },

    // Item actions
    loadItems(newItems: ReviewItem[]) {
      items = newItems;
      selectedIds = new Set();
      currentItemId = newItems.length > 0 ? newItems[0].id : null;
    },

    addItem(item: ReviewItem) {
      items = [...items, item];
    },

    updateItem(id: string, updates: Partial<ReviewItem>) {
      items = items.map((item) =>
        item.id === id
          ? {
              ...item,
              ...updates,
              reviewedAt: updates.status ? now() : item.reviewedAt,
            }
          : item,
      );
    },

    removeItem(id: string) {
      items = items.filter((i) => i.id !== id);
      selectedIds = new Set([...selectedIds].filter((sid) => sid !== id));
      if (currentItemId === id) {
        currentItemId = items.length > 0 ? items[0].id : null;
      }
    },

    clearItems() {
      items = [];
      selectedIds = new Set();
      currentItemId = null;
    },

    // Annotation actions
    updateAnnotation(
      itemId: string,
      annotationId: string,
      updates: Partial<Annotation>,
    ) {
      items = items.map((item) =>
        item.id === itemId
          ? {
              ...item,
              annotations: item.annotations.map((ann) =>
                ann.id === annotationId ? { ...ann, ...updates } : ann,
              ),
              status: "modified" as ReviewStatus,
            }
          : item,
      );
    },

    addAnnotation(
      itemId: string,
      annotation: Omit<Annotation, "id">,
    ): Annotation {
      const newAnnotation: Annotation = {
        ...annotation,
        id: generateId(),
      };
      items = items.map((item) =>
        item.id === itemId
          ? {
              ...item,
              annotations: [...item.annotations, newAnnotation],
              status: "modified" as ReviewStatus,
            }
          : item,
      );
      return newAnnotation;
    },

    removeAnnotation(itemId: string, annotationId: string) {
      items = items.map((item) =>
        item.id === itemId
          ? {
              ...item,
              annotations: item.annotations.filter(
                (ann) => ann.id !== annotationId,
              ),
              status: "modified" as ReviewStatus,
            }
          : item,
      );
    },

    // Selection actions
    selectItem(id: string) {
      selectedIds = new Set([...selectedIds, id]);
    },

    deselectItem(id: string) {
      selectedIds = new Set([...selectedIds].filter((sid) => sid !== id));
    },

    toggleSelection(id: string) {
      if (selectedIds.has(id)) {
        selectedIds = new Set([...selectedIds].filter((sid) => sid !== id));
      } else {
        selectedIds = new Set([...selectedIds, id]);
      }
    },

    selectAll() {
      selectedIds = new Set(filteredItems.map((i) => i.id));
    },

    clearSelection() {
      selectedIds = new Set();
    },

    // Batch operations
    approveSelected() {
      items = items.map((item) =>
        selectedIds.has(item.id)
          ? { ...item, status: "approved" as ReviewStatus, reviewedAt: now() }
          : item,
      );
      selectedIds = new Set();
    },

    rejectSelected() {
      items = items.map((item) =>
        selectedIds.has(item.id)
          ? { ...item, status: "rejected" as ReviewStatus, reviewedAt: now() }
          : item,
      );
      selectedIds = new Set();
    },

    approveItem(id: string) {
      items = items.map((item) =>
        item.id === id
          ? { ...item, status: "approved" as ReviewStatus, reviewedAt: now() }
          : item,
      );
    },

    rejectItem(id: string) {
      items = items.map((item) =>
        item.id === id
          ? { ...item, status: "rejected" as ReviewStatus, reviewedAt: now() }
          : item,
      );
    },

    // Navigation
    setCurrentItem(id: string | null) {
      currentItemId = id;
    },

    nextItem() {
      if (!currentItemId || filteredItems.length === 0) return;
      const currentIndex = filteredItems.findIndex(
        (i) => i.id === currentItemId,
      );
      if (currentIndex < filteredItems.length - 1) {
        currentItemId = filteredItems[currentIndex + 1].id;
      }
    },

    previousItem() {
      if (!currentItemId || filteredItems.length === 0) return;
      const currentIndex = filteredItems.findIndex(
        (i) => i.id === currentItemId,
      );
      if (currentIndex > 0) {
        currentItemId = filteredItems[currentIndex - 1].id;
      }
    },

    // Filtering
    setFilter<K extends keyof ReviewFilters>(key: K, value: ReviewFilters[K]) {
      filters = { ...filters, [key]: value };
    },

    resetFilters() {
      filters = { ...DEFAULT_FILTERS };
    },

    setLoading(loading: boolean) {
      isLoading = loading;
    },

    // Reset
    reset() {
      items = [];
      selectedIds = new Set();
      filters = { ...DEFAULT_FILTERS };
      isLoading = false;
      currentItemId = null;
    },
  };
}

// ============================================================================
// Context Helpers
// ============================================================================

/**
 * Initialize review state and set it in Svelte context.
 * Call this in the root +layout.svelte.
 */
export function setReviewState(): ReviewState {
  const state = createReviewState();
  setContext(REVIEW_STATE_KEY, state);
  return state;
}

/**
 * Get review state from Svelte context.
 * Call this in child components that need review state.
 */
export function getReviewState(): ReviewState {
  return getContext<ReviewState>(REVIEW_STATE_KEY);
}
