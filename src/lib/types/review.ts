/**
 * Review type definitions for the review queue and annotation editing.
 */

// ============================================================================
// Annotation Types
// ============================================================================

export type AnnotationType = "detection" | "segmentation" | "classification";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Annotation {
  id: string;
  type: AnnotationType;
  label: string;
  confidence: number;
  bbox?: BoundingBox;
  maskUrl?: string;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Review Item
// ============================================================================

export type ReviewStatus = "pending" | "approved" | "rejected" | "modified";

export interface ReviewItem {
  id: string;
  filePath: string;
  fileName: string;
  thumbnailUrl?: string;
  annotations: Annotation[];
  status: ReviewStatus;
  createdAt: string;
  reviewedAt?: string;
  notes?: string;
}

// ============================================================================
// Review Filters
// ============================================================================

export interface ReviewFilters {
  status: ReviewStatus | "all";
  label: string | null;
  minConfidence: number;
  maxConfidence: number;
  searchQuery: string;
}

// ============================================================================
// Review State Interface
// ============================================================================

export interface ReviewState {
  readonly items: ReviewItem[];
  readonly selectedIds: Set<string>;
  readonly filters: ReviewFilters;
  readonly isLoading: boolean;
  readonly currentItemId: string | null;

  // Derived
  readonly filteredItems: ReviewItem[];
  readonly selectedCount: number;
  readonly pendingCount: number;
  readonly approvedCount: number;
  readonly rejectedCount: number;
  readonly currentItem: ReviewItem | null;
  readonly hasSelection: boolean;

  // Item actions
  loadItems(items: ReviewItem[]): void;
  addItem(item: ReviewItem): void;
  updateItem(id: string, updates: Partial<ReviewItem>): void;
  removeItem(id: string): void;
  clearItems(): void;

  // Annotation actions
  updateAnnotation(
    itemId: string,
    annotationId: string,
    updates: Partial<Annotation>,
  ): void;
  addAnnotation(itemId: string, annotation: Omit<Annotation, "id">): Annotation;
  removeAnnotation(itemId: string, annotationId: string): void;

  // Selection actions
  selectItem(id: string): void;
  deselectItem(id: string): void;
  toggleSelection(id: string): void;
  selectAll(): void;
  clearSelection(): void;

  // Batch operations
  approveSelected(): void;
  rejectSelected(): void;
  approveItem(id: string): void;
  rejectItem(id: string): void;

  // Navigation
  setCurrentItem(id: string | null): void;
  nextItem(): void;
  previousItem(): void;

  // Filtering
  setFilter<K extends keyof ReviewFilters>(
    key: K,
    value: ReviewFilters[K],
  ): void;
  resetFilters(): void;
  setLoading(loading: boolean): void;

  // Reset
  reset(): void;
}
