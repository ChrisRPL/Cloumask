/**
 * Review type definitions for the review queue and annotation editing.
 * Aligned with backend models in backend/src/backend/api/models/review.py
 */

// ============================================================================
// Annotation Types
// ============================================================================

/**
 * Type of annotation (aligned with backend)
 */
export type AnnotationType = "bbox" | "polygon" | "mask";

/**
 * 2D point for polygon annotations (normalized 0-1 coordinates)
 */
export interface Point {
  x: number;
  y: number;
}

/**
 * Bounding box with normalized coordinates (0-1)
 */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Single annotation on an image
 */
export interface Annotation {
  id: string;
  type: AnnotationType;
  label: string;
  confidence: number;
  bbox?: BoundingBox;
  polygon?: Point[];
  maskUrl?: string;
  color: string;
  visible: boolean;
}

// ============================================================================
// Review Item
// ============================================================================

/**
 * Status of a review item
 */
export type ReviewStatus = "pending" | "approved" | "rejected" | "modified";

/**
 * Image dimensions in pixels
 */
export interface ImageDimensions {
  width: number;
  height: number;
}

/**
 * Single item in the review queue
 */
export interface ReviewItem {
  id: string;
  filePath: string;
  fileName: string;
  dimensions: ImageDimensions;
  thumbnailUrl: string;
  annotations: Annotation[];
  originalAnnotations: Annotation[];
  status: ReviewStatus;
  reviewedAt?: string;
  flagged: boolean;
  flagReason?: string;
}

// ============================================================================
// Review Filters
// ============================================================================

/**
 * Filters for the review queue list
 */
export interface ReviewFilters {
  status: ReviewStatus | "all";
  label: string | null;
  minConfidence: number;
  maxConfidence: number;
  searchQuery: string;
}

// ============================================================================
// API Types (for backend communication)
// ============================================================================

/**
 * Payload for updating a review item
 */
export interface ReviewItemUpdate {
  status?: ReviewStatus;
  annotations?: Annotation[];
  flagged?: boolean;
  flagReason?: string;
}

/**
 * Payload for creating a new annotation
 */
export interface AnnotationCreate {
  type: AnnotationType;
  label: string;
  confidence?: number;
  bbox?: BoundingBox;
  polygon?: Point[];
  maskUrl?: string;
  color?: string;
  visible?: boolean;
}

/**
 * Payload for updating an annotation
 */
export interface AnnotationUpdate {
  label?: string;
  confidence?: number;
  bbox?: BoundingBox;
  polygon?: Point[];
  color?: string;
  visible?: boolean;
}

/**
 * Batch operation request
 */
export interface BatchRequest {
  itemIds: string[];
}

/**
 * Batch operation response
 */
export interface BatchResponse {
  successCount: number;
  failedCount: number;
  failedIds: string[];
}

/**
 * Paginated response for review items
 */
export interface ReviewItemsResponse {
  items: ReviewItem[];
  total: number;
  skip: number;
  limit: number;
}

// ============================================================================
// Review State Interface
// ============================================================================

/**
 * Review state interface for the Svelte store
 */
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

// ============================================================================
// Canvas Types (for annotation editing)
// ============================================================================

/**
 * Canvas transform state for zoom/pan
 */
export interface CanvasTransform {
  x: number;
  y: number;
  scale: number;
}

/**
 * Interaction mode for bounding box editing
 */
export type BoxInteractionMode =
  | "idle"
  | "dragging-box"
  | "resizing-nw"
  | "resizing-n"
  | "resizing-ne"
  | "resizing-e"
  | "resizing-se"
  | "resizing-s"
  | "resizing-sw"
  | "resizing-w";

/**
 * Resize handle positions
 */
export type ResizeHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
