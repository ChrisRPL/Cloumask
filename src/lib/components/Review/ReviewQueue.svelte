<script lang="ts" module>
	export interface ReviewQueueProps {
		executionId?: string;
		projectId?: string | null;
		onDone?: () => void;
		class?: string;
	}
</script>

<script lang="ts">
	import { onDestroy, untrack } from 'svelte';
	import { convertFileSrc } from '@tauri-apps/api/core';
	import { cn } from '$lib/utils.js';
	import { toLocalImageUrl } from '$lib/utils/local-image';
	import { isTauri } from '$lib/utils/tauri';
	import { getReviewState } from '$lib/stores/review.svelte.js';
	import { getKeyboardState } from '$lib/stores/keyboard.svelte.js';
	import { getExecutionState } from '$lib/stores/execution.svelte.js';
	import type { Annotation, ReviewItem, ReviewStatus } from '$lib/types/review';

	// Components
	import ReviewHeader from './ReviewHeader.svelte';
	import FilterBar from './FilterBar.svelte';
	import SplitPane from './layout/SplitPane.svelte';
	import ItemList from './list/ItemList.svelte';
	import AnnotationCanvas from './canvas/AnnotationCanvas.svelte';
	import CanvasToolbar, { type DrawingTool } from './canvas/CanvasToolbar.svelte';
	import ZoomControls from './canvas/ZoomControls.svelte';
	import AnnotationDetails from './details/AnnotationDetails.svelte';
	import ActionBar from './actions/ActionBar.svelte';

	let { executionId = 'current', projectId = null, onDone, class: className }: ReviewQueueProps =
		$props();

	const isDesktopTauri = isTauri();
	const safeConvertFileSrc: ((path: string) => string) | null =
		typeof convertFileSrc === 'function' ? convertFileSrc : null;

	type BackendAnnotation = {
		id: string;
		type: 'bbox' | 'polygon' | 'mask';
		label: string;
		confidence: number;
		bbox?: { x: number; y: number; width: number; height: number };
		polygon?: Array<{ x: number; y: number }>;
		mask_url?: string | null;
		color?: string;
		visible?: boolean;
	};

	type BackendReviewItem = {
		id: string;
		file_path: string;
		file_name: string;
		dimensions: { width: number; height: number };
		thumbnail_url: string;
		annotations?: BackendAnnotation[];
		original_annotations?: BackendAnnotation[];
		status?: string;
		reviewed_at?: string | null;
		flagged?: boolean;
		flag_reason?: string | null;
	};

	function normalizeStatus(status: string | undefined): ReviewStatus {
		if (status === 'approved' || status === 'rejected' || status === 'modified') return status;
		return 'pending';
	}

	function normalizeAnnotation(annotation: BackendAnnotation): Annotation {
		return {
			id: annotation.id,
			type: annotation.type,
			label: annotation.label,
			confidence: annotation.confidence,
			bbox: annotation.bbox,
			polygon: annotation.polygon,
			maskUrl: annotation.mask_url ?? undefined,
			color: annotation.color ?? '#166534',
			visible: annotation.visible ?? true
		};
	}

	function normalizeItem(item: BackendReviewItem): ReviewItem {
		return {
			id: item.id,
			filePath: item.file_path,
			fileName: item.file_name,
			dimensions: item.dimensions,
			thumbnailUrl: item.thumbnail_url,
			annotations: (item.annotations ?? []).map(normalizeAnnotation),
			originalAnnotations: (item.original_annotations ?? []).map(normalizeAnnotation),
			status: normalizeStatus(item.status),
			reviewedAt: item.reviewed_at ?? undefined,
			flagged: item.flagged ?? false,
			flagReason: item.flag_reason ?? undefined
		};
	}

	function toBackendAnnotation(annotation: Annotation): BackendAnnotation {
		return {
			id: annotation.id,
			type: annotation.type,
			label: annotation.label,
			confidence: annotation.confidence,
			bbox: annotation.bbox,
			polygon: annotation.polygon,
			mask_url: annotation.maskUrl ?? null,
			color: annotation.color,
			visible: annotation.visible
		};
	}

	// Review state from context
	const reviewState = getReviewState();
	const keyboard = getKeyboardState();
	const execution = getExecutionState();

	// Local UI state
	let isEditMode = $state(false);
	let drawingTool = $state<DrawingTool>('select');
	let selectedAnnotationId = $state<string | null>(null);
	let zoom = $state(1);

	// Undo/redo stacks (command pattern)
	let undoStack = $state<Command[]>([]);
	let redoStack = $state<Command[]>([]);

	// Command interface for undo/redo
	interface Command {
		execute: () => void;
		undo: () => void;
		description: string;
	}

	// Derived values
	const currentItem = $derived(reviewState.currentItem);
	const currentIndex = $derived.by(() => {
		if (!currentItem) return -1;
		return reviewState.filteredItems.findIndex((i) => i.id === currentItem.id);
	});
	const canPrev = $derived(currentIndex > 0);
	const canNext = $derived(currentIndex < reviewState.filteredItems.length - 1);
	const imageUrl = $derived.by(() => {
		if (!currentItem) return '';
		// In desktop mode we can render local file paths directly.
		if (isDesktopTauri && safeConvertFileSrc) {
			return safeConvertFileSrc(currentItem.filePath);
		}
		// Browser mode: route local files through backend image endpoint.
		return toLocalImageUrl(currentItem.filePath || currentItem.thumbnailUrl);
	});

	// Keep selection valid when filters hide the current item.
	$effect(() => {
		const filtered = reviewState.filteredItems;
		const currentId = reviewState.currentItemId;

		if (filtered.length === 0) {
			if (currentId !== null) {
				reviewState.setCurrentItem(null);
			}
			return;
		}

		if (!currentId || !filtered.some((item) => item.id === currentId)) {
			reviewState.setCurrentItem(filtered[0].id);
		}
	});

	// Available labels for dropdown
	const availableLabels = ['person', 'face', 'license_plate', 'vehicle', 'sign', 'other'];

	// Debounce timer for saving
	let saveTimeout: ReturnType<typeof setTimeout> | null = null;
	let lastSavedAnnotations: string = '';

	// Debounced save function
	async function saveAnnotations(itemId: string, annotations: Annotation[]) {
		try {
			await fetch(`http://localhost:8765/api/review/items/${itemId}`, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					annotations: annotations.map(toBackendAnnotation)
				})
			});
		} catch (error) {
			console.error('Error saving annotations:', error);
			// TODO: Show error toast
		}
	}

	function debouncedSave(itemId: string, annotations: Annotation[]) {
		const serialized = JSON.stringify(annotations);
		if (serialized === lastSavedAnnotations) return;

		if (saveTimeout) clearTimeout(saveTimeout);
		saveTimeout = setTimeout(() => {
			lastSavedAnnotations = serialized;
			saveAnnotations(itemId, annotations);
		}, 500);
	}

	// Watch for annotation changes and auto-save
	$effect(() => {
		if (currentItem && currentItem.annotations.length > 0) {
			debouncedSave(currentItem.id, currentItem.annotations);
		}
	});

	// Cleanup timeout on destroy
	onDestroy(() => {
		if (saveTimeout) clearTimeout(saveTimeout);
	});

	// ============================================================================
	// Command Pattern for Undo/Redo
	// ============================================================================

	function executeCommand(cmd: Command) {
		cmd.execute();
		undoStack = [...undoStack, cmd].slice(-50); // Max 50 commands
		redoStack = [];
	}

	function undo() {
		const cmd = undoStack.at(-1);
		if (!cmd) return;
		cmd.undo();
		undoStack = undoStack.slice(0, -1);
		redoStack = [...redoStack, cmd];
	}

	function redo() {
		const cmd = redoStack.at(-1);
		if (!cmd) return;
		cmd.execute();
		redoStack = redoStack.slice(0, -1);
		undoStack = [...undoStack, cmd];
	}

	// ============================================================================
	// Actions
	// ============================================================================

	function approveCurrentItem() {
		if (!currentItem) return;

		const itemId = currentItem.id;
		const previousStatus = currentItem.status;

		executeCommand({
			execute: () => reviewState.approveItem(itemId),
			undo: () => reviewState.updateItem(itemId, { status: previousStatus }),
			description: `Approve item ${itemId.slice(0, 8)}`
		});

		// Auto-advance to next item
		reviewState.nextItem();
	}

	function rejectCurrentItem() {
		if (!currentItem) return;

		const itemId = currentItem.id;
		const previousStatus = currentItem.status;

		executeCommand({
			execute: () => reviewState.rejectItem(itemId),
			undo: () => reviewState.updateItem(itemId, { status: previousStatus }),
			description: `Reject item ${itemId.slice(0, 8)}`
		});

		// Auto-advance to next item
		reviewState.nextItem();
	}

	function toggleEditMode() {
		isEditMode = !isEditMode;
		if (!isEditMode) {
			drawingTool = 'select';
		}
	}

	function handleToolChange(tool: DrawingTool) {
		drawingTool = tool;
		if (tool !== 'select') {
			isEditMode = true;
		}
	}

	function handleAnnotationCreate(annotation: Annotation) {
		if (!currentItem) return;

		const itemId = currentItem.id;
		let createdAnnotation: Annotation | null = null;

		executeCommand({
			execute: () => {
				createdAnnotation = reviewState.addAnnotation(itemId, annotation);
			},
			undo: () => {
				if (createdAnnotation) {
					reviewState.removeAnnotation(itemId, createdAnnotation.id);
				}
			},
			description: `Create ${annotation.type} annotation`
		});
	}

	function handleAnnotationUpdate(annotation: Annotation) {
		if (!currentItem) return;

		const itemId = currentItem.id;
		const previousAnnotation = currentItem.annotations.find((a) => a.id === annotation.id);
		if (!previousAnnotation) return;

		executeCommand({
			execute: () => reviewState.updateAnnotation(itemId, annotation.id, annotation),
			undo: () => reviewState.updateAnnotation(itemId, annotation.id, previousAnnotation),
			description: `Update annotation ${annotation.id.slice(0, 8)}`
		});
	}

	function handleAnnotationDelete(annotationId: string) {
		if (!currentItem) return;

		const itemId = currentItem.id;
		const annotation = currentItem.annotations.find((a) => a.id === annotationId);
		if (!annotation) return;

		executeCommand({
			execute: () => reviewState.removeAnnotation(itemId, annotationId),
			undo: () => reviewState.addAnnotation(itemId, annotation),
			description: `Delete annotation ${annotationId.slice(0, 8)}`
		});

		selectedAnnotationId = null;
	}

	function handleAnnotationLabelChange(annotationId: string, label: string) {
		if (!currentItem) return;

		const itemId = currentItem.id;
		const annotation = currentItem.annotations.find((a) => a.id === annotationId);
		if (!annotation) return;

		const previousLabel = annotation.label;

		executeCommand({
			execute: () => reviewState.updateAnnotation(itemId, annotationId, { label }),
			undo: () => reviewState.updateAnnotation(itemId, annotationId, { label: previousLabel }),
			description: `Change label to "${label}"`
		});
	}

	function handleAnnotationVisibilityToggle(annotationId: string) {
		if (!currentItem) return;

		const annotation = currentItem.annotations.find((a) => a.id === annotationId);
		if (!annotation) return;

		reviewState.updateAnnotation(currentItem.id, annotationId, { visible: !annotation.visible });
	}

	function handleClearAllAnnotations() {
		if (!currentItem || currentItem.annotations.length === 0) return;

		const itemId = currentItem.id;
		const previousAnnotations = [...currentItem.annotations];

		executeCommand({
			execute: () => {
				for (const ann of previousAnnotations) {
					reviewState.removeAnnotation(itemId, ann.id);
				}
			},
			undo: () => {
				for (const ann of previousAnnotations) {
					reviewState.addAnnotation(itemId, ann);
				}
			},
			description: `Clear all annotations`
		});
	}

	function handleApproveAll() {
		const pendingIds = reviewState.filteredItems
			.filter((i) => i.status === 'pending')
			.map((i) => i.id);

		if (pendingIds.length === 0) return;

		const previousStates = pendingIds.map((id) => ({
			id,
			status: reviewState.items.find((i) => i.id === id)?.status ?? 'pending'
		}));

		executeCommand({
			execute: () => {
				for (const id of pendingIds) {
					reviewState.approveItem(id);
				}
			},
			undo: () => {
				for (const state of previousStates) {
					reviewState.updateItem(state.id, { status: state.status });
				}
			},
			description: `Approve all ${pendingIds.length} pending items`
		});
	}

	// ============================================================================
	// Keyboard Shortcuts (registered with keyboard store for scope awareness)
	// ============================================================================

	// NOTE: We use untrack() to prevent this effect from tracking registeredShortcuts reads
	// inside keyboard.register(). Without untrack, register() reads the shortcuts map,
	// which would cause an infinite loop (effect_update_depth_exceeded error).
	$effect(() => {
		const unregisterFns: (() => void)[] = [];

		untrack(() => {
			// Approve current item (with undo support)
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'a',
						action: approveCurrentItem,
						scope: 'review',
						description: 'Approve and advance',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			// Reject current item (with undo support)
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'r',
						action: rejectCurrentItem,
						scope: 'review',
						description: 'Reject and advance',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			// Toggle edit mode
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'e',
						action: toggleEditMode,
						scope: 'review',
						description: 'Toggle edit mode',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			// Tool shortcuts (when in edit mode)
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'v',
						action: () => {
							if (isEditMode) drawingTool = 'select';
						},
						scope: 'review',
						description: 'Select tool',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'b',
						action: () => {
							if (isEditMode) drawingTool = 'rectangle';
						},
						scope: 'review',
						description: 'Rectangle tool',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'p',
						action: () => {
							if (isEditMode) drawingTool = 'polygon';
						},
						scope: 'review',
						description: 'Polygon tool',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			// Undo/Redo
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'ctrl+z',
						action: undo,
						scope: 'review',
						description: 'Undo',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			// Redo (Ctrl+Shift+Z)
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'ctrl+shift+z',
						action: redo,
						scope: 'review',
						description: 'Redo',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			// Redo (Ctrl+Y - alternative)
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'ctrl+y',
						action: redo,
						scope: 'review',
						description: 'Redo',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			// Delete annotation (Delete key)
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'delete',
						action: () => {
							if (selectedAnnotationId && isEditMode) {
								handleAnnotationDelete(selectedAnnotationId);
							}
						},
						scope: 'review',
						description: 'Delete annotation',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			// Delete annotation (Backspace - alternative)
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'backspace',
						action: () => {
							if (selectedAnnotationId && isEditMode) {
								handleAnnotationDelete(selectedAnnotationId);
							}
						},
						scope: 'review',
						description: 'Delete annotation',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			// Escape - exit edit mode
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: 'escape',
						action: () => {
							if (isEditMode) {
								isEditMode = false;
								drawingTool = 'select';
							}
							selectedAnnotationId = null;
						},
						scope: 'review',
						description: 'Exit edit mode',
						category: 'Review',
						priority: 10, // Lower than global escape
					});
					return () => keyboard.unregister(id);
				})()
			);

			// Zoom in (+)
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: '+',
						action: () => {
							zoom = Math.min(zoom * 1.25, 5);
						},
						scope: 'review',
						description: 'Zoom in',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			// Zoom in (= - alternative for keyboards where + requires shift)
			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: '=',
						action: () => {
							zoom = Math.min(zoom * 1.25, 5);
						},
						scope: 'review',
						description: 'Zoom in',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: '-',
						action: () => {
							zoom = Math.max(zoom / 1.25, 0.1);
						},
						scope: 'review',
						description: 'Zoom out',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);

			unregisterFns.push(
				(() => {
					const id = keyboard.register({
						combo: '0',
						action: () => {
							zoom = 1;
						},
						scope: 'review',
						description: 'Reset zoom',
						category: 'Review',
					});
					return () => keyboard.unregister(id);
				})()
			);
		}); // end untrack

		return () => {
			untrack(() => {
				for (const fn of unregisterFns) fn();
			});
		};
	});

	// ============================================================================
	// Data Loading
	// ============================================================================
	let lastLoadedContextKey = $state<string | null>(null);

	async function loadReviewItems() {
		const execId = executionId?.trim();
		if (!execId) {
			reviewState.loadItems([]);
			reviewState.setLoading(false);
			return;
		}

		reviewState.setLoading(true);
		try {
			const params = new URLSearchParams({ execution_id: execId });
			const activeProjectId = projectId?.trim();
			if (activeProjectId) {
				params.set('project_id', activeProjectId);
			}

			const response = await fetch(`http://localhost:8765/api/review/items?${params.toString()}`);
			if (!response.ok) {
				throw new Error(`Failed to load review items: ${response.statusText}`);
			}
			const data = await response.json();
			const items = Array.isArray(data.items) ? (data.items as BackendReviewItem[]) : [];
			reviewState.loadItems(items.map(normalizeItem));
		} catch (error) {
			console.error('Error loading review items:', error);
			// TODO: Show error toast
		} finally {
			reviewState.setLoading(false);
		}
	}

	// Reload whenever execution context changes.
	$effect(() => {
		const execId = executionId?.trim();
		if (!execId) return;

		const activeProjectId = projectId?.trim() ?? '__default__';
		const contextKey = `${activeProjectId}:${execId}`;
		if (contextKey === lastLoadedContextKey) return;

		lastLoadedContextKey = contextKey;
		void loadReviewItems();
	});

	// Polling: Auto-refresh items during active execution
	let pollInterval: ReturnType<typeof setInterval> | null = null;
	
	$effect(() => {
		// Clean up any existing interval
		if (pollInterval) {
			clearInterval(pollInterval);
			pollInterval = null;
		}
		
		// Start polling if execution is running
		if (execution.isRunning) {
			pollInterval = setInterval(() => {
				void loadReviewItems();
			}, 5000); // Poll every 5 seconds
		}
		
		// Cleanup function
		return () => {
			if (pollInterval) {
				clearInterval(pollInterval);
				pollInterval = null;
			}
		};
	});
	
	// Also reload when execution completes
	$effect(() => {
		if (execution.status === 'completed' || execution.status === 'failed') {
			void loadReviewItems();
		}
	});
</script>

<div
	class={cn(
		'flex flex-col h-full',
		'bg-background text-foreground',
		'font-mono',
		className
	)}
>
	<!-- Header -->
	<ReviewHeader
		total={reviewState.items.length}
		pending={reviewState.pendingCount}
		approved={reviewState.approvedCount}
		rejected={reviewState.rejectedCount}
		currentIndex={currentIndex}
		onApproveAll={handleApproveAll}
		{onDone}
	/>

	<!-- Filter Bar -->
	<FilterBar
		statusFilter={reviewState.filters.status}
		searchQuery={reviewState.filters.searchQuery}
		minConfidence={reviewState.filters.minConfidence}
		onStatusChange={(status) => reviewState.setFilter('status', status)}
		onSearchChange={(query) => reviewState.setFilter('searchQuery', query)}
		onConfidenceChange={(min, max) => {
			reviewState.setFilter('minConfidence', min);
			reviewState.setFilter('maxConfidence', max);
		}}
		onReset={() => reviewState.resetFilters()}
	/>

	<!-- Main Content -->
	<SplitPane class="flex-1 min-h-0" leftWidth={340} minLeftWidth={260} maxLeftWidth={520}>
		{#snippet left()}
			<ItemList
				items={reviewState.filteredItems}
				selectedItemId={currentItem?.id ?? null}
				isLoading={reviewState.isLoading}
				onItemSelect={(id) => reviewState.setCurrentItem(id)}
				onLoadMore={() => {
					// TODO: Implement pagination
				}}
			/>
		{/snippet}

		{#snippet right()}
			<div class="flex flex-col h-full">
				<!-- Canvas Area -->
				<div class="relative flex-1 min-h-0">
					{#if currentItem}
						<AnnotationCanvas
							{imageUrl}
							annotations={currentItem.annotations}
							{selectedAnnotationId}
							{isEditMode}
							drawingTool={drawingTool === 'select' ? 'rectangle' : drawingTool}
							{zoom}
							onAnnotationCreate={handleAnnotationCreate}
							onAnnotationUpdate={handleAnnotationUpdate}
							onAnnotationDelete={handleAnnotationDelete}
							onSelectionChange={(id) => (selectedAnnotationId = id)}
							class="w-full h-full"
						/>

						<!-- Canvas Toolbar -->
						<CanvasToolbar
							activeTool={drawingTool}
							{isEditMode}
							onToolChange={handleToolChange}
							onToggleEdit={toggleEditMode}
							onClearAll={handleClearAllAnnotations}
							class="absolute top-2 left-2"
						/>

						<!-- Zoom Controls -->
						<ZoomControls
							{zoom}
							onZoomChange={(z) => (zoom = z)}
							onFitToView={() => (zoom = 1)}
							onResetZoom={() => (zoom = 1)}
							class="absolute bottom-2 right-2"
						/>
					{:else}
						<div class="flex items-center justify-center h-full">
							<div class="text-center text-muted-foreground">
								<p class="text-sm font-mono">No item selected</p>
								<p class="text-xs mt-1">Select an item from the list to view</p>
							</div>
						</div>
					{/if}
				</div>

				<!-- Annotation Details Panel -->
				{#if currentItem}
					<AnnotationDetails
						annotations={currentItem.annotations}
						{selectedAnnotationId}
						{availableLabels}
						onAnnotationSelect={(id) => (selectedAnnotationId = id)}
						onAnnotationLabelChange={handleAnnotationLabelChange}
						onAnnotationDelete={handleAnnotationDelete}
						onAnnotationVisibilityToggle={handleAnnotationVisibilityToggle}
						onAddAnnotation={() => {
							isEditMode = true;
							drawingTool = 'rectangle';
						}}
						class="h-40 flex-shrink-0"
					/>
				{/if}
			</div>
		{/snippet}
	</SplitPane>

	<!-- Action Bar -->
	<ActionBar
		{currentIndex}
		total={reviewState.filteredItems.length}
		{canPrev}
		{canNext}
		{isEditMode}
		onPrev={() => reviewState.previousItem()}
		onNext={() => reviewState.nextItem()}
		onApprove={approveCurrentItem}
		onReject={rejectCurrentItem}
		onToggleEdit={toggleEditMode}
	/>
</div>
