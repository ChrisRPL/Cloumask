<script lang="ts" module>
	export interface ReviewQueueProps {
		executionId: string;
		onDone?: () => void;
		class?: string;
	}
</script>

<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { convertFileSrc } from '@tauri-apps/api/core';
	import { cn } from '$lib/utils.js';
	import { getReviewState } from '$lib/stores/review.svelte.js';
	import type { Annotation, ReviewItem } from '$lib/types/review';

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

	let { executionId, onDone, class: className }: ReviewQueueProps = $props();

	// Review state from context
	const reviewState = getReviewState();

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
		// Use Tauri's convertFileSrc to serve local files
		return convertFileSrc(currentItem.filePath);
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
				body: JSON.stringify({ annotations })
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
	// Keyboard Shortcuts
	// ============================================================================

	function isInputFocused(target: EventTarget | null): boolean {
		if (!target || !(target instanceof HTMLElement)) return false;
		const tagName = target.tagName.toLowerCase();
		return (
			tagName === 'input' ||
			tagName === 'textarea' ||
			tagName === 'select' ||
			target.isContentEditable
		);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (isInputFocused(e.target)) return;

		const key = e.key.toLowerCase();
		const ctrl = e.ctrlKey || e.metaKey;

		switch (key) {
			// Navigation
			case 'j':
			case 'arrowdown':
				e.preventDefault();
				reviewState.nextItem();
				break;
			case 'k':
			case 'arrowup':
				e.preventDefault();
				reviewState.previousItem();
				break;

			// Actions
			case 'a':
				if (!ctrl) {
					e.preventDefault();
					approveCurrentItem();
				}
				break;
			case 'r':
				if (!ctrl) {
					e.preventDefault();
					rejectCurrentItem();
				}
				break;
			case 'e':
				e.preventDefault();
				toggleEditMode();
				break;

			// Tools (when in edit mode)
			case 'v':
				if (isEditMode) {
					e.preventDefault();
					drawingTool = 'select';
				}
				break;
			case 'b':
				if (isEditMode) {
					e.preventDefault();
					drawingTool = 'rectangle';
				}
				break;
			case 'p':
				if (isEditMode) {
					e.preventDefault();
					drawingTool = 'polygon';
				}
				break;

			// Undo/Redo
			case 'z':
				if (ctrl && !e.shiftKey) {
					e.preventDefault();
					undo();
				} else if (ctrl && e.shiftKey) {
					e.preventDefault();
					redo();
				}
				break;
			case 'y':
				if (ctrl) {
					e.preventDefault();
					redo();
				}
				break;

			// Delete
			case 'delete':
			case 'backspace':
				if (selectedAnnotationId && isEditMode) {
					e.preventDefault();
					handleAnnotationDelete(selectedAnnotationId);
				}
				break;

			// Escape
			case 'escape':
				e.preventDefault();
				if (isEditMode) {
					isEditMode = false;
					drawingTool = 'select';
				}
				selectedAnnotationId = null;
				break;

			// Zoom
			case '+':
			case '=':
				e.preventDefault();
				zoom = Math.min(zoom * 1.25, 5);
				break;
			case '-':
				e.preventDefault();
				zoom = Math.max(zoom / 1.25, 0.1);
				break;
			case '0':
				e.preventDefault();
				zoom = 1; // Fit to view
				break;
			case '1':
				e.preventDefault();
				zoom = 1; // 100%
				break;
		}
	}

	// ============================================================================
	// Data Loading
	// ============================================================================

	async function loadReviewItems() {
		reviewState.setLoading(true);
		try {
			const response = await fetch(
				`http://localhost:8765/api/review/items?execution_id=${executionId}`
			);
			if (!response.ok) {
				throw new Error(`Failed to load review items: ${response.statusText}`);
			}
			const data = await response.json();
			reviewState.loadItems(data.items);
		} catch (error) {
			console.error('Error loading review items:', error);
			// TODO: Show error toast
		} finally {
			reviewState.setLoading(false);
		}
	}

	// Load items on mount
	onMount(() => {
		loadReviewItems();
	});
</script>

<svelte:window on:keydown={handleKeydown} />

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
	<SplitPane class="flex-1 min-h-0">
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
						class="h-48 flex-shrink-0"
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
