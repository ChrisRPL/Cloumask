<script lang="ts" module>
	export interface AnnotationCanvasProps {
		imageUrl: string;
		fallbackImageUrl?: string;
		annotations: import('$lib/types/review').Annotation[];
		selectedAnnotationId?: string | null;
		isEditMode?: boolean;
		drawingTool?: 'rectangle' | 'polygon';
		zoom?: number;
		onAnnotationCreate?: (annotation: import('$lib/types/review').Annotation) => void;
		onAnnotationUpdate?: (annotation: import('$lib/types/review').Annotation) => void;
		onAnnotationDelete?: (annotationId: string) => void;
		onSelectionChange?: (annotationId: string | null) => void;
		class?: string;
	}
</script>

<script lang="ts">
	import { onDestroy } from 'svelte';
	import { createImageAnnotator, type ImageAnnotator } from '@annotorious/annotorious';
	import '@annotorious/annotorious/annotorious.css';
	import { cn } from '$lib/utils.js';
	import type { Annotation, BoundingBox, Point } from '$lib/types/review';

	let {
		imageUrl,
		fallbackImageUrl = '',
		annotations = [],
		selectedAnnotationId = null,
		isEditMode = false,
		drawingTool = 'rectangle',
		zoom = 1,
		onAnnotationCreate,
		onAnnotationUpdate,
		onAnnotationDelete,
		onSelectionChange,
		class: className
	}: AnnotationCanvasProps = $props();

	let containerRef: HTMLDivElement | undefined = $state();
	let imageRef: HTMLImageElement | undefined = $state();
	let annotator: ImageAnnotator | null = $state(null);
	let imageDimensions = $state({ width: 0, height: 0 });
	let isImageLoaded = $state(false);
	let hasImageError = $state(false);
	let activeImageUrl = $state('');

	// Convert our annotation format to W3C Web Annotation format
	function toW3CAnnotation(ann: Annotation): object | null {
		if (!imageDimensions.width || !imageDimensions.height) return null;

		const w3c: Record<string, unknown> = {
			'@context': 'http://www.w3.org/ns/anno.jsonld',
			id: ann.id,
			type: 'Annotation',
			body: [
				{
					type: 'TextualBody',
					purpose: 'tagging',
					value: ann.label
				}
			]
		};

		if (ann.type === 'bbox' && ann.bbox) {
			// Convert normalized (0-1) to pixel coordinates
			const x = Math.round(ann.bbox.x * imageDimensions.width);
			const y = Math.round(ann.bbox.y * imageDimensions.height);
			const width = Math.round(ann.bbox.width * imageDimensions.width);
			const height = Math.round(ann.bbox.height * imageDimensions.height);

			w3c.target = {
				selector: {
					type: 'FragmentSelector',
					conformsTo: 'http://www.w3.org/TR/media-frags/',
					value: `xywh=pixel:${x},${y},${width},${height}`
				}
			};
		} else if (ann.type === 'polygon' && ann.polygon) {
			// Convert normalized points to pixel SVG path
			const points = ann.polygon
				.map((p) => {
					const px = Math.round(p.x * imageDimensions.width);
					const py = Math.round(p.y * imageDimensions.height);
					return `${px},${py}`;
				})
				.join(' ');

			w3c.target = {
				selector: {
					type: 'SvgSelector',
					value: `<svg><polygon points="${points}"/></svg>`
				}
			};
		}

		return w3c;
	}

	// Convert W3C annotation back to our format
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	function fromW3CAnnotation(w3c: any): Annotation | null {
		if (!imageDimensions.width || !imageDimensions.height) return null;

		const target = w3c.target as Record<string, unknown> | undefined;
		const selector = target?.selector as Record<string, unknown> | undefined;

		if (!selector) return null;

		// Extract label from body
		const bodies = w3c.body as Array<Record<string, string>> | undefined;
		const tagBody = bodies?.find((b) => b.purpose === 'tagging');
		const label = tagBody?.value || 'unknown';

		const id = (w3c.id as string) || crypto.randomUUID();

		if (selector.type === 'FragmentSelector') {
			// Parse bbox from xywh=pixel:x,y,w,h
			const value = selector.value as string;
			const match = value.match(/xywh=pixel:(\d+),(\d+),(\d+),(\d+)/);
			if (!match) return null;

			const [, x, y, width, height] = match.map(Number);

			const bbox: BoundingBox = {
				x: x / imageDimensions.width,
				y: y / imageDimensions.height,
				width: width / imageDimensions.width,
				height: height / imageDimensions.height
			};

			return {
				id,
				type: 'bbox',
				label,
				confidence: 1.0,
				bbox,
				polygon: undefined,
				maskUrl: undefined,
				color: '#166534',
				visible: true
			};
		} else if (selector.type === 'SvgSelector') {
			// Parse polygon from SVG
			const value = selector.value as string;
			const pointsMatch = value.match(/points="([^"]+)"/);
			if (!pointsMatch) return null;

			const polygon: Point[] = pointsMatch[1].split(' ').map((pair) => {
				const [px, py] = pair.split(',').map(Number);
				return {
					x: px / imageDimensions.width,
					y: py / imageDimensions.height
				};
			});

			return {
				id,
				type: 'polygon',
				label,
				confidence: 1.0,
				bbox: undefined,
				polygon,
				maskUrl: undefined,
				color: '#166534',
				visible: true
			};
		}

		return null;
	}

	// Initialize Annotorious when image loads
	function initAnnotator() {
		if (!imageRef || !isImageLoaded || annotator) return;

		annotator = createImageAnnotator(imageRef, {
			autoSave: true,
			drawingEnabled: isEditMode
		});

		// Set initial drawing tool
		annotator.setDrawingTool(drawingTool);

		// Load existing annotations
		const w3cAnnotations = annotations
			.map((ann) => toW3CAnnotation(ann))
			.filter((a): a is object => a !== null);

		if (w3cAnnotations.length > 0) {
			annotator.setAnnotations(w3cAnnotations);
		}

		// Event handlers - use 'any' to bypass Annotorious type mismatches
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		annotator.on('createAnnotation', (w3c: any) => {
			const converted = fromW3CAnnotation(w3c);
			if (converted) {
				onAnnotationCreate?.(converted);
			}
		});

		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		annotator.on('updateAnnotation', (w3c: any, _previous: any) => {
			const converted = fromW3CAnnotation(w3c);
			if (converted) {
				onAnnotationUpdate?.(converted);
			}
		});

		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		annotator.on('deleteAnnotation', (w3c: any) => {
			const id = w3c.id as string;
			if (id) {
				onAnnotationDelete?.(id);
			}
		});

		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		annotator.on('selectionChanged', (selected: any[]) => {
			const selectedId = selected.length > 0 ? (selected[0].id as string) : null;
			onSelectionChange?.(selectedId);
		});
	}

	function resetCanvasState() {
		if (annotator) {
			annotator.destroy();
			annotator = null;
		}
		imageDimensions = { width: 0, height: 0 };
		isImageLoaded = false;
	}

	// Handle image load
	function handleImageLoad() {
		if (imageRef) {
			imageDimensions = {
				width: imageRef.naturalWidth,
				height: imageRef.naturalHeight
			};
			isImageLoaded = true;
			initAnnotator();
		}
	}

	function handleImageError() {
		const fallback = fallbackImageUrl?.trim();
		if (fallback && activeImageUrl !== fallback) {
			activeImageUrl = fallback;
			isImageLoaded = false;
			hasImageError = false;
			return;
		}

		hasImageError = true;
		resetCanvasState();
	}

	// Reset image state when source changes.
	$effect(() => {
		imageUrl;
		fallbackImageUrl;

		hasImageError = false;
		activeImageUrl = imageUrl;
		resetCanvasState();
	});

	// Sync edit mode
	$effect(() => {
		if (annotator) {
			annotator.setDrawingEnabled(isEditMode);
		}
	});

	// Sync drawing tool
	$effect(() => {
		if (annotator) {
			annotator.setDrawingTool(drawingTool);
		}
	});

	// Sync annotations when they change externally
	$effect(() => {
		if (annotator && isImageLoaded && annotations) {
			const w3cAnnotations = annotations
				.map((ann) => toW3CAnnotation(ann))
				.filter((a): a is object => a !== null);
			annotator.setAnnotations(w3cAnnotations);
		}
	});

	// Cleanup
	onDestroy(() => {
		resetCanvasState();
	});

	// Handle keyboard shortcuts for canvas
	function handleKeydown(e: KeyboardEvent) {
		if (!annotator) return;

		if (e.key === 'Escape') {
			annotator.cancelDrawing();
		} else if (e.key === 'Delete' || e.key === 'Backspace') {
			// Delete selected annotation - handled by parent
		}
	}
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions a11y_no_noninteractive_tabindex -->
<div
	bind:this={containerRef}
	class={cn(
		'relative w-full h-full overflow-hidden',
		'bg-muted/30 rounded-lg',
		'focus:outline-none focus-visible:ring-2 focus-visible:ring-primary',
		className
	)}
	tabindex="-1"
	role="application"
	aria-label="Annotation canvas"
	onkeydown={handleKeydown}
>
	{#if !isImageLoaded && !hasImageError}
		<div class="absolute inset-0 flex items-center justify-center">
			<div class="flex flex-col items-center gap-3 text-muted-foreground">
				<div
					class="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"
				></div>
				<span class="text-xs font-mono">Loading image...</span>
			</div>
		</div>
	{/if}

	{#if hasImageError}
		<div class="absolute inset-0 flex items-center justify-center">
			<div class="flex flex-col items-center gap-2 text-muted-foreground">
				<span class="text-sm font-mono">Failed to load image</span>
				<span class="text-xs">Try selecting another review item</span>
			</div>
		</div>
	{/if}

	<div
		class="flex items-center justify-center w-full h-full overflow-auto"
		style="transform: scale({zoom}); transform-origin: center;"
	>
		<img
			bind:this={imageRef}
			src={activeImageUrl}
			alt="Annotation target"
			class={cn('max-w-full max-h-full object-contain', !isImageLoaded && 'opacity-0')}
			onload={handleImageLoad}
			onerror={handleImageError}
		/>
	</div>

	{#if isEditMode}
		<div
			class="absolute top-2 left-2 px-2 py-1 text-[10px] font-mono bg-primary text-primary-foreground rounded"
		>
			EDIT MODE
		</div>
	{/if}
</div>

<style>
	/* Custom Annotorious styling for terminal aesthetic */
	:global(.a9s-annotationlayer) {
		cursor: crosshair;
	}

	:global(.a9s-annotation) {
		cursor: pointer;
	}

	:global(.a9s-annotation.selected) {
		cursor: move;
	}

	:global(.a9s-handle) {
		fill: #166534 !important;
		stroke: #faf7f0 !important;
		stroke-width: 1.5px !important;
	}

	:global(.a9s-selection-mask) {
		fill: rgba(22, 101, 52, 0.1);
	}
</style>
