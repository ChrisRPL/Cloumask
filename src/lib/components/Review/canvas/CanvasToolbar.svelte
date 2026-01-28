<script lang="ts" module>
	export type DrawingTool = 'rectangle' | 'polygon' | 'select';

	export interface CanvasToolbarProps {
		activeTool: DrawingTool;
		isEditMode: boolean;
		onToolChange?: (tool: DrawingTool) => void;
		onToggleEdit?: () => void;
		onClearAll?: () => void;
		class?: string;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button/index.js';
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import { Square, Pentagon, MousePointer2, Pencil, Trash2 } from '@lucide/svelte';

	let {
		activeTool,
		isEditMode,
		onToolChange,
		onToggleEdit,
		onClearAll,
		class: className
	}: CanvasToolbarProps = $props();

	const tools = [
		{
			id: 'select' as const,
			icon: MousePointer2,
			label: 'Select',
			shortcut: 'V'
		},
		{
			id: 'rectangle' as const,
			icon: Square,
			label: 'Rectangle',
			shortcut: 'R'
		},
		{
			id: 'polygon' as const,
			icon: Pentagon,
			label: 'Polygon',
			shortcut: 'P'
		}
	];
</script>

<div
	class={cn(
		'flex items-center gap-1 p-1.5',
		'bg-background/95 backdrop-blur-sm',
		'border border-border rounded-lg shadow-sm',
		className
	)}
>
	<!-- Edit Mode Toggle -->
	<Tooltip.Provider>
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button
					variant={isEditMode ? 'default' : 'ghost'}
					size="sm"
					onclick={onToggleEdit}
					class={cn('h-8 w-8 p-0', isEditMode && 'bg-primary text-primary-foreground')}
				>
					<Pencil class="w-4 h-4" />
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content side="bottom" class="font-mono text-xs">
				<p>
					{isEditMode ? 'Exit Edit Mode' : 'Enter Edit Mode'}
					<kbd class="ml-1 px-1 py-0.5 bg-muted rounded text-[10px]">E</kbd>
				</p>
			</Tooltip.Content>
		</Tooltip.Root>
	</Tooltip.Provider>

	<div class="w-px h-6 bg-border mx-1"></div>

	<!-- Drawing Tools -->
	{#each tools as tool (tool.id)}
		<Tooltip.Provider>
			<Tooltip.Root>
				<Tooltip.Trigger>
					<Button
						variant={activeTool === tool.id ? 'secondary' : 'ghost'}
						size="sm"
						onclick={() => onToolChange?.(tool.id)}
						disabled={!isEditMode && tool.id !== 'select'}
						class={cn(
							'h-8 w-8 p-0',
							activeTool === tool.id && 'bg-muted',
							!isEditMode && tool.id !== 'select' && 'opacity-50'
						)}
					>
						<tool.icon class="w-4 h-4" />
					</Button>
				</Tooltip.Trigger>
				<Tooltip.Content side="bottom" class="font-mono text-xs">
					<p>
						{tool.label}
						<kbd class="ml-1 px-1 py-0.5 bg-muted rounded text-[10px]">{tool.shortcut}</kbd>
					</p>
				</Tooltip.Content>
			</Tooltip.Root>
		</Tooltip.Provider>
	{/each}

	<div class="w-px h-6 bg-border mx-1"></div>

	<!-- Clear All -->
	<Tooltip.Provider>
		<Tooltip.Root>
			<Tooltip.Trigger>
				<Button
					variant="ghost"
					size="sm"
					onclick={onClearAll}
					disabled={!isEditMode}
					class={cn(
						'h-8 w-8 p-0 text-destructive hover:text-destructive hover:bg-destructive/10',
						!isEditMode && 'opacity-50'
					)}
				>
					<Trash2 class="w-4 h-4" />
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content side="bottom" class="font-mono text-xs">
				<p>Clear All Annotations</p>
			</Tooltip.Content>
		</Tooltip.Root>
	</Tooltip.Provider>
</div>
