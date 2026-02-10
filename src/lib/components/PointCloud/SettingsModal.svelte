<script lang="ts" module>
	export interface SettingsModalProps {
		class?: string;
		open?: boolean;
	}
</script>

<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { Switch } from '$lib/components/ui/switch';
	import { Label } from '$lib/components/ui/label';
	import { Separator } from '$lib/components/ui/separator';
	import { Input } from '$lib/components/ui/input';
	import { cn } from '$lib/utils';
	import { getPointCloudState } from '$lib/stores/pointcloud.svelte';
	import { X } from '@lucide/svelte';

	let { class: className, open = $bindable(false) }: SettingsModalProps = $props();

	const pcState = getPointCloudState();

	const budgetMin = 100_000;
	const budgetMax = 5_000_000;
	const budgetStep = 50_000;

	const presetColors = [
		{ name: 'Forest Dark', value: '#0c3b1f' },
		{ name: 'Forest', value: '#166534' },
		{ name: 'Cream', value: '#FAF7F0' },
		{ name: 'Slate', value: '#1f2937' },
		{ name: 'Charcoal', value: '#111827' },
		{ name: 'Midnight', value: '#0b1120' },
	];

	function handleBudgetInput(event: Event) {
		const target = event.target as HTMLInputElement;
		const value = Number(target.value);
		if (!Number.isNaN(value)) {
			pcState.setLodPointBudget(value);
		}
	}

	function handleColorInput(event: Event) {
		const target = event.target as HTMLInputElement;
		pcState.setBackgroundColor(target.value);
	}
</script>

<Dialog.Root bind:open={open}>
	<Dialog.Content
		class={cn(
			'max-w-lg p-0 bg-background border border-border font-mono',
			className,
		)}
	>
		<div class="flex items-center justify-between border-b border-border px-5 py-4">
			<div>
				<Dialog.Title class="text-sm font-semibold text-foreground">Viewer Settings</Dialog.Title>
				<Dialog.Description class="text-xs text-muted-foreground">
					Tune performance, overlays, and appearance.
				</Dialog.Description>
			</div>
			<button
				type="button"
				class="h-8 w-8 flex items-center justify-center rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
				onclick={() => (open = false)}
			>
				<X class="h-4 w-4" />
			</button>
		</div>

		<div class="px-5 py-4 space-y-5">
			<section class="space-y-4">
				<div class="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Performance</div>
				<div class="flex items-center justify-between gap-4">
					<div class="space-y-1">
						<Label class="text-xs font-mono">Enable LOD</Label>
						<p class="text-[10px] text-muted-foreground">
							Adaptive point decimation for large datasets.
						</p>
					</div>
					<Switch
						checked={pcState.lodEnabled}
						onCheckedChange={(checked) => pcState.setLodEnabled(checked)}
					/>
				</div>

				<div class="space-y-2">
					<div class="flex items-center justify-between">
						<Label class="text-xs font-mono">Point Budget</Label>
						<span class="text-[10px] text-muted-foreground">
							{pcState.lodPointBudget.toLocaleString()} points
						</span>
					</div>
					<input
						type="range"
						min={budgetMin}
						max={budgetMax}
						step={budgetStep}
						value={pcState.lodPointBudget}
						disabled={!pcState.lodEnabled}
						oninput={handleBudgetInput}
						class="w-full h-1.5 bg-border rounded-full appearance-none cursor-pointer accent-primary disabled:opacity-50"
					/>
					<div class="flex justify-between text-[10px] text-muted-foreground">
						<span>{(budgetMin / 1_000).toFixed(0)}k</span>
						<span>{(budgetMax / 1_000_000).toFixed(1)}M</span>
					</div>
				</div>
			</section>

			<Separator />

			<section class="space-y-4">
				<div class="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Overlays</div>
				<div class="flex items-center justify-between gap-4">
					<div class="space-y-1">
						<Label class="text-xs font-mono">Show Labels</Label>
						<p class="text-[10px] text-muted-foreground">
							Annotate bounding boxes with class and confidence.
						</p>
					</div>
					<Switch
						checked={pcState.showLabels}
						onCheckedChange={(checked) => pcState.setShowLabels(checked)}
					/>
				</div>
			</section>

			<Separator />

			<section class="space-y-4">
				<div class="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Appearance</div>
				<div class="space-y-2">
					<Label class="text-xs font-mono">Background Color</Label>
					<div class="grid grid-cols-6 gap-2">
						{#each presetColors as color}
							<button
								type="button"
								title={color.name}
								class={cn(
									'h-7 w-7 rounded-md border transition-transform hover:scale-105',
									pcState.backgroundColor === color.value ? 'border-primary' : 'border-border',
								)}
								style={`background-color: ${color.value}`}
								onclick={() => pcState.setBackgroundColor(color.value)}
							></button>
						{/each}
					</div>
					<div class="flex items-center gap-2">
						<input
							type="color"
							value={pcState.backgroundColor}
							oninput={handleColorInput}
							class="h-8 w-12 rounded border border-border bg-transparent cursor-pointer"
						/>
						<Input
							value={pcState.backgroundColor}
							oninput={handleColorInput}
							class="h-8 font-mono text-xs"
						/>
					</div>
				</div>
			</section>

			<Separator />

			<section class="space-y-3">
				<div class="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Shortcuts</div>
				<div class="grid grid-cols-2 gap-2 text-[10px]">
					<span class="text-muted-foreground">1-3</span>
					<span>Navigate</span>
					<span class="text-muted-foreground">G/A/B</span>
					<span>Toggles</span>
					<span class="text-muted-foreground">C</span>
					<span>Color mode</span>
					<span class="text-muted-foreground">+/-</span>
					<span>Point size</span>
					<span class="text-muted-foreground">R</span>
					<span>Reset camera</span>
					<span class="text-muted-foreground">S</span>
					<span>Screenshot</span>
				</div>
			</section>
		</div>

		<div class="px-5 py-3 border-t border-border bg-muted/40 text-[10px] text-muted-foreground">
			Settings apply immediately to the active viewport.
		</div>
	</Dialog.Content>
</Dialog.Root>
