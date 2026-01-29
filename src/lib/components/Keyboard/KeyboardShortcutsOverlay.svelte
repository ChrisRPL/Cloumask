<script lang="ts">
	import { cn } from '$lib/utils';
	import { getKeyboardState, CATEGORY_ORDER } from '$lib/stores/keyboard.svelte';
	import { formatComboString, formatSequence } from '$lib/utils/keyboard';
	import FocusTrap from './FocusTrap.svelte';

	const keyboard = getKeyboardState();

	// Get sorted categories with their shortcuts
	const sortedCategories = $derived.by(() => {
		const categories: Array<{
			name: string;
			shortcuts: Array<{ description: string; combo: string }>;
		}> = [];

		// Sort categories by predefined order
		for (const category of CATEGORY_ORDER) {
			const shortcuts = keyboard.shortcutsByCategory.get(category);
			if (shortcuts && shortcuts.length > 0) {
				categories.push({
					name: category,
					shortcuts: shortcuts.map((s) => ({
						description: s.description,
						combo:
							typeof s.combo === 'string'
								? formatComboString(s.combo, keyboard.platform)
								: formatSequence(s.combo, keyboard.platform),
					})),
				});
			}
		}

		return categories;
	});
</script>

{#if keyboard.isHelpOverlayOpen}
	<!-- Backdrop -->
	<div
		class="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm animate-fade-in"
		onclick={() => keyboard.closeHelpOverlay()}
		onkeydown={(e) => e.key === 'Escape' && keyboard.closeHelpOverlay()}
		role="button"
		tabindex="-1"
		aria-label="Close keyboard shortcuts"
	></div>

	<!-- Modal -->
	<FocusTrap active={true} onEscape={() => keyboard.closeHelpOverlay()}>
		<div
			class={cn(
				'fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50',
				'w-full max-w-2xl max-h-[80vh]',
				'bg-card border border-border rounded-lg shadow-xl',
				'animate-scale-in overflow-hidden',
				'flex flex-col'
			)}
			role="dialog"
			aria-labelledby="shortcuts-title"
			aria-modal="true"
		>
			<!-- Header -->
			<div class="flex items-center justify-between px-6 py-4 border-b border-border">
				<h2 id="shortcuts-title" class="text-lg font-medium text-foreground">
					Keyboard Shortcuts
				</h2>
				<button
					onclick={() => keyboard.closeHelpOverlay()}
					class={cn(
						'p-1.5 rounded-md',
						'text-muted-foreground hover:text-foreground',
						'hover:bg-muted transition-colors'
					)}
					aria-label="Close"
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="18"
						height="18"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
					>
						<path d="M18 6 6 18" />
						<path d="m6 6 12 12" />
					</svg>
				</button>
			</div>

			<!-- Content -->
			<div class="flex-1 overflow-y-auto px-6 py-4">
				<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
					{#each sortedCategories as category}
						<div>
							<h3
								class={cn(
									'text-sm font-medium text-primary mb-3 pb-2',
									'border-b border-border/50'
								)}
							>
								{category.name}
							</h3>
							<div class="space-y-2">
								{#each category.shortcuts as shortcut}
									<div class="flex items-center justify-between text-sm">
										<span class="text-foreground">{shortcut.description}</span>
										<kbd
											class={cn(
												'inline-flex items-center justify-center',
												'px-2 py-0.5 min-w-[2rem]',
												'bg-muted border border-border rounded',
												'text-xs font-mono text-muted-foreground',
												'whitespace-nowrap'
											)}
										>
											{shortcut.combo}
										</kbd>
									</div>
								{/each}
							</div>
						</div>
					{/each}
				</div>

				{#if sortedCategories.length === 0}
					<p class="text-center text-muted-foreground py-8">
						No shortcuts registered
					</p>
				{/if}
			</div>

			<!-- Footer -->
			<div
				class={cn(
					'px-6 py-3 border-t border-border',
					'bg-muted/30 text-sm text-muted-foreground',
					'flex items-center justify-between'
				)}
			>
				<span>Current scope: <span class="font-medium text-foreground">{keyboard.activeScope}</span></span>
				<span>Press <kbd class="px-1.5 py-0.5 bg-muted border border-border rounded text-xs font-mono">Esc</kbd> to close</span>
			</div>
		</div>
	</FocusTrap>
{/if}
