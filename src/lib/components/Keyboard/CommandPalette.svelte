<script lang="ts">
	import { cn } from '$lib/utils';
	import { getKeyboardState } from '$lib/stores/keyboard.svelte';
	import { fuzzyScore } from '$lib/utils/keyboard';
	import FocusTrap from './FocusTrap.svelte';

	const keyboard = getKeyboardState();

	let searchQuery = $state('');
	let selectedIndex = $state(0);
	let inputRef = $state<HTMLInputElement | null>(null);

	// Get all commands and filter by search query
	const filteredCommands = $derived.by(() => {
		const commands = keyboard.getAllCommands();

		if (!searchQuery.trim()) {
			return commands;
		}

		const query = searchQuery.toLowerCase();

		// Score and filter commands
		const scored = commands
			.map((cmd) => {
				const labelScore = fuzzyScore(query, cmd.label);
				const descScore = cmd.description ? fuzzyScore(query, cmd.description) * 0.8 : 0;
				const keywordScore = cmd.keywords
					? Math.max(...cmd.keywords.map((k) => fuzzyScore(query, k))) * 0.6
					: 0;

				return {
					command: cmd,
					score: Math.max(labelScore, descScore, keywordScore),
				};
			})
			.filter((item) => item.score > 0)
			.sort((a, b) => b.score - a.score);

		return scored.map((item) => item.command);
	});

	// Group commands by category
	const groupedCommands = $derived.by(() => {
		const groups = new Map<string, typeof filteredCommands>();

		for (const cmd of filteredCommands) {
			const category = cmd.category ?? 'General';
			const list = groups.get(category) ?? [];
			list.push(cmd);
			groups.set(category, list);
		}

		return groups;
	});

	// Flatten for index-based navigation
	const flatCommands = $derived(filteredCommands);

	// Reset selection when commands change
	$effect(() => {
		// eslint-disable-next-line @typescript-eslint/no-unused-expressions
		filteredCommands; // dependency
		selectedIndex = 0;
	});

	// Focus input when opened
	$effect(() => {
		if (keyboard.isCommandPaletteOpen && inputRef) {
			searchQuery = '';
			selectedIndex = 0;
			// Small delay to ensure DOM is ready
			setTimeout(() => inputRef?.focus(), 10);
		}
	});

	function handleKeydown(event: KeyboardEvent) {
		switch (event.key) {
			case 'ArrowDown':
				event.preventDefault();
				selectedIndex = Math.min(selectedIndex + 1, flatCommands.length - 1);
				scrollToSelected();
				break;

			case 'ArrowUp':
				event.preventDefault();
				selectedIndex = Math.max(selectedIndex - 1, 0);
				scrollToSelected();
				break;

			case 'Enter':
				event.preventDefault();
				if (flatCommands[selectedIndex]) {
					executeCommand(flatCommands[selectedIndex]);
				}
				break;

			case 'Escape':
				event.preventDefault();
				keyboard.closeCommandPalette();
				break;
		}
	}

	function executeCommand(command: (typeof flatCommands)[0]) {
		keyboard.closeCommandPalette();
		command.action();
	}

	function scrollToSelected() {
		const element = document.querySelector(`[data-command-index="${selectedIndex}"]`);
		element?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
	}
</script>

{#if keyboard.isCommandPaletteOpen}
	<!-- Backdrop -->
	<div
		class="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm animate-fade-in"
		onclick={() => keyboard.closeCommandPalette()}
		onkeydown={(e) => e.key === 'Escape' && keyboard.closeCommandPalette()}
		role="button"
		tabindex="-1"
		aria-label="Close command palette"
	></div>

	<!-- Modal -->
	<FocusTrap active={true} onEscape={() => keyboard.closeCommandPalette()}>
		<div
			class={cn(
				'fixed left-1/2 top-[20%] -translate-x-1/2 z-50',
				'w-full max-w-lg',
				'bg-card/95 backdrop-blur-md border border-border rounded-lg shadow-2xl',
				'animate-scale-in overflow-hidden',
				'flex flex-col'
			)}
			role="dialog"
			aria-labelledby="palette-title"
			aria-modal="true"
			tabindex="-1"
			onkeydown={handleKeydown}
		>
			<!-- Search Input -->
			<div class="flex items-center px-4 border-b border-border">
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
					class="text-muted-foreground shrink-0"
				>
					<circle cx="11" cy="11" r="8" />
					<path d="m21 21-4.3-4.3" />
				</svg>
				<input
					bind:this={inputRef}
					bind:value={searchQuery}
					type="text"
					placeholder="Type a command or search..."
					class={cn(
						'flex-1 px-3 py-3',
						'bg-transparent border-0 outline-none',
						'text-foreground placeholder:text-muted-foreground',
						'text-sm font-mono'
					)}
					id="palette-title"
					aria-label="Search commands"
				/>
				<kbd
					class={cn(
						'px-1.5 py-0.5 bg-muted border border-border rounded',
						'text-xs font-mono text-muted-foreground'
					)}
				>
					Esc
				</kbd>
			</div>

			<!-- Command List -->
			<div class="max-h-80 overflow-y-auto">
				{#if flatCommands.length === 0}
					<div class="px-4 py-8 text-center text-muted-foreground text-sm">
						{searchQuery ? 'No commands found' : 'No commands available'}
					</div>
				{:else}
					{#each groupedCommands as [category, commands]}
						<div class="py-1">
							<div
								class={cn(
									'px-4 py-1.5',
									'text-xs font-medium text-muted-foreground uppercase tracking-wide'
								)}
							>
								{category}
							</div>
							{#each commands as command, i}
								{@const globalIndex = flatCommands.indexOf(command)}
								<button
									data-command-index={globalIndex}
									onclick={() => executeCommand(command)}
									onmouseenter={() => (selectedIndex = globalIndex)}
									class={cn(
										'w-full px-4 py-2 flex items-center justify-between',
										'text-left text-sm transition-colors',
										globalIndex === selectedIndex
											? 'bg-primary text-primary-foreground'
											: 'text-foreground hover:bg-muted'
									)}
								>
									<span class="truncate">{command.label}</span>
									{#if command.shortcut}
										<kbd
											class={cn(
												'ml-2 px-1.5 py-0.5 rounded text-xs font-mono shrink-0',
												globalIndex === selectedIndex
													? 'bg-primary-foreground/20 text-primary-foreground'
													: 'bg-muted text-muted-foreground border border-border'
											)}
										>
											{command.shortcut}
										</kbd>
									{/if}
								</button>
							{/each}
						</div>
					{/each}
				{/if}
			</div>

			<!-- Footer -->
			<div
				class={cn(
					'px-4 py-2 border-t border-border',
					'bg-muted/30 text-xs text-muted-foreground',
					'flex items-center justify-between'
				)}
			>
				<div class="flex items-center gap-3">
					<span class="flex items-center gap-1">
						<kbd class="px-1 py-0.5 bg-muted border border-border rounded font-mono">↑↓</kbd>
						Navigate
					</span>
					<span class="flex items-center gap-1">
						<kbd class="px-1 py-0.5 bg-muted border border-border rounded font-mono">↵</kbd>
						Select
					</span>
				</div>
				<span>{flatCommands.length} command{flatCommands.length === 1 ? '' : 's'}</span>
			</div>
		</div>
	</FocusTrap>
{/if}
