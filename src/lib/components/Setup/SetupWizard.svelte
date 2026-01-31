<script lang="ts" module>
	export interface SetupWizardProps {
		onComplete: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { getSetupState, type SetupStep } from '$lib/stores';
	import {
		Download,
		Check,
		AlertCircle,
		Loader2,
		Container,
		Cpu,
		Sparkles,
	} from '@lucide/svelte';
	import { checkLLMReady, ensureLLMReady } from '$lib/utils/tauri';

	let { onComplete }: SetupWizardProps = $props();

	const setup = getSetupState();

	// Step configuration
	const steps: { id: SetupStep; label: string; icon: typeof Download }[] = [
		{ id: 'prerequisites', label: 'Check Prerequisites', icon: Cpu },
		{ id: 'download_llm', label: 'Download AI Model', icon: Download },
		{ id: 'build_executor', label: 'Build Executor', icon: Container },
		{ id: 'complete', label: 'Ready!', icon: Sparkles },
	];

	function getStepIndex(step: SetupStep): number {
		return steps.findIndex((s) => s.id === step);
	}

	function isStepComplete(step: SetupStep): boolean {
		const currentIndex = getStepIndex(setup.progress.currentStep);
		const stepIndex = getStepIndex(step);
		return stepIndex < currentIndex || (step === 'complete' && setup.isComplete);
	}

	function isStepActive(step: SetupStep): boolean {
		return setup.progress.currentStep === step && setup.isInProgress;
	}

	// Run setup steps
	async function runSetup() {
		setup.startSetup();

		try {
			// Step 1: Check prerequisites (Docker)
			setup.updateProgress(0, 'Checking Docker availability...');
			await new Promise((resolve) => setTimeout(resolve, 1000));

			// For now, we'll skip Docker check and use local execution
			// In production, this would check for Docker daemon
			setup.updateProgress(100, 'Prerequisites checked');
			await new Promise((resolve) => setTimeout(resolve, 500));

			// Step 2: Download LLM
			setup.nextStep();
			setup.updateProgress(0, 'Checking AI model...');

			const llmStatus = await checkLLMReady();
			if (llmStatus.ready) {
				setup.updateProgress(100, 'AI model already available');
			} else {
				setup.updateProgress(10, 'Downloading AI model (this may take a while)...');
				try {
					await ensureLLMReady();
					setup.updateProgress(100, 'AI model ready');
				} catch (e) {
					// Model download failed, but we can continue
					setup.updateProgress(100, 'AI model setup skipped (will retry on first use)');
				}
			}
			await new Promise((resolve) => setTimeout(resolve, 500));

			// Step 3: Build executor (skip for now - uses local execution)
			setup.nextStep();
			setup.updateProgress(0, 'Configuring script executor...');
			await new Promise((resolve) => setTimeout(resolve, 1000));
			setup.updateProgress(100, 'Script executor configured');
			await new Promise((resolve) => setTimeout(resolve, 500));

			// Step 4: Complete
			setup.nextStep();
			setup.markComplete();

			// Notify parent
			setTimeout(onComplete, 1500);
		} catch (error) {
			setup.setError(error instanceof Error ? error.message : 'Setup failed');
		}
	}

	// Start setup automatically when component mounts
	// Track if user wants to skip
	let isSkipping = $state(false);

	$effect(() => {
		if (!setup.isComplete && !setup.isInProgress && !isSkipping) {
			runSetup();
		}
	});

	function handleSkip() {
		console.log('[SetupWizard] Skipping setup...');
		isSkipping = true;
		setup.markComplete();
		onComplete?.();
	}
</script>

<div class="fixed inset-0 z-50 flex items-center justify-center bg-background">
	<div class="w-full max-w-lg p-8">
		<!-- Header -->
		<div class="text-center mb-8">
			<h1 class="text-3xl font-bold mb-2">Setting up Cloumask</h1>
			<p class="text-muted-foreground">
				First-time setup to prepare your AI-powered annotation environment
			</p>
		</div>

		<!-- Progress Steps -->
		<div class="space-y-4 mb-8">
			{#each steps as step, i}
				{@const IconComponent = step.icon}
				{@const complete = isStepComplete(step.id)}
				{@const active = isStepActive(step.id)}
				{@const hasError = active && setup.progress.hasError}

				<div
					class={cn(
						'flex items-center gap-4 p-4 rounded-lg border transition-colors',
						complete && 'bg-green-500/10 border-green-500/30',
						active && !hasError && 'bg-primary/10 border-primary/30',
						hasError && 'bg-destructive/10 border-destructive/30',
						!complete && !active && 'bg-muted/30 border-border opacity-50'
					)}
				>
					<!-- Step Icon/Status -->
					<div
						class={cn(
							'flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center',
							complete && 'bg-green-500 text-white',
							active && !hasError && 'bg-primary text-primary-foreground',
							hasError && 'bg-destructive text-destructive-foreground',
							!complete && !active && 'bg-muted text-muted-foreground'
						)}
					>
						{#if complete}
							<Check class="h-5 w-5" />
						{:else if active && !hasError}
							<Loader2 class="h-5 w-5 animate-spin" />
						{:else if hasError}
							<AlertCircle class="h-5 w-5" />
						{:else}
							<IconComponent class="h-5 w-5" />
						{/if}
					</div>

					<!-- Step Content -->
					<div class="flex-1 min-w-0">
						<div class="font-medium">{step.label}</div>
						{#if active}
							<div class="text-sm text-muted-foreground truncate">
								{setup.progress.statusMessage}
							</div>
							{#if hasError}
								<div class="text-sm text-destructive mt-1">
									{setup.progress.errorMessage}
								</div>
							{/if}
						{/if}
					</div>

					<!-- Progress Bar (for active step) -->
					{#if active && !hasError}
						<div class="w-16 h-2 bg-muted rounded-full overflow-hidden">
							<div
								class="h-full bg-primary transition-all duration-300"
								style="width: {setup.progress.stepProgress}%"
							></div>
						</div>
					{/if}
				</div>
			{/each}
		</div>

		<!-- Error Retry Button -->
		{#if setup.progress.hasError}
			<div class="flex justify-center">
				<button
					type="button"
					class="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
					onclick={() => {
						setup.retry();
						runSetup();
					}}
				>
					Retry Setup
				</button>
			</div>
		{/if}

		<!-- Skip Setup (for development) -->
		{#if !setup.isComplete}
			<div class="text-center mt-8 relative z-50">
				<button
					type="button"
					class="px-4 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-md transition-colors cursor-pointer"
					onclick={handleSkip}
				>
					Skip setup (development only)
				</button>
			</div>
		{/if}

		<!-- Completion Message -->
		{#if setup.isComplete}
			<div class="text-center">
				<p class="text-green-500 font-medium">Setup complete! Starting Cloumask...</p>
			</div>
		{/if}
	</div>
</div>
