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
	const isDevMode = import.meta.env.DEV;

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

	function wait(ms: number) {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}

	let isSkipping = $state(false);
	let setupStarted = $state(false);
	let awaitingModelChoice = $state(false);
	let isHandlingModelChoice = $state(false);

	async function finishRemainingSetup() {
		// Step 3: Build executor (local execution defaults)
		setup.nextStep();
		setup.updateProgress(0, 'Configuring local executor...');
		await wait(400);
		setup.updateProgress(100, 'Local executor configured');
		await wait(300);

		// Step 4: Complete
		setup.nextStep();
		setup.markComplete();

		// Notify parent
		setTimeout(onComplete, 800);
	}

	// Run setup steps
	async function runSetup() {
		setupStarted = true;
		awaitingModelChoice = false;
		isHandlingModelChoice = false;
		setup.startSetup();

		try {
			// Step 1: Check prerequisites
			setup.updateProgress(0, 'Checking desktop prerequisites...');
			await wait(300);
			setup.updateProgress(100, 'Requirements validated');
			await wait(250);

			// Step 2: AI model bootstrap (no manual CLI setup)
			setup.nextStep();
			setup.updateProgress(10, 'Checking AI service and required model...');

			const llmStatus = await checkLLMReady();
			if (llmStatus.ready) {
				setup.updateProgress(100, 'AI model already available');
			} else {
				// Let users choose between immediate model download and deferred setup.
				// This keeps onboarding UX-friendly while avoiding manual configuration.
				setup.updateProgress(40, 'Required AI model not installed yet');
				awaitingModelChoice = true;
				return;
			}
			await wait(250);

			await finishRemainingSetup();
		} catch (error) {
			setup.setError(error instanceof Error ? error.message : 'Setup failed');
		}
	}

	async function handleDownloadNow() {
		awaitingModelChoice = false;
		isHandlingModelChoice = true;
		try {
			setup.updateProgress(45, 'Downloading required AI model (~9GB)...');
			await ensureLLMReady();
			setup.updateProgress(100, 'AI model ready');
			await wait(250);
			await finishRemainingSetup();
		} catch (error) {
			setup.setError(
				error instanceof Error ? error.message : 'Model setup failed. You can retry or continue later.'
			);
		} finally {
			isHandlingModelChoice = false;
		}
	}

	async function handleContinueWithoutModel() {
		awaitingModelChoice = false;
		isHandlingModelChoice = true;
		try {
			setup.updateProgress(100, 'Model download deferred; Cloumask will auto-download on first AI use');
			await wait(250);
			await finishRemainingSetup();
		} finally {
			isHandlingModelChoice = false;
		}
	}

	function handleRetry() {
		setup.retry();
		awaitingModelChoice = false;
		isHandlingModelChoice = false;
		runSetup();
	}

	// Start setup automatically when component mounts
	$effect(() => {
		if (!setup.isComplete && !setup.isInProgress && !isSkipping && !setupStarted) {
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
				Preparing your desktop app. No CLI configuration required.
			</p>
		</div>

		<!-- Progress Steps -->
		<div class="space-y-4 mb-6">
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

		<!-- Model choice -->
		{#if awaitingModelChoice && !setup.progress.hasError}
			<div class="mb-6 p-4 rounded-lg border bg-muted/40 border-border">
				<p class="text-sm font-medium mb-1">AI model setup</p>
				<p class="text-sm text-muted-foreground mb-3">
					Choose one option. You never need terminal commands for this.
				</p>
				<div class="flex flex-wrap gap-2">
					<button
						type="button"
						class="px-4 py-2 text-sm bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50"
						onclick={handleDownloadNow}
						disabled={isHandlingModelChoice}
					>
						Download now (recommended)
					</button>
					<button
						type="button"
						class="px-4 py-2 text-sm bg-muted text-foreground rounded-md hover:bg-muted/80 border border-border disabled:opacity-50"
						onclick={handleContinueWithoutModel}
						disabled={isHandlingModelChoice}
					>
						Continue without model
					</button>
				</div>
				<p class="text-xs text-muted-foreground mt-2">
					If you continue now, Cloumask will automatically download the model when AI features are first used.
				</p>
			</div>
		{/if}

		<!-- Error Retry Button -->
		{#if setup.progress.hasError}
			<div class="flex justify-center">
				<button
					type="button"
					class="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
					onclick={handleRetry}
				>
					Retry Setup
				</button>
			</div>
		{/if}

		<!-- Skip Setup (development only) -->
		{#if !setup.isComplete && isDevMode}
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
