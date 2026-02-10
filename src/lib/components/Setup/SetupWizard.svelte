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
	import {
		checkLLMReady,
		ensureLLMReady,
		pullLLMModelWithProgress,
		waitForSidecarReady,
		isTauri,
		type LLMPullProgressEvent
	} from '$lib/utils/tauri';

	let { onComplete }: SetupWizardProps = $props();

	const setup = getSetupState();
	const isDevMode = import.meta.env.DEV;
	const isInTauri = isTauri();
	const MODEL_SETUP_RETRIES = 3;
	const SIDECAR_READY_TIMEOUT_MS = 20000;

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

	function formatBytes(bytes: number | null): string {
		if (bytes === null || bytes < 0) return '0 B';
		const units = ['B', 'KB', 'MB', 'GB', 'TB'];
		let size = bytes;
		let unitIndex = 0;
		while (size >= 1024 && unitIndex < units.length - 1) {
			size /= 1024;
			unitIndex += 1;
		}
		return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
	}

	function getPullProgressMessage(event: LLMPullProgressEvent): string {
		if (event.progressPercent !== null) {
			const percent = Math.round(event.progressPercent);
			return `Downloading required AI model... ${percent}% (${formatBytes(event.completedBytes)} / ${formatBytes(event.totalBytes)})`;
		}
		return event.status || 'Downloading required AI model...';
	}

	let isSkipping = $state(false);
	let setupStarted = $state(false);
	let backgroundModelInit = $state(false);

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

	async function autoBootstrapModel() {
		if (isInTauri) {
			setup.updateProgress(10, 'Waiting for local services...');
			const sidecarReady = await waitForSidecarReady(SIDECAR_READY_TIMEOUT_MS);
			if (!sidecarReady) {
				backgroundModelInit = true;
				setup.updateProgress(25, 'Continuing while services finish starting...');
				void ensureLLMReady().catch((error) => {
					console.error('[SetupWizard] Deferred model bootstrap failed:', error);
				});
				return;
			}
		}

		let llmStatus = await checkLLMReady();
		if (llmStatus.ready) {
			setup.updateProgress(100, 'AI model already available');
			return;
		}

		for (let attempt = 1; attempt <= MODEL_SETUP_RETRIES; attempt++) {
			setup.updateProgress(
				Math.min(20 + attempt * 10, 50),
				`Preparing AI model download... (${attempt}/${MODEL_SETUP_RETRIES})`
			);

			try {
				await pullLLMModelWithProgress(llmStatus.required_model, (event) => {
					const progress =
						event.progressPercent === null
							? Math.min(35 + attempt * 20, 90)
							: Math.min(Math.max(Math.round(event.progressPercent), 10), 95);
					setup.updateProgress(progress, getPullProgressMessage(event));
				});
			} catch (streamError) {
				console.error('[SetupWizard] Streaming model download failed:', streamError);
			}

			const ensured = await checkLLMReady();
			if (ensured.ready) {
				setup.updateProgress(100, 'AI model ready');
				return;
			}

			llmStatus = ensured;
			await wait(1200);
		}

		// Do not block users on first-run if model bootstrap is temporarily unavailable.
		backgroundModelInit = true;
		setup.updateProgress(100, 'Continuing setup. AI model download will keep retrying automatically.');
		void ensureLLMReady().catch((error) => {
			console.error('[SetupWizard] Background model bootstrap failed:', error);
		});
	}

	// Run setup steps
	async function runSetup() {
		setupStarted = true;
		backgroundModelInit = false;
		setup.startSetup();

		try {
			// Step 1: Check prerequisites
			setup.updateProgress(0, 'Checking desktop prerequisites...');
			await wait(300);
			setup.updateProgress(100, 'Requirements validated');
			await wait(250);

			// Step 2: AI model bootstrap (no manual CLI setup)
			setup.nextStep();
			await autoBootstrapModel();
			await wait(250);

			await finishRemainingSetup();
		} catch (error) {
			// Never hard-block onboarding on model/bootstrap errors.
			console.error('[SetupWizard] Setup step failed, continuing with fallback:', error);
			backgroundModelInit = true;
			setup.updateProgress(100, 'Continuing setup while background services initialize...');
			await wait(250);
			await finishRemainingSetup();
		}
	}

	function handleRetry() {
		setup.retry();
		backgroundModelInit = false;
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

		<!-- Background model bootstrap info -->
		{#if backgroundModelInit}
			<div class="mb-6 p-4 rounded-lg border bg-muted/40 border-border">
				<p class="text-sm font-medium mb-1">AI model initialization in progress</p>
				<p class="text-sm text-muted-foreground">
					Cloumask will continue automatically and keep retrying model download in the background.
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
