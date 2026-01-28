<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import { Separator } from '$lib/components/ui/separator';
	import {
		getAppInfo,
		getSidecarStatus,
		checkHealth,
		restartSidecar,
		getLLMStatus,
		listLLMModels,
		isTauri,
	} from '$lib/utils/tauri';
	import { open } from '@tauri-apps/plugin-shell';
	import type {
		AppInfo,
		HealthResponse,
		SidecarStatus,
		LLMStatus,
		LLMModelsResponse,
		HealthStatus,
	} from '$lib/types';
	import { getUIState, VIEWS } from '$lib/stores/ui.svelte';
	import { getSetupState } from '$lib/stores/setup.svelte';
	import { ViewPlaceholder } from '$lib/components/Layout';
	import { ChatPanel } from '$lib/components/Chat';
	import { PlanEditor } from '$lib/components/Plan';
	import { ExecutionView } from '$lib/components/Execution';
	import { ReviewQueue } from '$lib/components/Review';
	import { SetupWizard } from '$lib/components/Setup';

	// Get state from context
	const ui = getUIState();
	const setup = getSetupState();

	// Get current view config
	const currentViewConfig = $derived(VIEWS.find((v) => v.id === ui.currentView) ?? VIEWS[0]);

	// State with Svelte 5 runes
	let appInfo = $state<AppInfo | null>(null);
	let sidecarStatus = $state<SidecarStatus | null>(null);
	let healthResponse = $state<HealthResponse | null>(null);
	let llmStatus = $state<LLMStatus | null>(null);
	let llmModels = $state<LLMModelsResponse | null>(null);
	let error = $state<string | null>(null);
	let loading = $state(true);

	// Initialize once - isTauri() result never changes during runtime
	const isInTauri = isTauri();

	// Derived status computations
	const frontendStatus: HealthStatus = 'healthy';

	const rustStatus = $derived<HealthStatus>(
		appInfo ? 'healthy' : loading ? 'loading' : 'unhealthy'
	);

	const pythonStatus = $derived.by<HealthStatus>(() => {
		if (loading) return 'loading';
		if (!sidecarStatus?.running) return 'unhealthy';
		if (!healthResponse) return 'not_loaded';
		return healthResponse.status;
	});

	const llmHealthStatus = $derived.by<HealthStatus>(() => {
		if (loading) return 'loading';
		if (!llmStatus) return 'not_loaded';
		return llmStatus.available ? 'healthy' : 'unhealthy';
	});

	// Badge variant mapping
	function getBadgeVariant(
		status: HealthStatus
	): 'default' | 'secondary' | 'destructive' | 'outline' {
		switch (status) {
			case 'healthy':
				return 'default';
			case 'loading':
				return 'secondary';
			case 'degraded':
				return 'outline';
			case 'unhealthy':
			case 'not_loaded':
			default:
				return 'destructive';
		}
	}

	// Refresh all status
	async function refreshStatus() {
		if (!isInTauri) return;

		loading = true;
		error = null;

		try {
			const [appResult, sidecarResult, healthResult, llmResult, modelsResult] =
				await Promise.allSettled([
					getAppInfo(),
					getSidecarStatus(),
					checkHealth(),
					getLLMStatus(),
					listLLMModels(),
				]);

			if (appResult.status === 'fulfilled') {
				appInfo = appResult.value;
			}

			if (sidecarResult.status === 'fulfilled') {
				sidecarStatus = sidecarResult.value;
			}

			if (healthResult.status === 'fulfilled') {
				healthResponse = healthResult.value;
			} else {
				// Health check failed but sidecar might still be starting
				error = sidecarStatus?.running ? 'Sidecar is starting...' : 'Python sidecar not running';
			}

			if (llmResult.status === 'fulfilled') {
				llmStatus = llmResult.value;
			}

			if (modelsResult.status === 'fulfilled') {
				llmModels = modelsResult.value;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
		} finally {
			loading = false;
		}
	}

	// Restart sidecar handler
	async function handleRestartSidecar() {
		loading = true;
		error = null;

		try {
			await restartSidecar();
			// Wait for sidecar to restart, then refresh
			await new Promise((resolve) => setTimeout(resolve, 2000));
			await refreshStatus();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to restart sidecar';
			loading = false;
		}
	}

	// Initialize on mount and setup auto-refresh
	$effect(() => {
		if (isInTauri) {
			refreshStatus();

			// Auto-refresh every 30 seconds
			const interval = setInterval(refreshStatus, 30000);

			return () => clearInterval(interval);
		} else {
			loading = false;
		}
	});
</script>

{#if !setup.isComplete}
	<!-- First-Time Setup Wizard -->
	<SetupWizard onComplete={() => setup.markComplete()} />
{:else if ui.currentView === 'settings'}
	<!-- Settings View: System Status Dashboard -->
	<div class="flex flex-col items-center justify-start min-h-full p-8 gap-8 overflow-auto">
		<div class="text-center">
			<h1 class="text-2xl font-bold text-foreground mb-2">Settings</h1>
			<p class="text-muted-foreground">System status and configuration</p>
		</div>

		<!-- System Status Card -->
		<Card.Root class="w-full max-w-md">
			<Card.Header>
				<Card.Title class="flex items-center justify-between">
					System Status
					<Button variant="ghost" size="sm" onclick={refreshStatus} disabled={loading}>
						{loading ? 'Checking...' : 'Refresh'}
					</Button>
				</Card.Title>
				<Card.Description>Foundation module verification</Card.Description>
			</Card.Header>
			<Card.Content class="space-y-4">
				{#if !isInTauri}
					<p class="text-muted-foreground text-sm">
						Running outside Tauri - IPC commands unavailable.
					</p>
				{:else}
					<!-- Frontend Status -->
					<div class="flex items-center justify-between">
						<span class="text-muted-foreground">Frontend (Svelte 5)</span>
						<Badge variant={getBadgeVariant(frontendStatus)}>
							{frontendStatus}
						</Badge>
					</div>

					<!-- Rust Status -->
					<div class="flex items-center justify-between">
						<span class="text-muted-foreground">Rust Core (Tauri)</span>
						<Badge variant={getBadgeVariant(rustStatus)}>
							{rustStatus}
						</Badge>
					</div>

					<!-- Python Status -->
					<div class="flex items-center justify-between">
						<span class="text-muted-foreground">Python Sidecar</span>
						<Badge variant={getBadgeVariant(pythonStatus)}>
							{pythonStatus}
						</Badge>
					</div>

					<!-- LLM Status -->
					<div class="flex items-center justify-between">
						<span class="text-muted-foreground">AI Service</span>
						<Badge variant={getBadgeVariant(llmHealthStatus)}>
							{llmStatus?.available ? 'Connected' : llmHealthStatus}
						</Badge>
					</div>

					<Separator />

					<!-- Sidecar Details -->
					{#if sidecarStatus}
						<div class="text-sm text-muted-foreground space-y-1 font-mono">
							<p>Process: {sidecarStatus.running ? 'Running' : 'Stopped'}</p>
							<p>URL: {sidecarStatus.url}</p>
							<p>Port: {sidecarStatus.port}</p>
						</div>
					{/if}

					<!-- Health Details -->
					{#if healthResponse}
						<div class="text-sm text-muted-foreground space-y-1 font-mono">
							<p>Version: {healthResponse.version}</p>
							<p>Last check: {new Date(healthResponse.timestamp).toLocaleTimeString()}</p>
						</div>
					{/if}

					<!-- LLM Error -->
					{#if llmStatus && !llmStatus.available && llmStatus.error}
						<div class="p-3 rounded-md bg-muted/50 border border-muted-foreground/20">
							<p class="text-sm text-muted-foreground">AI Service: {llmStatus.error}</p>
						</div>
					{/if}

					<!-- Error Display -->
					{#if error}
						<div class="p-3 rounded-md bg-destructive/10 border border-destructive/20">
							<p class="text-sm text-destructive">{error}</p>
						</div>
					{/if}
				{/if}
			</Card.Content>
		</Card.Root>

		<!-- AI Models Card -->
		{#if llmModels && llmModels.models.length > 0}
			<Card.Root class="w-full max-w-md">
				<Card.Header>
					<Card.Title>AI Models</Card.Title>
					<Card.Description>Default: {llmModels.default_model}</Card.Description>
				</Card.Header>
				<Card.Content>
					<div class="space-y-2">
						{#each llmModels.models as model}
							<div class="flex justify-between text-sm">
								<span class="font-mono text-foreground">{model.name}</span>
								<span class="text-muted-foreground">{model.size}</span>
							</div>
						{/each}
					</div>
				</Card.Content>
			</Card.Root>
		{/if}

		<!-- App Info Card -->
		{#if appInfo}
			<Card.Root class="w-full max-w-md">
				<Card.Header>
					<Card.Title>Application Info</Card.Title>
				</Card.Header>
				<Card.Content class="space-y-2 text-sm">
					<div class="grid grid-cols-2 gap-2 font-mono">
						<span class="text-muted-foreground">Name</span>
						<span class="text-foreground">{appInfo.name}</span>
						<span class="text-muted-foreground">Version</span>
						<span class="text-foreground">{appInfo.version}</span>
						<span class="text-muted-foreground">Platform</span>
						<span class="text-foreground">{appInfo.platform}</span>
						<span class="text-muted-foreground">Arch</span>
						<span class="text-foreground">{appInfo.arch}</span>
						<span class="text-muted-foreground">Mode</span>
						<span class="text-foreground">{appInfo.debug ? 'Development' : 'Production'}</span>
					</div>
				</Card.Content>
			</Card.Root>
		{/if}

		<!-- Actions -->
		{#if isInTauri}
			<div class="flex gap-4">
				<Button onclick={handleRestartSidecar} disabled={loading}>Restart Sidecar</Button>
				<Button variant="secondary" onclick={() => open('http://localhost:8765/docs')}>
					API Docs
				</Button>
			</div>
		{/if}
	</div>
{:else if ui.currentView === 'chat'}
	<!-- Chat View -->
	<ChatPanel class="h-full" />
{:else if ui.currentView === 'plan'}
	<!-- Plan Editor View -->
	<PlanEditor class="h-full" />
{:else if ui.currentView === 'execute'}
	<!-- Execution View -->
	<ExecutionView class="h-full" />
{:else if ui.currentView === 'review'}
	<!-- Review Queue View -->
	<ReviewQueue onDone={() => ui.setView('execute')} class="h-full" />
{:else}
	<!-- Other Views: Show Placeholder -->
	<ViewPlaceholder view={currentViewConfig} />
{/if}
