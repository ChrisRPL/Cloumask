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
		isTauri,
	} from '$lib/utils/tauri';
	import type { AppInfo, HealthResponse, SidecarStatus, HealthStatus } from '$lib/types';

	// State with Svelte 5 runes
	let appInfo = $state<AppInfo | null>(null);
	let sidecarStatus = $state<SidecarStatus | null>(null);
	let healthResponse = $state<HealthResponse | null>(null);
	let error = $state<string | null>(null);
	let loading = $state(true);
	let isInTauri = $state(false);

	// Derived status computations
	const frontendStatus: HealthStatus = 'healthy';

	const rustStatus = $derived<HealthStatus>(appInfo ? 'healthy' : loading ? 'loading' : 'unhealthy');

	const pythonStatus = $derived.by<HealthStatus>(() => {
		if (loading) return 'loading';
		if (!sidecarStatus?.running) return 'unhealthy';
		if (!healthResponse) return 'not_loaded';
		return healthResponse.status;
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
			const [appResult, sidecarResult, healthResult] = await Promise.allSettled([
				getAppInfo(),
				getSidecarStatus(),
				checkHealth(),
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
		isInTauri = isTauri();

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

<main class="flex flex-col items-center justify-center min-h-screen p-8 gap-8">
	<!-- Header -->
	<div class="text-center">
		<img src="/assets/icon_large.png" alt="Cloumask" class="h-16 mx-auto mb-4 object-contain" />
		<h1 class="text-4xl font-bold text-foreground mb-2">Cloumask</h1>
		<p class="text-muted-foreground">Local-first AI for computer vision data processing</p>
		<div class="flex gap-2 justify-center mt-4">
			<Badge variant="secondary">Tauri 2.0</Badge>
			<Badge variant="secondary">Svelte 5</Badge>
			<Badge variant="secondary">Python</Badge>
		</div>
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

				<!-- Error Display -->
				{#if error}
					<div class="p-3 rounded-md bg-destructive/10 border border-destructive/20">
						<p class="text-sm text-destructive">{error}</p>
					</div>
				{/if}
			{/if}
		</Card.Content>
	</Card.Root>

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
			<Button variant="secondary" onclick={() => window.open('http://localhost:8765/docs', '_blank')}
				>API Docs</Button
			>
		</div>
	{/if}
</main>
