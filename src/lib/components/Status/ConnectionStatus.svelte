<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface ConnectionStatusProps {
		class?: string;
		/** Show compact version (dot only) */
		compact?: boolean;
	}
</script>

<script lang="ts">
	import { getSSEState, type ConnectionState } from '$lib/stores/sse.svelte';
	import { MAX_RECONNECT_ATTEMPTS } from '$lib/utils/sse';
	import * as Tooltip from '$lib/components/ui/tooltip';

	let { class: className, compact = false }: ConnectionStatusProps = $props();

	// Get SSE state from context
	let sseState: ReturnType<typeof getSSEState> | null = null;

	try {
		sseState = getSSEState();
	} catch {
		// Context not available, show disconnected state
	}

	// Status configuration for each connection state
	const statusConfig: Record<
		ConnectionState,
		{
			label: string;
			color: string;
			pulse: boolean;
		}
	> = {
		disconnected: {
			label: 'Disconnected',
			color: 'bg-muted-foreground',
			pulse: false
		},
		connecting: {
			label: 'Connecting...',
			color: 'bg-accent-warning',
			pulse: true
		},
		connected: {
			label: 'Connected',
			color: 'bg-forest-light',
			pulse: false
		},
		reconnecting: {
			label: 'Reconnecting...',
			color: 'bg-accent-warning',
			pulse: true
		},
		error: {
			label: 'Connection Error',
			color: 'bg-destructive',
			pulse: false
		}
	};

	// Get current state config, default to disconnected if not available
	const currentState = $derived(sseState?.connectionInfo.state ?? 'disconnected');
	const config = $derived(statusConfig[currentState]);
	const connectionInfo = $derived(sseState?.connectionInfo);
</script>

<Tooltip.Root>
	<Tooltip.Trigger
		data-slot="connection-status"
		class={cn(
			'flex items-center rounded-md border border-border bg-card text-foreground transition-colors duration-200 select-none',
			compact ? 'px-2.5 py-1' : 'gap-2 px-3 py-1.5 text-xs font-medium',
			'cursor-default select-none',
			className
		)}
		aria-label={`Connection status: ${config.label}`}
	>
		<!-- Status indicator dot -->
		<span class="relative flex h-2 w-2">
			{#if config.pulse}
				<span
					class={cn(
						'absolute inline-flex h-full w-full rounded-full opacity-75',
						config.color,
						'animate-ping'
					)}
				></span>
			{/if}
			<span
				class={cn('relative inline-flex h-2 w-2 rounded-full', config.color)}
			></span>
		</span>

		{#if !compact}
			<span class="text-muted-foreground">{config.label}</span>
		{/if}
	</Tooltip.Trigger>

	<Tooltip.Content side="bottom" class="max-w-[220px] font-mono text-xs">
		<div class="space-y-1.5">
			<p class="font-medium">{config.label}</p>

			{#if connectionInfo?.threadId}
				<div class="flex items-center gap-2 opacity-80">
					<span class="text-[10px] uppercase tracking-wider opacity-70">Thread</span>
					<code class="text-[11px] bg-background/20 px-1 rounded">
						{connectionInfo.threadId.slice(0, 8)}
					</code>
				</div>
			{/if}

			{#if connectionInfo?.error}
				<div class="text-red-300 text-[11px] leading-tight">
					{connectionInfo.error}
				</div>
			{/if}

			{#if connectionInfo && connectionInfo.reconnectAttempts > 0}
				<div class="flex items-center gap-2 opacity-80">
					<span class="text-[10px] uppercase tracking-wider opacity-70">Retry</span>
					<span class="text-[11px]">{connectionInfo.reconnectAttempts}/{MAX_RECONNECT_ATTEMPTS}</span>
				</div>
			{/if}

			{#if connectionInfo?.lastHeartbeat}
				<div class="flex items-center gap-2 opacity-80">
					<span class="text-[10px] uppercase tracking-wider opacity-70">Last ping</span>
					<span class="text-[11px]">
						{new Date(connectionInfo.lastHeartbeat).toLocaleTimeString()}
					</span>
				</div>
			{/if}
		</div>
	</Tooltip.Content>
</Tooltip.Root>
