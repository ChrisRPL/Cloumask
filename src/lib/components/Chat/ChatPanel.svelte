<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface ChatPanelProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { onMount } from 'svelte';
	import { getAgentState } from '$lib/stores/agent.svelte';
	import { getSSEState } from '$lib/stores/sse.svelte';
	import { getPipelineState } from '$lib/stores/pipeline.svelte';
	import { getUIState } from '$lib/stores/ui.svelte';
	import { createThread, sendMessage } from '$lib/utils/tauri';
	import type { UserDecision } from '$lib/types/agent';

	import ChatHeader from './ChatHeader.svelte';
	import MessageList from './MessageList.svelte';
	import StreamingIndicator from './StreamingIndicator.svelte';
	import ClarificationForm from './ClarificationForm.svelte';
	import PlanPreviewCard from './PlanPreviewCard.svelte';
	import ChatInput from './ChatInput.svelte';

	let { class: className }: ChatPanelProps = $props();

	// Get stores
	const agent = getAgentState();
	const sse = getSSEState();
	const pipeline = getPipelineState();
	const ui = getUIState();

	// Local state
	let inputValue = $state('');
	let isInitializing = $state(false);
	let initError = $state<string | null>(null);

	// Derived state
	const isInputDisabled = $derived(agent.isBusy || !sse.isConnected || isInitializing);
	const showClarification = $derived(agent.pendingClarification !== null);
	const showPlanPreview = $derived(
		pipeline.steps.length > 0 &&
		['planning', 'awaiting_approval'].includes(agent.phase)
	);

	// Initialize thread and SSE connection
	async function initializeChat() {
		// Guard against concurrent initialization
		if (isInitializing) return;
		// Already initialized and connected
		if (agent.threadId && sse.isConnected) return;

		isInitializing = true;
		initError = null;

		try {
			let threadId = agent.threadId;

			// Create new thread if needed
			if (!threadId) {
				const thread = await createThread();
				threadId = thread.thread_id;
				agent.setThreadId(threadId);
			}

			// Connect SSE using the local threadId to avoid race condition
			sse.connect(threadId);
		} catch (error) {
			console.error('[ChatPanel] Failed to initialize:', error);
			initError = error instanceof Error ? error.message : 'Failed to connect';
		} finally {
			isInitializing = false;
		}
	}

	// Send user message
	async function handleSend(content: string) {
		if (!content.trim() || !agent.threadId) return;

		// Add user message to store
		agent.addMessage({
			role: 'user',
			content: content.trim()
		});

		// Clear input
		inputValue = '';

		// Send to backend
		try {
			await sendMessage(agent.threadId, { content: content.trim() });
		} catch (error) {
			console.error('[ChatPanel] Failed to send message:', error);
			agent.setError(error instanceof Error ? error.message : 'Failed to send message');
		}
	}

	// Handle clarification response
	async function handleClarificationSubmit(response: { decision: UserDecision; selected?: string[] }) {
		if (!agent.threadId || !agent.pendingClarification) return;

		// Clear clarification
		agent.setClarification(null);

		// Send decision to backend
		try {
			const content = response.selected?.join(', ') || response.decision;
			await sendMessage(agent.threadId, {
				content,
				decision: response.decision
			});
		} catch (error) {
			console.error('[ChatPanel] Failed to submit clarification:', error);
			agent.setError(error instanceof Error ? error.message : 'Failed to submit response');
		}
	}

	// Handle clarification cancel
	function handleClarificationCancel() {
		agent.setClarification(null);
		// Optionally send cancel to backend
		if (agent.threadId) {
			sendMessage(agent.threadId, {
				content: 'Cancelled',
				decision: 'cancel'
			}).catch(console.error);
		}
	}

	// Clear conversation
	function handleClear() {
		agent.startNewConversation();
		pipeline.clearPipeline();
		// Reconnect with new thread
		initializeChat();
	}

	// Navigate to plan editor
	function handleViewPlan() {
		ui.setView('plan');
	}

	// Export conversation (placeholder)
	function handleExport() {
		// Convert messages to markdown
		const markdown = agent.messages
			.map((msg) => {
				const prefix = msg.role === 'user' ? '**You:**' : '**Agent:**';
				return `${prefix}\n${msg.content}\n`;
			})
			.join('\n---\n\n');

		// Copy to clipboard
		navigator.clipboard.writeText(markdown).catch(console.error);
	}

	// Initialize on mount
	onMount(() => {
		initializeChat();

		// Cleanup on unmount - don't disconnect, just let SSE continue
		return () => {
			// SSE connection persists across view switches
		};
	});
</script>

<div class={cn('flex flex-col h-full bg-background', className)}>
	<!-- Header -->
	<ChatHeader
		phase={agent.phase}
		isConnected={sse.isConnected}
		onClear={handleClear}
		onExport={handleExport}
	/>

	<!-- Messages -->
	<MessageList
		messages={agent.messages}
		isStreaming={agent.isStreaming}
		class="flex-1"
	/>

	<!-- Streaming indicator -->
	{#if agent.isStreaming}
		<StreamingIndicator phase={agent.phase} />
	{/if}

	<!-- Plan preview (when planning/awaiting approval) -->
	{#if showPlanPreview}
		<div class="px-4 pb-2">
			<PlanPreviewCard
				steps={pipeline.sortedSteps}
				onViewPlan={handleViewPlan}
			/>
		</div>
	{/if}

	<!-- Clarification form (when pending) -->
	{#if showClarification && agent.pendingClarification}
		<div class="px-4 pb-2">
			<ClarificationForm
				clarification={agent.pendingClarification}
				disabled={!sse.isConnected}
				onSubmit={handleClarificationSubmit}
				onCancel={handleClarificationCancel}
			/>
		</div>
	{/if}

	<!-- Error display -->
	{#if agent.lastError}
		<div class="px-4 py-2 mx-4 mb-2 rounded bg-destructive/10 border border-destructive/20 text-destructive text-sm">
			{agent.lastError}
		</div>
	{/if}

	<!-- Connection error -->
	{#if initError}
		<div class="px-4 py-2 mx-4 mb-2 rounded bg-amber-500/10 border border-amber-500/20 text-amber-600 text-sm flex items-center justify-between">
			<span>{initError}</span>
			<button
				class="text-xs underline"
				onclick={initializeChat}
			>
				Retry
			</button>
		</div>
	{/if}

	<!-- Input -->
	<ChatInput
		value={inputValue}
		disabled={isInputDisabled}
		placeholder={isInitializing ? 'Connecting...' : 'Type a message...'}
		onSend={handleSend}
		onValueChange={(v) => (inputValue = v)}
	/>
</div>
