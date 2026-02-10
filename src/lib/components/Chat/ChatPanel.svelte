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
	import { createThread, sendMessage, checkLLMReady, ensureLLMReady, isTauri } from '$lib/utils/tauri';
	import type { UserDecision } from '$lib/types/agent';
	import type { LLMReadyResponse } from '$lib/types/commands';

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
	const isInTauri = isTauri();

	// Local state
	let inputValue = $state('');
	let isInitializing = $state(false);
	let initError = $state<string | null>(null);
	let llmStatus = $state<LLMReadyResponse | null>(null);
	let isCheckingLLM = $state(false);
	let isPullingModel = $state(false);
	let autoPullTriggered = $state(false);

	// Derived state
	const llmNotReady = $derived(llmStatus !== null && !llmStatus.ready);
	const isInputDisabled = $derived(agent.isBusy || !sse.isConnected || isInitializing || llmNotReady);
	const showClarification = $derived(agent.pendingClarification !== null);
	const showPlanPreview = $derived(
		pipeline.steps.length > 0 &&
		['planning', 'awaiting_approval'].includes(agent.phase)
	);

	// Check LLM service readiness
	async function checkLLM() {
		isCheckingLLM = true;
		try {
			llmStatus = await checkLLMReady();
		} catch (error) {
			console.error('[ChatPanel] Failed to check LLM service:', error);
			llmStatus = {
				ready: false,
				service_running: false,
				required_model: 'qwen3:14b',
				model_available: false,
				error: 'Failed to connect to backend'
			};
		} finally {
			isCheckingLLM = false;
		}
	}

	// Pull the required model
	async function handlePullModel() {
		if (!llmStatus || isPullingModel) return;

		isPullingModel = true;
		try {
			const ensured = await ensureLLMReady();
			llmStatus = ensured;
			// Allow re-attempt if backend reports still not ready.
			if (!ensured.ready) {
				autoPullTriggered = false;
			}
		} catch (error) {
			console.error('[ChatPanel] Failed to download model:', error);
			autoPullTriggered = false;
		} finally {
			isPullingModel = false;
		}
	}

	// Auto-download model when service is up but model is missing.
	$effect(() => {
		if (
			isInTauri &&
			llmStatus &&
			llmStatus.service_running &&
			!llmStatus.model_available &&
			!isPullingModel &&
			!autoPullTriggered
		) {
			autoPullTriggered = true;
			void handlePullModel();
		}
	});

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
		// Check LLM service readiness first
		checkLLM();
		// Initialize chat connection
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

	<!-- AI service not ready warning -->
	{#if llmNotReady && llmStatus}
		<div class="px-4 py-3 mx-4 mb-2 rounded bg-amber-500/10 border border-amber-500/20 text-amber-700 text-sm">
			<div class="flex items-start gap-3">
				<div class="flex-1">
					{#if !llmStatus.service_running}
						<p class="font-medium">AI service is starting...</p>
						<p class="text-xs mt-1 text-amber-600">
							The language model is initializing. This may take a moment on first launch.
						</p>
					{:else if !llmStatus.model_available}
						<p class="font-medium">Downloading AI model...</p>
						<p class="text-xs mt-1 text-amber-600">
							First-time setup requires downloading the AI model (~9GB).
						</p>
					{:else}
						<p class="font-medium">AI service not ready</p>
						<p class="text-xs mt-1 text-amber-600">{llmStatus.error}</p>
					{/if}
				</div>
				<div class="flex gap-2">
					{#if llmStatus.service_running && !llmStatus.model_available}
						<button
							class="px-3 py-1 text-xs rounded bg-forest text-white hover:bg-forest-dark disabled:opacity-50"
							onclick={handlePullModel}
							disabled={isPullingModel}
						>
							{isPullingModel ? 'Downloading...' : 'Download Model'}
						</button>
					{/if}
					<button
						class="px-2 py-1 text-xs underline"
						onclick={checkLLM}
						disabled={isCheckingLLM}
					>
						{isCheckingLLM ? 'Checking...' : 'Refresh'}
					</button>
				</div>
			</div>
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
