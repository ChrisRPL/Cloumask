<script lang="ts" module>
	import { cn } from '$lib/utils.js';
	import type { Message } from '$lib/types/agent';

	export interface MessageBubbleProps {
		message: Message;
		isLatest?: boolean;
		class?: string;
	}

	// Role prefixes for terminal aesthetic
	const rolePrefix: Record<string, string> = {
		user: '>',
		assistant: '<',
		system: '#',
		tool: '$'
	};

	// Format timestamp to HH:MM
	function formatTime(timestamp: string): string {
		try {
			const date = new Date(timestamp);
			return date.toLocaleTimeString('en-US', {
				hour: '2-digit',
				minute: '2-digit',
				hour12: false
			});
		} catch {
			return '';
		}
	}
</script>

<script lang="ts">
	import MessageContent from './MessageContent.svelte';

	let { message, isLatest = false, class: className }: MessageBubbleProps = $props();

	const isUser = $derived(message.role === 'user');
	const isSystem = $derived(message.role === 'system');
	const prefix = $derived(rolePrefix[message.role] || '');
	const time = $derived(formatTime(message.timestamp));
</script>

<div
	class={cn(
		'flex flex-col gap-1',
		isUser ? 'items-end' : 'items-start',
		isSystem && 'items-center',
		className
	)}
>
	<!-- Prefix and timestamp -->
	<div
		class={cn(
			'flex items-center gap-2 text-xs text-muted-foreground',
			isUser && 'flex-row-reverse'
		)}
	>
		<span class="font-medium text-forest-light">{prefix}</span>
		<span class="tabular-nums opacity-60">{time}</span>
	</div>

	<!-- Message content -->
	<div
		class={cn(
			'max-w-[85%] px-3 py-2 rounded-md',
			isUser && 'bg-secondary/40 border border-border',
			!isUser && !isSystem && 'bg-transparent',
			isSystem && 'bg-muted/30 text-center max-w-full px-4 py-1 text-xs'
		)}
	>
		<MessageContent
			content={message.content}
			role={message.role}
			isStreaming={isLatest && message.isStreaming}
		/>
	</div>
</div>
