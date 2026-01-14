<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export type LogoSize = 'sm' | 'md' | 'lg';

	export interface LogoProps {
		size?: LogoSize;
		showText?: boolean;
		class?: string;
	}

	const sizeMap: Record<LogoSize, { icon: number; text: string }> = {
		sm: { icon: 40, text: 'text-base' },
		md: { icon: 48, text: 'text-lg' },
		lg: { icon: 64, text: 'text-xl' },
	};
</script>

<script lang="ts">
	let { size = 'md', showText = false, class: className }: LogoProps = $props();

	const dimensions = $derived(sizeMap[size]);
</script>

<div class={cn('flex items-center gap-2', className)}>
	<img
		src="/assets/icon.png"
		alt="Cloumask"
		width={dimensions.icon}
		height={dimensions.icon}
		class="shrink-0 object-contain rounded-md"
	/>

	{#if showText}
		<span class={cn('font-semibold tracking-tight text-foreground', dimensions.text)}>
			Cloumask
		</span>
	{/if}
</div>
