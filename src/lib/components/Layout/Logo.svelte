<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export type LogoSize = 'sm' | 'md' | 'lg';

	export interface LogoProps {
		size?: LogoSize;
		showText?: boolean;
		class?: string;
	}

	const sizeMap: Record<LogoSize, { icon: number; text: string }> = {
		sm: { icon: 24, text: 'text-base' },
		md: { icon: 32, text: 'text-lg' },
		lg: { icon: 40, text: 'text-xl' },
	};
</script>

<script lang="ts">
	let { size = 'md', showText = false, class: className }: LogoProps = $props();

	const dimensions = $derived(sizeMap[size]);
</script>

<div class={cn('flex items-center gap-2', className)}>
	<!-- SVG Logo Mark: Abstract cloud/mask shape -->
	<svg
		width={dimensions.icon}
		height={dimensions.icon}
		viewBox="0 0 32 32"
		fill="none"
		xmlns="http://www.w3.org/2000/svg"
		class="shrink-0"
		aria-hidden="true"
	>
		<!-- Background circle -->
		<circle cx="16" cy="16" r="14" class="fill-primary" />
		<!-- Cloud/data points pattern -->
		<g class="fill-primary-foreground">
			<!-- Main mask/cloud shape -->
			<path
				d="M8 18c0-2.2 1.8-4 4-4h.5c.3-2.3 2.2-4 4.5-4 2.5 0 4.5 2 4.5 4.5 0 .2 0 .3-.02.5H22c1.7 0 3 1.3 3 3s-1.3 3-3 3H12c-2.2 0-4-1.8-4-4z"
				fill-opacity="0.9"
			/>
			<!-- Data points -->
			<circle cx="10" cy="12" r="1.5" fill-opacity="0.6" />
			<circle cx="14" cy="9" r="1" fill-opacity="0.4" />
			<circle cx="22" cy="11" r="1.5" fill-opacity="0.6" />
		</g>
	</svg>

	{#if showText}
		<span class={cn('font-semibold tracking-tight text-foreground', dimensions.text)}>
			Cloumask
		</span>
	{/if}
</div>
