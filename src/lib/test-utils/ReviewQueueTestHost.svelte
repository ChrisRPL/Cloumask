<script lang="ts" module>
	export interface ReviewQueueTestHostProps {
		executionId?: string;
		projectId?: string | null;
	}
</script>

<script lang="ts">
	import ReviewQueue from '$lib/components/Review/ReviewQueue.svelte';
	import { setReviewState } from '$lib/stores/review.svelte';
	import { setKeyboardState } from '$lib/stores/keyboard.svelte';
	import { setExecutionState } from '$lib/stores/execution.svelte';

	let { executionId = 'current', projectId = null }: ReviewQueueTestHostProps = $props();

	setReviewState();
	setKeyboardState();
	setExecutionState();
</script>

<ReviewQueue {executionId} {projectId} class="h-full" />
