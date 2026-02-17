<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { setExecutionState } from '$lib/stores/execution.svelte';
	import type { PreviewItem } from '$lib/types/execution';
	import PointCloudViewer from './PointCloudViewer.svelte';

	let { selectedPointcloudPreview = null }: { selectedPointcloudPreview?: PreviewItem | null } =
		$props();

	const execution = setExecutionState();

	$effect(() => {
		execution.setSelectedPointcloudPreview(selectedPointcloudPreview);
	});
</script>

<Tooltip.Provider>
	<PointCloudViewer />
</Tooltip.Provider>
