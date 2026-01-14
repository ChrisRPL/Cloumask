<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface ProjectSelectorProps {
		class?: string;
	}
</script>

<script lang="ts">
	import { FolderOpen, Plus } from 'lucide-svelte';
	import * as Select from '$lib/components/ui/select';
	import { getUIState } from '$lib/stores/ui';

	let { class: className }: ProjectSelectorProps = $props();

	const ui = getUIState();

	// For now, use a mock project list since backend isn't connected
	const mockProjects = [
		{ id: '1', name: 'My Project', path: '/data/my-project' },
		{ id: '2', name: 'Dataset v2', path: '/data/dataset-v2' },
	];

	let selectedValue = $state<string | undefined>(undefined);

	function handleValueChange(value: string | undefined) {
		if (value === 'new') {
			// TODO: Open new project dialog
			console.log('Create new project');
			return;
		}

		selectedValue = value;
		const project = mockProjects.find((p) => p.id === value);
		if (project) {
			ui.setProject({ ...project, lastOpened: new Date() });
		}
	}
</script>

<Select.Root type="single" bind:value={selectedValue} onValueChange={handleValueChange}>
	<Select.Trigger class={cn('w-[180px] bg-background/50', className)} size="sm">
		{#if ui.currentProject}
			<span class="flex items-center gap-2 truncate">
				<FolderOpen class="size-4 shrink-0 text-muted-foreground" />
				<span class="truncate">{ui.currentProject.name}</span>
			</span>
		{:else}
			<span class="text-muted-foreground">Select project...</span>
		{/if}
	</Select.Trigger>
	<Select.Portal>
		<Select.Content>
			<Select.Group>
				<Select.GroupHeading>Recent Projects</Select.GroupHeading>
				{#each mockProjects as project (project.id)}
					<Select.Item value={project.id}>
						<FolderOpen class="size-4 text-muted-foreground" />
						{project.name}
					</Select.Item>
				{/each}
			</Select.Group>
			<Select.Separator />
			<Select.Item value="new">
				<Plus class="size-4 text-muted-foreground" />
				New Project...
			</Select.Item>
		</Select.Content>
	</Select.Portal>
</Select.Root>
