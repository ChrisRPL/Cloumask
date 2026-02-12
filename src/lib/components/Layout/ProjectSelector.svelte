<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface ProjectSelectorProps {
		class?: string;
	}
</script>

<script lang="ts">
	import Icon from '$lib/components/ui/icons.svelte';
	import * as Select from '$lib/components/ui/select';
	import { getUIState } from '$lib/stores/ui.svelte';
	import type { Project } from '$lib/types/ui';

	let { class: className }: ProjectSelectorProps = $props();

	const ui = getUIState();

	// Local defaults for first-run.
	const defaultProjects: Project[] = [
		{ id: '1', name: 'My Project', path: '/data/my-project' },
		{ id: '2', name: 'Dataset v2', path: '/data/dataset-v2' },
	];

	let selectedValue = $state<string | undefined>(undefined);

	const availableProjects = $derived.by(() => {
		const byId = new Map<string, Project>();
		for (const project of ui.recentProjects) {
			byId.set(project.id, project);
		}
		for (const project of defaultProjects) {
			if (!byId.has(project.id)) byId.set(project.id, project);
		}
		return [...byId.values()];
	});

	$effect(() => {
		selectedValue = ui.currentProject?.id;
	});

	function slugifyProjectName(name: string): string {
		return name
			.toLowerCase()
			.trim()
			.replace(/[^a-z0-9]+/g, '-')
			.replace(/^-+|-+$/g, '');
	}

	function promptForNewProject(): Project | null {
		const name = window.prompt('Project name');
		if (!name || !name.trim()) return null;

		const safeName = slugifyProjectName(name) || 'new-project';
		const path =
			window.prompt('Project path', `/data/${safeName}`)?.trim() ?? '';
		if (!path) return null;

		return {
			id: crypto.randomUUID(),
			name: name.trim(),
			path,
			lastOpened: new Date()
		};
	}

	function handleValueChange(value: string | undefined) {
		if (value === 'new') {
			const project = promptForNewProject();
			if (project) {
				ui.setProject(project);
				selectedValue = project.id;
			} else {
				selectedValue = ui.currentProject?.id;
			}
			return;
		}

		selectedValue = value;
		const project = availableProjects.find((p) => p.id === value);
		if (project) {
			ui.setProject({ ...project, lastOpened: new Date() });
		}
	}
</script>

<Select.Root type="single" bind:value={selectedValue} onValueChange={handleValueChange}>
	<Select.Trigger class={cn('w-[180px] bg-background/50', className)} size="sm">
		{#if ui.currentProject}
			<span class="flex items-center gap-2 truncate">
				<Icon name="folder-open" class="size-4 shrink-0 text-muted-foreground" />
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
				{#each availableProjects as project (project.id)}
					<Select.Item value={project.id}>
						<Icon name="folder-open" class="size-4 text-muted-foreground" />
						{project.name}
					</Select.Item>
				{/each}
			</Select.Group>
			<Select.Separator />
			<Select.Item value="new">
				<Icon name="plus" class="size-4 text-muted-foreground" />
				New Project...
			</Select.Item>
		</Select.Content>
	</Select.Portal>
</Select.Root>
