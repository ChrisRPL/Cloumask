<script lang="ts" module>
	import { cn } from '$lib/utils.js';

	export interface ProjectSelectorProps {
		class?: string;
	}
</script>

<script lang="ts">
	import Icon from '$lib/components/ui/icons.svelte';
	import * as Select from '$lib/components/ui/select';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import { getUIState } from '$lib/stores/ui.svelte';
	import type { Project } from '$lib/types/ui';

	let { class: className }: ProjectSelectorProps = $props();

	const ui = getUIState();

	let selectedValue = $state<string | undefined>(undefined);
	let dialogOpen = $state(false);
	let newProjectName = $state('');
	let newProjectPath = $state('');

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

	function openNewProjectDialog() {
		newProjectName = '';
		newProjectPath = '';
		dialogOpen = true;
	}

	function handleCreateProject() {
		const name = newProjectName.trim();
		if (!name) return;

		const safeName = slugifyProjectName(name) || 'new-project';
		const path = newProjectPath.trim() || `/data/${safeName}`;

		const project: Project = {
			id: crypto.randomUUID(),
			name,
			path,
			lastOpened: new Date(),
		};

		ui.setProject(project);
		selectedValue = project.id;
		dialogOpen = false;
	}

	function handleValueChange(value: string | undefined) {
		if (value === 'new') {
			openNewProjectDialog();
			// Reset the select value so it doesn't stay on 'new'
			selectedValue = ui.currentProject?.id;
			return;
		}

		selectedValue = value;
		const project = ui.recentProjects.find((p) => p.id === value);
		if (project) {
			ui.setProject({ ...project, lastOpened: new Date() });
		}
	}

	function handleNameInput(e: Event) {
		const input = e.target as HTMLInputElement;
		newProjectName = input.value;
		// Auto-generate the path if user hasn't manually edited it
		const slug = slugifyProjectName(input.value) || 'new-project';
		newProjectPath = `/data/${slug}`;
	}
</script>

<Select.Root type="single" bind:value={selectedValue} onValueChange={handleValueChange}>
	<Select.Trigger class={cn('w-full min-w-0 bg-background/50', className)} size="sm">
		{#if ui.currentProject}
			<span class="flex min-w-0 items-center gap-2 truncate">
				<Icon name="folder-open" class="size-4 shrink-0 text-muted-foreground" />
				<span class="truncate">{ui.currentProject.name}</span>
			</span>
		{:else}
			<span class="block truncate text-muted-foreground">Select project...</span>
		{/if}
	</Select.Trigger>
	<Select.Portal>
		<Select.Content>
			{#if ui.recentProjects.length > 0}
				<Select.Group>
					<Select.GroupHeading>Recent Projects</Select.GroupHeading>
					{#each ui.recentProjects as project (project.id)}
						<Select.Item value={project.id}>
							<Icon name="folder-open" class="size-4 text-muted-foreground" />
							{project.name}
						</Select.Item>
					{/each}
				</Select.Group>
				<Select.Separator />
			{/if}
			<Select.Item value="new">
				<Icon name="plus" class="size-4 text-muted-foreground" />
				New Project...
			</Select.Item>
		</Select.Content>
	</Select.Portal>
</Select.Root>

<Dialog.Root bind:open={dialogOpen}>
	<Dialog.Portal>
		<Dialog.Content class="sm:max-w-md">
			<Dialog.Header>
				<Dialog.Title>New Project</Dialog.Title>
				<Dialog.Description>Create a new project to organize your images and pipelines.</Dialog.Description>
			</Dialog.Header>
			<div class="space-y-4 py-4">
				<div class="space-y-2">
					<label for="project-name" class="text-sm font-medium text-foreground">Project Name</label>
					<input
						id="project-name"
						type="text"
						class="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
						placeholder="e.g. Street Anonymization"
						value={newProjectName}
						oninput={handleNameInput}
						onkeydown={(e) => { if (e.key === 'Enter') handleCreateProject(); }}
					/>
				</div>
				<div class="space-y-2">
					<label for="project-path" class="text-sm font-medium text-foreground">Project Path</label>
					<input
						id="project-path"
						type="text"
						class="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm text-foreground shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring font-mono text-xs"
						placeholder="/data/my-project"
						value={newProjectPath}
						oninput={(e) => { newProjectPath = (e.target as HTMLInputElement).value; }}
						onkeydown={(e) => { if (e.key === 'Enter') handleCreateProject(); }}
					/>
				</div>
			</div>
			<Dialog.Footer>
				<Dialog.Close>
					<Button variant="outline" size="sm">Cancel</Button>
				</Dialog.Close>
				<Button
					size="sm"
					onclick={handleCreateProject}
					disabled={!newProjectName.trim()}
				>
					Create Project
				</Button>
			</Dialog.Footer>
		</Dialog.Content>
	</Dialog.Portal>
</Dialog.Root>
