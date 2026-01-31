<script lang="ts" module>
	import type { PipelineStep, StepConfig as StepConfigType } from '$lib/types/pipeline';

	export interface CustomStepConfigProps {
		step: PipelineStep;
		class?: string;
		onUpdate?: (updates: Partial<StepConfigType>) => void;
		onClose?: () => void;
	}
</script>

<script lang="ts">
	import { cn } from '$lib/utils.js';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Label } from '$lib/components/ui/label';
	import {
		X,
		Sparkles,
		CheckCircle2,
		AlertCircle,
		Save,
		Loader2,
		RefreshCw
	} from '@lucide/svelte';
	import {
		generateScript,
		validateScript,
		saveScript,
		suggestScriptName
	} from '$lib/utils/scripts.js';
	import type { ValidateScriptResponse } from '$lib/types/scripts.js';

	let { step, class: className, onUpdate, onClose }: CustomStepConfigProps = $props();

	// Script builder state
	let prompt = $state('');
	let code = $state('');
	let scriptName = $state('');
	let description = $state('');

	// Loading states
	let isGenerating = $state(false);
	let isValidating = $state(false);
	let isSaving = $state(false);

	// Results
	let validation = $state<ValidateScriptResponse | null>(null);
	let error = $state<string | null>(null);
	let success = $state<string | null>(null);
	let savedPath = $state<string | null>(null);

	// Initialize from existing config
	$effect(() => {
		if (step.config.params?.script) {
			savedPath = step.config.params.script as string;
		}
	});

	// Derived states
	const hasCode = $derived(code.trim().length > 0);
	const isValid = $derived(validation?.valid === true);
	const hasValidationErrors = $derived(
		validation !== null && validation.errors.length > 0
	);

	/**
	 * Generate a script from the prompt using AI.
	 */
	async function handleGenerate() {
		if (!prompt.trim()) return;

		isGenerating = true;
		error = null;
		success = null;
		validation = null;

		try {
			const response = await generateScript({ prompt });
			code = response.script;

			// Auto-suggest script name if empty
			if (!scriptName) {
				scriptName = suggestScriptName(prompt);
			}

			// Auto-validate generated code
			await handleValidate();

			success = response.explanation || 'Script generated successfully';
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to generate script';
		} finally {
			isGenerating = false;
		}
	}

	/**
	 * Validate the current code.
	 */
	async function handleValidate() {
		if (!code.trim()) return;

		isValidating = true;
		error = null;

		try {
			validation = await validateScript(code);
			if (!validation.valid) {
				error = `Validation failed: ${validation.errors.length} error(s)`;
			}
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to validate script';
		} finally {
			isValidating = false;
		}
	}

	/**
	 * Save the script and apply to step config.
	 */
	async function handleSave() {
		if (!code.trim() || !scriptName.trim()) return;

		isSaving = true;
		error = null;
		success = null;

		try {
			const response = await saveScript({
				name: scriptName,
				content: code,
				description: description || prompt,
				overwrite: true
			});

			savedPath = response.path;
			success = `Script saved to ${response.path}`;

			// Update step config with script path
			onUpdate?.({
				params: {
					...step.config.params,
					script: response.path
				}
			});
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to save script';
		} finally {
			isSaving = false;
		}
	}

	/**
	 * Handle code changes from editor.
	 */
	function handleCodeChange(newValue: string) {
		code = newValue;
		// Clear validation when code changes
		validation = null;
	}

	/**
	 * Clear and start fresh.
	 */
	function handleClear() {
		prompt = '';
		code = '';
		scriptName = '';
		description = '';
		validation = null;
		error = null;
		success = null;
	}
</script>

<aside
	class={cn(
		'w-[600px] border-l border-border bg-card/50 flex flex-col',
		'animate-in slide-in-from-right duration-200',
		className
	)}
>
	<!-- Header -->
	<header class="flex items-center justify-between px-4 py-3 border-b border-border">
		<div class="flex items-center gap-2 min-w-0">
			<Sparkles class="h-4 w-4 text-forest-light" />
			<span class="font-medium">Custom Script Builder</span>
		</div>
		<button
			onclick={onClose}
			class="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
			title="Close (Esc)"
		>
			<X class="h-4 w-4" />
		</button>
	</header>

	<!-- Content -->
	<div class="flex-1 overflow-auto p-4 space-y-4">
		<!-- Prompt Section -->
		<div class="space-y-2">
			<Label for="prompt" class="text-sm font-medium">
				Describe your script
			</Label>
			<Textarea
				id="prompt"
				bind:value={prompt}
				placeholder="e.g., Convert all images to grayscale and resize to 512x512 while preserving aspect ratio"
				class="min-h-[80px] font-mono text-sm"
				disabled={isGenerating}
			/>
			<div class="flex items-center gap-2">
				<Button
					variant="default"
					size="sm"
					onclick={handleGenerate}
					disabled={!prompt.trim() || isGenerating}
				>
					{#if isGenerating}
						<Loader2 class="h-4 w-4 mr-2 animate-spin" />
						Generating...
					{:else}
						<Sparkles class="h-4 w-4 mr-2" />
						Generate Script
					{/if}
				</Button>
				{#if hasCode}
					<Button variant="ghost" size="sm" onclick={handleClear}>
						<RefreshCw class="h-4 w-4 mr-2" />
						Clear
					</Button>
				{/if}
			</div>
		</div>

		<!-- Code Editor Section -->
		{#if hasCode || isGenerating}
			<div class="space-y-2">
				<div class="flex items-center justify-between">
					<Label class="text-sm font-medium">Generated Code</Label>
					<div class="flex items-center gap-2">
						{#if validation}
							{#if validation.valid}
								<span class="flex items-center gap-1 text-xs text-green-600">
									<CheckCircle2 class="h-3 w-3" />
									Valid
								</span>
							{:else}
								<span class="flex items-center gap-1 text-xs text-destructive">
									<AlertCircle class="h-3 w-3" />
									{validation.errors.length} error(s)
								</span>
							{/if}
						{/if}
						<Button
							variant="outline"
							size="sm"
							onclick={handleValidate}
							disabled={!hasCode || isValidating}
						>
							{#if isValidating}
								<Loader2 class="h-3 w-3 mr-1 animate-spin" />
							{:else}
								<CheckCircle2 class="h-3 w-3 mr-1" />
							{/if}
							Validate
						</Button>
					</div>
				</div>

				<!-- Simple code textarea (Monaco had loading issues in Tauri) -->
				<textarea
					bind:value={code}
					oninput={(e) => handleCodeChange(e.currentTarget.value)}
					class="w-full h-[300px] p-3 font-mono text-sm bg-[#1e1e1e] text-[#d4d4d4] border border-border rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-ring"
					spellcheck="false"
					placeholder="# Generated Python code will appear here..."
				></textarea>

				<!-- Validation Errors -->
				{#if hasValidationErrors && validation}
					<div class="rounded-md border border-destructive/50 bg-destructive/10 p-3 space-y-1">
						{#each validation.errors as err}
							<p class="text-xs text-destructive font-mono">
								{#if err.line}
									Line {err.line}: {err.message}
								{:else}
									{err.message}
								{/if}
							</p>
						{/each}
					</div>
				{/if}

				<!-- Warnings -->
				{#if validation && validation.warnings.length > 0}
					<div class="rounded-md border border-yellow-500/50 bg-yellow-500/10 p-3 space-y-1">
						{#each validation.warnings as warn}
							<p class="text-xs text-yellow-600 font-mono">
								{#if warn.line}
									Line {warn.line}: {warn.message}
								{:else}
									{warn.message}
								{/if}
							</p>
						{/each}
					</div>
				{/if}
			</div>
		{/if}

		<!-- Save Section -->
		{#if hasCode}
			<div class="space-y-3 pt-2 border-t border-border">
				<div class="grid grid-cols-2 gap-3">
					<div class="space-y-1">
						<Label for="scriptName" class="text-xs">Script Name</Label>
						<Input
							id="scriptName"
							bind:value={scriptName}
							placeholder="my_script"
							class="font-mono text-sm"
						/>
					</div>
					<div class="space-y-1">
						<Label for="description" class="text-xs">Description (optional)</Label>
						<Input
							id="description"
							bind:value={description}
							placeholder="Brief description..."
							class="text-sm"
						/>
					</div>
				</div>
			</div>
		{/if}

		<!-- Status Messages -->
		{#if error}
			<div class="rounded-md border border-destructive/50 bg-destructive/10 p-3">
				<p class="text-sm text-destructive">{error}</p>
			</div>
		{/if}

		{#if success}
			<div class="rounded-md border border-green-500/50 bg-green-500/10 p-3">
				<p class="text-sm text-green-600">{success}</p>
			</div>
		{/if}

		<!-- Saved Script Path -->
		{#if savedPath}
			<div class="rounded-md border border-border bg-muted/30 p-3">
				<p class="text-xs text-muted-foreground mb-1">Script path:</p>
				<code class="text-xs font-mono text-foreground">{savedPath}</code>
			</div>
		{/if}
	</div>

	<!-- Footer Actions -->
	<footer class="flex items-center justify-between px-4 py-3 border-t border-border">
		<Button variant="ghost" size="sm" onclick={onClose}>
			Cancel
		</Button>
		<Button
			variant="default"
			size="sm"
			onclick={handleSave}
			disabled={!hasCode || !scriptName.trim() || isSaving || (validation !== null && !validation.valid)}
		>
			{#if isSaving}
				<Loader2 class="h-4 w-4 mr-2 animate-spin" />
				Saving...
			{:else}
				<Save class="h-4 w-4 mr-2" />
				Save & Apply
			{/if}
		</Button>
	</footer>
</aside>
