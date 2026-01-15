<script lang="ts" module>
	export interface MonacoEditorProps {
		value: string;
		language?: string;
		theme?: 'vs-dark' | 'vs' | 'hc-black';
		readonly?: boolean;
		height?: string;
		class?: string;
		onValueChange?: (value: string) => void;
	}
</script>

<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { cn } from '$lib/utils.js';

	let {
		value = $bindable(''),
		language = 'python',
		theme = 'vs-dark',
		readonly = false,
		height = '400px',
		class: className,
		onValueChange,
	}: MonacoEditorProps = $props();

	let container: HTMLDivElement;
	let editor: import('monaco-editor').editor.IStandaloneCodeEditor | null = null;
	let monaco: typeof import('monaco-editor') | null = null;
	let isUpdating = false;

	onMount(async () => {
		// Dynamic import to avoid SSR issues
		monaco = await import('monaco-editor');

		// Configure Monaco for Vite (self-hosted workers)
		// This uses blob URLs for workers which works well with Vite
		self.MonacoEnvironment = {
			getWorker: function (_moduleId: string, label: string) {
				const getWorkerModule = (_moduleUrl: string, _label: string) => {
					return new Worker(
						new URL(
							'monaco-editor/esm/vs/editor/editor.worker.js',
							import.meta.url
						),
						{ type: 'module' }
					);
				};

				switch (label) {
					case 'json':
						return getWorkerModule('vs/language/json/json.worker', label);
					case 'typescript':
					case 'javascript':
						return getWorkerModule('vs/language/typescript/ts.worker', label);
					default:
						return getWorkerModule('vs/editor/editor.worker', label);
				}
			},
		};

		editor = monaco.editor.create(container, {
			value,
			language,
			theme,
			readOnly: readonly,
			automaticLayout: true,
			minimap: { enabled: false },
			fontSize: 13,
			fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
			lineNumbers: 'on',
			scrollBeyondLastLine: false,
			wordWrap: 'on',
			tabSize: 4,
			insertSpaces: true,
			renderWhitespace: 'selection',
			bracketPairColorization: { enabled: true },
			padding: { top: 12, bottom: 12 },
			smoothScrolling: true,
			cursorBlinking: 'smooth',
			cursorSmoothCaretAnimation: 'on',
		});

		// Listen for content changes
		editor.onDidChangeModelContent(() => {
			if (!isUpdating && editor) {
				const newValue = editor.getValue();
				value = newValue;
				onValueChange?.(newValue);
			}
		});
	});

	onDestroy(() => {
		editor?.dispose();
		editor = null;
	});

	// Update editor when value prop changes externally
	$effect(() => {
		if (editor && value !== editor.getValue()) {
			isUpdating = true;
			editor.setValue(value);
			isUpdating = false;
		}
	});

	// Update editor options when props change
	$effect(() => {
		if (editor) {
			editor.updateOptions({ readOnly: readonly });
		}
	});

	$effect(() => {
		if (editor && monaco) {
			monaco.editor.setModelLanguage(editor.getModel()!, language);
		}
	});

	$effect(() => {
		if (editor && monaco) {
			monaco.editor.setTheme(theme);
		}
	});
</script>

<div
	class={cn(
		'w-full rounded-md border border-border overflow-hidden',
		'bg-[#1e1e1e]',
		className
	)}
	style:height
>
	<div bind:this={container} class="h-full w-full"></div>
</div>
