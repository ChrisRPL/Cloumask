<script lang="ts">
	import { setSettingsState } from '../settings.svelte';

	const state = setSettingsState();
</script>

<span data-testid="theme">{state.settings.theme}</span>
