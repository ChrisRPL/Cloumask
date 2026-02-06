---
name: svelte5-development
description: Guide for developing Svelte 5 components and stores in Cloumask using runes, context, and reactive patterns. Use when creating components, managing state, or working with the frontend.
---

# Svelte 5 Development

## Quick Start

When creating Svelte 5 components:

1. Use `$state` for reactive state
2. Use `$derived` for computed values
3. Use `$effect` for side effects
4. Use context for shared state
5. Use runes in `<script>` tags (not `<script setup>`)

## Runes Basics

Svelte 5 uses runes for reactivity:

```svelte
<script>
  // State rune
  let count = $state(0);
  
  // Derived rune
  let doubled = $derived(count * 2);
  
  // Effect rune
  $effect(() => {
    console.log(`Count is ${count}`);
  });
  
  function increment() {
    count++;
  }
</script>

<button onclick={increment}>
  Count: {count}, Doubled: {doubled}
</button>
```

## State Management Pattern

Use state factories for complex state:

```typescript
// stores/my-store.svelte.ts
import { getContext, setContext } from 'svelte';

const MY_STATE_KEY = Symbol('my-state');

export interface MyState {
  readonly value: number;
  setValue(v: number): void;
  reset(): void;
}

export function createMyState(): MyState {
  let value = $state(0);
  
  return {
    get value() {
      return value;
    },
    setValue(v: number) {
      value = v;
    },
    reset() {
      value = 0;
    },
  };
}

export function setMyState(): MyState {
  const state = createMyState();
  setContext(MY_STATE_KEY, state);
  return state;
}

export function getMyState(): MyState {
  return getContext<MyState>(MY_STATE_KEY);
}
```

## Component Structure

Components follow this pattern:

```svelte
<script lang="ts">
  import { getAgentState } from '$lib/stores/agent.svelte';
  import type { Message } from '$lib/types/agent';
  
  // Props
  interface Props {
    message: Message;
    highlight?: boolean;
  }
  
  let { message, highlight = false }: Props = $props();
  
  // State
  let isExpanded = $state(false);
  
  // Derived
  const displayText = $derived(
    isExpanded ? message.content : message.content.slice(0, 100)
  );
  
  // Effects
  $effect(() => {
    if (message.role === 'user') {
      // Side effect when message changes
    }
  });
  
  // Functions
  function toggleExpand() {
    isExpanded = !isExpanded;
  }
</script>

<div class:highlight>
  <p>{displayText}</p>
  <button onclick={toggleExpand}>
    {isExpanded ? 'Collapse' : 'Expand'}
  </button>
</div>
```

## Props

Use `$props()` for component props:

```svelte
<script>
  interface Props {
    title: string;
    count?: number;
    items?: string[];
  }
  
  let { title, count = 0, items = [] }: Props = $props();
</script>

<h1>{title}</h1>
<p>Count: {count}</p>
{#each items as item}
  <p>{item}</p>
{/each}
```

## Context Pattern

Use context for shared state:

```svelte
<!-- Parent component -->
<script>
  import { setAgentState } from '$lib/stores/agent.svelte';
  
  const agentState = setAgentState();
</script>

<!-- Child component -->
<script>
  import { getAgentState } from '$lib/stores/agent.svelte';
  
  const agentState = getAgentState();
  
  // Use agentState.messages, agentState.phase, etc.
</script>
```

## Derived Values

Use `$derived` for computed values:

```svelte
<script>
  let items = $state([1, 2, 3, 4, 5]);
  
  // Simple derived
  const sum = $derived(items.reduce((a, b) => a + b, 0));
  
  // Complex derived
  const filtered = $derived.by(() => {
    return items.filter(x => x > 2);
  });
</script>

<p>Sum: {sum}</p>
<p>Filtered: {filtered.join(', ')}</p>
```

## Effects

Use `$effect` for side effects:

```svelte
<script>
  let count = $state(0);
  
  // Effect runs when dependencies change
  $effect(() => {
    console.log(`Count changed to ${count}`);
    
    // Cleanup function
    return () => {
      console.log('Effect cleanup');
    };
  });
  
  // Effect with explicit dependencies
  $effect(() => {
    const timer = setInterval(() => {
      count++;
    }, 1000);
    
    return () => clearInterval(timer);
  });
</script>
```

## Tauri Commands

Invoke Tauri commands:

```svelte
<script>
  import { invoke } from '@tauri-apps/api/core';
  
  let data = $state<string | null>(null);
  let loading = $state(false);
  
  async function loadData() {
    loading = true;
    try {
      data = await invoke<string>('get_data');
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      loading = false;
    }
  }
</script>

{#if loading}
  <p>Loading...</p>
{:else if data}
  <p>{data}</p>
{/else}
  <button onclick={loadData}>Load</button>
{/if}
```

## Event Listeners

Listen to Tauri events:

```svelte
<script>
  import { listen } from '@tauri-apps/api/event';
  import { onMount } from 'svelte';
  
  let progress = $state(0);
  
  onMount(() => {
    const unlisten = listen<{ current: number; total: number }>(
      'progress',
      (event) => {
        progress = (event.payload.current / event.payload.total) * 100;
      }
    );
    
    return () => {
      unlisten.then(fn => fn());
    };
  });
</script>

<progress value={progress} max={100} />
```

## SSE Streaming

Handle SSE events:

```svelte
<script>
  import { getAgentState } from '$lib/stores/agent.svelte';
  import { connectSSE } from '$lib/utils/sse';
  
  const agentState = getAgentState();
  
  onMount(() => {
    const cleanup = connectSSE({
      onMessage: (message) => {
        agentState.addMessage(message);
      },
      onPhaseChange: (phase) => {
        agentState.setPhase(phase);
      },
    });
    
    return cleanup;
  });
</script>
```

## Conditional Rendering

Use Svelte conditionals:

```svelte
{#if condition}
  <p>Shown when condition is true</p>
{:else if otherCondition}
  <p>Shown when otherCondition is true</p>
{:else}
  <p>Default content</p>
{/if}
```

## Loops

Iterate over arrays:

```svelte
{#each items as item, index (item.id)}
  <div>
    {index + 1}: {item.name}
  </div>
{:else}
  <p>No items</p>
{/each}
```

## Async Blocks

Handle promises:

```svelte
<script>
  let promise = $state(Promise.resolve('initial'));
  
  async function loadData() {
    promise = fetch('/api/data').then(r => r.json());
  }
</script>

{#await promise}
  <p>Loading...</p>
{:then data}
  <p>{data}</p>
{:catch error}
  <p>Error: {error.message}</p>
{/await}
```

## Key Blocks

Re-render on key changes:

```svelte
{#key selectedId}
  <ExpensiveComponent id={selectedId} />
{/key}
```

## Component Composition

Compose components:

```svelte
<!-- Parent.svelte -->
<script>
  import Child from './Child.svelte';
  
  let value = $state(42);
</script>

<Child {value} />

<!-- Child.svelte -->
<script>
  interface Props {
    value: number;
  }
  
  let { value }: Props = $props();
</script>

<p>Value: {value}</p>
```

## Slots

Use slots for content projection:

```svelte
<!-- Card.svelte -->
<div class="card">
  <header>
    <slot name="header">Default header</slot>
  </header>
  <main>
    <slot />
  </main>
</div>

<!-- Usage -->
<Card>
  <svelte:fragment slot="header">
    <h1>Custom Header</h1>
  </svelte:fragment>
  <p>Main content</p>
</Card>
```

## Transitions

Use transitions for animations:

```svelte
<script>
  import { fade, slide } from 'svelte/transition';
  
  let visible = $state(true);
</script>

{#if visible}
  <div transition:fade={{ duration: 300 }}>
    Content
  </div>
{/if}

<!-- Or with slide -->
<div in:slide={{ axis: 'y' }}>
  Slides in
</div>
```

## Actions

Use actions for DOM interactions:

```svelte
<script>
  function clickOutside(node: HTMLElement) {
    function handleClick(event: MouseEvent) {
      if (!node.contains(event.target as Node)) {
        node.dispatchEvent(new CustomEvent('clickoutside'));
      }
    }
    
    document.addEventListener('click', handleClick);
    
    return {
      destroy() {
        document.removeEventListener('click', handleClick);
      }
    };
  }
</script>

<div use:clickOutside on:clickoutside={() => console.log('Clicked outside')}>
  Content
</div>
```

## Best Practices

1. **Use runes in `<script>`** - Not `<script setup>`
2. **Use context for shared state** - Avoid prop drilling
3. **Use `$derived` for computed values** - More efficient than functions
4. **Clean up effects** - Return cleanup functions
5. **Use TypeScript** - Type props and state
6. **Extract stores** - Complex state in separate files
7. **Use key blocks** - Force re-render when needed
8. **Handle loading states** - Show loading indicators
9. **Clean up event listeners** - In `onMount` cleanup
10. **Use slots** - For flexible component composition

## Common Patterns

### Loading State

```svelte
<script>
  let loading = $state(false);
  let data = $state<string | null>(null);
  
  async function load() {
    loading = true;
    try {
      data = await fetchData();
    } finally {
      loading = false;
    }
  }
</script>

{#if loading}
  <Spinner />
{:else if data}
  <Display {data} />
{:else}
  <button onclick={load}>Load</button>
{/if}
```

### Form Handling

```svelte
<script>
  let formData = $state({
    name: '',
    email: '',
  });
  
  function handleSubmit() {
    // Submit form
    console.log(formData);
  }
</script>

<form onsubmit={handleSubmit}>
  <input
    type="text"
    bind:value={formData.name}
    placeholder="Name"
  />
  <input
    type="email"
    bind:value={formData.email}
    placeholder="Email"
  />
  <button type="submit">Submit</button>
</form>
```

### Debounced Input

```svelte
<script>
  let search = $state('');
  let debouncedSearch = $state('');
  
  $effect(() => {
    const timer = setTimeout(() => {
      debouncedSearch = search;
    }, 300);
    
    return () => clearTimeout(timer);
  });
</script>

<input bind:value={search} />
<p>Searching for: {debouncedSearch}</p>
```

## Additional Resources

- See `src/lib/stores/agent.svelte.ts` for state management example
- See `src/lib/components/Chat/` for component examples
- See `src/lib/utils/sse.ts` for SSE handling
