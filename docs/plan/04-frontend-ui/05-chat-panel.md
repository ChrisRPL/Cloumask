# Chat Panel

> **Status:** 🔴 Not Started
> **Priority:** P1 (High - primary user interaction)
> **Dependencies:** 01-design-system, 02-core-layout, 03-stores-state, 04-tauri-sse-integration
> **Estimated Complexity:** High

## Overview

Implement the conversational chat interface where users interact with the AI agent. Supports streaming text responses, inline clarification questions, quick-reply buttons, plan previews, and file attachments.

## Goals

- [ ] Message list with user/agent distinction
- [ ] Streaming text display (character by character)
- [ ] Inline clarification questions with options
- [ ] Quick-reply suggestion buttons
- [ ] Plan preview cards in messages
- [ ] File/folder attachment support
- [ ] Message input with keyboard shortcuts

## Technical Design

### Chat Panel Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CHAT HEADER                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  💬 Chat with Cloumask Agent              [Clear] [Export]        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  MESSAGE LIST (scrollable)                                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │ 🤖 Agent                                          10:30 AM  │  │  │
│  │  │ Hello! I can help you process your images. What would       │  │  │
│  │  │ you like to do today?                                       │  │  │
│  │  │                                                             │  │  │
│  │  │ [Anonymize faces] [Auto-label] [Convert format]             │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                                          👤 You   10:31 AM  │  │  │
│  │  │  I need to anonymize faces in my dataset                    │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │ 🤖 Agent                                          10:31 AM  │  │  │
│  │  │ I'll create a pipeline to detect and blur faces. Which      │  │  │
│  │  │ blur method do you prefer?                                  │  │  │
│  │  │                                                             │  │  │
│  │  │ ○ Gaussian blur (natural look)                              │  │  │
│  │  │ ○ Pixelate (obvious anonymization)                          │  │  │
│  │  │ ○ Solid fill (complete removal)                             │  │  │
│  │  │                                                             │  │  │
│  │  │ [Confirm selection]                                         │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│  INPUT AREA                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  [📎]  Type your message...                              [Send]  │  │
│  │                                                                   │  │
│  │  📁 dataset/ (1,234 files)                                  [×]  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Message Types

```typescript
// Different message bubble variants
type MessageVariant =
  | 'text'           // Plain text message
  | 'streaming'      // Text being streamed
  | 'clarification'  // Question with options
  | 'plan_preview'   // Pipeline plan card
  | 'file_list'      // Attached files summary
  | 'error'          // Error message
  | 'system';        // System notification

// Quick reply structure
interface QuickReply {
  label: string;
  value: string;
  icon?: string;
}

// Clarification question
interface Clarification {
  question: string;
  options: ClarificationOption[];
  multi_select: boolean;
  required: boolean;
}

interface ClarificationOption {
  id: string;
  label: string;
  description?: string;
  selected?: boolean;
}
```

### Component Hierarchy

```
ChatPanel.svelte
├── ChatHeader.svelte
│   ├── Title
│   ├── ClearButton
│   └── ExportButton
├── MessageList.svelte
│   └── MessageBubble.svelte (× n)
│       ├── MessageContent.svelte
│       ├── QuickReplyButtons.svelte
│       ├── ClarificationForm.svelte
│       ├── PlanPreviewCard.svelte
│       └── StreamingIndicator.svelte
└── ChatInput.svelte
    ├── AttachmentButton.svelte
    ├── TextArea
    ├── AttachmentPreview.svelte
    └── SendButton.svelte
```

## Implementation Tasks

- [ ] **ChatPanel Container**
  - [ ] Create `ChatPanel.svelte` with flex column layout
  - [ ] Connect to `agentStore` for messages
  - [ ] Handle SSE events for streaming updates
  - [ ] Implement scroll-to-bottom on new messages
  - [ ] Add keyboard focus management

- [ ] **ChatHeader Component**
  - [ ] Create `ChatHeader.svelte`
  - [ ] Add "Clear conversation" button with confirmation
  - [ ] Add "Export chat" button (markdown/JSON)
  - [ ] Show agent state indicator (thinking, typing, idle)

- [ ] **MessageList Component**
  - [ ] Create `MessageList.svelte` with virtual scrolling
  - [ ] Implement auto-scroll behavior
  - [ ] Add scroll-to-bottom button when scrolled up
  - [ ] Handle message grouping by time

- [ ] **MessageBubble Component**
  - [ ] Create `MessageBubble.svelte` with variants
  - [ ] User messages: right-aligned, violet accent
  - [ ] Agent messages: left-aligned, dark card
  - [ ] System messages: centered, muted
  - [ ] Add timestamp display
  - [ ] Add copy-to-clipboard button

- [ ] **Streaming Text Display**
  - [ ] Implement character-by-character rendering
  - [ ] Add typing indicator (animated dots)
  - [ ] Handle markdown rendering in streamed text
  - [ ] Smooth cursor animation

- [ ] **ClarificationForm Component**
  - [ ] Create `ClarificationForm.svelte`
  - [ ] Radio buttons for single select
  - [ ] Checkboxes for multi-select
  - [ ] Option descriptions as tooltips
  - [ ] Submit button with validation

- [ ] **QuickReplyButtons Component**
  - [ ] Create `QuickReplyButtons.svelte`
  - [ ] Horizontal button row with overflow scroll
  - [ ] Icon + label format
  - [ ] Click sends message immediately

- [ ] **PlanPreviewCard Component**
  - [ ] Create `PlanPreviewCard.svelte`
  - [ ] Show pipeline steps summary
  - [ ] "View full plan" link to Plan Editor
  - [ ] Approve/Edit buttons inline

- [ ] **ChatInput Component**
  - [ ] Create `ChatInput.svelte`
  - [ ] Auto-expanding textarea
  - [ ] Enter to send, Shift+Enter for newline
  - [ ] Disabled state when agent is busy
  - [ ] Character count (optional limit)

- [ ] **AttachmentButton Component**
  - [ ] Create `AttachmentButton.svelte`
  - [ ] Open file/folder picker via Tauri
  - [ ] Show selected files as pills
  - [ ] Remove attachment button
  - [ ] File count and total size display

- [ ] **Markdown Rendering**
  - [ ] Install `marked` or `markdown-it`
  - [ ] Render code blocks with syntax highlighting
  - [ ] Render links, lists, bold, italic
  - [ ] Sanitize HTML output

## Acceptance Criteria

- [ ] Messages display correctly with user/agent distinction
- [ ] Streaming text appears smoothly without flicker
- [ ] Clarification questions are answerable inline
- [ ] Quick replies send messages on click
- [ ] Files can be attached and displayed
- [ ] Enter sends message, Shift+Enter adds newline
- [ ] Conversation can be cleared and exported
- [ ] Auto-scroll works, with manual scroll override

## Files to Create/Modify

```
src/lib/components/Chat/
├── ChatPanel.svelte
├── ChatHeader.svelte
├── MessageList.svelte
├── MessageBubble.svelte
├── MessageContent.svelte
├── StreamingIndicator.svelte
├── QuickReplyButtons.svelte
├── ClarificationForm.svelte
├── PlanPreviewCard.svelte
├── ChatInput.svelte
├── AttachmentButton.svelte
├── AttachmentPreview.svelte
└── index.ts
```

## Streaming Text Implementation

```svelte
<!-- StreamingText.svelte -->
<script lang="ts">
  const { content, isStreaming } = $props<{
    content: string;
    isStreaming: boolean;
  }>();

  let displayedContent = $state('');
  let charIndex = $state(0);

  $effect(() => {
    if (isStreaming && charIndex < content.length) {
      const timer = setTimeout(() => {
        displayedContent = content.slice(0, charIndex + 1);
        charIndex++;
      }, 20); // 20ms per character ≈ 50 chars/second

      return () => clearTimeout(timer);
    } else if (!isStreaming) {
      displayedContent = content;
    }
  });
</script>

<span class="streaming-text">
  {displayedContent}
  {#if isStreaming}
    <span class="cursor animate-blink">|</span>
  {/if}
</span>
```

## Message Store Integration

```typescript
// In ChatPanel.svelte
import { agentStore } from '$lib/stores/agent';
import { sseStore } from '$lib/stores/sse';

$effect(() => {
  // Subscribe to SSE for message streaming
  if ($sseStore.isConnected) {
    sseStore.on('agent_message', (event) => {
      if (event.is_partial) {
        // Update last message with partial content
        agentStore.updateLastMessage(event.content);
      } else {
        // Complete message
        agentStore.addMessage({
          role: 'agent',
          content: event.content,
          metadata: event.metadata,
        });
      }
    });
  }
});
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line |
| `Ctrl+/` | Focus chat input |
| `Escape` | Clear input / close attachment |
| `↑` | Edit last message (when input empty) |

## Accessibility

- [ ] Messages have proper ARIA roles (`role="log"`, `aria-live="polite"`)
- [ ] Input has `aria-label` and placeholder
- [ ] Buttons have descriptive labels
- [ ] Focus is managed on new messages
- [ ] Screen reader announces new messages

## Notes

- Consider using `svelte-virtual-list` for large message histories
- Debounce scroll events to prevent performance issues
- Cache rendered markdown to avoid re-parsing
- Test with 1000+ messages for performance
- Export format should be human-readable markdown
