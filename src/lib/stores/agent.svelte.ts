/**
 * Agent state management using Svelte 5 runes and context.
 *
 * Provides centralized state for chat messages, conversation state,
 * and agent phase tracking. The store structure is designed to receive
 * SSE events from the backend (connection handled in separate module).
 */

import { getContext, setContext } from "svelte";
import type {
  Message,
  MessageRole,
  AgentPhase,
  AgentState,
  ClarificationRequest,
  ToolCall,
  UserDecision,
} from "$lib/types/agent";

// Re-export types for convenience
export type {
  Message,
  MessageRole,
  AgentPhase,
  AgentState,
  ClarificationRequest,
  ToolCall,
  UserDecision,
};

// ============================================================================
// Constants
// ============================================================================

const AGENT_STATE_KEY = Symbol("agent-state");

// ============================================================================
// Helpers
// ============================================================================

function generateId(): string {
  return crypto.randomUUID();
}

function now(): string {
  return new Date().toISOString();
}

// ============================================================================
// State Factory
// ============================================================================

/**
 * Creates agent state using Svelte 5 runes.
 * Call this at the root layout to initialize the state.
 */
export function createAgentState(): AgentState {
  // Reactive state
  let threadId = $state<string | null>(null);
  let messages = $state<Message[]>([]);
  let phase = $state<AgentPhase>("idle");
  let isConnected = $state(false);
  let isStreaming = $state(false);
  let pendingClarification = $state<ClarificationRequest | null>(null);
  let lastError = $state<string | null>(null);

  // Derived values
  const isBusy = $derived(
    ["understanding", "planning", "executing"].includes(phase),
  );
  const needsInput = $derived(
    ["awaiting_approval", "checkpoint"].includes(phase),
  );
  const lastMessage = $derived(
    messages.length > 0 ? messages[messages.length - 1] : null,
  );

  return {
    // Getters
    get threadId() {
      return threadId;
    },
    get messages() {
      return messages;
    },
    get phase() {
      return phase;
    },
    get isConnected() {
      return isConnected;
    },
    get isStreaming() {
      return isStreaming;
    },
    get pendingClarification() {
      return pendingClarification;
    },
    get lastError() {
      return lastError;
    },

    // Derived getters
    get isBusy() {
      return isBusy;
    },
    get needsInput() {
      return needsInput;
    },
    get lastMessage() {
      return lastMessage;
    },

    // State setters
    setThreadId(id: string | null) {
      threadId = id;
    },

    setPhase(newPhase: AgentPhase) {
      phase = newPhase;
      // Clear error when moving to non-error phase
      if (newPhase !== "error") {
        lastError = null;
      }
    },

    setConnected(connected: boolean) {
      isConnected = connected;
    },

    setStreaming(streaming: boolean) {
      isStreaming = streaming;
    },

    setError(error: string | null) {
      lastError = error;
      if (error) {
        phase = "error";
      }
    },

    // Message actions
    addMessage(message: Omit<Message, "id" | "timestamp">): Message {
      const newMessage: Message = {
        ...message,
        id: generateId(),
        timestamp: now(),
      };
      messages = [...messages, newMessage];
      return newMessage;
    },

    updateMessage(id: string, updates: Partial<Message>) {
      messages = messages.map((msg) =>
        msg.id === id ? { ...msg, ...updates } : msg,
      );
    },

    clearMessages() {
      messages = [];
    },

    // Clarification actions
    setClarification(clarification: ClarificationRequest | null) {
      pendingClarification = clarification;
      if (clarification) {
        phase = "awaiting_approval";
      }
    },

    // Conversation management
    startNewConversation() {
      threadId = generateId();
      messages = [];
      phase = "idle";
      isStreaming = false;
      pendingClarification = null;
      lastError = null;
    },

    reset() {
      threadId = null;
      messages = [];
      phase = "idle";
      isConnected = false;
      isStreaming = false;
      pendingClarification = null;
      lastError = null;
    },
  };
}

// ============================================================================
// Context Helpers
// ============================================================================

/**
 * Initialize agent state and set it in Svelte context.
 * Call this in the root +layout.svelte.
 */
export function setAgentState(): AgentState {
  const state = createAgentState();
  setContext(AGENT_STATE_KEY, state);
  return state;
}

/**
 * Get agent state from Svelte context.
 * Call this in child components that need agent state.
 */
export function getAgentState(): AgentState {
  return getContext<AgentState>(AGENT_STATE_KEY);
}
