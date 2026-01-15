/**
 * Agent type definitions for chat messages and conversation state.
 *
 * These types mirror the Python backend types in backend/src/backend/agent/state.py
 * to ensure type-safe communication between frontend and backend.
 */

// ============================================================================
// Message Types
// ============================================================================

export type MessageRole = "user" | "assistant" | "system" | "tool";

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string; // ISO 8601
  toolCalls?: ToolCall[];
  toolCallId?: string;
  isStreaming?: boolean;
}

// ============================================================================
// Agent Phase State Machine
// ============================================================================

/**
 * Agent phase represents the current state in the conversation flow:
 * idle -> understanding -> planning -> awaiting_approval -> executing -> checkpoint -> complete
 */
export type AgentPhase =
  | "idle"
  | "understanding"
  | "planning"
  | "awaiting_approval"
  | "executing"
  | "checkpoint"
  | "complete"
  | "error";

// ============================================================================
// User Decisions
// ============================================================================

export type UserDecision = "approve" | "edit" | "cancel" | "retry";

export type ClarificationInputType =
  | "plan_approval"
  | "checkpoint_approval"
  | "clarification";

export interface ClarificationRequest {
  id: string;
  prompt: string;
  options?: string[];
  inputType: ClarificationInputType;
}

// ============================================================================
// Agent State Interface
// ============================================================================

export interface AgentState {
  readonly threadId: string | null;
  readonly messages: Message[];
  readonly phase: AgentPhase;
  readonly isConnected: boolean;
  readonly isStreaming: boolean;
  readonly pendingClarification: ClarificationRequest | null;
  readonly lastError: string | null;

  // Derived
  readonly isBusy: boolean;
  readonly needsInput: boolean;
  readonly lastMessage: Message | null;

  // Actions
  setThreadId(id: string | null): void;
  setPhase(phase: AgentPhase): void;
  setConnected(connected: boolean): void;
  setStreaming(streaming: boolean): void;
  setError(error: string | null): void;

  // Message actions
  addMessage(message: Omit<Message, "id" | "timestamp">): Message;
  updateMessage(id: string, updates: Partial<Message>): void;
  clearMessages(): void;

  // Clarification actions
  setClarification(clarification: ClarificationRequest | null): void;

  // Conversation management
  startNewConversation(): void;
  reset(): void;
}
