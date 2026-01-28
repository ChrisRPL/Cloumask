/**
 * Svelte stores for Cloumask application state.
 *
 * Each store provides centralized state management for a specific domain:
 * - UI: Sidebar, navigation, project selection
 * - Settings: User preferences, keyboard shortcuts, model defaults
 * - Agent: Chat messages, conversation state, agent phase
 * - Pipeline: Plan editor steps, configuration, reordering
 * - Execution: Progress tracking, statistics, checkpoints
 * - Review: Review queue items, filtering, batch operations
 */

// UI Store
export {
  createUIState,
  setUIState,
  getUIState,
  VIEWS,
  SIDEBAR_COLLAPSED_WIDTH,
  SIDEBAR_EXPANDED_WIDTH,
} from "./ui.svelte";
export type { ViewId, ViewConfig, Project, UIState } from "./ui.svelte";

// Settings Store
export {
  createSettingsState,
  setSettingsState,
  getSettingsState,
  DEFAULT_SETTINGS,
} from "./settings.svelte";
export type {
  Settings,
  SettingsState,
  Theme,
  KeyboardShortcuts,
  ModelDefaults,
} from "./settings.svelte";

// Agent Store
export { createAgentState, setAgentState, getAgentState } from "./agent.svelte";
export type {
  Message,
  MessageRole,
  AgentPhase,
  AgentState,
  ClarificationRequest,
  ToolCall,
  UserDecision,
} from "./agent.svelte";

// Pipeline Store
export {
  createPipelineState,
  setPipelineState,
  getPipelineState,
} from "./pipeline.svelte";
export type {
  PipelineStep,
  PipelineState,
  StepType,
  StepStatus,
  StepConfig,
} from "./pipeline.svelte";

// Execution Store
export {
  createExecutionState,
  setExecutionState,
  getExecutionState,
} from "./execution.svelte";
export type {
  ExecutionState,
  ExecutionStatus,
  ExecutionProgress,
  ExecutionStats,
  ExecutionError,
  CheckpointInfo,
  CheckpointTrigger,
  QualityMetrics,
} from "./execution.svelte";

// Review Store
export {
  createReviewState,
  setReviewState,
  getReviewState,
} from "./review.svelte";
export type {
  ReviewState,
  ReviewItem,
  ReviewFilters,
  ReviewStatus,
  Annotation,
  AnnotationType,
  BoundingBox,
} from "./review.svelte";

// Setup Store
export {
  createSetupState,
  setSetupState,
  getSetupState,
} from "./setup.svelte";
export type {
  SetupState,
  SetupStep,
  SetupProgress,
} from "./setup.svelte";
