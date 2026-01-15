/**
 * Plan Editor type definitions for step configuration schemas.
 *
 * These types extend the base pipeline types with configuration
 * metadata for the visual plan editor UI.
 */

import type { StepType } from "./pipeline";

// ============================================================================
// Configuration Field Schema
// ============================================================================

export type ConfigFieldType =
  | "select"
  | "number"
  | "slider"
  | "checkbox"
  | "text"
  | "multiselect";

export interface ConfigFieldOption {
  value: string;
  label: string;
}

export interface ConfigFieldSchema {
  key: string;
  label: string;
  type: ConfigFieldType;
  options?: ConfigFieldOption[];
  min?: number;
  max?: number;
  step?: number;
  default: unknown;
  description?: string;
}

// ============================================================================
// Step Type Configuration
// ============================================================================

export interface StepTypeConfig {
  type: StepType;
  label: string;
  icon: string;
  prefix: string;
  configSchema: ConfigFieldSchema[];
}

// ============================================================================
// Plan Editor Context
// ============================================================================

export interface PlanEditorContext {
  isAwaitingApproval: boolean;
  canStart: boolean;
  estimatedTimeMs: number | null;
  configPanelOpen: boolean;
  configPanelStepId: string | null;
}

// ============================================================================
// Time Estimation
// ============================================================================

export interface TimeFactors {
  baseMs: number;
  modelMultiplier: number;
  gpuSpeedup: number;
}
