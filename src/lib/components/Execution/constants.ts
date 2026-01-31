/**
 * Constants for ExecutionView components.
 * Status labels, checkpoint triggers, and keyboard shortcuts.
 */

import type { ExecutionStatus, CheckpointTrigger } from '$lib/types/execution';

// ============================================================================
// Status Display Configuration
// ============================================================================

export const STATUS_DISPLAY: Record<ExecutionStatus, { label: string; color: string }> = {
	idle: { label: 'idle', color: 'text-muted-foreground' },
	running: { label: 'executing', color: 'text-forest-light' },
	paused: { label: 'paused', color: 'text-amber-500' },
	checkpoint: { label: 'checkpoint', color: 'text-amber-400' },
	completed: { label: 'complete', color: 'text-green-600' },
	failed: { label: 'failed', color: 'text-destructive' },
	cancelled: { label: 'cancelled', color: 'text-muted-foreground' },
};

// ============================================================================
// Checkpoint Trigger Labels
// ============================================================================

export const CHECKPOINT_TRIGGERS: Record<CheckpointTrigger, string> = {
	percentage: 'Progress milestone reached',
	quality_drop: 'Quality drop detected (confidence -15%)',
	error_rate: 'Error rate exceeded threshold (>5%)',
	critical_step: 'Critical step requires review',
};

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

export const KEYBOARD_SHORTCUTS = [
	{ key: 'Space', action: 'Pause/Resume' },
	{ key: 'Esc', action: 'Cancel' },
	{ key: 'R', action: 'Review' },
	{ key: 'Enter', action: 'Continue' },
	{ key: 'E', action: 'Errors' },
] as const;

// ============================================================================
// Animation Classes
// ============================================================================

export const STEP_STATUS_CLASSES = {
	completed: 'bg-primary',
	running: 'bg-primary animate-pulse ring-2 ring-primary/50',
	pending: 'bg-muted',
	failed: 'bg-destructive',
	skipped: 'bg-muted/50',
} as const;

export const STEP_LINE_CLASSES = {
	completed: 'bg-primary',
	running: 'bg-primary/50',
	pending: 'bg-muted',
	failed: 'bg-destructive/50',
	skipped: 'bg-muted/30',
} as const;
