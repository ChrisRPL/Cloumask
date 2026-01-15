/**
 * Plan Editor constants and step type configurations.
 *
 * Defines metadata for each step type including icons, labels,
 * terminal prefixes, and configuration schemas for the UI.
 */

import type { StepType } from "$lib/types/pipeline";
import type {
  StepTypeConfig,
  ConfigFieldSchema,
  TimeFactors,
} from "$lib/types/plan-editor";

// ============================================================================
// Step Type Configurations
// ============================================================================

export const STEP_TYPE_CONFIGS: Record<StepType, StepTypeConfig> = {
  detection: {
    type: "detection",
    label: "Detection",
    icon: "Search",
    prefix: "[D]",
    configSchema: [
      {
        key: "model",
        label: "Model",
        type: "select",
        options: [
          { value: "yolo11m", label: "YOLO11m (fast)" },
          { value: "yolo-world", label: "YOLO-World (open-vocab)" },
          { value: "rt-detr", label: "RT-DETR (accurate)" },
          { value: "scrfd", label: "SCRFD (faces)" },
        ],
        default: "yolo11m",
        description: "Detection model to use",
      },
      {
        key: "confidence",
        label: "Confidence",
        type: "slider",
        min: 0.1,
        max: 1.0,
        step: 0.05,
        default: 0.5,
        description: "Minimum confidence threshold",
      },
      {
        key: "classes",
        label: "Classes",
        type: "text",
        default: "",
        description: "Comma-separated class names to detect",
      },
    ],
  },
  segmentation: {
    type: "segmentation",
    label: "Segmentation",
    icon: "Scissors",
    prefix: "[S]",
    configSchema: [
      {
        key: "model",
        label: "Model",
        type: "select",
        options: [
          { value: "sam2", label: "SAM2 (point prompt)" },
          { value: "sam3", label: "SAM3 (text prompt)" },
          { value: "mobile-sam", label: "MobileSAM (fast)" },
        ],
        default: "sam2",
        description: "Segmentation model to use",
      },
      {
        key: "prompt_type",
        label: "Prompt Type",
        type: "select",
        options: [
          { value: "auto", label: "Auto (from detection)" },
          { value: "point", label: "Point" },
          { value: "box", label: "Bounding Box" },
          { value: "text", label: "Text" },
        ],
        default: "auto",
        description: "How to prompt the segmentation model",
      },
      {
        key: "mask_threshold",
        label: "Mask Threshold",
        type: "slider",
        min: 0.0,
        max: 1.0,
        step: 0.05,
        default: 0.5,
        description: "Threshold for mask generation",
      },
    ],
  },
  anonymization: {
    type: "anonymization",
    label: "Anonymization",
    icon: "EyeOff",
    prefix: "[A]",
    configSchema: [
      {
        key: "method",
        label: "Method",
        type: "select",
        options: [
          { value: "blur", label: "Gaussian Blur" },
          { value: "pixelate", label: "Pixelate" },
          { value: "mask", label: "Solid Mask" },
          { value: "inpaint", label: "Inpaint" },
        ],
        default: "blur",
        description: "Anonymization method",
      },
      {
        key: "intensity",
        label: "Intensity",
        type: "slider",
        min: 1,
        max: 10,
        step: 1,
        default: 5,
        description: "Blur/pixelation intensity",
      },
      {
        key: "fill_color",
        label: "Fill Color",
        type: "text",
        default: "#000000",
        description: "Color for solid mask (hex)",
      },
    ],
  },
  export: {
    type: "export",
    label: "Export",
    icon: "Download",
    prefix: "[E]",
    configSchema: [
      {
        key: "format",
        label: "Format",
        type: "select",
        options: [
          { value: "coco", label: "COCO JSON" },
          { value: "yolo", label: "YOLO TXT" },
          { value: "pascal", label: "Pascal VOC" },
          { value: "csv", label: "CSV" },
        ],
        default: "coco",
        description: "Export annotation format",
      },
      {
        key: "include_images",
        label: "Include Images",
        type: "checkbox",
        default: true,
        description: "Copy images to output directory",
      },
      {
        key: "zip",
        label: "Create ZIP",
        type: "checkbox",
        default: false,
        description: "Package output as ZIP archive",
      },
    ],
  },
  classification: {
    type: "classification",
    label: "Classification",
    icon: "Tag",
    prefix: "[C]",
    configSchema: [
      {
        key: "model",
        label: "Model",
        type: "select",
        options: [
          { value: "clip", label: "CLIP" },
          { value: "resnet", label: "ResNet" },
        ],
        default: "clip",
        description: "Classification model",
      },
      {
        key: "labels",
        label: "Labels",
        type: "text",
        default: "",
        description: "Comma-separated classification labels",
      },
      {
        key: "top_k",
        label: "Top K",
        type: "number",
        min: 1,
        max: 10,
        step: 1,
        default: 3,
        description: "Number of top predictions to return",
      },
    ],
  },
  custom: {
    type: "custom",
    label: "Custom",
    icon: "Wand2",
    prefix: "[*]",
    configSchema: [
      {
        key: "script",
        label: "Script Path",
        type: "text",
        default: "",
        description: "Path to custom processing script",
      },
    ],
  },
};

// ============================================================================
// Status Display
// ============================================================================

export const STATUS_LABELS: Record<string, string> = {
  pending: "[pending]",
  running: "[running]",
  completed: "[done]",
  failed: "[failed]",
  skipped: "[skip]",
};

export const STATUS_COLORS: Record<string, string> = {
  pending: "text-muted-foreground",
  running: "text-forest-light",
  completed: "text-green-600",
  failed: "text-destructive",
  skipped: "text-muted-foreground/50",
};

// ============================================================================
// Time Estimation Factors
// ============================================================================

export const TIME_FACTORS: Record<StepType, TimeFactors> = {
  detection: { baseMs: 50, modelMultiplier: 1.0, gpuSpeedup: 10 },
  segmentation: { baseMs: 200, modelMultiplier: 1.5, gpuSpeedup: 15 },
  anonymization: { baseMs: 20, modelMultiplier: 1.0, gpuSpeedup: 5 },
  classification: { baseMs: 30, modelMultiplier: 1.0, gpuSpeedup: 8 },
  export: { baseMs: 10, modelMultiplier: 1.0, gpuSpeedup: 1 },
  custom: { baseMs: 100, modelMultiplier: 1.0, gpuSpeedup: 1 },
};

// ============================================================================
// Helpers
// ============================================================================

/**
 * Get configuration schema for a step type.
 */
export function getStepSchema(type: StepType): ConfigFieldSchema[] {
  return STEP_TYPE_CONFIGS[type]?.configSchema ?? [];
}

/**
 * Get default configuration values for a step type.
 */
export function getDefaultConfig(type: StepType): Record<string, unknown> {
  const schema = getStepSchema(type);
  const config: Record<string, unknown> = {};
  for (const field of schema) {
    config[field.key] = field.default;
  }
  return config;
}

/**
 * Estimate pipeline execution time in milliseconds.
 */
export function estimatePipelineTime(
  steps: Array<{ type: StepType; status?: string }>,
  itemCount: number,
  hasGpu: boolean
): number {
  return steps
    .filter((s) => s.status !== "skipped")
    .reduce((total, step) => {
      const factors = TIME_FACTORS[step.type];
      if (!factors) return total;
      const stepTime = factors.baseMs * factors.modelMultiplier;
      const adjustedTime = hasGpu ? stepTime / factors.gpuSpeedup : stepTime;
      return total + adjustedTime * itemCount;
    }, 0);
}

/**
 * Format milliseconds to human-readable duration.
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return "<1s";
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (minutes < 60) {
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
}
