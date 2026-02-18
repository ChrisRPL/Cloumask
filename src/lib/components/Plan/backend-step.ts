import type { PipelineStep, StepType } from "$lib/types/pipeline";
import { getDefaultConfig } from "./constants";

const FALLBACK_DATA_PATH = "/tmp";

const BACKEND_TOOL_NAMES = new Set([
  "scan_directory",
  "detect",
  "segment",
  "anonymize",
  "export",
  "convert_format",
  "find_duplicates",
  "label_qa",
  "split_dataset",
  "review",
  "custom_script",
  "pointcloud_stats",
  "process_pointcloud",
  "detect_3d",
  "project_3d_to_2d",
  "anonymize_pointcloud",
  "extract_rosbag",
]);

const STEP_TYPE_TOOL_NAME: Record<StepType, string> = {
  detection: "detect",
  segmentation: "segment",
  anonymization: "anonymize",
  export: "export",
  classification: "label_qa",
  utility: "scan_directory",
  custom: "custom_script",
};

const STEP_TYPE_DESCRIPTION: Record<StepType, string> = {
  detection: "Detect objects in the dataset",
  segmentation: "Segment objects in the dataset",
  anonymization: "Anonymize sensitive regions in images",
  export: "Export annotations to target format",
  classification: "Run label QA checks",
  utility: "Run utility/data-management step",
  custom: "Run custom processing script",
};

function readString(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function trimTrailingSlashes(value: string): string {
  return value.replace(/[\/\\]+$/, "");
}

function findMostRecentPath(
  steps: PipelineStep[],
  keys: string[],
  predicate?: (step: PipelineStep) => boolean,
): string | null {
  const ordered = [...steps].sort((a, b) => b.order - a.order);
  for (const step of ordered) {
    if (predicate && !predicate(step)) continue;
    const params = step.config?.params ?? {};
    for (const key of keys) {
      const candidate = readString(params[key]);
      if (candidate) return candidate;
    }
  }
  return null;
}

function inferBasePath(
  existingSteps: PipelineStep[],
  currentProjectPath: string | null,
): string {
  const projectPath = readString(currentProjectPath);
  if (projectPath) return projectPath;

  const inferred = findMostRecentPath(existingSteps, [
    "input_path",
    "source_path",
    "path",
    "image_dir",
  ]);
  if (inferred) return inferred;

  return FALLBACK_DATA_PATH;
}

function inferDetectOutputPath(existingSteps: PipelineStep[], basePath: string): string {
  const existing = findMostRecentPath(
    existingSteps,
    ["output_path"],
    (step) => resolveBackendToolName(step) === "detect",
  );
  if (existing) return existing;

  const sanitized = trimTrailingSlashes(basePath) || basePath;
  return `${sanitized}_detections_yolo`;
}

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function stringOrUndefined(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function buildRequiredParams(
  type: StepType,
  options: { existingSteps: PipelineStep[]; currentProjectPath: string | null },
): Record<string, unknown> {
  const { existingSteps, currentProjectPath } = options;
  const basePath = inferBasePath(existingSteps, currentProjectPath);
  const detectOutputPath = inferDetectOutputPath(existingSteps, basePath);
  const sanitizedBasePath = trimTrailingSlashes(basePath) || basePath;

  switch (type) {
    case "detection":
      return {
        input_path: basePath,
        classes: [],
        confidence: 0.5,
        save_annotations: true,
        output_path: detectOutputPath,
      };

    case "segmentation":
      return {
        input_path: basePath,
        prompt: "object",
        confidence: 0.25,
      };

    case "anonymization":
      return {
        input_path: basePath,
        output_path: `${sanitizedBasePath}_anonymized`,
        target: "all",
        mode: "blur",
        blur_strength: 5,
      };

    case "export":
      return {
        source_path: detectOutputPath,
        output_path: `${trimTrailingSlashes(detectOutputPath) || detectOutputPath}_export`,
        output_format: "yolo",
      };

    case "classification":
      return {
        path: detectOutputPath,
        generate_report: true,
      };

    case "utility":
      return {
        path: basePath,
        recursive: true,
      };

    case "custom":
      return {
        script: "",
        input_path: basePath,
        output_path: `${sanitizedBasePath}_custom_output`,
      };
  }
}

export function mapStepTypeToBackendTool(type: StepType): string {
  return STEP_TYPE_TOOL_NAME[type];
}

export function resolveBackendToolName(
  step: Pick<PipelineStep, "toolName" | "type">,
): string {
  const normalizedToolName = step.toolName.trim().toLowerCase();
  if (BACKEND_TOOL_NAMES.has(normalizedToolName)) {
    return normalizedToolName;
  }
  return mapStepTypeToBackendTool(step.type);
}

export function createStepDraft(
  type: StepType,
  options: {
    existingSteps: PipelineStep[];
    currentProjectPath: string | null;
  },
): Omit<PipelineStep, "id" | "order" | "status"> {
  const defaults = getDefaultConfig(type);
  const requiredParams = buildRequiredParams(type, options);
  const params = { ...defaults, ...requiredParams };

  return {
    toolName: mapStepTypeToBackendTool(type),
    type,
    description: STEP_TYPE_DESCRIPTION[type],
    config: {
      model: stringOrUndefined(params.model),
      confidence: numberOrUndefined(params.confidence),
      params,
    },
    critical: false,
  };
}
