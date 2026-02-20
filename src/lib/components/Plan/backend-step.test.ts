import { describe, expect, it } from "vitest";
import type { PipelineStep, StepType } from "$lib/types/pipeline";
import {
  createStepDraft,
  mapStepTypeToBackendTool,
  resolveBackendToolName,
} from "./backend-step";

function makeStep(overrides: Partial<PipelineStep> = {}): PipelineStep {
  return {
    id: "step-1",
    toolName: "detect",
    type: "detection",
    description: "Detect objects",
    config: { params: {} },
    status: "pending",
    order: 0,
    ...overrides,
  };
}

describe("plan backend step mapping", () => {
  it("maps each step type to a backend tool", () => {
    const expectations: Array<[StepType, string]> = [
      ["detection", "detect"],
      ["segmentation", "segment"],
      ["anonymization", "anonymize"],
      ["export", "export"],
      ["classification", "label_qa"],
      ["utility", "scan_directory"],
      ["custom", "custom_script"],
    ];

    for (const [stepType, expectedTool] of expectations) {
      expect(mapStepTypeToBackendTool(stepType)).toBe(expectedTool);
    }
  });

  it("keeps known backend tool names unchanged", () => {
    const existingStep = makeStep({
      toolName: "scan_directory",
      type: "custom",
    });

    expect(resolveBackendToolName(existingStep)).toBe("scan_directory");
  });

  it("normalizes UI aliases to backend tool names", () => {
    const uiAliasStep = makeStep({
      toolName: "segmentation",
      type: "segmentation",
    });

    expect(resolveBackendToolName(uiAliasStep)).toBe("segment");
  });

  it("creates detection draft with required defaults and project input path", () => {
    const draft = createStepDraft("detection", {
      existingSteps: [],
      currentProjectPath: "/data/images",
    });

    expect(draft.toolName).toBe("detect");
    expect(draft.config.model).toBe("sam3");
    expect(draft.config.params.input_path).toBe("/data/images");
    expect(draft.config.params.output_path).toBe("/data/images_detections_yolo");
    expect(draft.config.params.save_annotations).toBe(true);
  });

  it("uses SAM3 defaults for segmentation and anonymization drafts", () => {
    const segmentationDraft = createStepDraft("segmentation", {
      existingSteps: [],
      currentProjectPath: "/data/images",
    });
    const anonymizationDraft = createStepDraft("anonymization", {
      existingSteps: [],
      currentProjectPath: "/data/images",
    });

    expect(segmentationDraft.config.model).toBe("sam3");
    expect(anonymizationDraft.config.model).toBe("sam3");
  });

  it("creates export draft using prior detection output when available", () => {
    const detectStep = makeStep({
      id: "step-detect",
      toolName: "detect",
      type: "detection",
      order: 1,
      config: {
        params: {
          input_path: "/dataset/raw",
          output_path: "/dataset/raw_detections_yolo",
        },
      },
    });

    const draft = createStepDraft("export", {
      existingSteps: [detectStep],
      currentProjectPath: null,
    });

    expect(draft.toolName).toBe("export");
    expect(draft.config.params.source_path).toBe("/dataset/raw_detections_yolo");
    expect(draft.config.params.output_path).toBe(
      "/dataset/raw_detections_yolo_export",
    );
    expect(draft.config.params.output_format).toBe("yolo");
  });
});
