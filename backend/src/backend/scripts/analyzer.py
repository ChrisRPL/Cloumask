"""
Script behavior analyzer.

Extracts structured behavior from Python code using AST analysis
combined with LLM for natural language descriptions.
"""

import ast
import re
from dataclasses import dataclass

from backend.scripts.behavior import (
    BehaviorInput,
    BehaviorOutput,
    ResourceUsage,
    ScriptBehavior,
)


@dataclass
class ASTAnalysisResult:
    """Results from AST analysis of a script."""

    imports: list[str]
    file_extensions_read: set[str]
    file_extensions_written: set[str]
    cv_operations: list[str]
    has_loops: bool
    has_gpu_calls: bool
    function_calls: list[str]


class ScriptBehaviorAnalyzer:
    """
    Analyzes Python scripts to extract structured behavior.

    Uses AST analysis for deterministic extraction of:
    - File types read/written
    - CV operations used
    - Resource requirements

    Optionally uses LLM for natural language descriptions.
    """

    # Common CV library patterns
    CV_LIBRARIES = {
        "cv2": "OpenCV",
        "opencv": "OpenCV",
        "PIL": "Pillow",
        "pillow": "Pillow",
        "skimage": "scikit-image",
        "scipy.ndimage": "SciPy",
        "torch": "PyTorch",
        "torchvision": "PyTorch Vision",
        "tensorflow": "TensorFlow",
        "numpy": "NumPy",
    }

    # File extension patterns
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ANNOTATION_EXTENSIONS = {".json", ".xml", ".txt", ".csv"}
    POINTCLOUD_EXTENSIONS = {".pcd", ".ply", ".las", ".laz", ".xyz"}

    # CV operation patterns (function names -> descriptions)
    CV_OPERATION_PATTERNS = {
        r"resize": "Resize images",
        r"crop": "Crop images",
        r"rotate": "Rotate images",
        r"flip": "Flip images",
        r"blur|gaussian|median": "Apply blur/smoothing",
        r"threshold|binarize": "Apply thresholding",
        r"edge|canny|sobel": "Detect edges",
        r"contour": "Find contours",
        r"detect|yolo|rcnn": "Object detection",
        r"segment|mask|sam": "Segmentation",
        r"face": "Face detection/processing",
        r"ocr|text": "Text recognition (OCR)",
        r"transform|warp|perspective": "Geometric transformation",
        r"color|rgb|bgr|hsv|gray": "Color space conversion",
        r"histogram|equalize": "Histogram processing",
        r"morpholog|dilate|erode": "Morphological operations",
        r"denoise|noise": "Noise reduction",
        r"sharpen": "Sharpening",
        r"augment": "Data augmentation",
    }

    def __init__(self, use_llm: bool = False):
        """
        Initialize the analyzer.

        Args:
            use_llm: Whether to use LLM for natural language descriptions.
                     If False, uses pattern-based descriptions only.
        """
        self.use_llm = use_llm

    def analyze(self, code: str, prompt: str | None = None) -> ScriptBehavior:
        """
        Analyze a script and extract its behavior.

        Args:
            code: Python source code to analyze.
            prompt: Optional original prompt used to generate the script.

        Returns:
            ScriptBehavior with inputs, outputs, operations, etc.
        """
        # AST analysis for deterministic extraction
        ast_result = self._analyze_ast(code)

        # Build behavior from AST analysis
        behavior = self._build_behavior_from_ast(ast_result, code)

        # TODO: Optionally enhance with LLM descriptions
        # if self.use_llm and prompt:
        #     behavior = await self._enhance_with_llm(behavior, code, prompt)

        return behavior

    def _analyze_ast(self, code: str) -> ASTAnalysisResult:
        """Analyze code using Python AST."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Return empty result for invalid code
            return ASTAnalysisResult(
                imports=[],
                file_extensions_read=set(),
                file_extensions_written=set(),
                cv_operations=[],
                has_loops=False,
                has_gpu_calls=False,
                function_calls=[],
            )

        imports: list[str] = []
        file_extensions_read: set[str] = set()
        file_extensions_written: set[str] = set()
        function_calls: list[str] = []
        has_loops = False
        has_gpu_calls = False

        for node in ast.walk(tree):
            # Collect imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

            # Detect loops
            if isinstance(node, (ast.For, ast.While)):
                has_loops = True

            # Collect function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    function_calls.append(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    function_calls.append(node.func.id)

            # Look for file operations with extensions
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self._extract_file_extensions(
                    node.value, file_extensions_read, file_extensions_written
                )

        # Detect GPU usage
        gpu_patterns = ["cuda", "gpu", "device", "to('cuda", 'to("cuda']
        code_lower = code.lower()
        has_gpu_calls = any(pattern in code_lower for pattern in gpu_patterns)

        # Extract CV operations from function calls
        cv_operations = self._extract_cv_operations(function_calls, code)

        return ASTAnalysisResult(
            imports=imports,
            file_extensions_read=file_extensions_read,
            file_extensions_written=file_extensions_written,
            cv_operations=cv_operations,
            has_loops=has_loops,
            has_gpu_calls=has_gpu_calls,
            function_calls=function_calls,
        )

    def _extract_file_extensions(
        self,
        value: str,
        read_set: set[str],
        write_set: set[str],
    ) -> None:
        """Extract file extensions from string constants."""
        # Look for file extension patterns
        ext_pattern = r"\.[a-zA-Z0-9]{2,4}(?=['\"\s\)]|$)"
        matches = re.findall(ext_pattern, value)

        for ext in matches:
            ext_lower = ext.lower()
            # Heuristic: if in common read context, add to read set
            # This is simplified - real analysis would track variable flow
            if ext_lower in self.IMAGE_EXTENSIONS | self.VIDEO_EXTENSIONS:
                read_set.add(ext_lower)
            elif ext_lower in self.ANNOTATION_EXTENSIONS:
                # Annotations could be read or written
                read_set.add(ext_lower)
                write_set.add(ext_lower)

    def _extract_cv_operations(
        self, function_calls: list[str], code: str
    ) -> list[str]:
        """Extract CV operations from function calls and code patterns."""
        operations: list[str] = []
        code_lower = code.lower()

        for pattern, description in self.CV_OPERATION_PATTERNS.items():
            if re.search(pattern, code_lower) and description not in operations:
                operations.append(description)

        return operations

    def _build_behavior_from_ast(
        self, ast_result: ASTAnalysisResult, code: str
    ) -> ScriptBehavior:
        """Build ScriptBehavior from AST analysis results."""
        # Determine inputs
        inputs: list[BehaviorInput] = []

        if ast_result.file_extensions_read & self.IMAGE_EXTENSIONS:
            inputs.append(
                BehaviorInput(
                    name="Images",
                    types=sorted(ast_result.file_extensions_read & self.IMAGE_EXTENSIONS),
                    description="Input images to process",
                )
            )

        if ast_result.file_extensions_read & self.VIDEO_EXTENSIONS:
            inputs.append(
                BehaviorInput(
                    name="Videos",
                    types=sorted(ast_result.file_extensions_read & self.VIDEO_EXTENSIONS),
                    description="Input videos to process",
                )
            )

        if ast_result.file_extensions_read & self.ANNOTATION_EXTENSIONS:
            inputs.append(
                BehaviorInput(
                    name="Annotations",
                    types=sorted(ast_result.file_extensions_read & self.ANNOTATION_EXTENSIONS),
                    description="Existing annotations",
                    required=False,
                )
            )

        # Default to images if no specific types detected
        if not inputs:
            inputs.append(
                BehaviorInput(
                    name="Files",
                    types=[".jpg", ".png"],
                    description="Input files to process",
                )
            )

        # Determine outputs
        outputs: list[BehaviorOutput] = []

        if ast_result.file_extensions_written & self.IMAGE_EXTENSIONS:
            outputs.append(
                BehaviorOutput(
                    name="Processed Images",
                    types=sorted(ast_result.file_extensions_written & self.IMAGE_EXTENSIONS),
                    description="Output images",
                )
            )

        if ast_result.file_extensions_written & self.ANNOTATION_EXTENSIONS:
            outputs.append(
                BehaviorOutput(
                    name="Annotations",
                    types=sorted(ast_result.file_extensions_written & self.ANNOTATION_EXTENSIONS),
                    description="Generated annotations",
                )
            )

        # Default output
        if not outputs:
            outputs.append(
                BehaviorOutput(
                    name="Results",
                    types=[".json"],
                    description="Processing results",
                )
            )

        # Determine resource usage
        cpu_level = "high" if ast_result.has_loops else "medium"
        if any(lib in str(ast_result.imports) for lib in ["torch", "tensorflow"]):
            cpu_level = "high"

        resource_usage = ResourceUsage(
            cpu=cpu_level,  # type: ignore
            memory="medium" if ast_result.has_loops else "low",  # type: ignore
            gpu=ast_result.has_gpu_calls,
        )

        # Estimate time
        estimated_time = None
        if ast_result.has_loops:
            estimated_time = "~1-5s per file"
        elif ast_result.has_gpu_calls:
            estimated_time = "~0.5-2s per file (GPU)"

        # Warnings
        warnings: list[str] = []
        if ast_result.has_gpu_calls:
            warnings.append("Requires GPU for optimal performance")
        if not ast_result.cv_operations:
            warnings.append("No CV operations detected - verify script behavior")

        return ScriptBehavior(
            inputs=inputs,
            outputs=outputs,
            operations=ast_result.cv_operations or ["Process files"],
            warnings=warnings,
            estimated_time=estimated_time,
            resource_usage=resource_usage,
        )


# Convenience function
def analyze_script(code: str, prompt: str | None = None) -> ScriptBehavior:
    """
    Analyze a script and extract its behavior.

    Convenience function that creates an analyzer and runs analysis.

    Args:
        code: Python source code.
        prompt: Optional original prompt.

    Returns:
        ScriptBehavior describing what the script does.
    """
    analyzer = ScriptBehaviorAnalyzer()
    return analyzer.analyze(code, prompt)
