"""
3D point cloud anonymization pipeline.

Removes or noises points belonging to faces detected via multi-view
projection. The pipeline:
1. Generates virtual cameras around the scene
2. Renders depth images from each viewpoint
3. Runs face detection (SCRFD / YuNet) on each view
4. Lifts detected face regions back to 3D point indices
5. Applies anonymization (remove / noise) to those points
6. Optionally verifies no faces remain in the output

Implements spec: 05-point-cloud/07-anonymization-3d

Example:
    from backend.cv.anonymization_3d import PointCloudAnonymizer

    anonymizer = PointCloudAnonymizer()
    anonymizer.load("cuda")
    result = anonymizer.anonymize("scene.pcd", "scene_anon.pcd")
    anonymizer.unload()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from backend.cv.face_3d_projection import (
    find_points_in_2d_box,
    project_points_to_camera,
    render_depth_image,
)
from backend.cv.virtual_camera import VirtualCamera, generate_virtual_cameras

if TYPE_CHECKING:
    import open3d as o3d

    from backend.cv.base import BaseModelWrapper
    from backend.cv.types import FaceDetectionResult

logger = logging.getLogger(__name__)

# Anonymization mode type
AnonymizationMode3D = Literal["remove", "noise"]


@dataclass
class Anonymization3DResult:
    """
    Result of a 3D point cloud anonymization operation.

    Attributes:
        output_path: Path to the anonymized point cloud.
        original_point_count: Points in the input cloud.
        anonymized_point_count: Points in the output cloud.
        face_regions_found: Total face detections across all views.
        points_removed: Points removed (mode="remove").
        points_noised: Points with added noise (mode="noise").
        verification_passed: Whether re-detection found zero faces.
        processing_time_ms: Total wall-clock time in milliseconds.
        views_processed: Number of virtual camera views evaluated.
        mode: Anonymization mode used.
    """

    output_path: str
    original_point_count: int
    anonymized_point_count: int
    face_regions_found: int
    points_removed: int
    points_noised: int
    verification_passed: bool
    processing_time_ms: float
    views_processed: int = 0
    mode: str = "remove"


class PointCloudAnonymizer:
    """
    Anonymize faces in 3D point clouds via multi-view projection.

    Uses virtual cameras to render depth images, detects faces with
    SCRFD (GPU) or YuNet (CPU), then maps detections back to 3D
    points for removal or noise injection.

    Example:
        anonymizer = PointCloudAnonymizer()
        anonymizer.load("cuda")
        result = anonymizer.anonymize("input.pcd", "output.pcd")
        print(f"Removed {result.points_removed} face points")
        anonymizer.unload()
    """

    def __init__(self) -> None:
        """Initialise the anonymizer (models are loaded lazily via load())."""
        self._face_detector: BaseModelWrapper[FaceDetectionResult] | None = None
        self._device: str = "cpu"
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Whether the face detector is loaded."""
        return self._is_loaded

    def load(self, device: str = "auto") -> None:
        """
        Load face detection model for depth-image inference.

        Args:
            device: Target device ("cuda", "cpu", "mps", or "auto").
        """
        if self._is_loaded:
            logger.warning("PointCloudAnonymizer already loaded, skipping")
            return

        from backend.cv.device import select_device
        from backend.cv.faces import get_face_detector

        if device == "auto":
            device = select_device()

        self._device = device
        self._face_detector = get_face_detector(realtime=False)

        try:
            self._face_detector.load(device)
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                logger.warning("GPU OOM loading face detector, falling back to CPU")
                self._face_detector.load("cpu")
                self._device = "cpu"
            else:
                raise

        self._is_loaded = True
        logger.info("PointCloudAnonymizer loaded on %s", self._device)

    def unload(self) -> None:
        """Unload the face detector and free resources."""
        if self._face_detector is not None:
            self._face_detector.unload()
            self._face_detector = None
        self._is_loaded = False
        logger.info("PointCloudAnonymizer unloaded")

    def anonymize(
        self,
        pcd_path: str | Path,
        output_path: str | Path,
        *,
        mode: AnonymizationMode3D = "remove",
        num_views: int = 8,
        face_confidence: float = 0.4,
        face_margin: float = 1.2,
        noise_sigma: float = 0.1,
        verify: bool = True,
        resolution: tuple[int, int] = (640, 480),
    ) -> Anonymization3DResult:
        """
        Anonymize faces in a point cloud.

        Args:
            pcd_path: Path to the input point cloud file.
            output_path: Path for the anonymized output.
            mode: "remove" deletes face points; "noise" adds Gaussian noise.
            num_views: Number of virtual camera viewpoints.
            face_confidence: Minimum face detection confidence.
            face_margin: Factor to expand face bounding boxes (1.2 = 20%).
            noise_sigma: Standard deviation of noise for "noise" mode (metres).
            verify: Re-run detection on output to confirm removal.
            resolution: Virtual camera image resolution (width, height).

        Returns:
            Anonymization3DResult with statistics.

        Raises:
            RuntimeError: If the anonymizer is not loaded.
            FileNotFoundError: If the input file does not exist.
        """
        if not self._is_loaded or self._face_detector is None:
            raise RuntimeError("PointCloudAnonymizer not loaded. Call load() first.")

        import open3d as o3d

        start_time = time.perf_counter()

        pcd_path = Path(pcd_path)
        output_path = Path(output_path)

        if not pcd_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {pcd_path}")

        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
        original_count = len(points)

        if original_count == 0:
            # Nothing to anonymize
            o3d.io.write_point_cloud(str(output_path), pcd)
            return Anonymization3DResult(
                output_path=str(output_path),
                original_point_count=0,
                anonymized_point_count=0,
                face_regions_found=0,
                points_removed=0,
                points_noised=0,
                verification_passed=True,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                views_processed=0,
                mode=mode,
            )

        # Compute bounds
        bounds = (points.min(axis=0), points.max(axis=0))

        # Generate virtual cameras
        cameras = generate_virtual_cameras(bounds, num_views, resolution)

        # Detect faces in each view and collect point indices
        face_point_indices: set[int] = set()
        face_regions_found = 0

        for cam_idx, cam in enumerate(cameras):
            indices, det_count = self._detect_faces_in_view(
                points, cam, face_confidence, face_margin
            )
            if len(indices) > 0:
                logger.debug(
                    "View %d: found points for face regions (%d pts)",
                    cam_idx,
                    len(indices),
                )
            face_point_indices.update(indices)
            face_regions_found += det_count

        # Apply anonymization
        face_indices_arr = np.array(sorted(face_point_indices), dtype=np.intp)
        new_pcd, points_removed, points_noised = self._apply_anonymization(
            pcd, points, face_indices_arr, mode, noise_sigma
        )

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_path), new_pcd)

        # Verification pass
        verification_passed = True
        if verify and face_regions_found > 0:
            verification_passed = self.verify(new_pcd, num_views, resolution, face_confidence)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        result = Anonymization3DResult(
            output_path=str(output_path),
            original_point_count=original_count,
            anonymized_point_count=len(new_pcd.points),
            face_regions_found=face_regions_found,
            points_removed=points_removed,
            points_noised=points_noised,
            verification_passed=verification_passed,
            processing_time_ms=processing_time_ms,
            views_processed=num_views,
            mode=mode,
        )

        logger.info(
            "3D anonymization complete: %d face regions, %d pts removed, "
            "%d pts noised, verification=%s, %.0fms",
            result.face_regions_found,
            result.points_removed,
            result.points_noised,
            result.verification_passed,
            result.processing_time_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_faces_in_view(
        self,
        points: np.ndarray,
        camera: VirtualCamera,
        confidence: float,
        margin: float,
    ) -> tuple[set[int], int]:
        """
        Detect faces in a single virtual camera view and return 3D point indices.

        Renders a depth image, writes it to a temporary file for the face
        detector, then maps detected 2D boxes back to 3D point indices.

        Returns:
            (point_indices, detection_count) tuple.
        """
        import tempfile

        import cv2

        assert self._face_detector is not None

        # Render depth image
        depth_image = render_depth_image(points, camera, point_radius=2)

        # Skip empty views
        if depth_image.max() == 0:
            return set(), 0

        # Write temp image for face detector (expects a file path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, depth_image)

        try:
            result = self._face_detector.predict(
                tmp_path, confidence=confidence
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        detection_count = len(result.faces)

        if not result.faces:
            return set(), 0

        # Project all points to this camera
        points_2d, _depths, valid = project_points_to_camera(points, camera)

        # Collect point indices for each face detection
        W, H = camera.resolution
        collected: set[int] = set()

        for face in result.faces:
            # Convert normalised bbox to pixel coords
            bbox = face.bbox
            x_min = (bbox.x - bbox.width / 2) * W
            y_min = (bbox.y - bbox.height / 2) * H
            x_max = (bbox.x + bbox.width / 2) * W
            y_max = (bbox.y + bbox.height / 2) * H

            indices = find_points_in_2d_box(
                points_2d, valid, (x_min, y_min, x_max, y_max), margin=margin
            )
            collected.update(indices.tolist())

        return collected, detection_count

    @staticmethod
    def _apply_anonymization(
        pcd: o3d.geometry.PointCloud,
        points: np.ndarray,
        face_indices: np.ndarray,
        mode: AnonymizationMode3D,
        noise_sigma: float,
    ) -> tuple[o3d.geometry.PointCloud, int, int]:
        """
        Apply remove or noise anonymization to face points.

        Returns:
            (new_pcd, points_removed, points_noised)
        """
        import open3d as o3d

        if len(face_indices) == 0:
            return pcd, 0, 0

        has_colors = pcd.has_colors()
        colors = np.asarray(pcd.colors) if has_colors else None

        if mode == "remove":
            mask = np.ones(len(points), dtype=bool)
            mask[face_indices] = False
            new_points = points[mask]
            new_colors = colors[mask] if colors is not None else None
            points_removed = int(len(face_indices))
            points_noised = 0

        elif mode == "noise":
            new_points = points.copy()
            noise = np.random.default_rng().normal(
                0, noise_sigma, (len(face_indices), 3)
            )
            new_points[face_indices] += noise
            new_colors = colors.copy() if colors is not None else None
            points_removed = 0
            points_noised = int(len(face_indices))

        else:
            raise ValueError(f"Unknown anonymization mode: {mode}")

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_points)
        if new_colors is not None:
            new_pcd.colors = o3d.utility.Vector3dVector(new_colors)

        # Preserve normals if present
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            if mode == "remove":
                mask = np.ones(len(points), dtype=bool)
                mask[face_indices] = False
                new_pcd.normals = o3d.utility.Vector3dVector(normals[mask])
            else:
                new_pcd.normals = o3d.utility.Vector3dVector(normals.copy())

        return new_pcd, points_removed, points_noised

    def verify(
        self,
        pcd: o3d.geometry.PointCloud,
        num_views: int = 8,
        resolution: tuple[int, int] = (640, 480),
        confidence: float = 0.4,
    ) -> bool:
        """
        Verify that a point cloud contains no detectable faces.

        Re-projects the cloud to virtual camera views and runs face
        detection on each rendered depth image.

        Args:
            pcd: Open3D PointCloud to verify.
            num_views: Number of virtual camera viewpoints.
            resolution: Virtual camera image resolution (width, height).
            confidence: Face detection confidence threshold.

        Returns:
            True if no faces are detected across all views.

        Raises:
            RuntimeError: If the anonymizer is not loaded.
        """
        if not self._is_loaded or self._face_detector is None:
            raise RuntimeError("PointCloudAnonymizer not loaded. Call load() first.")
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return True

        bounds = (points.min(axis=0), points.max(axis=0))
        cameras = generate_virtual_cameras(bounds, num_views, resolution)

        for cam in cameras:
            depth_image = render_depth_image(points, camera=cam, point_radius=2)
            if depth_image.max() == 0:
                continue

            import tempfile

            import cv2

            assert self._face_detector is not None

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, depth_image)

            try:
                result = self._face_detector.predict(tmp_path, confidence=confidence)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            if result.faces:
                logger.warning(
                    "Verification failed: %d faces still detected", len(result.faces)
                )
                return False

        return True
