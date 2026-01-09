# Python Open3D Processing

> **Status:** 🔴 Not Started
> **Priority:** P1 (High)
> **Dependencies:** 01-foundation (FastAPI sidecar)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Implement Python-based point cloud processing using Open3D for geometry operations including loading, downsampling, filtering, and normal estimation. Provides FastAPI endpoints for the agent and frontend to invoke processing operations.

## Goals

- [ ] Load point clouds from PCD, PLY, LAS formats
- [ ] Voxel grid downsampling with configurable voxel size
- [ ] Random downsampling with target point count
- [ ] Statistical outlier removal
- [ ] Radius outlier removal
- [ ] Normal estimation with search radius
- [ ] FastAPI REST endpoints for all operations

## Technical Design

### Dependencies

```txt
# requirements.txt
open3d>=0.18.0
laspy>=2.5.0
numpy>=1.24.0
```

### PointCloudProcessor Class

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import open3d as o3d


@dataclass
class PointCloudStats:
    """Statistics about a point cloud."""
    point_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    has_colors: bool
    has_normals: bool
    has_intensity: bool


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    output_path: str
    original_count: int
    result_count: int
    operation: str
    parameters: dict


class PointCloudProcessor:
    """Open3D-based point cloud processing operations."""

    def load(self, path: str | Path) -> o3d.geometry.PointCloud:
        """Load point cloud from file."""
        path = Path(path)
        if path.suffix.lower() in ('.las', '.laz'):
            return self._load_las(path)
        return o3d.io.read_point_cloud(str(path))

    def save(self, pcd: o3d.geometry.PointCloud, path: str | Path) -> None:
        """Save point cloud to file."""
        o3d.io.write_point_cloud(str(path), pcd)

    def get_stats(self, pcd: o3d.geometry.PointCloud) -> PointCloudStats:
        """Get statistics about point cloud."""
        points = np.asarray(pcd.points)
        return PointCloudStats(
            point_count=len(points),
            bounds_min=tuple(points.min(axis=0)),
            bounds_max=tuple(points.max(axis=0)),
            has_colors=pcd.has_colors(),
            has_normals=pcd.has_normals(),
            has_intensity=False,  # Check custom attributes
        )

    def downsample_voxel(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float
    ) -> o3d.geometry.PointCloud:
        """Voxel grid downsampling."""
        return pcd.voxel_down_sample(voxel_size)

    def downsample_random(
        self,
        pcd: o3d.geometry.PointCloud,
        target_count: int
    ) -> o3d.geometry.PointCloud:
        """Random downsampling to target point count."""
        ratio = target_count / len(pcd.points)
        if ratio >= 1.0:
            return pcd
        return pcd.random_down_sample(ratio)

    def remove_statistical_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Remove statistical outliers."""
        return pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    def remove_radius_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        radius: float,
        min_neighbors: int = 2
    ) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Remove radius outliers."""
        return pcd.remove_radius_outlier(min_neighbors, radius)

    def estimate_normals(
        self,
        pcd: o3d.geometry.PointCloud,
        search_radius: float = 0.1,
        max_nn: int = 30
    ) -> o3d.geometry.PointCloud:
        """Estimate normals using hybrid search."""
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius, max_nn=max_nn
            )
        )
        return pcd

    def _load_las(self, path: Path) -> o3d.geometry.PointCloud:
        """Load LAS/LAZ file using laspy."""
        import laspy
        las = laspy.read(str(path))
        points = np.vstack([las.x, las.y, las.z]).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Add colors if available
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            colors = np.vstack([las.red, las.green, las.blue]).T / 65535.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd
```

### FastAPI Routes

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/pointcloud", tags=["pointcloud"])


class DownsampleRequest(BaseModel):
    input_path: str
    output_path: str
    method: str  # "voxel" | "random"
    voxel_size: float | None = None
    target_count: int | None = None


class FilterRequest(BaseModel):
    input_path: str
    output_path: str
    method: str  # "statistical" | "radius"
    nb_neighbors: int = 20
    std_ratio: float = 2.0
    radius: float | None = None
    min_neighbors: int = 2


@router.post("/downsample")
async def downsample(request: DownsampleRequest) -> ProcessingResult:
    """Downsample point cloud."""
    processor = PointCloudProcessor()
    pcd = processor.load(request.input_path)
    original_count = len(pcd.points)

    if request.method == "voxel":
        if request.voxel_size is None:
            raise HTTPException(400, "voxel_size required for voxel method")
        result = processor.downsample_voxel(pcd, request.voxel_size)
    elif request.method == "random":
        if request.target_count is None:
            raise HTTPException(400, "target_count required for random method")
        result = processor.downsample_random(pcd, request.target_count)
    else:
        raise HTTPException(400, f"Unknown method: {request.method}")

    processor.save(result, request.output_path)
    return ProcessingResult(
        output_path=request.output_path,
        original_count=original_count,
        result_count=len(result.points),
        operation="downsample",
        parameters={"method": request.method},
    )
```

## Implementation Tasks

- [ ] **Setup Open3D environment**
  - [ ] Add open3d, laspy to requirements.txt
  - [ ] Verify GPU acceleration availability
  - [ ] Create cv/pointcloud.py module

- [ ] **Implement PointCloudProcessor**
  - [ ] load() with format detection
  - [ ] save() with format inference from extension
  - [ ] get_stats() for metadata extraction

- [ ] **Implement downsampling**
  - [ ] Voxel grid downsampling
  - [ ] Random subsampling
  - [ ] Uniform downsampling (every nth point)

- [ ] **Implement filtering**
  - [ ] Statistical outlier removal
  - [ ] Radius outlier removal
  - [ ] Pass-through filter (crop by bounds)

- [ ] **Implement normal estimation**
  - [ ] Hybrid KDTree search
  - [ ] Orient normals consistently
  - [ ] Normal visualization support

- [ ] **FastAPI integration**
  - [ ] Create routes/pointcloud.py
  - [ ] Register router in main.py
  - [ ] Add request/response models

- [ ] **Testing**
  - [ ] Unit tests for each operation
  - [ ] Integration tests with sample data
  - [ ] Performance benchmarks

## Files to Create/Modify

| Path | Action | Purpose |
|------|--------|---------|
| `backend/requirements.txt` | Modify | Add open3d, laspy |
| `backend/cv/pointcloud.py` | Create | PointCloudProcessor class |
| `backend/cv/__init__.py` | Modify | Export pointcloud module |
| `backend/api/routes/pointcloud.py` | Create | FastAPI endpoints |
| `backend/api/main.py` | Modify | Register pointcloud router |
| `backend/tests/cv/test_pointcloud.py` | Create | Unit tests |
| `backend/tests/fixtures/sample.pcd` | Create | Test fixture |

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/pointcloud/stats` | Get point cloud metadata |
| POST | `/pointcloud/downsample` | Downsample point cloud |
| POST | `/pointcloud/filter` | Apply outlier filtering |
| POST | `/pointcloud/normals` | Estimate normals |
| POST | `/pointcloud/convert` | Convert between formats |

### Request/Response Models

```python
class StatsRequest(BaseModel):
    path: str

class StatsResponse(BaseModel):
    point_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    has_colors: bool
    has_normals: bool
    attributes: list[str]

class NormalsRequest(BaseModel):
    input_path: str
    output_path: str
    search_radius: float = 0.1
    max_nn: int = 30
```

## Acceptance Criteria

- [ ] Downsample 1M points to 100K in <1 second
- [ ] Statistical outlier removal identifies 5%+ outliers in noisy data
- [ ] Radius filtering removes isolated points correctly
- [ ] Normal vectors have unit magnitude (1.0 +/- 0.001)
- [ ] LAS/LAZ files load with correct coordinates
- [ ] All endpoints return proper error messages for invalid inputs
- [ ] `pytest tests/cv/test_pointcloud.py -v` passes
- [ ] `ruff check backend/cv/pointcloud.py` passes

## Testing Strategy

```python
import pytest
import numpy as np
from cv.pointcloud import PointCloudProcessor

@pytest.fixture
def processor():
    return PointCloudProcessor()

@pytest.fixture
def sample_pcd(tmp_path):
    """Create sample point cloud for testing."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    points = np.random.rand(10000, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    path = tmp_path / "sample.pcd"
    o3d.io.write_point_cloud(str(path), pcd)
    return path

def test_voxel_downsample(processor, sample_pcd):
    pcd = processor.load(sample_pcd)
    result = processor.downsample_voxel(pcd, voxel_size=0.1)
    assert len(result.points) < len(pcd.points)

def test_normal_estimation(processor, sample_pcd):
    pcd = processor.load(sample_pcd)
    result = processor.estimate_normals(pcd)
    assert result.has_normals()
    normals = np.asarray(result.normals)
    magnitudes = np.linalg.norm(normals, axis=1)
    assert np.allclose(magnitudes, 1.0, atol=0.001)
```

## Performance Considerations

- Open3D uses multi-threading for KDTree operations
- Consider chunked processing for >10M points
- GPU acceleration available for some operations (check `o3d.core.Device`)
- Cache loaded point clouds in memory for repeated operations
- Profile with `cProfile` for bottleneck identification

## Related Sub-Specs

- [01-rust-io.md](./01-rust-io.md) - Rust-side I/O alternative
- [03-rosbag-extraction.md](./03-rosbag-extraction.md) - ROS data source
- [07-anonymization-3d.md](./07-anonymization-3d.md) - Uses filtering operations
