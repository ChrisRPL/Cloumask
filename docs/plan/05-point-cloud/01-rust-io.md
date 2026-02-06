# Rust Point Cloud I/O

> **Status:** 🟢 Complete
> **Priority:** P1 (High)
> **Dependencies:** 01-foundation (Tauri IPC)
> **Parent:** [SPEC.md](./SPEC.md)

## Overview

Implement Rust-based point cloud I/O using the `pasture` crate for high-performance reading, writing, and streaming of PCD, PLY, and LAS/LAZ formats. Provides Tauri IPC commands for frontend integration and streaming support for large files.

## Goals

- [ ] Read PCD files (ASCII and binary)
- [ ] Read PLY files (ASCII and binary)
- [ ] Read LAS/LAZ files (compressed and uncompressed)
- [ ] Stream large point clouds in chunks to frontend
- [ ] Convert between formats preserving all attributes
- [ ] Implement Rust-side voxel decimation

## Technical Design

### pasture Integration

```toml
# Cargo.toml
[dependencies]
pasture-core = "0.4"
pasture-io = { version = "0.4", features = ["las", "ply", "pcd"] }
```

### Point Cloud Data Structure

```rust
use pasture_core::containers::BorrowedBuffer;
use pasture_core::layout::PointLayout;

/// Unified point cloud representation for IPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloudData {
    /// Number of points
    pub count: usize,
    /// Point positions as flat [x0, y0, z0, x1, y1, z1, ...] array
    pub positions: Vec<f64>,
    /// Optional intensity values
    pub intensity: Option<Vec<f32>>,
    /// Optional RGB colors as [r0, g0, b0, r1, g1, b1, ...]
    pub colors: Option<Vec<u8>>,
    /// Bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
    pub bounds: [f64; 6],
    /// Point attributes present in the file
    pub attributes: Vec<String>,
}

/// Metadata without loading full point data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloudMetadata {
    pub path: String,
    pub format: String,
    pub point_count: usize,
    pub bounds: Option<[f64; 6]>,
    pub attributes: Vec<String>,
    pub file_size_bytes: u64,
}
```

### Streaming Architecture

```
┌─────────────────┐    Chunks (100K pts)    ┌─────────────────┐
│   pasture I/O   │ ───────────────────────▶│  Tauri Channel  │
│   (streaming)   │                         │   (IPC event)   │
└─────────────────┘                         └─────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │    Frontend     │
                                            │  (incremental)  │
                                            └─────────────────┘
```

### Format Detection

```rust
pub fn detect_format(path: &Path) -> Result<PointFormat, Error> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("pcd") => Ok(PointFormat::Pcd),
        Some("ply") => Ok(PointFormat::Ply),
        Some("las") => Ok(PointFormat::Las),
        Some("laz") => Ok(PointFormat::Laz),
        _ => Err(Error::UnknownFormat),
    }
}
```

## Implementation Tasks

- [ ] **Setup pasture crate**
  - [ ] Add pasture-core and pasture-io dependencies
  - [ ] Configure feature flags for las, ply, pcd
  - [ ] Create pointcloud module structure

- [ ] **Implement readers**
  - [ ] PCD reader (ASCII + binary modes)
  - [ ] PLY reader (ASCII + binary modes)
  - [ ] LAS/LAZ reader using pasture-io las feature
  - [ ] Unified PointCloudData output format

- [ ] **Implement streaming**
  - [ ] Chunk iterator for large files (configurable chunk size)
  - [ ] Tauri event emission for progress updates
  - [ ] Memory-efficient buffer management
  - [ ] Cancellation support via AppHandle

- [ ] **Implement format conversion**
  - [ ] PCD → PLY conversion
  - [ ] PLY → PCD conversion
  - [ ] LAS → PCD/PLY conversion
  - [ ] Attribute mapping between formats

- [ ] **Implement decimation**
  - [ ] Voxel grid downsampling
  - [ ] Random subsampling
  - [ ] Configurable target point count

- [ ] **Tauri IPC commands**
  - [ ] `read_pointcloud_metadata` - Quick file inspection
  - [ ] `read_pointcloud` - Full load for small files
  - [ ] `stream_pointcloud` - Chunked streaming for large files
  - [ ] `convert_pointcloud` - Format conversion
  - [ ] `decimate_pointcloud` - Downsampling

## Files to Create/Modify

| Path | Action | Purpose |
|------|--------|---------|
| `src-tauri/Cargo.toml` | Modify | Add pasture dependencies |
| `src-tauri/src/pointcloud/mod.rs` | Create | Module exports |
| `src-tauri/src/pointcloud/io.rs` | Create | Read/write with pasture |
| `src-tauri/src/pointcloud/convert.rs` | Create | Format conversion |
| `src-tauri/src/pointcloud/decimate.rs` | Create | Downsampling algorithms |
| `src-tauri/src/pointcloud/types.rs` | Create | Data structures |
| `src-tauri/src/commands/pointcloud.rs` | Create | Tauri IPC handlers |
| `src-tauri/src/commands/mod.rs` | Modify | Export pointcloud commands |
| `src-tauri/src/main.rs` | Modify | Register pointcloud commands |

## API Reference

### Tauri Commands

```rust
#[tauri::command]
async fn read_pointcloud_metadata(path: String) -> Result<PointCloudMetadata, String>;

#[tauri::command]
async fn read_pointcloud(path: String) -> Result<PointCloudData, String>;

#[tauri::command]
async fn stream_pointcloud(
    app: AppHandle,
    path: String,
    chunk_size: Option<usize>,
) -> Result<String, String>; // Returns stream_id

#[tauri::command]
async fn convert_pointcloud(
    input_path: String,
    output_path: String,
) -> Result<PointCloudMetadata, String>;

#[tauri::command]
async fn decimate_pointcloud(
    input_path: String,
    output_path: String,
    target_count: usize,
    method: String, // "voxel" | "random"
) -> Result<PointCloudMetadata, String>;
```

### Frontend Events

```typescript
// Streaming events emitted to frontend
interface PointCloudChunk {
  stream_id: string;
  chunk_index: number;
  total_chunks: number;
  positions: number[];  // Float64Array compatible
  intensity?: number[];
  colors?: number[];
}

// Listen for chunks
listen<PointCloudChunk>('pointcloud:chunk', (event) => {
  appendToViewer(event.payload);
});
```

## Acceptance Criteria

- [ ] Load 10M point PCD file in <2 seconds
- [ ] Stream chunks of 100K points without blocking UI
- [ ] Convert PCD to PLY preserving XYZ, intensity, RGB attributes
- [ ] LAS/LAZ files load with correct coordinate system
- [ ] Voxel decimation reduces 1M to 100K points in <500ms
- [ ] `cargo clippy -- -D warnings` passes in src-tauri
- [ ] `cargo test` passes for pointcloud module

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_pcd_ascii() {
        let data = read_pointcloud("fixtures/sample_ascii.pcd").unwrap();
        assert_eq!(data.count, 1000);
        assert!(data.attributes.contains(&"intensity".to_string()));
    }

    #[test]
    fn test_format_conversion() {
        convert_pointcloud("fixtures/input.pcd", "fixtures/output.ply").unwrap();
        let meta = read_pointcloud_metadata("fixtures/output.ply").unwrap();
        assert_eq!(meta.format, "ply");
    }

    #[test]
    fn test_decimation() {
        let result = decimate_pointcloud(
            "fixtures/large.pcd",
            "fixtures/small.pcd",
            10000,
            "voxel",
        ).unwrap();
        assert!(result.point_count <= 10000);
    }
}
```

## Performance Considerations

- Use memory-mapped I/O for files >100MB
- Prefer binary formats over ASCII for speed
- Stream in 100K point chunks (balance between overhead and responsiveness)
- Consider parallel chunk processing for multi-core systems
- Profile with `cargo flamegraph` for bottleneck identification

## Related Sub-Specs

- [06-threejs-viewer.md](./06-threejs-viewer.md) - Consumes streamed point data
- [02-python-open3d.md](./02-python-open3d.md) - Alternative processing path
