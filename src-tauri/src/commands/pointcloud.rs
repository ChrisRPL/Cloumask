//! Point cloud Tauri IPC commands.
//!
//! Provides commands for reading, streaming, converting, and decimating point clouds.

use tauri::{AppHandle, Emitter};

use crate::pointcloud::{
    convert, io, ConversionOptions, DecimationMethod, PointCloudChunk, PointCloudData,
    PointCloudMetadata, StreamConfig,
};

// ============================================================================
// Metadata Command
// ============================================================================

/// Read point cloud metadata without loading all points.
///
/// Fast inspection of file format, point count, bounds, and available attributes.
#[tauri::command]
pub async fn read_pointcloud_metadata(path: String) -> Result<PointCloudMetadata, String> {
    tokio::task::spawn_blocking(move || io::read_metadata(&path))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}

// ============================================================================
// Full Read Command (Small Files)
// ============================================================================

/// Read entire point cloud into memory.
///
/// Use for files under ~1M points. For larger files, use stream_pointcloud.
#[tauri::command]
pub async fn read_pointcloud(path: String) -> Result<PointCloudData, String> {
    tokio::task::spawn_blocking(move || io::read_pointcloud(&path))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}

// ============================================================================
// Streaming Command (Large Files)
// ============================================================================

/// Stream point cloud in chunks via Tauri events.
///
/// Emits `pointcloud:chunk` events with PointCloudChunk payloads.
/// Emits `pointcloud:complete` when done or `pointcloud:error` on failure.
///
/// Returns immediately with metadata; chunks are delivered asynchronously.
#[tauri::command]
pub async fn stream_pointcloud(
    app: AppHandle,
    path: String,
    config: Option<StreamConfig>,
) -> Result<PointCloudMetadata, String> {
    let config = config.unwrap_or_default();

    // Read full point cloud data first (blocks)
    let data = tokio::task::spawn_blocking({
        let path = path.clone();
        move || io::read_pointcloud(&path)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;

    let metadata = data.metadata.clone();

    // Spawn background task for streaming chunks
    tokio::task::spawn(async move {
        let chunks = io::create_chunks(&data, config.chunk_size);

        for chunk in chunks {
            if let Err(e) = app.emit("pointcloud:chunk", &chunk) {
                log::error!("Failed to emit chunk: {}", e);
                let _ = app.emit("pointcloud:error", e.to_string());
                return;
            }

            // Small delay to prevent flooding the frontend
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        }

        // Signal completion
        let _ = app.emit("pointcloud:complete", &data.metadata);
    });

    Ok(metadata)
}

// ============================================================================
// Conversion Command
// ============================================================================

/// Convert point cloud to another format.
///
/// Optionally applies decimation during conversion.
#[tauri::command]
pub async fn convert_pointcloud(
    input_path: String,
    output_path: String,
    options: ConversionOptions,
) -> Result<PointCloudMetadata, String> {
    tokio::task::spawn_blocking(move || {
        convert::convert_pointcloud(&input_path, &output_path, options)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())
}

// ============================================================================
// Decimation Command
// ============================================================================

/// Decimate a point cloud and save to a new file.
///
/// Supports voxel grid, random, and uniform decimation methods.
#[tauri::command]
pub async fn decimate_pointcloud(
    input_path: String,
    output_path: String,
    method: DecimationMethod,
) -> Result<PointCloudMetadata, String> {
    tokio::task::spawn_blocking(move || {
        // Determine output format from extension
        let output_format = io::detect_format(std::path::Path::new(&output_path))?;

        // Use convert_pointcloud with decimation option
        let options = ConversionOptions {
            target_format: output_format,
            preserve_intensity: true,
            preserve_rgb: true,
            preserve_classification: true,
            decimation: Some(method),
        };

        convert::convert_pointcloud(&input_path, &output_path, options)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())
}

// ============================================================================
// Streaming Event Types (for frontend reference)
// ============================================================================

/// Event emitted for each chunk during streaming.
/// Event name: "pointcloud:chunk"
/// Payload: PointCloudChunk
#[allow(dead_code)]
pub type StreamChunkEvent = PointCloudChunk;

/// Event emitted when streaming completes successfully.
/// Event name: "pointcloud:complete"
/// Payload: PointCloudMetadata
#[allow(dead_code)]
pub type StreamCompleteEvent = PointCloudMetadata;

/// Event emitted when streaming encounters an error.
/// Event name: "pointcloud:error"
/// Payload: String (error message)
#[allow(dead_code)]
pub type StreamErrorEvent = String;
