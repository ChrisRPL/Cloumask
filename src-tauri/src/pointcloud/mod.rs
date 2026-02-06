//! Native Rust point cloud I/O using pasture.
//!
//! Supports reading/writing PCD, PLY, LAS/LAZ formats with:
//! - Fast metadata inspection
//! - Chunked streaming for large files
//! - Format conversion
//! - Voxel grid decimation

pub mod convert;
pub mod decimate;
pub mod error;
pub mod io;
pub mod types;

pub use types::*;

// Re-export error type for external use
#[allow(unused_imports)]
pub use error::PointCloudError;
