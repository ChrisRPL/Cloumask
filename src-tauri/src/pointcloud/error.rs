//! Point cloud error types.

use thiserror::Error;

/// Errors that can occur during point cloud operations.
#[derive(Error, Debug)]
pub enum PointCloudError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Failed to read file: {0}")]
    ReadError(String),

    #[error("Failed to write file: {0}")]
    WriteError(String),

    #[error("Invalid point cloud data: {0}")]
    InvalidData(String),

    #[error("Decimation failed: {0}")]
    DecimationError(String),

    #[error("Conversion failed: {0}")]
    ConversionError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),
}

impl From<PointCloudError> for String {
    fn from(err: PointCloudError) -> Self {
        err.to_string()
    }
}
