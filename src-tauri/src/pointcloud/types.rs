//! Point cloud data types for Tauri IPC.

use serde::{Deserialize, Serialize};

/// Supported point cloud file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PointCloudFormat {
    Pcd,
    Ply,
    Las,
    Laz,
}

impl PointCloudFormat {
    /// Detect format from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "pcd" => Some(Self::Pcd),
            "ply" => Some(Self::Ply),
            "las" => Some(Self::Las),
            "laz" => Some(Self::Laz),
            _ => None,
        }
    }

    /// Get the file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Pcd => "pcd",
            Self::Ply => "ply",
            Self::Las => "las",
            Self::Laz => "laz",
        }
    }
}

/// 3D axis-aligned bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bounds3D {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Default for Bounds3D {
    fn default() -> Self {
        Self::new()
    }
}

impl Bounds3D {
    /// Create an empty bounds (inverted for extension).
    pub fn new() -> Self {
        Self {
            min: [f64::MAX, f64::MAX, f64::MAX],
            max: [f64::MIN, f64::MIN, f64::MIN],
        }
    }

    /// Extend bounds to include a point.
    pub fn extend(&mut self, point: [f64; 3]) {
        for (i, &p) in point.iter().enumerate() {
            self.min[i] = self.min[i].min(p);
            self.max[i] = self.max[i].max(p);
        }
    }

    /// Check if bounds are valid (have been extended at least once).
    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] && self.min[1] <= self.max[1] && self.min[2] <= self.max[2]
    }

    /// Get the center of the bounds.
    #[allow(dead_code)]
    pub fn center(&self) -> [f64; 3] {
        [
            (self.min[0] + self.max[0]) / 2.0,
            (self.min[1] + self.max[1]) / 2.0,
            (self.min[2] + self.max[2]) / 2.0,
        ]
    }

    /// Get the size of the bounds.
    #[allow(dead_code)]
    pub fn size(&self) -> [f64; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }
}

/// Point cloud metadata (fast to compute, no full file read for some formats).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloudMetadata {
    /// File path.
    pub path: String,
    /// Detected format.
    pub format: PointCloudFormat,
    /// Total number of points.
    pub point_count: u64,
    /// File size in bytes.
    pub file_size_bytes: u64,
    /// Available attributes (e.g., ["position", "intensity", "rgb", "classification"]).
    pub attributes: Vec<String>,
    /// 3D bounding box (may require scanning file for some formats).
    pub bounds: Option<Bounds3D>,
    /// Whether file has intensity values.
    pub has_intensity: bool,
    /// Whether file has RGB color.
    pub has_rgb: bool,
    /// Whether file has classification (LAS-specific).
    pub has_classification: bool,
}

/// A chunk of point cloud data for streaming.
/// Uses flat arrays for efficient serialization to frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloudChunk {
    /// Chunk index (0-based).
    pub chunk_index: u32,
    /// Total number of chunks.
    pub total_chunks: u32,
    /// Number of points in this chunk.
    pub point_count: u32,
    /// Flat XYZ positions: [x0, y0, z0, x1, y1, z1, ...].
    pub positions: Vec<f32>,
    /// Optional intensity values (normalized 0-1).
    pub intensities: Option<Vec<f32>>,
    /// Optional RGB colors as packed u32 (0xRRGGBB).
    pub colors: Option<Vec<u32>>,
    /// Optional classification values (LAS-specific).
    pub classifications: Option<Vec<u8>>,
}

/// Complete point cloud data (for small files loaded at once).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloudData {
    /// Metadata about the point cloud.
    pub metadata: PointCloudMetadata,
    /// Flat XYZ positions: [x0, y0, z0, x1, y1, z1, ...].
    pub positions: Vec<f32>,
    /// Optional intensity values (normalized 0-1).
    pub intensities: Option<Vec<f32>>,
    /// Optional RGB colors as packed u32 (0xRRGGBB).
    pub colors: Option<Vec<u32>>,
    /// Optional classification values (LAS-specific).
    pub classifications: Option<Vec<u8>>,
}

/// Decimation method for downsampling point clouds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecimationMethod {
    /// Voxel grid downsampling with specified voxel size.
    VoxelGrid { voxel_size: f64 },
    /// Random sampling keeping specified percentage (0.0-1.0).
    Random { keep_ratio: f64 },
    /// Keep every Nth point.
    Uniform { step: u32 },
}

/// Options for format conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionOptions {
    /// Target format.
    pub target_format: PointCloudFormat,
    /// Whether to preserve intensity values.
    pub preserve_intensity: bool,
    /// Whether to preserve RGB colors.
    pub preserve_rgb: bool,
    /// Whether to preserve classification.
    pub preserve_classification: bool,
    /// Optional decimation to apply during conversion.
    pub decimation: Option<DecimationMethod>,
}

/// Configuration for streaming point cloud data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Number of points per chunk (default: 100,000).
    pub chunk_size: u32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self { chunk_size: 100_000 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_from_extension() {
        assert_eq!(
            PointCloudFormat::from_extension("las"),
            Some(PointCloudFormat::Las)
        );
        assert_eq!(
            PointCloudFormat::from_extension("LAZ"),
            Some(PointCloudFormat::Laz)
        );
        assert_eq!(
            PointCloudFormat::from_extension("PLY"),
            Some(PointCloudFormat::Ply)
        );
        assert_eq!(
            PointCloudFormat::from_extension("pcd"),
            Some(PointCloudFormat::Pcd)
        );
        assert_eq!(PointCloudFormat::from_extension("xyz"), None);
    }

    #[test]
    fn test_bounds_extend() {
        let mut bounds = Bounds3D::new();
        assert!(!bounds.is_valid());

        bounds.extend([0.0, 0.0, 0.0]);
        bounds.extend([10.0, 20.0, 30.0]);

        assert!(bounds.is_valid());
        assert_eq!(bounds.min, [0.0, 0.0, 0.0]);
        assert_eq!(bounds.max, [10.0, 20.0, 30.0]);
        assert_eq!(bounds.center(), [5.0, 10.0, 15.0]);
        assert_eq!(bounds.size(), [10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_decimation_serde() {
        let voxel = DecimationMethod::VoxelGrid { voxel_size: 0.1 };
        let json = serde_json::to_string(&voxel).unwrap();
        assert!(json.contains("voxel_grid"));

        let random = DecimationMethod::Random { keep_ratio: 0.5 };
        let json = serde_json::to_string(&random).unwrap();
        assert!(json.contains("random"));
    }
}
