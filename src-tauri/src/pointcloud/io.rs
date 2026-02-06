//! Point cloud file readers.
//!
//! Supports reading PCD, PLY, LAS, and LAZ formats.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use pasture_core::containers::{BorrowedBuffer, BorrowedBufferExt, VectorBuffer};
use pasture_core::layout::attributes::{CLASSIFICATION, COLOR_RGB, INTENSITY, POSITION_3D};
use pasture_core::nalgebra::Vector3;
use pasture_io::base::read_all;
use ply_rs_bw::parser::Parser as PlyParser;
use ply_rs_bw::ply::{DefaultElement, Property};

use crate::pointcloud::error::PointCloudError;
use crate::pointcloud::types::*;

/// Detect point cloud format from file extension.
pub fn detect_format(path: &Path) -> Result<PointCloudFormat, PointCloudError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| PointCloudError::UnsupportedFormat("No file extension".into()))?;

    PointCloudFormat::from_extension(ext)
        .ok_or_else(|| PointCloudError::UnsupportedFormat(format!("Unknown extension: {}", ext)))
}

/// Read point cloud metadata without loading all point data.
pub fn read_metadata(path: &str) -> Result<PointCloudMetadata, PointCloudError> {
    let path = Path::new(path);

    if !path.exists() {
        return Err(PointCloudError::FileNotFound(path.display().to_string()));
    }

    let format = detect_format(path)?;
    let file_size_bytes = std::fs::metadata(path)?.len();

    match format {
        PointCloudFormat::Las | PointCloudFormat::Laz => read_las_metadata(path, file_size_bytes),
        PointCloudFormat::Ply => read_ply_metadata(path, file_size_bytes),
        PointCloudFormat::Pcd => read_pcd_metadata(path, file_size_bytes),
    }
}

/// Read entire point cloud into memory.
pub fn read_pointcloud(path: &str) -> Result<PointCloudData, PointCloudError> {
    let path = Path::new(path);

    if !path.exists() {
        return Err(PointCloudError::FileNotFound(path.display().to_string()));
    }

    let format = detect_format(path)?;

    match format {
        PointCloudFormat::Las | PointCloudFormat::Laz => read_las_pointcloud(path),
        PointCloudFormat::Ply => read_ply_pointcloud(path),
        PointCloudFormat::Pcd => read_pcd_pointcloud(path),
    }
}

/// Create chunks from point cloud data for streaming.
pub fn create_chunks(data: &PointCloudData, chunk_size: u32) -> Vec<PointCloudChunk> {
    let point_count = data.metadata.point_count as usize;
    let chunk_count = (point_count as f64 / chunk_size as f64).ceil() as u32;
    let mut chunks = Vec::with_capacity(chunk_count as usize);

    for chunk_idx in 0..chunk_count {
        let start_point = (chunk_idx * chunk_size) as usize;
        let end_point = ((chunk_idx + 1) * chunk_size).min(point_count as u32) as usize;
        let chunk_point_count = (end_point - start_point) as u32;

        // Extract positions for this chunk
        let start_pos = start_point * 3;
        let end_pos = end_point * 3;
        let positions = data.positions[start_pos..end_pos].to_vec();

        // Extract optional attributes
        let intensities = data
            .intensities
            .as_ref()
            .map(|v| v[start_point..end_point].to_vec());

        let colors = data
            .colors
            .as_ref()
            .map(|v| v[start_point..end_point].to_vec());

        let classifications = data
            .classifications
            .as_ref()
            .map(|v| v[start_point..end_point].to_vec());

        chunks.push(PointCloudChunk {
            chunk_index: chunk_idx,
            total_chunks: chunk_count,
            point_count: chunk_point_count,
            positions,
            intensities,
            colors,
            classifications,
        });
    }

    chunks
}

// ============================================================================
// LAS/LAZ readers (using pasture-io)
// ============================================================================

fn read_las_metadata(path: &Path, file_size_bytes: u64) -> Result<PointCloudMetadata, PointCloudError> {
    let buffer: VectorBuffer =
        read_all(path).map_err(|e| PointCloudError::ReadError(e.to_string()))?;

    let layout = buffer.point_layout();
    let point_count = buffer.len() as u64;

    let has_intensity = layout.has_attribute(&INTENSITY);
    let has_rgb = layout.has_attribute(&COLOR_RGB);
    let has_classification = layout.has_attribute(&CLASSIFICATION);

    let mut attributes = vec!["position".to_string()];
    if has_intensity {
        attributes.push("intensity".to_string());
    }
    if has_rgb {
        attributes.push("rgb".to_string());
    }
    if has_classification {
        attributes.push("classification".to_string());
    }

    // Compute bounds
    let mut bounds = Bounds3D::new();
    for pos in buffer.view_attribute::<Vector3<f64>>(&POSITION_3D) {
        bounds.extend([pos.x, pos.y, pos.z]);
    }

    let format = detect_format(path)?;

    Ok(PointCloudMetadata {
        path: path.display().to_string(),
        format,
        point_count,
        file_size_bytes,
        attributes,
        bounds: if bounds.is_valid() { Some(bounds) } else { None },
        has_intensity,
        has_rgb,
        has_classification,
    })
}

fn read_las_pointcloud(path: &Path) -> Result<PointCloudData, PointCloudError> {
    let buffer: VectorBuffer =
        read_all(path).map_err(|e| PointCloudError::ReadError(e.to_string()))?;

    let layout = buffer.point_layout();
    let point_count = buffer.len();

    let has_intensity = layout.has_attribute(&INTENSITY);
    let has_rgb = layout.has_attribute(&COLOR_RGB);
    let has_classification = layout.has_attribute(&CLASSIFICATION);

    // Extract positions as f32
    let mut positions = Vec::with_capacity(point_count * 3);
    let mut bounds = Bounds3D::new();
    for pos in buffer.view_attribute::<Vector3<f64>>(&POSITION_3D) {
        bounds.extend([pos.x, pos.y, pos.z]);
        positions.push(pos.x as f32);
        positions.push(pos.y as f32);
        positions.push(pos.z as f32);
    }

    // Extract intensity (normalize u16 to 0-1 f32)
    let intensities = if has_intensity {
        let mut vals = Vec::with_capacity(point_count);
        for intensity in buffer.view_attribute::<u16>(&INTENSITY) {
            vals.push(intensity as f32 / 65535.0);
        }
        Some(vals)
    } else {
        None
    };

    // Extract RGB (pack into u32)
    let colors = if has_rgb {
        let mut vals = Vec::with_capacity(point_count);
        for rgb in buffer.view_attribute::<Vector3<u16>>(&COLOR_RGB) {
            // LAS uses 16-bit RGB, convert to 8-bit and pack
            let r = (rgb.x >> 8) as u32;
            let g = (rgb.y >> 8) as u32;
            let b = (rgb.z >> 8) as u32;
            vals.push((r << 16) | (g << 8) | b);
        }
        Some(vals)
    } else {
        None
    };

    // Extract classification
    let classifications = if has_classification {
        let mut vals = Vec::with_capacity(point_count);
        for class in buffer.view_attribute::<u8>(&CLASSIFICATION) {
            vals.push(class);
        }
        Some(vals)
    } else {
        None
    };

    let mut attributes = vec!["position".to_string()];
    if has_intensity {
        attributes.push("intensity".to_string());
    }
    if has_rgb {
        attributes.push("rgb".to_string());
    }
    if has_classification {
        attributes.push("classification".to_string());
    }

    let format = detect_format(path)?;
    let file_size_bytes = std::fs::metadata(path)?.len();

    Ok(PointCloudData {
        metadata: PointCloudMetadata {
            path: path.display().to_string(),
            format,
            point_count: point_count as u64,
            file_size_bytes,
            attributes,
            bounds: if bounds.is_valid() { Some(bounds) } else { None },
            has_intensity,
            has_rgb,
            has_classification,
        },
        positions,
        intensities,
        colors,
        classifications,
    })
}

// ============================================================================
// PLY reader (using ply-rs-bw)
// ============================================================================

fn read_ply_metadata(path: &Path, file_size_bytes: u64) -> Result<PointCloudMetadata, PointCloudError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let parser = PlyParser::<DefaultElement>::new();
    let header = parser
        .read_header(&mut reader)
        .map_err(|e| PointCloudError::ReadError(format!("PLY header parse error: {:?}", e)))?;

    // Find vertex element
    let vertex_elem = header
        .elements
        .get("vertex")
        .ok_or_else(|| PointCloudError::InvalidData("PLY file missing vertex element".into()))?;

    let point_count = vertex_elem.count as u64;

    // Check for properties
    let has_intensity = vertex_elem.properties.contains_key("intensity")
        || vertex_elem.properties.contains_key("scalar_intensity");
    let has_rgb = vertex_elem.properties.contains_key("red")
        || vertex_elem.properties.contains_key("r");

    let mut attributes = vec!["position".to_string()];
    if has_intensity {
        attributes.push("intensity".to_string());
    }
    if has_rgb {
        attributes.push("rgb".to_string());
    }

    Ok(PointCloudMetadata {
        path: path.display().to_string(),
        format: PointCloudFormat::Ply,
        point_count,
        file_size_bytes,
        attributes,
        bounds: None, // Would need to read all points to compute
        has_intensity,
        has_rgb,
        has_classification: false,
    })
}

fn read_ply_pointcloud(path: &Path) -> Result<PointCloudData, PointCloudError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let parser = PlyParser::<DefaultElement>::new();
    let header = parser
        .read_header(&mut reader)
        .map_err(|e| PointCloudError::ReadError(format!("PLY header parse error: {:?}", e)))?;

    let payload = parser
        .read_payload(&mut reader, &header)
        .map_err(|e| PointCloudError::ReadError(format!("PLY payload parse error: {:?}", e)))?;

    let vertices = payload
        .get("vertex")
        .ok_or_else(|| PointCloudError::InvalidData("PLY file missing vertex data".into()))?;

    let point_count = vertices.len();
    let mut positions = Vec::with_capacity(point_count * 3);
    let mut bounds = Bounds3D::new();

    // Check for attribute names (PLY uses various conventions)
    let has_intensity = vertices
        .first()
        .map(|v| v.contains_key("intensity") || v.contains_key("scalar_intensity"))
        .unwrap_or(false);
    let has_rgb = vertices
        .first()
        .map(|v| v.contains_key("red") || v.contains_key("r"))
        .unwrap_or(false);

    let mut intensities: Option<Vec<f32>> = if has_intensity {
        Some(Vec::with_capacity(point_count))
    } else {
        None
    };
    let mut colors: Option<Vec<u32>> = if has_rgb {
        Some(Vec::with_capacity(point_count))
    } else {
        None
    };

    for vertex in vertices {
        let x = get_float_property(vertex, "x")?;
        let y = get_float_property(vertex, "y")?;
        let z = get_float_property(vertex, "z")?;

        bounds.extend([x, y, z]);
        positions.push(x as f32);
        positions.push(y as f32);
        positions.push(z as f32);

        if let Some(ref mut vals) = intensities {
            let intensity = get_float_property_optional(vertex, "intensity")
                .or_else(|| get_float_property_optional(vertex, "scalar_intensity"))
                .unwrap_or(0.0);
            vals.push(intensity as f32);
        }

        if let Some(ref mut vals) = colors {
            let r = get_u8_property_optional(vertex, "red")
                .or_else(|| get_u8_property_optional(vertex, "r"))
                .unwrap_or(128) as u32;
            let g = get_u8_property_optional(vertex, "green")
                .or_else(|| get_u8_property_optional(vertex, "g"))
                .unwrap_or(128) as u32;
            let b = get_u8_property_optional(vertex, "blue")
                .or_else(|| get_u8_property_optional(vertex, "b"))
                .unwrap_or(128) as u32;
            vals.push((r << 16) | (g << 8) | b);
        }
    }

    let mut attributes = vec!["position".to_string()];
    if has_intensity {
        attributes.push("intensity".to_string());
    }
    if has_rgb {
        attributes.push("rgb".to_string());
    }

    let file_size_bytes = std::fs::metadata(path)?.len();

    Ok(PointCloudData {
        metadata: PointCloudMetadata {
            path: path.display().to_string(),
            format: PointCloudFormat::Ply,
            point_count: point_count as u64,
            file_size_bytes,
            attributes,
            bounds: if bounds.is_valid() { Some(bounds) } else { None },
            has_intensity,
            has_rgb,
            has_classification: false,
        },
        positions,
        intensities,
        colors,
        classifications: None,
    })
}

fn get_float_property(vertex: &DefaultElement, key: &str) -> Result<f64, PointCloudError> {
    vertex
        .get(key)
        .and_then(|p| match p {
            Property::Float(v) => Some(*v as f64),
            Property::Double(v) => Some(*v),
            Property::Int(v) => Some(*v as f64),
            Property::UInt(v) => Some(*v as f64),
            _ => None,
        })
        .ok_or_else(|| PointCloudError::InvalidData(format!("Missing property: {}", key)))
}

fn get_float_property_optional(vertex: &DefaultElement, key: &str) -> Option<f64> {
    vertex.get(key).and_then(|p| match p {
        Property::Float(v) => Some(*v as f64),
        Property::Double(v) => Some(*v),
        Property::Int(v) => Some(*v as f64),
        Property::UInt(v) => Some(*v as f64),
        _ => None,
    })
}

fn get_u8_property_optional(vertex: &DefaultElement, key: &str) -> Option<u8> {
    vertex.get(key).and_then(|p| match p {
        Property::UChar(v) => Some(*v),
        Property::Int(v) => Some(*v as u8),
        Property::UInt(v) => Some(*v as u8),
        _ => None,
    })
}

// ============================================================================
// PCD reader (ASCII format)
// ============================================================================

fn read_pcd_metadata(path: &Path, file_size_bytes: u64) -> Result<PointCloudMetadata, PointCloudError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut point_count = 0u64;
    let mut fields: Vec<String> = Vec::new();
    let mut data_type = String::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.starts_with("FIELDS") {
            fields = line
                .split_whitespace()
                .skip(1)
                .map(|s| s.to_lowercase())
                .collect();
        } else if line.starts_with("POINTS") {
            point_count = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        } else if line.starts_with("DATA") {
            data_type = line
                .split_whitespace()
                .nth(1)
                .unwrap_or("ascii")
                .to_lowercase();
            break;
        }
    }

    if data_type != "ascii" {
        return Err(PointCloudError::UnsupportedFormat(
            "Only ASCII PCD files are currently supported".into(),
        ));
    }

    let has_intensity = fields.contains(&"intensity".to_string());
    let has_rgb = fields.contains(&"rgb".to_string()) || fields.contains(&"rgba".to_string());

    let mut attributes = vec!["position".to_string()];
    if has_intensity {
        attributes.push("intensity".to_string());
    }
    if has_rgb {
        attributes.push("rgb".to_string());
    }

    Ok(PointCloudMetadata {
        path: path.display().to_string(),
        format: PointCloudFormat::Pcd,
        point_count,
        file_size_bytes,
        attributes,
        bounds: None,
        has_intensity,
        has_rgb,
        has_classification: false,
    })
}

fn read_pcd_pointcloud(path: &Path) -> Result<PointCloudData, PointCloudError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut fields: Vec<String> = Vec::new();
    let mut in_data = false;

    let mut positions = Vec::new();
    let mut bounds = Bounds3D::new();
    let mut intensities_raw = Vec::new();
    let mut colors_raw = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if !in_data {
            if line.starts_with("FIELDS") {
                fields = line
                    .split_whitespace()
                    .skip(1)
                    .map(|s| s.to_lowercase())
                    .collect();
            } else if line.starts_with("POINTS") {
                let expected_count: usize = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                positions.reserve(expected_count * 3);
            } else if line.starts_with("DATA") {
                let data_type = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("ascii")
                    .to_lowercase();

                if data_type != "ascii" {
                    return Err(PointCloudError::UnsupportedFormat(
                        "Only ASCII PCD files are currently supported".into(),
                    ));
                }
                in_data = true;
            }
        } else {
            // Parse point data
            let values: Vec<&str> = line.split_whitespace().collect();
            if values.len() < 3 {
                continue;
            }

            // Find field indices
            let x_idx = fields.iter().position(|f| f == "x").unwrap_or(0);
            let y_idx = fields.iter().position(|f| f == "y").unwrap_or(1);
            let z_idx = fields.iter().position(|f| f == "z").unwrap_or(2);
            let intensity_idx = fields.iter().position(|f| f == "intensity");
            let rgb_idx = fields
                .iter()
                .position(|f| f == "rgb" || f == "rgba");

            let x: f64 = values
                .get(x_idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            let y: f64 = values
                .get(y_idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            let z: f64 = values
                .get(z_idx)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);

            bounds.extend([x, y, z]);
            positions.push(x as f32);
            positions.push(y as f32);
            positions.push(z as f32);

            if let Some(idx) = intensity_idx {
                if let Some(v) = values.get(idx).and_then(|v| v.parse::<f32>().ok()) {
                    intensities_raw.push(v);
                }
            }

            if let Some(idx) = rgb_idx {
                // PCD stores RGB as a single float (bitcasted from u32)
                if let Some(v) = values.get(idx).and_then(|v| v.parse::<f32>().ok()) {
                    let packed = v.to_bits();
                    colors_raw.push(packed);
                }
            }
        }
    }

    let has_intensity = !intensities_raw.is_empty();
    let has_rgb = !colors_raw.is_empty();

    let mut attributes = vec!["position".to_string()];
    if has_intensity {
        attributes.push("intensity".to_string());
    }
    if has_rgb {
        attributes.push("rgb".to_string());
    }

    let actual_point_count = positions.len() / 3;
    let file_size_bytes = std::fs::metadata(path)?.len();

    Ok(PointCloudData {
        metadata: PointCloudMetadata {
            path: path.display().to_string(),
            format: PointCloudFormat::Pcd,
            point_count: actual_point_count as u64,
            file_size_bytes,
            attributes,
            bounds: if bounds.is_valid() { Some(bounds) } else { None },
            has_intensity,
            has_rgb,
            has_classification: false,
        },
        positions,
        intensities: if has_intensity {
            Some(intensities_raw)
        } else {
            None
        },
        colors: if has_rgb { Some(colors_raw) } else { None },
        classifications: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format() {
        assert_eq!(
            detect_format(Path::new("test.las")).unwrap(),
            PointCloudFormat::Las
        );
        assert_eq!(
            detect_format(Path::new("test.LAZ")).unwrap(),
            PointCloudFormat::Laz
        );
        assert_eq!(
            detect_format(Path::new("test.ply")).unwrap(),
            PointCloudFormat::Ply
        );
        assert_eq!(
            detect_format(Path::new("test.pcd")).unwrap(),
            PointCloudFormat::Pcd
        );
        assert!(detect_format(Path::new("test.xyz")).is_err());
        assert!(detect_format(Path::new("noext")).is_err());
    }

    #[test]
    fn test_create_chunks() {
        let data = PointCloudData {
            metadata: PointCloudMetadata {
                path: "test.las".into(),
                format: PointCloudFormat::Las,
                point_count: 5,
                file_size_bytes: 100,
                attributes: vec!["position".into()],
                bounds: None,
                has_intensity: false,
                has_rgb: false,
                has_classification: false,
            },
            positions: vec![
                0.0, 0.0, 0.0, // point 0
                1.0, 1.0, 1.0, // point 1
                2.0, 2.0, 2.0, // point 2
                3.0, 3.0, 3.0, // point 3
                4.0, 4.0, 4.0, // point 4
            ],
            intensities: None,
            colors: None,
            classifications: None,
        };

        let chunks = create_chunks(&data, 2);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].point_count, 2);
        assert_eq!(chunks[1].point_count, 2);
        assert_eq!(chunks[2].point_count, 1);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[2].total_chunks, 3);
    }
}
