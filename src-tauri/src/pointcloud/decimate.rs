//! Point cloud decimation algorithms.
//!
//! Provides voxel grid, random, and uniform downsampling.

use std::collections::HashMap;

use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::pointcloud::error::PointCloudError;
use crate::pointcloud::types::{DecimationMethod, PointCloudData};

/// Apply decimation to point cloud data.
pub fn decimate(
    data: &PointCloudData,
    method: DecimationMethod,
) -> Result<PointCloudData, PointCloudError> {
    match method {
        DecimationMethod::VoxelGrid { voxel_size } => voxel_grid_decimate(data, voxel_size),
        DecimationMethod::Random { keep_ratio } => random_decimate(data, keep_ratio),
        DecimationMethod::Uniform { step } => uniform_decimate(data, step),
    }
}

/// Voxel grid downsampling.
///
/// Groups points into voxels and keeps one representative point per voxel.
fn voxel_grid_decimate(
    data: &PointCloudData,
    voxel_size: f64,
) -> Result<PointCloudData, PointCloudError> {
    if voxel_size <= 0.0 {
        return Err(PointCloudError::DecimationError(
            "Voxel size must be positive".into(),
        ));
    }

    let point_count = data.metadata.point_count as usize;
    if point_count == 0 {
        return Ok(data.clone());
    }

    // Map voxel indices to point indices (keep first point in each voxel)
    let mut voxel_map: HashMap<(i64, i64, i64), usize> = HashMap::new();

    for i in 0..point_count {
        let x = data.positions[i * 3] as f64;
        let y = data.positions[i * 3 + 1] as f64;
        let z = data.positions[i * 3 + 2] as f64;

        let voxel_key = (
            (x / voxel_size).floor() as i64,
            (y / voxel_size).floor() as i64,
            (z / voxel_size).floor() as i64,
        );

        // Keep first point in each voxel
        voxel_map.entry(voxel_key).or_insert(i);
    }

    // Collect and sort indices for deterministic output
    let mut kept_indices: Vec<usize> = voxel_map.into_values().collect();
    kept_indices.sort_unstable();

    extract_points_by_indices(data, &kept_indices)
}

/// Random downsampling.
///
/// Randomly samples points, keeping the specified ratio.
fn random_decimate(
    data: &PointCloudData,
    keep_ratio: f64,
) -> Result<PointCloudData, PointCloudError> {
    if !(0.0..=1.0).contains(&keep_ratio) {
        return Err(PointCloudError::DecimationError(
            "Keep ratio must be in range [0, 1]".into(),
        ));
    }

    let point_count = data.metadata.point_count as usize;
    if point_count == 0 || keep_ratio == 0.0 {
        return Ok(create_empty_pointcloud(data));
    }
    if keep_ratio == 1.0 {
        return Ok(data.clone());
    }

    let keep_count = (point_count as f64 * keep_ratio).ceil() as usize;

    let mut indices: Vec<usize> = (0..point_count).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);
    indices.truncate(keep_count);
    indices.sort_unstable(); // Sort for cache-friendly access

    extract_points_by_indices(data, &indices)
}

/// Uniform downsampling.
///
/// Keeps every Nth point.
fn uniform_decimate(data: &PointCloudData, step: u32) -> Result<PointCloudData, PointCloudError> {
    if step == 0 {
        return Err(PointCloudError::DecimationError(
            "Step must be positive".into(),
        ));
    }

    let point_count = data.metadata.point_count as usize;
    if point_count == 0 {
        return Ok(data.clone());
    }

    let indices: Vec<usize> = (0..point_count).step_by(step as usize).collect();

    extract_points_by_indices(data, &indices)
}

/// Extract points at given indices into a new PointCloudData.
fn extract_points_by_indices(
    data: &PointCloudData,
    indices: &[usize],
) -> Result<PointCloudData, PointCloudError> {
    let new_point_count = indices.len();

    // Extract positions
    let mut positions = Vec::with_capacity(new_point_count * 3);
    let mut bounds = crate::pointcloud::types::Bounds3D::new();

    for &idx in indices {
        let x = data.positions[idx * 3];
        let y = data.positions[idx * 3 + 1];
        let z = data.positions[idx * 3 + 2];
        bounds.extend([x as f64, y as f64, z as f64]);
        positions.push(x);
        positions.push(y);
        positions.push(z);
    }

    // Extract optional attributes
    let intensities = data.intensities.as_ref().map(|vals| {
        indices.iter().map(|&idx| vals[idx]).collect()
    });

    let colors = data.colors.as_ref().map(|vals| {
        indices.iter().map(|&idx| vals[idx]).collect()
    });

    let classifications = data.classifications.as_ref().map(|vals| {
        indices.iter().map(|&idx| vals[idx]).collect()
    });

    let mut metadata = data.metadata.clone();
    metadata.point_count = new_point_count as u64;
    metadata.bounds = if bounds.is_valid() { Some(bounds) } else { None };

    Ok(PointCloudData {
        metadata,
        positions,
        intensities,
        colors,
        classifications,
    })
}

/// Create an empty point cloud with the same attributes as the input.
fn create_empty_pointcloud(data: &PointCloudData) -> PointCloudData {
    let mut metadata = data.metadata.clone();
    metadata.point_count = 0;
    metadata.bounds = None;

    PointCloudData {
        metadata,
        positions: Vec::new(),
        intensities: if data.intensities.is_some() {
            Some(Vec::new())
        } else {
            None
        },
        colors: if data.colors.is_some() {
            Some(Vec::new())
        } else {
            None
        },
        classifications: if data.classifications.is_some() {
            Some(Vec::new())
        } else {
            None
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointcloud::types::{PointCloudFormat, PointCloudMetadata};

    fn create_test_data(point_count: usize) -> PointCloudData {
        let mut positions = Vec::with_capacity(point_count * 3);
        for i in 0..point_count {
            positions.push(i as f32);
            positions.push(i as f32);
            positions.push(i as f32);
        }

        PointCloudData {
            metadata: PointCloudMetadata {
                path: "test.las".into(),
                format: PointCloudFormat::Las,
                point_count: point_count as u64,
                file_size_bytes: 100,
                attributes: vec!["position".into()],
                bounds: None,
                has_intensity: false,
                has_rgb: false,
                has_classification: false,
            },
            positions,
            intensities: None,
            colors: None,
            classifications: None,
        }
    }

    #[test]
    fn test_voxel_grid_decimate() {
        let data = create_test_data(100);
        let result = voxel_grid_decimate(&data, 10.0).unwrap();

        // With voxel size 10, points 0-9 go in one voxel, 10-19 in another, etc.
        assert!(result.metadata.point_count < data.metadata.point_count);
        assert_eq!(result.metadata.point_count, 10); // 0-9, 10-19, ..., 90-99
    }

    #[test]
    fn test_voxel_grid_invalid_size() {
        let data = create_test_data(10);
        assert!(voxel_grid_decimate(&data, 0.0).is_err());
        assert!(voxel_grid_decimate(&data, -1.0).is_err());
    }

    #[test]
    fn test_random_decimate() {
        let data = create_test_data(100);
        let result = random_decimate(&data, 0.5).unwrap();

        // Should keep approximately 50% of points
        assert!(result.metadata.point_count >= 45 && result.metadata.point_count <= 55);
    }

    #[test]
    fn test_random_decimate_edge_cases() {
        let data = create_test_data(100);

        // Keep all
        let result = random_decimate(&data, 1.0).unwrap();
        assert_eq!(result.metadata.point_count, 100);

        // Keep none
        let result = random_decimate(&data, 0.0).unwrap();
        assert_eq!(result.metadata.point_count, 0);
    }

    #[test]
    fn test_random_decimate_invalid_ratio() {
        let data = create_test_data(10);
        assert!(random_decimate(&data, 1.5).is_err());
        assert!(random_decimate(&data, -0.1).is_err());
    }

    #[test]
    fn test_uniform_decimate() {
        let data = create_test_data(100);
        let result = uniform_decimate(&data, 2).unwrap();

        // Should keep every 2nd point
        assert_eq!(result.metadata.point_count, 50);
    }

    #[test]
    fn test_uniform_decimate_invalid_step() {
        let data = create_test_data(10);
        assert!(uniform_decimate(&data, 0).is_err());
    }

    #[test]
    fn test_decimate_preserves_attributes() {
        let mut data = create_test_data(10);
        data.intensities = Some((0..10).map(|i| i as f32 / 10.0).collect());
        data.colors = Some((0..10).map(|i| i as u32 * 1000).collect());

        let result = uniform_decimate(&data, 2).unwrap();

        assert!(result.intensities.is_some());
        assert!(result.colors.is_some());
        assert_eq!(result.intensities.as_ref().unwrap().len(), 5);
        assert_eq!(result.colors.as_ref().unwrap().len(), 5);
    }
}
