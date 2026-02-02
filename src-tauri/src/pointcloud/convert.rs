//! Point cloud format conversion.
//!
//! Converts between PCD, PLY, LAS, and LAZ formats.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::pointcloud::decimate::decimate;
use crate::pointcloud::error::PointCloudError;
use crate::pointcloud::io::read_pointcloud;
use crate::pointcloud::types::{ConversionOptions, PointCloudData, PointCloudFormat, PointCloudMetadata};

/// Convert a point cloud file to another format.
///
/// Optionally applies decimation during conversion.
pub fn convert_pointcloud(
    input_path: &str,
    output_path: &str,
    options: ConversionOptions,
) -> Result<PointCloudMetadata, PointCloudError> {
    let input = Path::new(input_path);
    let output = Path::new(output_path);

    if !input.exists() {
        return Err(PointCloudError::FileNotFound(input.display().to_string()));
    }

    // Validate output extension matches target format
    let out_ext = output
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if out_ext.to_lowercase() != options.target_format.extension() {
        return Err(PointCloudError::ConversionError(format!(
            "Output path extension '{}' doesn't match target format '{}'",
            out_ext,
            options.target_format.extension()
        )));
    }

    // Read input
    let mut data = read_pointcloud(input_path)?;

    // Apply decimation if requested
    if let Some(method) = options.decimation {
        data = decimate(&data, method)?;
    }

    // Filter attributes based on options
    if !options.preserve_intensity {
        data.intensities = None;
    }
    if !options.preserve_rgb {
        data.colors = None;
    }
    if !options.preserve_classification {
        data.classifications = None;
    }

    // Write output
    match options.target_format {
        PointCloudFormat::Ply => write_ply(&data, output),
        PointCloudFormat::Pcd => write_pcd(&data, output),
        PointCloudFormat::Las | PointCloudFormat::Laz => {
            // LAS/LAZ writing requires pasture-io write support
            // For now, return an error suggesting PLY/PCD as alternatives
            Err(PointCloudError::ConversionError(
                "LAS/LAZ writing not yet implemented. Use PLY or PCD as target format.".into(),
            ))
        }
    }
}

/// Write point cloud data to PLY format (ASCII).
fn write_ply(data: &PointCloudData, path: &Path) -> Result<PointCloudMetadata, PointCloudError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let point_count = data.metadata.point_count;
    let has_intensity = data.intensities.is_some();
    let has_rgb = data.colors.is_some();

    // Write header
    writeln!(writer, "ply")?;
    writeln!(writer, "format ascii 1.0")?;
    writeln!(writer, "element vertex {}", point_count)?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;

    if has_intensity {
        writeln!(writer, "property float intensity")?;
    }
    if has_rgb {
        writeln!(writer, "property uchar red")?;
        writeln!(writer, "property uchar green")?;
        writeln!(writer, "property uchar blue")?;
    }

    writeln!(writer, "end_header")?;

    // Write point data
    for i in 0..point_count as usize {
        let x = data.positions[i * 3];
        let y = data.positions[i * 3 + 1];
        let z = data.positions[i * 3 + 2];

        write!(writer, "{} {} {}", x, y, z)?;

        if let Some(ref intensities) = data.intensities {
            write!(writer, " {}", intensities[i])?;
        }

        if let Some(ref colors) = data.colors {
            let packed = colors[i];
            let r = (packed >> 16) & 0xFF;
            let g = (packed >> 8) & 0xFF;
            let b = packed & 0xFF;
            write!(writer, " {} {} {}", r, g, b)?;
        }

        writeln!(writer)?;
    }

    writer.flush()?;

    // Build metadata for output file
    let file_size_bytes = std::fs::metadata(path)?.len();
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
        bounds: data.metadata.bounds.clone(),
        has_intensity,
        has_rgb,
        has_classification: false,
    })
}

/// Write point cloud data to PCD format (ASCII).
fn write_pcd(data: &PointCloudData, path: &Path) -> Result<PointCloudMetadata, PointCloudError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let point_count = data.metadata.point_count;
    let has_intensity = data.intensities.is_some();
    let has_rgb = data.colors.is_some();

    // Build FIELDS line
    let mut fields = vec!["x", "y", "z"];
    let mut sizes = vec!["4", "4", "4"];
    let mut types = vec!["F", "F", "F"];
    let mut counts = vec!["1", "1", "1"];

    if has_intensity {
        fields.push("intensity");
        sizes.push("4");
        types.push("F");
        counts.push("1");
    }
    if has_rgb {
        fields.push("rgb");
        sizes.push("4");
        types.push("F");
        counts.push("1");
    }

    // Write header
    writeln!(writer, "# .PCD v0.7 - Point Cloud Data file format")?;
    writeln!(writer, "VERSION 0.7")?;
    writeln!(writer, "FIELDS {}", fields.join(" "))?;
    writeln!(writer, "SIZE {}", sizes.join(" "))?;
    writeln!(writer, "TYPE {}", types.join(" "))?;
    writeln!(writer, "COUNT {}", counts.join(" "))?;
    writeln!(writer, "WIDTH {}", point_count)?;
    writeln!(writer, "HEIGHT 1")?;
    writeln!(writer, "VIEWPOINT 0 0 0 1 0 0 0")?;
    writeln!(writer, "POINTS {}", point_count)?;
    writeln!(writer, "DATA ascii")?;

    // Write point data
    for i in 0..point_count as usize {
        let x = data.positions[i * 3];
        let y = data.positions[i * 3 + 1];
        let z = data.positions[i * 3 + 2];

        write!(writer, "{} {} {}", x, y, z)?;

        if let Some(ref intensities) = data.intensities {
            write!(writer, " {}", intensities[i])?;
        }

        if let Some(ref colors) = data.colors {
            // PCD stores RGB as a float (bitcast from u32)
            let packed = colors[i];
            let rgb_float = f32::from_bits(packed);
            write!(writer, " {}", rgb_float)?;
        }

        writeln!(writer)?;
    }

    writer.flush()?;

    // Build metadata for output file
    let file_size_bytes = std::fs::metadata(path)?.len();
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
        bounds: data.metadata.bounds.clone(),
        has_intensity,
        has_rgb,
        has_classification: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointcloud::types::{Bounds3D, DecimationMethod};
    use std::io::Read;
    use tempfile::tempdir;

    fn create_test_data() -> PointCloudData {
        PointCloudData {
            metadata: PointCloudMetadata {
                path: "test.las".into(),
                format: PointCloudFormat::Las,
                point_count: 3,
                file_size_bytes: 100,
                attributes: vec!["position".into(), "rgb".into()],
                bounds: Some(Bounds3D {
                    min: [0.0, 0.0, 0.0],
                    max: [2.0, 2.0, 2.0],
                }),
                has_intensity: false,
                has_rgb: true,
                has_classification: false,
            },
            positions: vec![
                0.0, 0.0, 0.0, // point 0
                1.0, 1.0, 1.0, // point 1
                2.0, 2.0, 2.0, // point 2
            ],
            intensities: None,
            colors: Some(vec![
                0xFF0000, // red
                0x00FF00, // green
                0x0000FF, // blue
            ]),
            classifications: None,
        }
    }

    #[test]
    fn test_write_ply() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("output.ply");

        let data = create_test_data();
        let metadata = write_ply(&data, &output_path).unwrap();

        assert_eq!(metadata.format, PointCloudFormat::Ply);
        assert_eq!(metadata.point_count, 3);
        assert!(metadata.has_rgb);

        // Read and verify content
        let mut content = String::new();
        File::open(&output_path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();

        assert!(content.contains("ply"));
        assert!(content.contains("format ascii 1.0"));
        assert!(content.contains("element vertex 3"));
        assert!(content.contains("property uchar red"));
    }

    #[test]
    fn test_write_pcd() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("output.pcd");

        let data = create_test_data();
        let metadata = write_pcd(&data, &output_path).unwrap();

        assert_eq!(metadata.format, PointCloudFormat::Pcd);
        assert_eq!(metadata.point_count, 3);

        // Read and verify content
        let mut content = String::new();
        File::open(&output_path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();

        assert!(content.contains("VERSION 0.7"));
        assert!(content.contains("FIELDS x y z rgb"));
        assert!(content.contains("POINTS 3"));
        assert!(content.contains("DATA ascii"));
    }

    #[test]
    fn test_conversion_with_decimation() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("output.ply");

        // Create larger test data
        let mut data = create_test_data();
        data.metadata.point_count = 10;
        data.positions = (0..30).map(|i| (i / 3) as f32).collect();
        data.colors = Some((0..10).map(|i| i as u32 * 1000).collect());

        // We can't call convert_pointcloud directly since it reads from file,
        // but we can test the decimation + write flow
        let decimated = crate::pointcloud::decimate::decimate(
            &data,
            DecimationMethod::Uniform { step: 2 },
        )
        .unwrap();

        let metadata = write_ply(&decimated, &output_path).unwrap();
        assert_eq!(metadata.point_count, 5);
    }
}
