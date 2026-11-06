//! COLMAP sparse reconstructions: the data types, and readers and writers for the binary
//! (`.bin`) and text (`.txt`) formats. 2D point observations and 3D point tracks are parsed
//! past but not kept.

mod binary;
mod text;

pub use binary::write_model_bin;
pub use text::write_model_txt;

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// COLMAP camera models, with their parameter counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraModel {
    /// `f, cx, cy`
    SimplePinhole,
    /// `fx, fy, cx, cy`
    Pinhole,
    /// `f, cx, cy, k`
    SimpleRadial,
    /// `f, cx, cy, k1, k2`
    Radial,
    /// `fx, fy, cx, cy, k1, k2, p1, p2`
    OpenCv,
    /// `fx, fy, cx, cy, k1, k2, k3, k4`
    OpenCvFisheye,
    /// `fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6`
    FullOpenCv,
    /// `fx, fy, cx, cy, omega`
    Fov,
    /// `f, cx, cy, k`
    SimpleRadialFisheye,
    /// `f, cx, cy, k1, k2`
    RadialFisheye,
    /// `fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1`
    ThinPrismFisheye,
    /// `fx, fy, cx, cy, k0..k5, p0, p1, s0..s3` (COLMAP 3.11+)
    RadTanThinPrismFisheye,
}

impl CameraModel {
    /// Model from its numeric id in the binary format.
    pub fn from_id(id: i32) -> Option<CameraModel> {
        Some(match id {
            0 => CameraModel::SimplePinhole,
            1 => CameraModel::Pinhole,
            2 => CameraModel::SimpleRadial,
            3 => CameraModel::Radial,
            4 => CameraModel::OpenCv,
            5 => CameraModel::OpenCvFisheye,
            6 => CameraModel::FullOpenCv,
            7 => CameraModel::Fov,
            8 => CameraModel::SimpleRadialFisheye,
            9 => CameraModel::RadialFisheye,
            10 => CameraModel::ThinPrismFisheye,
            11 => CameraModel::RadTanThinPrismFisheye,
            _ => return None,
        })
    }

    /// Model from its name in the text format.
    pub fn from_name(name: &str) -> Option<CameraModel> {
        Some(match name {
            "SIMPLE_PINHOLE" => CameraModel::SimplePinhole,
            "PINHOLE" => CameraModel::Pinhole,
            "SIMPLE_RADIAL" => CameraModel::SimpleRadial,
            "RADIAL" => CameraModel::Radial,
            "OPENCV" => CameraModel::OpenCv,
            "OPENCV_FISHEYE" => CameraModel::OpenCvFisheye,
            "FULL_OPENCV" => CameraModel::FullOpenCv,
            "FOV" => CameraModel::Fov,
            "SIMPLE_RADIAL_FISHEYE" => CameraModel::SimpleRadialFisheye,
            "RADIAL_FISHEYE" => CameraModel::RadialFisheye,
            "THIN_PRISM_FISHEYE" => CameraModel::ThinPrismFisheye,
            "RAD_TAN_THIN_PRISM_FISHEYE" => CameraModel::RadTanThinPrismFisheye,
            _ => return None,
        })
    }

    /// Numeric id used by the binary format.
    pub fn id(self) -> i32 {
        match self {
            CameraModel::SimplePinhole => 0,
            CameraModel::Pinhole => 1,
            CameraModel::SimpleRadial => 2,
            CameraModel::Radial => 3,
            CameraModel::OpenCv => 4,
            CameraModel::OpenCvFisheye => 5,
            CameraModel::FullOpenCv => 6,
            CameraModel::Fov => 7,
            CameraModel::SimpleRadialFisheye => 8,
            CameraModel::RadialFisheye => 9,
            CameraModel::ThinPrismFisheye => 10,
            CameraModel::RadTanThinPrismFisheye => 11,
        }
    }

    /// Name used by the text format.
    pub fn name(self) -> &'static str {
        match self {
            CameraModel::SimplePinhole => "SIMPLE_PINHOLE",
            CameraModel::Pinhole => "PINHOLE",
            CameraModel::SimpleRadial => "SIMPLE_RADIAL",
            CameraModel::Radial => "RADIAL",
            CameraModel::OpenCv => "OPENCV",
            CameraModel::OpenCvFisheye => "OPENCV_FISHEYE",
            CameraModel::FullOpenCv => "FULL_OPENCV",
            CameraModel::Fov => "FOV",
            CameraModel::SimpleRadialFisheye => "SIMPLE_RADIAL_FISHEYE",
            CameraModel::RadialFisheye => "RADIAL_FISHEYE",
            CameraModel::ThinPrismFisheye => "THIN_PRISM_FISHEYE",
            CameraModel::RadTanThinPrismFisheye => "RAD_TAN_THIN_PRISM_FISHEYE",
        }
    }

    /// Number of intrinsic parameters.
    pub fn num_params(self) -> usize {
        match self {
            CameraModel::SimplePinhole => 3,
            CameraModel::Pinhole => 4,
            CameraModel::SimpleRadial => 4,
            CameraModel::Radial => 5,
            CameraModel::OpenCv => 8,
            CameraModel::OpenCvFisheye => 8,
            CameraModel::FullOpenCv => 12,
            CameraModel::Fov => 5,
            CameraModel::SimpleRadialFisheye => 4,
            CameraModel::RadialFisheye => 5,
            CameraModel::ThinPrismFisheye => 12,
            CameraModel::RadTanThinPrismFisheye => 16,
        }
    }

    /// Whether the model is distortion-free and therefore usable for training directly.
    pub fn is_pinhole(self) -> bool {
        matches!(self, CameraModel::SimplePinhole | CameraModel::Pinhole)
    }
}

/// A COLMAP camera (intrinsics).
#[derive(Clone, Debug, PartialEq)]
pub struct ColmapCamera {
    /// Camera id.
    pub id: u32,
    /// Camera model.
    pub model: CameraModel,
    /// Image width.
    pub width: u64,
    /// Image height.
    pub height: u64,
    /// Intrinsic parameters in the model's order.
    pub params: Vec<f64>,
}

/// A registered COLMAP image (extrinsics).
#[derive(Clone, Debug, PartialEq)]
pub struct ColmapImage {
    /// Image id.
    pub id: u32,
    /// World-to-camera rotation as a quaternion `(w, x, y, z)`.
    pub qvec: [f64; 4],
    /// World-to-camera translation.
    pub tvec: [f64; 3],
    /// Id of the camera used by this image.
    pub camera_id: u32,
    /// File name relative to the images directory.
    pub name: String,
}

/// A triangulated 3D point.
#[derive(Clone, Debug, PartialEq)]
pub struct ColmapPoint {
    /// Point id.
    pub id: u64,
    /// World position.
    pub xyz: [f64; 3],
    /// Colour.
    pub rgb: [u8; 3],
    /// Reprojection error.
    pub error: f64,
}

/// A sparse reconstruction.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColmapModel {
    /// Cameras keyed by id.
    pub cameras: HashMap<u32, ColmapCamera>,
    /// Registered images in file order.
    pub images: Vec<ColmapImage>,
    /// 3D points in file order.
    pub points: Vec<ColmapPoint>,
}

/// Read a sparse model directory containing `cameras`, `images` and `points3D` as either
/// `.bin` or `.txt` files.
pub fn read_model(dir: &Path) -> Result<ColmapModel> {
    let pick = |stem: &str| -> Result<PathBuf> {
        let bin = dir.join(format!("{stem}.bin"));
        if bin.is_file() {
            return Ok(bin);
        }
        let txt = dir.join(format!("{stem}.txt"));
        if txt.is_file() {
            return Ok(txt);
        }
        Err(Error::Invalid(format!(
            "neither {stem}.bin nor {stem}.txt found in {}",
            dir.display()
        )))
    };
    let cameras_path = pick("cameras")?;
    let images_path = pick("images")?;
    let points_path = pick("points3D")?;
    let is_bin = |p: &Path| p.extension().map(|e| e == "bin").unwrap_or(false);

    let cameras = if is_bin(&cameras_path) {
        binary::read_cameras(&cameras_path)?
    } else {
        text::read_cameras(&cameras_path)?
    };
    let images = if is_bin(&images_path) {
        binary::read_images(&images_path)?
    } else {
        text::read_images(&images_path)?
    };
    let points = if is_bin(&points_path) {
        binary::read_points(&points_path)?
    } else {
        text::read_points(&points_path)?
    };
    let cameras = cameras.into_iter().map(|c| (c.id, c)).collect();
    Ok(ColmapModel {
        cameras,
        images,
        points,
    })
}

/// Rotation matrix (row-major) to a unit quaternion `(w, x, y, z)`, numerically stable for
/// every rotation (Shepperd's method).
pub fn rotation_to_qvec(m: &[[f64; 3]; 3]) -> [f64; 4] {
    let tr = m[0][0] + m[1][1] + m[2][2];
    let (w, x, y, z);
    if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        w = 0.25 * s;
        x = (m[2][1] - m[1][2]) / s;
        y = (m[0][2] - m[2][0]) / s;
        z = (m[1][0] - m[0][1]) / s;
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        w = (m[2][1] - m[1][2]) / s;
        x = 0.25 * s;
        y = (m[0][1] + m[1][0]) / s;
        z = (m[0][2] + m[2][0]) / s;
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        w = (m[0][2] - m[2][0]) / s;
        x = (m[0][1] + m[1][0]) / s;
        y = 0.25 * s;
        z = (m[1][2] + m[2][1]) / s;
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        w = (m[1][0] - m[0][1]) / s;
        x = (m[0][2] + m[2][0]) / s;
        y = (m[1][2] + m[2][1]) / s;
        z = 0.25 * s;
    }
    let n = (w * w + x * x + y * y + z * z).sqrt();
    [w / n, x / n, y / n, z / n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Quat;
    use crate::Real;

    #[test]
    fn rotation_to_quaternion_round_trips_all_branches() {
        let cases: [[Real; 4]; 5] = [
            [0.9, 0.1, 0.2, 0.3],
            [0.05, 0.95, 0.1, 0.1],
            [0.05, 0.1, 0.95, 0.1],
            [0.05, 0.1, 0.1, 0.95],
            [-0.3, 0.5, -0.6, 0.4],
        ];
        for c in cases {
            let q = Quat::from_array(c).normalized();
            let r = q.to_mat();
            let mut m = [[0.0f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] = r.m[i][j] as f64;
                }
            }
            let back = rotation_to_qvec(&m);
            let r2 = Quat {
                w: back[0] as Real,
                x: back[1] as Real,
                y: back[2] as Real,
                z: back[3] as Real,
            }
            .to_mat();
            for i in 0..3 {
                for j in 0..3 {
                    assert!((r.m[i][j] - r2.m[i][j]).abs() < 1e-5, "case {c:?}");
                }
            }
        }
    }
}