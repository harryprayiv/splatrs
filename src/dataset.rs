//! Loading a COLMAP reconstruction plus its images into training views.

use crate::camera::{scene_extent, Camera};
use crate::colmap::{self, ColmapModel};
use crate::error::{Error, Result};
use crate::math::{Quat, V3};
use crate::Real;
use image::imageops::FilterType;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// One training image with its camera.
#[derive(Clone, Debug)]
pub struct View {
    /// Camera at the (possibly downscaled) image resolution.
    pub camera: Camera,
    /// RGB pixels, row-major, 8 bits per channel.
    pub rgb: Vec<u8>,
}

impl View {
    /// Pixels as `Real` in [0, 1].
    pub fn to_real(&self) -> Vec<Real> {
        self.rgb.iter().map(|&v| v as Real / 255.0).collect()
    }
}

/// A loaded dataset.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// Views in COLMAP image order.
    pub views: Vec<View>,
    /// Sparse points with colours, used to initialise the model.
    pub points: Vec<(V3, [u8; 3])>,
    /// Scene extent (see [`scene_extent`]).
    pub extent: Real,
}

/// Locate the sparse model directory under a COLMAP project root: `root/sparse/0`, then
/// `root/sparse`, unless an explicit directory is given.
pub fn locate_sparse(root: &Path, sparse: Option<&Path>) -> Result<PathBuf> {
    match sparse {
        Some(p) => Ok(p.to_path_buf()),
        None => {
            let a = root.join("sparse").join("0");
            let b = root.join("sparse");
            if a.is_dir() {
                Ok(a)
            } else if b.is_dir() {
                Ok(b)
            } else {
                Err(Error::Invalid(format!(
                    "no sparse/0 or sparse directory under {}",
                    root.display()
                )))
            }
        }
    }
}

/// Locate the sparse model and image directories under a COLMAP project root. Images are
/// looked for at `root/images` unless an explicit directory is given.
pub fn locate(
    root: &Path,
    sparse: Option<&Path>,
    images: Option<&Path>,
) -> Result<(PathBuf, PathBuf)> {
    let sparse = locate_sparse(root, sparse)?;
    let images = match images {
        Some(p) => p.to_path_buf(),
        None => root.join("images"),
    };
    if !images.is_dir() {
        return Err(Error::Invalid(format!(
            "images directory {} does not exist",
            images.display()
        )));
    }
    Ok((sparse, images))
}

/// Convert a COLMAP image plus its camera into a [`Camera`] at full resolution.
pub fn camera_from_colmap(model: &ColmapModel, img: &colmap::ColmapImage) -> Result<Camera> {
    let cam = model.cameras.get(&img.camera_id).ok_or_else(|| {
        Error::Invalid(format!(
            "image {} refers to unknown camera {}",
            img.name, img.camera_id
        ))
    })?;
    let (fx, fy, cx, cy) = match cam.model {
        colmap::CameraModel::SimplePinhole => {
            (cam.params[0], cam.params[0], cam.params[1], cam.params[2])
        }
        colmap::CameraModel::Pinhole => {
            (cam.params[0], cam.params[1], cam.params[2], cam.params[3])
        }
        other => {
            return Err(Error::UnsupportedCameraModel {
                id: cam.id,
                model: other.name().to_string(),
            });
        }
    };
    let q = Quat {
        w: img.qvec[0] as Real,
        x: img.qvec[1] as Real,
        y: img.qvec[2] as Real,
        z: img.qvec[3] as Real,
    }
    .normalized();
    Ok(Camera {
        name: img.name.clone(),
        width: cam.width as u32,
        height: cam.height as u32,
        fx: fx as Real,
        fy: fy as Real,
        cx: cx as Real,
        cy: cy as Real,
        rot: q.to_mat(),
        trans: V3::new(
            img.tvec[0] as Real,
            img.tvec[1] as Real,
            img.tvec[2] as Real,
        ),
    })
}

/// Load the model and every registered image, downscaling by the integer `downscale`.
/// Views are ordered by image name, as the reference loader orders them, so that the
/// `--eval` hold-out picks the same images.
pub fn load(sparse: &Path, images: &Path, downscale: u32) -> Result<Dataset> {
    let downscale = downscale.max(1);
    let mut model = colmap::read_model(sparse)?;
    if model.images.is_empty() {
        return Err(Error::Invalid(format!(
            "no registered images in {}",
            sparse.display()
        )));
    }
    model.images.sort_by(|a, b| a.name.cmp(&b.name));
    let views: Vec<View> = model
        .images
        .par_iter()
        .map(|img| -> Result<View> {
            let camera = camera_from_colmap(&model, img)?;
            let path = images.join(&img.name);
            let decoded = image::open(&path)
                .map_err(|e| Error::Image {
                    path: path.clone(),
                    source: e,
                })?
                .to_rgb8();
            let camera = match_image_size(&camera, decoded.width(), decoded.height()).map_err(
                |e| match e {
                    Error::ImageSizeMismatch {
                        expected_w,
                        expected_h,
                        actual_w,
                        actual_h,
                        ..
                    } => Error::ImageSizeMismatch {
                        path: path.clone(),
                        expected_w,
                        expected_h,
                        actual_w,
                        actual_h,
                    },
                    other => other,
                },
            )?;
            let (w, h) = (
                ((camera.width as f64 / downscale as f64).round() as u32).max(1),
                ((camera.height as f64 / downscale as f64).round() as u32).max(1),
            );
            let resized = if downscale > 1 {
                image::imageops::resize(&decoded, w, h, FilterType::Lanczos3)
            } else {
                decoded
            };
            Ok(View {
                camera: camera.resized(w, h),
                rgb: resized.into_raw(),
            })
        })
        .collect::<Result<Vec<View>>>()?;
    let points: Vec<(V3, [u8; 3])> = model
        .points
        .iter()
        .map(|p| {
            (
                V3::new(p.xyz[0] as Real, p.xyz[1] as Real, p.xyz[2] as Real),
                p.rgb,
            )
        })
        .collect();
    let cameras: Vec<Camera> = views.iter().map(|v| v.camera.clone()).collect();
    let extent = scene_extent(&cameras);
    Ok(Dataset {
        views,
        points,
        extent,
    })
}

/// Reconcile a camera with the size of the image file it belongs to. Datasets often ship
/// images resampled after COLMAP ran (the 3DGS `tandt_db` archive stores half-size images
/// against full-size cameras), so a uniformly scaled image is accepted and the intrinsics
/// are scaled to it. The two axis scale factors must agree to within one pixel of rounding;
/// anything else (a crop, a rotation, a different aspect ratio) is rejected, because the
/// intrinsics could not be trusted.
pub fn match_image_size(camera: &Camera, width: u32, height: u32) -> Result<Camera> {
    if width == camera.width && height == camera.height {
        return Ok(camera.clone());
    }
    let (cw, ch) = (camera.width as f64, camera.height as f64);
    let (w, h) = (width as f64, height as f64);
    let width_from_height = cw * h / ch;
    let height_from_width = ch * w / cw;
    if width == 0
        || height == 0
        || (width_from_height - w).abs() > 1.0
        || (height_from_width - h).abs() > 1.0
    {
        return Err(Error::ImageSizeMismatch {
            path: PathBuf::from(&camera.name),
            expected_w: camera.width,
            expected_h: camera.height,
            actual_w: width,
            actual_h: height,
        });
    }
    Ok(camera.resized(width, height))
}

/// Split views into training and held-out sets the way the reference `--eval` flag does:
/// every `hold`-th view (by index) is held out.
pub fn split_eval(views: Vec<View>, hold: usize) -> (Vec<View>, Vec<View>) {
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (i, v) in views.into_iter().enumerate() {
        if hold > 0 && i % hold == 0 {
            test.push(v);
        } else {
            train.push(v);
        }
    }
    (train, test)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::M3;

    fn truck_camera() -> Camera {
        Camera {
            name: "000001.jpg".to_string(),
            width: 1957,
            height: 1091,
            fx: 1163.2547,
            fy: 1156.2804,
            cx: 978.5,
            cy: 545.5,
            rot: M3::IDENTITY,
            trans: V3::ZERO,
        }
    }

    #[test]
    fn same_size_is_unchanged() {
        let c = truck_camera();
        assert_eq!(match_image_size(&c, 1957, 1091).unwrap(), c);
    }

    #[test]
    fn half_size_images_rescale_intrinsics() {
        // The tandt_db archive: cameras at 1957x1091, images at 979x546.
        let c = match_image_size(&truck_camera(), 979, 546).unwrap();
        assert_eq!((c.width, c.height), (979, 546));
        let sx = 979.0 / 1957.0;
        let sy = 546.0 / 1091.0;
        assert!((c.fx as f64 - 1163.2547 * sx).abs() < 1e-3);
        assert!((c.fy as f64 - 1156.2804 * sy).abs() < 1e-3);
        assert!((c.cx as f64 - 978.5 * sx).abs() < 1e-3);
        assert!((c.cy as f64 - 545.5 * sy).abs() < 1e-3);
    }

    #[test]
    fn different_aspect_ratio_is_rejected() {
        // A crop or a rotated image must not be silently rescaled.
        assert!(match_image_size(&truck_camera(), 979, 600).is_err());
        assert!(match_image_size(&truck_camera(), 1091, 1957).is_err());
        assert!(match_image_size(&truck_camera(), 0, 0).is_err());
    }
}