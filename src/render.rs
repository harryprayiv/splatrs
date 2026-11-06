//! Rendering a model from a camera (forward pass only).

use crate::camera::Camera;
use crate::error::{Error, Result};
use crate::model::Model;
use crate::project::{project_all, ProjOpts};
use crate::rasterize::{bin, rasterize_forward, RasterSettings};
use crate::Real;
use std::path::Path;

/// Render `model` through `cam` with the given background. Returns `width * height * 3`
/// values in [0, 1] (colours are not clamped by the compositing itself, so values can
/// slightly exceed 1 when SH colours do).
pub fn render(model: &Model, cam: &Camera, opts: &ProjOpts, background: [Real; 3]) -> Vec<Real> {
    let projs = project_all(model, cam, opts);
    let binning = bin(&projs, cam.width, cam.height);
    let settings = RasterSettings {
        width: cam.width,
        height: cam.height,
        background,
    };
    rasterize_forward(&projs, &binning, &settings).image
}

/// Convert an image in [0, 1] to 8-bit RGB with rounding and clamping.
pub fn to_rgb8(image: &[Real]) -> Vec<u8> {
    image
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// Write an image in [0, 1] as a PNG.
pub fn save_png(path: &Path, image: &[Real], width: u32, height: u32) -> Result<()> {
    let buf = to_rgb8(image);
    let img = image::RgbImage::from_raw(width, height, buf)
        .ok_or_else(|| Error::Invalid("image buffer has the wrong size".to_string()))?;
    img.save(path).map_err(|e| Error::Image {
        path: path.to_path_buf(),
        source: e,
    })
}