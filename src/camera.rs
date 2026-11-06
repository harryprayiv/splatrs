//! Pinhole camera with COLMAP conventions.
//!
//! World-to-camera: `x_cam = rot * x_world + trans`. The camera looks down +z, x points
//! right and y points down. Pixel `(i, j)` has its centre at continuous image coordinates
//! `(i + 0.5, j + 0.5)`, and a camera-space point projects to
//! `u = fx * x / z + cx`, `v = fy * y / z + cy`.

use crate::math::{M3, V3};
use crate::Real;

/// A pinhole camera.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    /// Image file name, as recorded by COLMAP.
    pub name: String,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Focal length along x, in pixels.
    pub fx: Real,
    /// Focal length along y, in pixels.
    pub fy: Real,
    /// Principal point x, in continuous pixel coordinates.
    pub cx: Real,
    /// Principal point y, in continuous pixel coordinates.
    pub cy: Real,
    /// World-to-camera rotation.
    pub rot: M3,
    /// World-to-camera translation.
    pub trans: V3,
}

impl Camera {
    /// Transform a world point into camera space.
    pub fn world_to_cam(&self, p: V3) -> V3 {
        self.rot.mul_v(p) + self.trans
    }

    /// Camera centre in world coordinates: `-rot^T * trans`.
    pub fn center(&self) -> V3 {
        -(self.rot.transpose().mul_v(self.trans))
    }

    /// The same camera with its image resampled to `width x height`. Intrinsics scale
    /// with the actual ratio of the new and old sizes, so integer rounding of the size is
    /// accounted for exactly.
    pub fn resized(&self, width: u32, height: u32) -> Camera {
        let sx = width as Real / self.width as Real;
        let sy = height as Real / self.height as Real;
        Camera {
            name: self.name.clone(),
            width,
            height,
            fx: self.fx * sx,
            fy: self.fy * sy,
            cx: self.cx * sx,
            cy: self.cy * sy,
            rot: self.rot,
            trans: self.trans,
        }
    }

    /// Build a camera that looks from `eye` towards `target`, with `up` giving the
    /// world up direction. Used by the synthetic tests and examples.
    pub fn look_at(
        name: &str,
        width: u32,
        height: u32,
        fx: Real,
        fy: Real,
        eye: V3,
        target: V3,
        up: V3,
    ) -> Camera {
        // Camera axes in world coordinates: forward = +z, right = +x, down = +y.
        let forward = (target - eye).normalized();
        let right = forward.cross(up).normalized();
        let down = forward.cross(right);
        let rot = M3::from_rows(right, down, forward);
        let trans = -(rot.mul_v(eye));
        Camera {
            name: name.to_string(),
            width,
            height,
            fx,
            fy,
            cx: width as Real / 2.0,
            cy: height as Real / 2.0,
            rot,
            trans,
        }
    }
}

/// Scene extent used to scale learning rates and densification thresholds, following the
/// reference implementation: 1.1 times the largest distance from any camera centre to the
/// mean camera centre.
pub fn scene_extent(cameras: &[Camera]) -> Real {
    if cameras.is_empty() {
        return 1.0;
    }
    let mut center = V3::ZERO;
    for c in cameras {
        center += c.center();
    }
    center = center * (1.0 / cameras.len() as Real);
    let mut radius: Real = 0.0;
    for c in cameras {
        radius = radius.max((c.center() - center).norm());
    }
    let extent = radius * 1.1;
    if extent > 0.0 {
        extent
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn look_at_projects_target_to_principal_point() {
        let cam = Camera::look_at(
            "t",
            64,
            48,
            50.0,
            50.0,
            V3::new(1.0, 2.0, -5.0),
            V3::new(0.0, 0.0, 0.0),
            V3::new(0.0, -1.0, 0.0),
        );
        let p = cam.world_to_cam(V3::ZERO);
        assert!(p.x.abs() < 1e-5 && p.y.abs() < 1e-5 && p.z > 0.0);
        let c = cam.center();
        assert!((c - V3::new(1.0, 2.0, -5.0)).norm() < 1e-4);
    }
}