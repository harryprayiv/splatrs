//! Helpers shared by the integration tests.
#![allow(dead_code)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_range_loop)]

pub mod scene;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use splatrs::camera::Camera;
use splatrs::math::V3;
use splatrs::model::Gaussian;
use splatrs::sh::MAX_COEFFS;
use splatrs::Real;

/// Finite-difference step for O(1) losses.
#[cfg(feature = "f64")]
pub const H: Real = 1e-6;
/// Relative tolerance.
#[cfg(feature = "f64")]
pub const RTOL: Real = 1e-5;
/// Absolute tolerance, multiplied by a per-check scale.
#[cfg(feature = "f64")]
pub const ATOL: Real = 1e-8;

/// Finite-difference step for O(1) losses.
#[cfg(not(feature = "f64"))]
pub const H: Real = 1e-2;
/// Relative tolerance.
#[cfg(not(feature = "f64"))]
pub const RTOL: Real = 5e-2;
/// Absolute tolerance, multiplied by a per-check scale.
#[cfg(not(feature = "f64"))]
pub const ATOL: Real = 1e-3;

/// Assert that an analytic derivative matches a finite difference.
pub fn check(what: &str, analytic: Real, numeric: Real, scale: Real) {
    let tol = ATOL * scale + RTOL * analytic.abs().max(numeric.abs());
    assert!(
        (analytic - numeric).abs() <= tol,
        "{what}: analytic {analytic:.9e} vs finite difference {numeric:.9e} (tol {tol:.3e})"
    );
}

/// Central difference of `f` at `x`.
pub fn central<F: FnMut(Real) -> Real>(x: Real, h: Real, mut f: F) -> Real {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Seeded generator.
pub fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Uniform sample in `[lo, hi)`.
pub fn rnd(rng: &mut ChaCha8Rng, lo: Real, hi: Real) -> Real {
    lo + (hi - lo) * rng.random::<f64>() as Real
}

/// A random Gaussian near the origin with clearly positive colours.
pub fn random_gaussian(rng: &mut ChaCha8Rng) -> Gaussian {
    let mut sh = [[0.0; 3]; MAX_COEFFS];
    for k in 0..MAX_COEFFS {
        for c in 0..3 {
            sh[k][c] = if k == 0 {
                rnd(rng, 0.2, 1.5)
            } else {
                rnd(rng, -0.15, 0.15)
            };
        }
    }
    Gaussian {
        mean: [
            rnd(rng, -0.6, 0.6),
            rnd(rng, -0.4, 0.4),
            rnd(rng, -0.5, 0.5),
        ],
        log_scale: [
            rnd(rng, -2.5, -1.2),
            rnd(rng, -2.5, -1.2),
            rnd(rng, -2.5, -1.2),
        ],
        quat: [
            rnd(rng, 0.5, 1.5),
            rnd(rng, -0.7, 0.7),
            rnd(rng, -0.7, 0.7),
            rnd(rng, -0.7, 0.7),
        ],
        opacity: rnd(rng, -1.0, 2.0),
        sh,
    }
}

/// A camera three units from the origin looking at it.
pub fn test_camera(width: u32, height: u32) -> Camera {
    Camera::look_at(
        "cam",
        width,
        height,
        width as Real * 0.9,
        width as Real * 0.9,
        V3::new(0.3, -0.2, -3.0),
        V3::new(0.0, 0.0, 0.0),
        V3::new(0.0, -1.0, 0.0),
    )
}

/// Number of scalar parameters per Gaussian.
pub const NUM_PARAMS: usize = 59;

/// The Gaussian with scalar parameter `i` (0..59) shifted by `delta`.
pub fn with_param(g: &Gaussian, i: usize, delta: Real) -> Gaussian {
    let mut out = *g;
    match i {
        0..=2 => out.mean[i] += delta,
        3..=5 => out.log_scale[i - 3] += delta,
        6..=9 => out.quat[i - 6] += delta,
        10 => out.opacity += delta,
        _ => {
            let j = i - 11;
            out.sh[j / 3][j % 3] += delta;
        }
    }
    out
}

/// Scalar parameter `i` of a parameter-shaped record.
pub fn param(g: &Gaussian, i: usize) -> Real {
    match i {
        0..=2 => g.mean[i],
        3..=5 => g.log_scale[i - 3],
        6..=9 => g.quat[i - 6],
        10 => g.opacity,
        _ => {
            let j = i - 11;
            g.sh[j / 3][j % 3]
        }
    }
}

/// Human-readable name of scalar parameter `i`.
pub fn param_name(i: usize) -> String {
    match i {
        0..=2 => format!("mean[{i}]"),
        3..=5 => format!("log_scale[{}]", i - 3),
        6..=9 => format!("quat[{}]", i - 6),
        10 => "opacity".to_string(),
        _ => {
            let j = i - 11;
            format!("sh[{}][{}]", j / 3, j % 3)
        }
    }
}