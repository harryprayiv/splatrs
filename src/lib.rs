//! splatrs: 3D Gaussian splatting (Kerbl et al. 2023) as a CPU reference implementation.
//!
//! Every differentiable stage (spherical harmonics, EWA projection, tile-sorted alpha
//! compositing, L1 + SSIM loss) has a hand-derived backward pass that is checked against
//! central finite differences in the test suite. The crate builds with `f32` scalars by
//! default; the `f64` cargo feature switches every scalar to `f64` so the gradient checks
//! can use tight tolerances.
//!
//! Index-based loops are used deliberately in the numeric code, so the two clippy lints
//! that argue with that style are allowed crate-wide. Casts between `Real` and the fixed
//! `f32`/`f64` types used at file boundaries are identity casts in one of the two builds,
//! so `unnecessary_cast` is allowed as well.
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unnecessary_cast)]

/// Scalar type for all geometry, parameters and gradients.
#[cfg(not(feature = "f64"))]
pub type Real = f32;
/// Scalar type for all geometry, parameters and gradients.
#[cfg(feature = "f64")]
pub type Real = f64;

/// Atomic storage for one `Real`, used for race-free scatter of per-entry gradients.
#[cfg(not(feature = "f64"))]
pub type AtomicReal = std::sync::atomic::AtomicU32;
/// Atomic storage for one `Real`, used for race-free scatter of per-entry gradients.
#[cfg(feature = "f64")]
pub type AtomicReal = std::sync::atomic::AtomicU64;

/// Store a `Real` into its atomic slot (relaxed ordering; slots are written by one task only).
#[inline]
pub fn atomic_store(slot: &AtomicReal, value: Real) {
    slot.store(value.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// Load a `Real` from its atomic slot.
#[inline]
pub fn atomic_load(slot: &AtomicReal) -> Real {
    Real::from_bits(slot.load(std::sync::atomic::Ordering::Relaxed))
}

pub mod adam;
pub mod camera;
pub mod colmap;
pub mod dataset;
pub mod densify;
pub mod error;
pub mod loss;
pub mod math;
pub mod model;
pub mod ply;
pub mod project;
pub mod rasterize;
pub mod render;
pub mod sh;
pub mod train;

pub use error::{Error, Result};