//! Real spherical harmonics up to degree 3, in the basis and coefficient ordering used by
//! the reference 3DGS implementation (so exported coefficients load everywhere).
//!
//! A colour channel is `clamp(sum_k basis_k(dir) * coeff_k + 0.5, 0, inf)`.

use crate::math::V3;
use crate::Real;

/// Number of coefficients per channel at the maximum supported degree.
pub const MAX_COEFFS: usize = 16;

/// Maximum supported degree.
pub const MAX_DEGREE: usize = 3;

/// Degree-0 basis constant `1 / (2 sqrt(pi))`.
pub const SH_C0: Real = 0.282_094_791_773_878_14_f64 as Real;
const SH_C1: Real = 0.488_602_511_902_919_9_f64 as Real;
const SH_C2: [Real; 5] = [
    1.092_548_430_592_079_2_f64 as Real,
    -1.092_548_430_592_079_2_f64 as Real,
    0.315_391_565_252_520_05_f64 as Real,
    -1.092_548_430_592_079_2_f64 as Real,
    0.546_274_215_296_039_6_f64 as Real,
];
const SH_C3: [Real; 7] = [
    -0.590_043_589_926_643_5_f64 as Real,
    2.890_611_442_640_554_f64 as Real,
    -0.457_045_799_464_465_8_f64 as Real,
    0.373_176_332_590_115_4_f64 as Real,
    -0.457_045_799_464_465_8_f64 as Real,
    1.445_305_721_320_277_f64 as Real,
    -0.590_043_589_926_643_5_f64 as Real,
];

/// Number of coefficients per channel for a given degree: `(degree + 1)^2`.
pub const fn num_coeffs(degree: usize) -> usize {
    (degree + 1) * (degree + 1)
}

/// Convert an RGB value in [0, 1] to the degree-0 coefficient that reproduces it.
pub fn rgb_to_sh(c: Real) -> Real {
    (c - 0.5) / SH_C0
}

/// Evaluate the basis functions at unit direction `dir`. Entries beyond
/// `num_coeffs(degree)` are left at zero.
pub fn basis(degree: usize, dir: V3) -> [Real; MAX_COEFFS] {
    let mut b = [0.0; MAX_COEFFS];
    let (x, y, z) = (dir.x, dir.y, dir.z);
    b[0] = SH_C0;
    if degree >= 1 {
        b[1] = -SH_C1 * y;
        b[2] = SH_C1 * z;
        b[3] = -SH_C1 * x;
    }
    if degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, yz, xz) = (x * y, y * z, x * z);
        b[4] = SH_C2[0] * xy;
        b[5] = SH_C2[1] * yz;
        b[6] = SH_C2[2] * (2.0 * zz - xx - yy);
        b[7] = SH_C2[3] * xz;
        b[8] = SH_C2[4] * (xx - yy);
        if degree >= 3 {
            b[9] = SH_C3[0] * y * (3.0 * xx - yy);
            b[10] = SH_C3[1] * xy * z;
            b[11] = SH_C3[2] * y * (4.0 * zz - xx - yy);
            b[12] = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
            b[13] = SH_C3[4] * x * (4.0 * zz - xx - yy);
            b[14] = SH_C3[5] * z * (xx - yy);
            b[15] = SH_C3[6] * x * (xx - 3.0 * yy);
        }
    }
    b
}

/// Partial derivatives of every basis function with respect to the three components of
/// `dir` (treated as free variables; the projection onto the sphere is chained by the
/// caller). Entries beyond `num_coeffs(degree)` are zero.
pub fn basis_grad(degree: usize, dir: V3) -> [V3; MAX_COEFFS] {
    let mut g = [V3::ZERO; MAX_COEFFS];
    let (x, y, z) = (dir.x, dir.y, dir.z);
    if degree >= 1 {
        g[1] = V3::new(0.0, -SH_C1, 0.0);
        g[2] = V3::new(0.0, 0.0, SH_C1);
        g[3] = V3::new(-SH_C1, 0.0, 0.0);
    }
    if degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        g[4] = V3::new(SH_C2[0] * y, SH_C2[0] * x, 0.0);
        g[5] = V3::new(0.0, SH_C2[1] * z, SH_C2[1] * y);
        g[6] = V3::new(-2.0 * SH_C2[2] * x, -2.0 * SH_C2[2] * y, 4.0 * SH_C2[2] * z);
        g[7] = V3::new(SH_C2[3] * z, 0.0, SH_C2[3] * x);
        g[8] = V3::new(2.0 * SH_C2[4] * x, -2.0 * SH_C2[4] * y, 0.0);
        if degree >= 3 {
            g[9] = V3::new(
                SH_C3[0] * 6.0 * x * y,
                SH_C3[0] * (3.0 * xx - 3.0 * yy),
                0.0,
            );
            g[10] = V3::new(SH_C3[1] * y * z, SH_C3[1] * x * z, SH_C3[1] * x * y);
            g[11] = V3::new(
                SH_C3[2] * (-2.0 * x * y),
                SH_C3[2] * (4.0 * zz - xx - 3.0 * yy),
                SH_C3[2] * 8.0 * y * z,
            );
            g[12] = V3::new(
                SH_C3[3] * (-6.0 * x * z),
                SH_C3[3] * (-6.0 * y * z),
                SH_C3[3] * (6.0 * zz - 3.0 * xx - 3.0 * yy),
            );
            g[13] = V3::new(
                SH_C3[4] * (4.0 * zz - 3.0 * xx - yy),
                SH_C3[4] * (-2.0 * x * y),
                SH_C3[4] * 8.0 * x * z,
            );
            g[14] = V3::new(
                SH_C3[5] * 2.0 * x * z,
                -SH_C3[5] * 2.0 * y * z,
                SH_C3[5] * (xx - yy),
            );
            g[15] = V3::new(
                SH_C3[6] * (3.0 * xx - 3.0 * yy),
                SH_C3[6] * (-6.0 * x * y),
                0.0,
            );
        }
    }
    g
}

/// Evaluate the colour of one Gaussian for a unit view direction, before the `+0.5` offset
/// and clamp. `coeffs[k][c]` is coefficient `k` of channel `c`.
pub fn eval_raw(degree: usize, coeffs: &[[Real; 3]; MAX_COEFFS], dir: V3) -> [Real; 3] {
    let b = basis(degree, dir);
    let n = num_coeffs(degree);
    let mut out = [0.0; 3];
    for k in 0..n {
        for c in 0..3 {
            out[c] += b[k] * coeffs[k][c];
        }
    }
    out
}