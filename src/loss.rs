//! Photometric loss `(1 - lambda) * L1 + lambda * (1 - SSIM)` with its gradient with
//! respect to the rendered image.
//!
//! SSIM follows the reference implementation: an 11x11 Gaussian window with sigma 1.5,
//! applied as a zero-padded "same" convolution, constants `C1 = 0.01^2`, `C2 = 0.03^2`,
//! averaged over all pixels and channels.

use crate::Real;
use rayon::prelude::*;

/// Window edge of the SSIM filter.
pub const SSIM_WINDOW: usize = 11;

fn gaussian_kernel() -> [Real; SSIM_WINDOW] {
    let sigma: Real = 1.5;
    let mut k = [0.0; SSIM_WINDOW];
    let mut sum = 0.0;
    for i in 0..SSIM_WINDOW {
        let x = i as Real - (SSIM_WINDOW / 2) as Real;
        k[i] = (-(x * x) / (2.0 * sigma * sigma)).exp();
        sum += k[i];
    }
    for v in k.iter_mut() {
        *v /= sum;
    }
    k
}

/// Separable, zero-padded, same-size blur of a `width x height x channels` image.
/// The kernel is symmetric, so this operator is its own adjoint.
pub fn blur(
    src: &[Real],
    width: usize,
    height: usize,
    channels: usize,
    kernel: &[Real; SSIM_WINDOW],
) -> Vec<Real> {
    let half = (SSIM_WINDOW / 2) as isize;
    let row = width * channels;
    // Horizontal pass, parallel over rows.
    let mut tmp = vec![0.0; src.len()];
    tmp.par_chunks_mut(row).enumerate().for_each(|(y, out)| {
        let line = &src[y * row..(y + 1) * row];
        for x in 0..width {
            for c in 0..channels {
                let mut acc = 0.0;
                for k in 0..SSIM_WINDOW {
                    let xs = x as isize + k as isize - half;
                    if xs >= 0 && (xs as usize) < width {
                        acc += kernel[k] * line[xs as usize * channels + c];
                    }
                }
                out[x * channels + c] = acc;
            }
        }
    });
    // Vertical pass, parallel over rows of the output.
    let mut dst = vec![0.0; src.len()];
    dst.par_chunks_mut(row).enumerate().for_each(|(y, out)| {
        for k in 0..SSIM_WINDOW {
            let ys = y as isize + k as isize - half;
            if ys >= 0 && (ys as usize) < height {
                let line = &tmp[ys as usize * row..(ys as usize + 1) * row];
                let kv = kernel[k];
                for i in 0..row {
                    out[i] += kv * line[i];
                }
            }
        }
    });
    dst
}

/// Mean SSIM between `x` and `y` (both `width x height x 3` in [0, 1]) and `d(mean SSIM)/dx`.
pub fn ssim_with_grad(x: &[Real], y: &[Real], width: usize, height: usize) -> (Real, Vec<Real>) {
    let n = x.len();
    assert_eq!(n, y.len());
    assert_eq!(n, width * height * 3);
    let kernel = gaussian_kernel();
    let c1: Real = 0.01 * 0.01;
    let c2: Real = 0.03 * 0.03;

    let mu_x = blur(x, width, height, 3, &kernel);
    let mu_y = blur(y, width, height, 3, &kernel);
    let xx: Vec<Real> = x.par_iter().map(|v| v * v).collect();
    let yy: Vec<Real> = y.par_iter().map(|v| v * v).collect();
    let xy: Vec<Real> = x.par_iter().zip(y.par_iter()).map(|(a, b)| a * b).collect();
    let e_xx = blur(&xx, width, height, 3, &kernel);
    let e_yy = blur(&yy, width, height, 3, &kernel);
    let e_xy = blur(&xy, width, height, 3, &kernel);

    // Per-pixel SSIM and its partial derivatives with respect to mu_x, E[x^2] and E[xy].
    let mut d_mu = vec![0.0; n];
    let mut d_exx = vec![0.0; n];
    let mut d_exy = vec![0.0; n];
    let ssim_sum: Real = (0..n)
        .into_par_iter()
        .zip(d_mu.par_iter_mut())
        .zip(d_exx.par_iter_mut())
        .zip(d_exy.par_iter_mut())
        .map(|(((i, dmu), dexx), dexy)| {
            let (mx, my) = (mu_x[i], mu_y[i]);
            let sx2 = e_xx[i] - mx * mx;
            let sy2 = e_yy[i] - my * my;
            let sxy = e_xy[i] - mx * my;
            let n1 = 2.0 * mx * my + c1;
            let n2 = 2.0 * sxy + c2;
            let d1 = mx * mx + my * my + c1;
            let d2 = sx2 + sy2 + c2;
            let num = n1 * n2;
            let den = d1 * d2;
            let s = num / den;
            // dN/dmx = 2 my n2 + n1 * (-2 my); dD/dmx = 2 mx d2 + d1 * (-2 mx).
            let dn_dmx = 2.0 * my * n2 - 2.0 * my * n1;
            let dd_dmx = 2.0 * mx * d2 - 2.0 * mx * d1;
            *dmu = (dn_dmx * den - num * dd_dmx) / (den * den);
            *dexx = -num * d1 / (den * den);
            *dexy = 2.0 * n1 / den;
            s
        })
        .sum();
    let mean = ssim_sum / n as Real;

    // Adjoint of the blur is the blur itself.
    let b_mu = blur(&d_mu, width, height, 3, &kernel);
    let b_exx = blur(&d_exx, width, height, 3, &kernel);
    let b_exy = blur(&d_exy, width, height, 3, &kernel);
    let inv_n = 1.0 / n as Real;
    let grad: Vec<Real> = (0..n)
        .into_par_iter()
        .map(|i| (b_mu[i] + 2.0 * x[i] * b_exx[i] + y[i] * b_exy[i]) * inv_n)
        .collect();
    (mean, grad)
}

/// Combined loss and its gradient with respect to `pred`.
pub fn photometric_loss(
    pred: &[Real],
    target: &[Real],
    width: usize,
    height: usize,
    lambda_dssim: Real,
) -> (Real, Vec<Real>) {
    let n = pred.len();
    assert_eq!(n, target.len());
    let inv_n = 1.0 / n as Real;
    let l1: Real = pred
        .par_iter()
        .zip(target.par_iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<Real>()
        * inv_n;
    let (ssim, dssim) = ssim_with_grad(pred, target, width, height);
    let loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim);
    let w1 = (1.0 - lambda_dssim) * inv_n;
    let grad: Vec<Real> = (0..n)
        .into_par_iter()
        .map(|i| {
            let d = pred[i] - target[i];
            let sign = if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            };
            w1 * sign - lambda_dssim * dssim[i]
        })
        .collect();
    (loss, grad)
}

/// Peak signal-to-noise ratio in dB. Both images are clamped to [0, 1] first, as the
/// reference metric does, and the sum runs in `f64` so large images do not lose precision.
pub fn psnr(pred: &[Real], target: &[Real]) -> Real {
    assert_eq!(pred.len(), target.len());
    let mse: f64 = pred
        .iter()
        .zip(target)
        .map(|(p, t)| {
            let d = p.clamp(0.0, 1.0) as f64 - t.clamp(0.0, 1.0) as f64;
            d * d
        })
        .sum::<f64>()
        / pred.len().max(1) as f64;
    if mse <= 0.0 {
        return Real::INFINITY;
    }
    (-10.0 * mse.log10()) as Real
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_images_have_unit_ssim() {
        let (w, h) = (20, 14);
        let img: Vec<Real> = (0..w * h * 3)
            .map(|i| ((i * 7919) % 255) as Real / 255.0)
            .collect();
        let (s, g) = ssim_with_grad(&img, &img, w, h);
        assert!((s - 1.0).abs() < 1e-4, "ssim {s}");
        // At the maximum, the gradient must vanish.
        let gmax = g.iter().fold(0.0 as Real, |m, v| m.max(v.abs()));
        assert!(gmax < 1e-3, "gradient at optimum {gmax}");
    }
}