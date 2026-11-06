//! Finite-difference checks that need double precision: the rasterizer, the image losses
//! and the whole parameters-to-loss chain. Their losses sum thousands of pixel terms, so
//! single-precision central differences cannot resolve the gradients; the file is compiled
//! only with `cargo test --features f64`.
#![cfg(feature = "f64")]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_range_loop)]

mod common;

use common::{
    central, check, param, param_name, random_gaussian, rnd, rng, test_camera, with_param, H,
    NUM_PARAMS,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use splatrs::loss::{photometric_loss, ssim_with_grad};
use splatrs::model::Model;
use splatrs::project::{project_all, project_backward_all, tile_grid, Proj, ProjOpts};
use splatrs::rasterize::{bin, rasterize_backward, rasterize_forward, RasterSettings};
use splatrs::Real;

/// Step for conic entries: the exponent is quadratic in the pixel offset, so the step must
/// stay well inside the linear regime.
const H_CONIC: Real = 1e-6;

/// Screen-space Gaussians covering the whole image with distinct depths.
fn random_projs(rng: &mut ChaCha8Rng, width: u32, height: u32, n: usize) -> Vec<Proj> {
    let (tx, ty) = tile_grid(width, height);
    (0..n)
        .map(|i| {
            // Random positive-definite covariance via L L^T, inverted to a conic.
            let l00 = rnd(rng, 1.5, 4.0);
            let l10 = rnd(rng, -2.0, 2.0);
            let l11 = rnd(rng, 1.5, 4.0);
            let a = l00 * l00;
            let b = l00 * l10;
            let c = l10 * l10 + l11 * l11;
            let det = a * c - b * b;
            Proj {
                mean2d: [
                    rnd(rng, 4.0, width as Real - 4.0),
                    rnd(rng, 4.0, height as Real - 4.0),
                ],
                conic: [c / det, -b / det, a / det],
                depth: 1.0 + i as Real * 0.37,
                radius: 1000,
                opacity: rnd(rng, 0.3, 0.9),
                color: [rnd(rng, 0.0, 1.0), rnd(rng, 0.0, 1.0), rnd(rng, 0.0, 1.0)],
                tile_min: [0, 0],
                tile_max: [tx, ty],
            }
        })
        .collect()
}

fn raster_loss(projs: &[Proj], settings: &RasterSettings, w: &[Real]) -> Real {
    let binning = bin(projs, settings.width, settings.height);
    let out = rasterize_forward(projs, &binning, settings);
    out.image.iter().zip(w).map(|(a, b)| a * b).sum()
}

#[test]
fn rasterizer_gradient() {
    let mut rng = rng(21);
    let (width, height) = (40u32, 28u32);
    let settings = RasterSettings {
        width,
        height,
        background: [0.2, 0.5, 0.8],
    };
    let projs = random_projs(&mut rng, width, height, 9);
    let w: Vec<Real> = (0..(width * height * 3) as usize)
        .map(|_| rnd(&mut rng, -1.0, 1.0))
        .collect();
    let binning = bin(&projs, width, height);
    let out = rasterize_forward(&projs, &binning, &settings);
    let (grads, abs) = rasterize_backward(&projs, &binning, &settings, &out, &w);
    for i in 0..projs.len() {
        for field in 0..9 {
            let h = if (2..=4).contains(&field) { H_CONIC } else { H };
            let num = central(0.0, h, |e| {
                let mut p = projs.clone();
                match field {
                    0 | 1 => p[i].mean2d[field] += e,
                    2..=4 => p[i].conic[field - 2] += e,
                    5 => p[i].opacity += e,
                    _ => p[i].color[field - 6] += e,
                }
                raster_loss(&p, &settings, &w)
            });
            let analytic = match field {
                0 | 1 => grads[i].mean2d[field],
                2..=4 => grads[i].conic[field - 2],
                5 => grads[i].opacity,
                _ => grads[i].color[field - 6],
            };
            check(&format!("gaussian {i} field {field}"), analytic, num, 1.0);
        }
        // The absolute accumulator bounds the signed gradient.
        assert!(abs[i][0] + 1e-9 >= grads[i].mean2d[0].abs());
        assert!(abs[i][1] + 1e-9 >= grads[i].mean2d[1].abs());
    }
}

#[test]
fn ssim_gradient() {
    let mut rng = rng(31);
    let (w, h) = (18usize, 13usize);
    let x: Vec<Real> = (0..w * h * 3).map(|_| rnd(&mut rng, 0.0, 1.0)).collect();
    let y: Vec<Real> = (0..w * h * 3).map(|_| rnd(&mut rng, 0.0, 1.0)).collect();
    let (_, grad) = ssim_with_grad(&x, &y, w, h);
    for _ in 0..40 {
        let i = rng.random_range(0..x.len());
        let num = central(0.0, H, |e| {
            let mut xp = x.clone();
            xp[i] += e;
            ssim_with_grad(&xp, &y, w, h).0
        });
        check(&format!("ssim d/dx[{i}]"), grad[i], num, 1.0);
    }
}

#[test]
fn photometric_loss_gradient() {
    let mut rng = rng(32);
    let (w, h) = (16usize, 12usize);
    let x: Vec<Real> = (0..w * h * 3).map(|_| rnd(&mut rng, 0.0, 1.0)).collect();
    let y: Vec<Real> = (0..w * h * 3).map(|_| rnd(&mut rng, 0.0, 1.0)).collect();
    let (_, grad) = photometric_loss(&x, &y, w, h, 0.2);
    for _ in 0..40 {
        let i = rng.random_range(0..x.len());
        let num = central(0.0, H, |e| {
            let mut xp = x.clone();
            xp[i] += e;
            photometric_loss(&xp, &y, w, h, 0.2).0
        });
        check(&format!("loss d/dx[{i}]"), grad[i], num, 1.0);
    }
}

/// The whole chain: parameters -> projection -> rasterization -> photometric loss. The
/// D-SSIM weight is set to 1 so the loss is smooth: the L1 term's kinks at pixels where the
/// render crosses the target make central differences unreliable at tight tolerances, and
/// the L1 gradient is checked on its own above.
#[test]
fn end_to_end_gradient() {
    let mut rng = rng(41);
    let cam = test_camera(48, 32);
    let (w, h) = (cam.width, cam.height);
    let mut gaussians = Vec::new();
    for _ in 0..6 {
        let mut g = random_gaussian(&mut rng);
        // Big enough on screen to overlap each other.
        g.log_scale = [
            rnd(&mut rng, -1.6, -0.9),
            rnd(&mut rng, -1.6, -0.9),
            rnd(&mut rng, -1.6, -0.9),
        ];
        gaussians.push(g);
    }
    let model = Model {
        gaussians,
        sh_degree: 3,
        max_sh_degree: 3,
    };
    let target: Vec<Real> = (0..(w * h * 3) as usize)
        .map(|_| rnd(&mut rng, 0.0, 1.0))
        .collect();
    let opts = ProjOpts {
        antialias: true,
        sh_degree: 3,
        ..ProjOpts::default()
    };
    let settings = RasterSettings {
        width: w,
        height: h,
        background: [0.1, 0.1, 0.1],
    };

    let loss_of = |m: &Model| -> Real {
        let projs = project_all(m, &cam, &opts);
        let binning = bin(&projs, w, h);
        let out = rasterize_forward(&projs, &binning, &settings);
        photometric_loss(&out.image, &target, w as usize, h as usize, 1.0).0
    };

    let projs = project_all(&model, &cam, &opts);
    for p in &projs {
        assert!(p.visible());
    }
    let binning = bin(&projs, w, h);
    let out = rasterize_forward(&projs, &binning, &settings);
    let (_, dimage) = photometric_loss(&out.image, &target, w as usize, h as usize, 1.0);
    let (pgrads, _) = rasterize_backward(&projs, &binning, &settings, &out, &dimage);
    let grads = project_backward_all(&model, &cam, &opts, &pgrads);

    // The loss is O(1) while many gradients are O(1e-7), so the step is kept larger than
    // elsewhere and the absolute tolerance covers the double-precision rounding of the
    // difference quotient (about 1e-16 / h).
    let h: Real = 1e-5;
    for gi in 0..model.len() {
        for i in 0..NUM_PARAMS {
            let num = central(0.0, h, |e| {
                let mut m = model.clone();
                m.gaussians[gi] = with_param(&model.gaussians[gi], i, e);
                loss_of(&m)
            });
            check(
                &format!("gaussian {gi} {}", param_name(i)),
                param(&grads[gi], i),
                num,
                1.0,
            );
        }
    }
}