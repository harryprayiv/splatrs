//! Finite-difference checks of the spherical-harmonics and projection backward passes.
//! These losses and gradients are O(1), so the checks run in both the default f32 build and
//! with `--features f64`. The rasterizer and image-loss checks live in `gradcheck_f64.rs`.
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_range_loop)]

mod common;

use common::{
    central, check, param, param_name, random_gaussian, rnd, rng, test_camera, with_param, H,
    NUM_PARAMS,
};
use rand_chacha::ChaCha8Rng;
use splatrs::camera::Camera;
use splatrs::math::V3;
use splatrs::model::Gaussian;
use splatrs::project::{project_backward_one, project_one, Proj, ProjGrad, ProjOpts};
use splatrs::sh::{self, MAX_COEFFS};
use splatrs::Real;

#[test]
fn sh_basis_gradient() {
    let mut rng = rng(7);
    for _ in 0..20 {
        let d = V3::new(
            rnd(&mut rng, -1.0, 1.0),
            rnd(&mut rng, -1.0, 1.0),
            rnd(&mut rng, -1.0, 1.0),
        )
        .normalized();
        let g = sh::basis_grad(3, d);
        for k in 0..MAX_COEFFS {
            for axis in 0..3 {
                let num = central(0.0, H, |e| {
                    let mut p = d;
                    match axis {
                        0 => p.x += e,
                        1 => p.y += e,
                        _ => p.z += e,
                    }
                    sh::basis(3, p)[k]
                });
                check(&format!("basis[{k}] d/d{axis}"), g[k].get(axis), num, 1.0);
            }
        }
    }
}

/// Scalar loss over the screen quantities of one projected Gaussian.
fn proj_loss(p: &Proj, w: &ProjGrad) -> Real {
    let mut l = 0.0;
    for i in 0..2 {
        l += w.mean2d[i] * p.mean2d[i];
    }
    for i in 0..3 {
        l += w.conic[i] * p.conic[i];
        l += w.color[i] * p.color[i];
    }
    l + w.opacity * p.opacity
}

fn random_weights(rng: &mut ChaCha8Rng) -> ProjGrad {
    ProjGrad {
        mean2d: [rnd(rng, -1.0, 1.0), rnd(rng, -1.0, 1.0)],
        conic: [
            rnd(rng, -1.0, 1.0),
            rnd(rng, -1.0, 1.0),
            rnd(rng, -1.0, 1.0),
        ],
        opacity: rnd(rng, -1.0, 1.0),
        color: [
            rnd(rng, -1.0, 1.0),
            rnd(rng, -1.0, 1.0),
            rnd(rng, -1.0, 1.0),
        ],
    }
}

fn check_projection(
    cam: &Camera,
    g: &Gaussian,
    opts: &ProjOpts,
    rng: &mut ChaCha8Rng,
    label: &str,
) {
    let base = project_one(g, cam, opts);
    assert!(base.visible(), "{label}: test Gaussian must be visible");
    let w = random_weights(rng);
    let analytic = project_backward_one(g, cam, opts, &w);
    for i in 0..NUM_PARAMS {
        let num = central(0.0, H, |e| {
            let p = project_one(&with_param(g, i, e), cam, opts);
            assert!(p.visible(), "{label}: perturbed Gaussian left the view");
            proj_loss(&p, &w)
        });
        // The conic can be large (small Gaussians), so scale the absolute tolerance by it.
        let scale = 1.0 + base.conic[0].abs() + base.conic[2].abs();
        check(
            &format!("{label} {}", param_name(i)),
            param(&analytic, i),
            num,
            scale,
        );
    }
}

#[test]
fn projection_gradient_plain() {
    let mut rng = rng(11);
    let cam = test_camera(96, 64);
    let opts = ProjOpts {
        antialias: false,
        sh_degree: 3,
        ..ProjOpts::default()
    };
    for n in 0..12 {
        let g = random_gaussian(&mut rng);
        check_projection(&cam, &g, &opts, &mut rng, &format!("plain #{n}"));
    }
}

#[test]
fn projection_gradient_antialiased() {
    let mut rng = rng(12);
    let cam = test_camera(96, 64);
    let opts = ProjOpts {
        antialias: true,
        sh_degree: 3,
        ..ProjOpts::default()
    };
    for n in 0..12 {
        let g = random_gaussian(&mut rng);
        check_projection(&cam, &g, &opts, &mut rng, &format!("antialiased #{n}"));
    }
}

#[test]
fn projection_gradient_lower_degrees() {
    let mut rng = rng(13);
    let cam = test_camera(96, 64);
    for degree in 0..3 {
        let opts = ProjOpts {
            antialias: false,
            sh_degree: degree,
            ..ProjOpts::default()
        };
        let g = random_gaussian(&mut rng);
        check_projection(&cam, &g, &opts, &mut rng, &format!("degree {degree}"));
    }
}

#[test]
fn projection_gradient_with_jacobian_clamping() {
    // A large Gaussian far to the side of the frustum: its Jacobian point is clamped but its
    // footprint still reaches the image.
    let mut rng = rng(14);
    let cam = Camera::look_at(
        "cam",
        64,
        48,
        60.0,
        60.0,
        V3::new(0.0, 0.0, -2.0),
        V3::new(0.0, 0.0, 0.0),
        V3::new(0.0, -1.0, 0.0),
    );
    let opts = ProjOpts {
        antialias: true,
        sh_degree: 2,
        ..ProjOpts::default()
    };
    let mut g = random_gaussian(&mut rng);
    g.mean = [3.2, 0.1, 0.0];
    g.log_scale = [0.4, 0.4, 0.4];
    let t = cam.world_to_cam(g.mean_v());
    let lim = (cam.width as Real - cam.cx) / cam.fx + 0.3 * 0.5 * cam.width as Real / cam.fx;
    assert!(
        t.x / t.z > lim,
        "test setup: Gaussian must be past the clamp limit"
    );
    check_projection(&cam, &g, &opts, &mut rng, "clamped");
}