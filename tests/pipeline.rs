//! End-to-end training on synthetic scenes rendered by the renderer itself, and the
//! equivalence of the tiled rasterizer with a plain per-pixel compositor.
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_range_loop)]

mod common;

use common::rng;
use common::scene::{ground_truth, init_points, render_views, ring_cameras};
use splatrs::camera::scene_extent;
use splatrs::model::Model;
use splatrs::project::{project_all, ProjOpts, TILE};
use splatrs::rasterize::{bin, rasterize_forward, RasterSettings};
use splatrs::train::{evaluate, train, TrainConfig};
use splatrs::Real;

#[test]
fn synthetic_scene_trains_without_densification() {
    let mut rng = rng(100);
    let gt = ground_truth(&mut rng, 40);
    let cameras = ring_cameras(10, 64, 48, 2.6);
    let bg = [0.0; 3];
    let views = render_views(&gt, &cameras, bg);
    let extent = scene_extent(&cameras);
    let init = Model::from_points(&init_points(&gt, &mut rng), 1);
    let cfg = TrainConfig {
        iterations: 600,
        densify: false,
        sh_degree: 1,
        sh_degree_interval: 200,
        position_lr_max_steps: 600,
        log_interval: 0,
        save_iterations: vec![],
        seed: 3,
        ..TrainConfig::default()
    };
    let before = evaluate(&init, &views, &cfg);
    let mut losses = Vec::new();
    let trained = train(
        init,
        &views,
        extent,
        &cfg,
        |_| {},
        |iter, m| {
            assert_eq!(iter, 600);
            losses.push(m.len());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(losses, vec![40]);
    let after = evaluate(&trained, &views, &cfg);
    eprintln!("no densification: PSNR {before:.2} -> {after:.2} dB");
    assert!(
        after > before + 6.0,
        "PSNR did not improve enough: {before:.2} -> {after:.2}"
    );
    assert!(after > 24.0, "PSNR after training too low: {after:.2}");
}

#[test]
fn synthetic_scene_trains_with_densification() {
    let mut rng = rng(200);
    let gt = ground_truth(&mut rng, 30);
    let cameras = ring_cameras(8, 48, 36, 2.6);
    let bg = [1.0; 3];
    let views = render_views(&gt, &cameras, bg);
    let extent = scene_extent(&cameras);
    // Start from a third of the points so densification has work to do.
    let mut pts = init_points(&gt, &mut rng);
    pts.truncate(10);
    let init = Model::from_points(&pts, 1);
    let cfg = TrainConfig {
        iterations: 500,
        densify: true,
        densify_from_iter: 40,
        densify_until_iter: 400,
        densification_interval: 40,
        opacity_reset_interval: 200,
        densify_grad_threshold: 0.0005,
        white_background: true,
        sh_degree: 1,
        sh_degree_interval: 150,
        position_lr_max_steps: 500,
        log_interval: 50,
        save_iterations: vec![250],
        seed: 5,
        ..TrainConfig::default()
    };
    let before = evaluate(&init, &views, &cfg);
    let mut saves = Vec::new();
    let mut reports = 0;
    let trained = train(
        init,
        &views,
        extent,
        &cfg,
        |p| {
            reports += 1;
            assert!(p.loss.is_finite());
        },
        |iter, m| {
            saves.push((iter, m.len()));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(reports, 10);
    assert_eq!(saves.len(), 2);
    assert_eq!(saves[0].0, 250);
    assert_eq!(saves[1].0, 500);
    assert!(trained.len() > 10, "densification never added Gaussians");
    assert_eq!(trained.sh_degree, 1);
    let after = evaluate(&trained, &views, &cfg);
    eprintln!(
        "with densification: PSNR {before:.2} -> {after:.2} dB, {} -> {} gaussians",
        10,
        trained.len()
    );
    assert!(
        after > before + 4.0,
        "PSNR did not improve enough: {before:.2} -> {after:.2}"
    );
    for g in &trained.gaussians {
        for v in g.mean.iter().chain(g.log_scale.iter()).chain(g.quat.iter()) {
            assert!(v.is_finite());
        }
    }
}

/// Per-pixel reference compositor: every visible Gaussian whose tile rectangle covers the
/// pixel's tile, in depth order, with the same skip rules as the tiled rasterizer.
fn naive_render(
    projs: &[splatrs::project::Proj],
    width: u32,
    height: u32,
    bg: [Real; 3],
) -> Vec<Real> {
    let mut order: Vec<usize> = (0..projs.len()).filter(|&i| projs[i].visible()).collect();
    order.sort_by_key(|&i| ((projs[i].depth as f32).to_bits(), i));
    let mut image = vec![0.0; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let (tx, ty) = (x / TILE, y / TILE);
            let (px, py) = (x as Real + 0.5, y as Real + 0.5);
            let mut t: Real = 1.0;
            let mut c = [0.0; 3];
            for &i in &order {
                let p = &projs[i];
                if tx < p.tile_min[0]
                    || tx >= p.tile_max[0]
                    || ty < p.tile_min[1]
                    || ty >= p.tile_max[1]
                {
                    continue;
                }
                let dx = px - p.mean2d[0];
                let dy = py - p.mean2d[1];
                let sigma =
                    0.5 * (p.conic[0] * dx * dx + p.conic[2] * dy * dy) + p.conic[1] * dx * dy;
                if sigma < 0.0 {
                    continue;
                }
                let og = p.opacity * (-sigma).exp();
                if og < 1.0 / 255.0 {
                    continue;
                }
                let alpha = og.min(0.99);
                let test_t = t * (1.0 - alpha);
                if test_t < 1e-4 {
                    break;
                }
                for ch in 0..3 {
                    c[ch] += alpha * t * p.color[ch];
                }
                t = test_t;
            }
            let pix = (y * width + x) as usize;
            for ch in 0..3 {
                image[pix * 3 + ch] = c[ch] + t * bg[ch];
            }
        }
    }
    image
}

#[test]
fn tiled_rasterizer_matches_naive_compositing() {
    let mut rng = rng(500);
    let gt = ground_truth(&mut rng, 120);
    // Mix in a few large and a few nearly transparent Gaussians.
    let mut model = gt.clone();
    for (i, g) in model.gaussians.iter_mut().enumerate() {
        if i % 10 == 0 {
            g.log_scale = [-0.7, -0.9, -0.8];
        }
        if i % 7 == 0 {
            g.opacity = -5.0;
        }
    }
    let cam = ring_cameras(1, 70, 50, 2.4).remove(0);
    let opts = ProjOpts {
        sh_degree: 1,
        antialias: true,
        ..ProjOpts::default()
    };
    let projs = project_all(&model, &cam, &opts);
    let bg = [0.3, 0.6, 0.1];
    let binning = bin(&projs, cam.width, cam.height);
    let settings = RasterSettings {
        width: cam.width,
        height: cam.height,
        background: bg,
    };
    let tiled = rasterize_forward(&projs, &binning, &settings).image;
    let naive = naive_render(&projs, cam.width, cam.height, bg);
    let mut worst: Real = 0.0;
    for (a, b) in tiled.iter().zip(&naive) {
        worst = worst.max((a - b).abs());
    }
    assert!(worst < 1e-5, "tiled and naive renders differ by {worst}");
    assert!(projs.iter().filter(|p| p.visible()).count() > 100);
}