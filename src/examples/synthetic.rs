//! Generate a synthetic COLMAP-style dataset from random Gaussian blobs, rendered by
//! splatrs itself. Useful for checking the pipeline and measuring speed without running
//! COLMAP first:
//!
//! ```text
//! cargo run --release --example synthetic -- --out /tmp/synth --gaussians 2000 --views 30
//! splatrs train --data /tmp/synth --out /tmp/synth-out --iterations 2000
//! ```
//!
//! The ground-truth splat is written to `<out>/ground_truth.ply`.
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_range_loop)]

use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use splatrs::camera::Camera;
use splatrs::colmap::{
    rotation_to_qvec, write_model_bin, CameraModel, ColmapCamera, ColmapImage, ColmapModel,
    ColmapPoint,
};
use splatrs::math::V3;
use splatrs::model::{Gaussian, Model};
use splatrs::project::ProjOpts;
use splatrs::render::{render, save_png};
use splatrs::sh::{self, MAX_COEFFS};
use splatrs::{ply, Real};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Write a synthetic COLMAP dataset rendered from random Gaussians")]
struct Args {
    /// Output directory (gets `sparse/0`, `images` and `ground_truth.ply`).
    #[arg(long)]
    out: PathBuf,
    /// Number of Gaussians in the scene.
    #[arg(long, default_value_t = 500)]
    gaussians: usize,
    /// Number of cameras on the ring.
    #[arg(long, default_value_t = 24)]
    views: usize,
    /// Image width.
    #[arg(long, default_value_t = 320)]
    width: u32,
    /// Image height.
    #[arg(long, default_value_t = 240)]
    height: u32,
    /// Fraction of the Gaussians exported as initial points (the rest must be densified).
    #[arg(long, default_value_t = 0.5)]
    point_fraction: f64,
    /// Random seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

fn rnd(rng: &mut ChaCha8Rng, lo: Real, hi: Real) -> Real {
    lo + (hi - lo) * rng.random::<f64>() as Real
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);

    let mut gaussians = Vec::with_capacity(args.gaussians);
    for _ in 0..args.gaussians {
        let mut coeffs = [[0.0; 3]; MAX_COEFFS];
        for c in 0..3 {
            coeffs[0][c] = sh::rgb_to_sh(rnd(&mut rng, 0.05, 0.95));
        }
        for k in 1..4 {
            for c in 0..3 {
                coeffs[k][c] = rnd(&mut rng, -0.2, 0.2);
            }
        }
        gaussians.push(Gaussian {
            mean: [
                rnd(&mut rng, -1.0, 1.0),
                rnd(&mut rng, -0.6, 0.6),
                rnd(&mut rng, -1.0, 1.0),
            ],
            log_scale: [
                rnd(&mut rng, -3.5, -2.0),
                rnd(&mut rng, -3.5, -2.0),
                rnd(&mut rng, -3.5, -2.0),
            ],
            quat: [
                rnd(&mut rng, 0.5, 1.0),
                rnd(&mut rng, -0.5, 0.5),
                rnd(&mut rng, -0.5, 0.5),
                rnd(&mut rng, -0.5, 0.5),
            ],
            opacity: rnd(&mut rng, 0.0, 3.0),
            sh: coeffs,
        });
    }
    let gt = Model {
        gaussians,
        sh_degree: 1,
        max_sh_degree: 1,
    };

    let cameras: Vec<Camera> = (0..args.views)
        .map(|i| {
            let a = i as Real / args.views as Real * std::f64::consts::TAU as Real;
            let eye = V3::new(3.0 * a.cos(), -0.8 + 0.5 * (3.0 * a).sin(), 3.0 * a.sin());
            Camera::look_at(
                &format!("view_{i:03}.png"),
                args.width,
                args.height,
                args.width as Real * 1.1,
                args.width as Real * 1.1,
                eye,
                V3::ZERO,
                V3::new(0.0, -1.0, 0.0),
            )
        })
        .collect();

    let sparse = args.out.join("sparse").join("0");
    let images = args.out.join("images");
    std::fs::create_dir_all(&images)?;
    let opts = ProjOpts {
        sh_degree: gt.sh_degree,
        ..ProjOpts::default()
    };
    for cam in &cameras {
        let img = render(&gt, cam, &opts, [0.0; 3]);
        save_png(&images.join(&cam.name), &img, cam.width, cam.height)?;
    }

    let cam0 = &cameras[0];
    let mut colmap_cameras = HashMap::new();
    colmap_cameras.insert(
        1,
        ColmapCamera {
            id: 1,
            model: CameraModel::Pinhole,
            width: cam0.width as u64,
            height: cam0.height as u64,
            params: vec![
                cam0.fx as f64,
                cam0.fy as f64,
                cam0.cx as f64,
                cam0.cy as f64,
            ],
        },
    );
    let colmap_images = cameras
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut m = [[0.0f64; 3]; 3];
            for r in 0..3 {
                for k in 0..3 {
                    m[r][k] = c.rot.m[r][k] as f64;
                }
            }
            ColmapImage {
                id: i as u32 + 1,
                qvec: rotation_to_qvec(&m),
                tvec: [c.trans.x as f64, c.trans.y as f64, c.trans.z as f64],
                camera_id: 1,
                name: c.name.clone(),
            }
        })
        .collect();
    let n_points =
        ((args.gaussians as f64 * args.point_fraction).round() as usize).clamp(1, args.gaussians);
    let points = gt.gaussians[..n_points]
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let mut rgb = [0u8; 3];
            for c in 0..3 {
                rgb[c] = ((g.sh[0][c] * sh::SH_C0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            let jitter = V3::new(
                rnd(&mut rng, -0.02, 0.02),
                rnd(&mut rng, -0.02, 0.02),
                rnd(&mut rng, -0.02, 0.02),
            );
            let p = g.mean_v() + jitter;
            ColmapPoint {
                id: i as u64 + 1,
                xyz: [p.x as f64, p.y as f64, p.z as f64],
                rgb,
                error: 0.5,
            }
        })
        .collect();
    write_model_bin(
        &sparse,
        &ColmapModel {
            cameras: colmap_cameras,
            images: colmap_images,
            points,
        },
    )?;
    ply::write(&args.out.join("ground_truth.ply"), &gt)?;
    eprintln!(
        "wrote {} views of {}x{} and {} initial points to {}",
        args.views,
        args.width,
        args.height,
        n_points,
        args.out.display()
    );
    Ok(())
}