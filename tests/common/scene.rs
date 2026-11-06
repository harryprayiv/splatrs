//! Synthetic scenes and COLMAP-style dataset directories for the integration tests.
#![allow(dead_code)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_range_loop)]

use super::rnd;
use rand_chacha::ChaCha8Rng;
use splatrs::camera::Camera;
use splatrs::colmap::{
    rotation_to_qvec, CameraModel, ColmapCamera, ColmapImage, ColmapModel, ColmapPoint,
};
use splatrs::dataset::View;
use splatrs::math::V3;
use splatrs::model::{Gaussian, Model};
use splatrs::project::ProjOpts;
use splatrs::render::{render, to_rgb8};
use splatrs::sh::{self, MAX_COEFFS};
use splatrs::Real;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// A fresh temporary directory.
pub fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("splatrs-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Cameras on a ring around the origin, looking at it.
pub fn ring_cameras(n: usize, width: u32, height: u32, radius: Real) -> Vec<Camera> {
    (0..n)
        .map(|i| {
            let a = i as Real / n as Real * std::f64::consts::TAU as Real;
            let eye = V3::new(
                radius * a.cos(),
                -0.6 + 0.3 * (3.0 * a).sin(),
                radius * a.sin(),
            );
            Camera::look_at(
                &format!("view_{i:03}.png"),
                width,
                height,
                width as Real * 1.1,
                width as Real * 1.1,
                eye,
                V3::ZERO,
                V3::new(0.0, -1.0, 0.0),
            )
        })
        .collect()
}

/// A ground-truth scene of random blobs.
pub fn ground_truth(rng: &mut ChaCha8Rng, n: usize) -> Model {
    let mut gaussians = Vec::new();
    for _ in 0..n {
        let mut sh = [[0.0; 3]; MAX_COEFFS];
        for c in 0..3 {
            sh[0][c] = sh::rgb_to_sh(rnd(rng, 0.1, 0.9));
        }
        for k in 1..4 {
            for c in 0..3 {
                sh[k][c] = rnd(rng, -0.2, 0.2);
            }
        }
        gaussians.push(Gaussian {
            mean: [
                rnd(rng, -0.7, 0.7),
                rnd(rng, -0.5, 0.5),
                rnd(rng, -0.7, 0.7),
            ],
            log_scale: [
                rnd(rng, -2.6, -1.6),
                rnd(rng, -2.6, -1.6),
                rnd(rng, -2.6, -1.6),
            ],
            quat: [
                rnd(rng, 0.5, 1.0),
                rnd(rng, -0.5, 0.5),
                rnd(rng, -0.5, 0.5),
                rnd(rng, -0.5, 0.5),
            ],
            opacity: rnd(rng, 0.5, 3.0),
            sh,
        });
    }
    Model {
        gaussians,
        sh_degree: 1,
        max_sh_degree: 1,
    }
}

pub fn render_views(gt: &Model, cameras: &[Camera], bg: [Real; 3]) -> Vec<View> {
    let opts = ProjOpts {
        sh_degree: gt.sh_degree,
        ..ProjOpts::default()
    };
    cameras
        .iter()
        .map(|cam| View {
            camera: cam.clone(),
            rgb: to_rgb8(&render(gt, cam, &opts, bg)),
        })
        .collect()
}

/// Initial points: the true means with their DC colour, jittered.
pub fn init_points(gt: &Model, rng: &mut ChaCha8Rng) -> Vec<(V3, [u8; 3])> {
    gt.gaussians
        .iter()
        .map(|g| {
            let p = g.mean_v()
                + V3::new(
                    rnd(rng, -0.03, 0.03),
                    rnd(rng, -0.03, 0.03),
                    rnd(rng, -0.03, 0.03),
                );
            let mut rgb = [0u8; 3];
            for c in 0..3 {
                rgb[c] = ((g.sh[0][c] * sh::SH_C0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            (p, rgb)
        })
        .collect()
}

// ---------------------------------------------------------------------------------------
// COLMAP writers used to build test datasets (binary and text, same layout as COLMAP).
// ---------------------------------------------------------------------------------------

pub fn write_colmap_bin(dir: &Path, m: &ColmapModel) {
    let mut cams = Vec::new();
    cams.extend((m.cameras.len() as u64).to_le_bytes());
    let mut ids: Vec<_> = m.cameras.keys().copied().collect();
    ids.sort();
    for id in ids {
        let c = &m.cameras[&id];
        cams.extend(c.id.to_le_bytes());
        cams.extend(c.model.id().to_le_bytes());
        cams.extend(c.width.to_le_bytes());
        cams.extend(c.height.to_le_bytes());
        for p in &c.params {
            cams.extend(p.to_le_bytes());
        }
    }
    std::fs::write(dir.join("cameras.bin"), cams).unwrap();

    let mut imgs = Vec::new();
    imgs.extend((m.images.len() as u64).to_le_bytes());
    for (n, im) in m.images.iter().enumerate() {
        imgs.extend(im.id.to_le_bytes());
        for q in im.qvec {
            imgs.extend(q.to_le_bytes());
        }
        for t in im.tvec {
            imgs.extend(t.to_le_bytes());
        }
        imgs.extend(im.camera_id.to_le_bytes());
        imgs.extend(im.name.as_bytes());
        imgs.push(0);
        // A varying number of 2D observations, some without a 3D point.
        let n2d = (n % 3) as u64;
        imgs.extend(n2d.to_le_bytes());
        for k in 0..n2d {
            imgs.extend((10.5 + k as f64).to_le_bytes());
            imgs.extend((20.5 + k as f64).to_le_bytes());
            let id: i64 = if k == 0 { -1 } else { k as i64 };
            imgs.extend(id.to_le_bytes());
        }
    }
    std::fs::write(dir.join("images.bin"), imgs).unwrap();

    let mut pts = Vec::new();
    pts.extend((m.points.len() as u64).to_le_bytes());
    for (n, p) in m.points.iter().enumerate() {
        pts.extend(p.id.to_le_bytes());
        for v in p.xyz {
            pts.extend(v.to_le_bytes());
        }
        pts.extend(p.rgb);
        pts.extend(p.error.to_le_bytes());
        let track = (n % 4) as u64;
        pts.extend(track.to_le_bytes());
        for k in 0..track {
            pts.extend((k as i32 + 1).to_le_bytes());
            pts.extend((k as i32 * 7).to_le_bytes());
        }
    }
    std::fs::write(dir.join("points3D.bin"), pts).unwrap();
}

pub fn write_colmap_txt(dir: &Path, m: &ColmapModel) {
    let mut f = std::fs::File::create(dir.join("cameras.txt")).unwrap();
    writeln!(f, "# Camera list with one line of data per camera:").unwrap();
    writeln!(f, "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]").unwrap();
    let mut ids: Vec<_> = m.cameras.keys().copied().collect();
    ids.sort();
    for id in ids {
        let c = &m.cameras[&id];
        let params: Vec<String> = c.params.iter().map(|p| format!("{p}")).collect();
        writeln!(
            f,
            "{} {} {} {} {}",
            c.id,
            c.model.name(),
            c.width,
            c.height,
            params.join(" ")
        )
        .unwrap();
    }
    let mut f = std::fs::File::create(dir.join("images.txt")).unwrap();
    writeln!(f, "# Image list with two lines of data per image:").unwrap();
    for (n, im) in m.images.iter().enumerate() {
        writeln!(
            f,
            "{} {} {} {} {} {} {} {} {} {}",
            im.id,
            im.qvec[0],
            im.qvec[1],
            im.qvec[2],
            im.qvec[3],
            im.tvec[0],
            im.tvec[1],
            im.tvec[2],
            im.camera_id,
            im.name
        )
        .unwrap();
        // Observation lines: empty, or a few triples (one without a 3D point).
        match n % 3 {
            0 => writeln!(f).unwrap(),
            1 => writeln!(f, "10.5 20.5 -1").unwrap(),
            _ => writeln!(f, "10.5 20.5 -1 11.5 21.5 1").unwrap(),
        }
    }
    let mut f = std::fs::File::create(dir.join("points3D.txt")).unwrap();
    writeln!(f, "# 3D point list with one line of data per point:").unwrap();
    for (n, p) in m.points.iter().enumerate() {
        let track: Vec<String> = (0..n % 4).map(|k| format!("{} {}", k + 1, k * 7)).collect();
        writeln!(
            f,
            "{} {} {} {} {} {} {} {} {}",
            p.id,
            p.xyz[0],
            p.xyz[1],
            p.xyz[2],
            p.rgb[0],
            p.rgb[1],
            p.rgb[2],
            p.error,
            track.join(" ")
        )
        .unwrap();
    }
}

pub fn synthetic_colmap_model(cameras: &[Camera], points: &[(V3, [u8; 3])]) -> ColmapModel {
    let cam = &cameras[0];
    let mut colmap_cameras = HashMap::new();
    colmap_cameras.insert(
        1,
        ColmapCamera {
            id: 1,
            model: CameraModel::Pinhole,
            width: cam.width as u64,
            height: cam.height as u64,
            params: vec![cam.fx as f64, cam.fy as f64, cam.cx as f64, cam.cy as f64],
        },
    );
    let images = cameras
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
    let points = points
        .iter()
        .enumerate()
        .map(|(i, (p, rgb))| ColmapPoint {
            id: i as u64 + 1,
            xyz: [p.x as f64, p.y as f64, p.z as f64],
            rgb: *rgb,
            error: 0.5,
        })
        .collect();
    ColmapModel {
        cameras: colmap_cameras,
        images,
        points,
    }
}

/// Build a complete COLMAP-style dataset directory from a rendered synthetic scene.
pub fn write_dataset(dir: &Path, gt: &Model, cameras: &[Camera], points: &[(V3, [u8; 3])]) {
    let sparse = dir.join("sparse").join("0");
    let images = dir.join("images");
    std::fs::create_dir_all(&sparse).unwrap();
    std::fs::create_dir_all(&images).unwrap();
    write_colmap_bin(&sparse, &synthetic_colmap_model(cameras, points));
    let opts = ProjOpts {
        sh_degree: gt.sh_degree,
        ..ProjOpts::default()
    };
    for cam in cameras {
        let img = render(gt, cam, &opts, [0.0; 3]);
        splatrs::render::save_png(&images.join(&cam.name), &img, cam.width, cam.height).unwrap();
    }
}