//! File formats and the command-line interface: COLMAP round trips through both formats,
//! dataset loading with downscaling, and the `info`, `train` and `render` subcommands on a
//! synthetic project.
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_range_loop)]

mod common;

use common::rng;
use common::scene::{
    ground_truth, init_points, ring_cameras, synthetic_colmap_model, temp_dir, write_colmap_bin,
    write_colmap_txt, write_dataset,
};
use splatrs::camera::scene_extent;
use splatrs::colmap::{read_model, ColmapModel};
use splatrs::dataset;
use splatrs::math::V3;

fn assert_models_equal(a: &ColmapModel, b: &ColmapModel) {
    assert_eq!(a.cameras.len(), b.cameras.len());
    for (id, ca) in &a.cameras {
        let cb = &b.cameras[id];
        assert_eq!(ca.model, cb.model);
        assert_eq!((ca.width, ca.height), (cb.width, cb.height));
        assert_eq!(ca.params.len(), cb.params.len());
        for (x, y) in ca.params.iter().zip(&cb.params) {
            assert!((x - y).abs() < 1e-9);
        }
    }
    assert_eq!(a.images.len(), b.images.len());
    for (ia, ib) in a.images.iter().zip(&b.images) {
        assert_eq!(
            (ia.id, ia.camera_id, &ia.name),
            (ib.id, ib.camera_id, &ib.name)
        );
        for k in 0..4 {
            assert!((ia.qvec[k] - ib.qvec[k]).abs() < 1e-9);
        }
        for k in 0..3 {
            assert!((ia.tvec[k] - ib.tvec[k]).abs() < 1e-9);
        }
    }
    assert_eq!(a.points.len(), b.points.len());
    for (pa, pb) in a.points.iter().zip(&b.points) {
        assert_eq!((pa.id, pa.rgb), (pb.id, pb.rgb));
        for k in 0..3 {
            assert!((pa.xyz[k] - pb.xyz[k]).abs() < 1e-9);
        }
        assert!((pa.error - pb.error).abs() < 1e-9);
    }
}

#[test]
fn colmap_binary_and_text_round_trip() {
    let cameras = ring_cameras(5, 32, 24, 2.0);
    let points = vec![
        (V3::new(0.1, 0.2, 0.3), [1u8, 2, 3]),
        (V3::new(-0.5, 0.0, 0.25), [200u8, 100, 50]),
    ];
    let model = synthetic_colmap_model(&cameras, &points);
    let dir = temp_dir("colmap");
    let bin = dir.join("bin");
    let txt = dir.join("txt");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::create_dir_all(&txt).unwrap();
    write_colmap_bin(&bin, &model);
    write_colmap_txt(&txt, &model);
    let from_bin = read_model(&bin).unwrap();
    let from_txt = read_model(&txt).unwrap();
    assert_models_equal(&model, &from_bin);
    assert_models_equal(&model, &from_txt);
    // The camera reconstructed from the COLMAP record must match the original.
    for (im, cam) in from_bin.images.iter().zip(&cameras) {
        let back = dataset::camera_from_colmap(&from_bin, im).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((back.rot.m[i][j] - cam.rot.m[i][j]).abs() < 1e-5);
            }
        }
        assert!((back.trans - cam.trans).norm() < 1e-5);
        assert!((back.fx - cam.fx).abs() < 1e-5 && (back.cx - cam.cx).abs() < 1e-5);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn dataset_loads_and_downscales() {
    let mut rng = rng(300);
    let gt = ground_truth(&mut rng, 20);
    let cameras = ring_cameras(6, 40, 30, 2.5);
    let points = init_points(&gt, &mut rng);
    let dir = temp_dir("dataset");
    write_dataset(&dir, &gt, &cameras, &points);
    let (sparse, images) = dataset::locate(&dir, None, None).unwrap();
    let full = dataset::load(&sparse, &images, 1).unwrap();
    assert_eq!(full.views.len(), 6);
    assert_eq!(full.points.len(), 20);
    assert_eq!(full.views[0].rgb.len(), 40 * 30 * 3);
    assert!((full.extent - scene_extent(&cameras)).abs() < 1e-4);
    let half = dataset::load(&sparse, &images, 2).unwrap();
    let v = &half.views[0];
    assert_eq!((v.camera.width, v.camera.height), (20, 15));
    assert_eq!(v.rgb.len(), 20 * 15 * 3);
    assert!((v.camera.fx - cameras[0].fx * 0.5).abs() < 1e-4);
    assert!((v.camera.cx - cameras[0].cx * 0.5).abs() < 1e-4);
    // The downscaled image is the full image with half the detail, not something else.
    let full_mean: f64 =
        full.views[0].rgb.iter().map(|&v| v as f64).sum::<f64>() / full.views[0].rgb.len() as f64;
    let half_mean: f64 = v.rgb.iter().map(|&v| v as f64).sum::<f64>() / v.rgb.len() as f64;
    assert!(
        (full_mean - half_mean).abs() < 3.0,
        "{full_mean} vs {half_mean}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn cli_info_train_render() {
    let mut rng = rng(400);
    let gt = ground_truth(&mut rng, 25);
    let cameras = ring_cameras(6, 40, 30, 2.5);
    let points = init_points(&gt, &mut rng);
    let dir = temp_dir("cli");
    write_dataset(&dir, &gt, &cameras, &points);
    let exe = env!("CARGO_BIN_EXE_splatrs");
    let run = |args: &[&str]| {
        let out = std::process::Command::new(exe).args(args).output().unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success(), "command {:?} failed:\n{text}", args);
        text
    };
    let data = dir.to_str().unwrap();
    let info = run(&["info", "--data", data]);
    assert!(info.contains("ready for training"), "{info}");
    assert!(info.contains("registered images: 6"), "{info}");
    let out_dir = dir.join("out");
    let out = out_dir.to_str().unwrap();
    let log = run(&[
        "--threads",
        "2",
        "train",
        "--data",
        data,
        "--out",
        out,
        "--iterations",
        "30",
        "--no-densify",
        "--sh-degree",
        "1",
        "--log-interval",
        "10",
        "--save-at",
        "20",
    ]);
    assert!(log.contains("iter     30"), "{log}");
    let ply_20 = out_dir
        .join("point_cloud")
        .join("iteration_20")
        .join("point_cloud.ply");
    let ply_30 = out_dir
        .join("point_cloud")
        .join("iteration_30")
        .join("point_cloud.ply");
    assert!(ply_20.is_file() && ply_30.is_file());
    let loaded = splatrs::ply::read(&ply_30).unwrap();
    assert_eq!(loaded.len(), 25);
    assert_eq!(loaded.max_sh_degree, 1);
    let renders = dir.join("renders");
    run(&[
        "render",
        "--data",
        data,
        "--ply",
        ply_30.to_str().unwrap(),
        "--out",
        renders.to_str().unwrap(),
    ]);
    for cam in &cameras {
        assert!(
            renders.join(&cam.name).is_file(),
            "missing render for {}",
            cam.name
        );
    }
    let one = dir.join("one");
    run(&[
        "render",
        "--data",
        data,
        "--ply",
        ply_30.to_str().unwrap(),
        "--out",
        one.to_str().unwrap(),
        "--view",
        "view_002.png",
        "--downscale",
        "2",
    ]);
    let img = image::open(one.join("view_002.png")).unwrap();
    assert_eq!((img.width(), img.height()), (20, 15));
    std::fs::remove_dir_all(&dir).unwrap();
}