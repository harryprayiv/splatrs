//! Time the phases of one training iteration on a dataset and a trained `.ply`:
//!
//! ```text
//! cargo run --release --example profile -- <sparse dir> <images dir> <splat.ply>
//! ```
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_range_loop)]
use splatrs::dataset;
use splatrs::loss::photometric_loss;
use splatrs::project::{project_all, project_backward_all, ProjOpts};
use splatrs::rasterize::{bin, rasterize_backward, rasterize_forward, RasterSettings};
use std::time::Instant;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ds = dataset::load(
        std::path::Path::new(&args[1]),
        std::path::Path::new(&args[2]),
        1,
    )
    .unwrap();
    let model = splatrs::ply::read(std::path::Path::new(&args[3])).unwrap();
    let view = &ds.views[0];
    let cam = &view.camera;
    let (w, h) = (cam.width, cam.height);
    let opts = ProjOpts {
        sh_degree: model.sh_degree,
        ..ProjOpts::default()
    };
    let settings = RasterSettings {
        width: w,
        height: h,
        background: [0.0; 3],
    };
    for _ in 0..3 {
        let t0 = Instant::now();
        let projs = project_all(&model, cam, &opts);
        let t1 = Instant::now();
        let binning = bin(&projs, w, h);
        let t2 = Instant::now();
        let out = rasterize_forward(&projs, &binning, &settings);
        let t3 = Instant::now();
        let gt = view.to_real();
        let (_, dimage) = photometric_loss(&out.image, &gt, w as usize, h as usize, 0.2);
        let t4 = Instant::now();
        let (pg, _) = rasterize_backward(&projs, &binning, &settings, &out, &dimage);
        let t5 = Instant::now();
        let _g = project_backward_all(&model, cam, &opts, &pg);
        let t6 = Instant::now();
        let entries = binning.sorted.len();
        let vis = projs.iter().filter(|p| p.visible()).count();
        eprintln!("gaussians {} visible {} entries {} | project {:?} bin {:?} forward {:?} loss {:?} backward {:?} proj_bwd {:?}",
            model.len(), vis, entries, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5);
    }
}