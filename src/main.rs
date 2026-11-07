//! Command-line interface: inspect a COLMAP reconstruction, train a splat, render views.

use anyhow::{bail, Context};
use clap::{Args, Parser, Subcommand};
use splatrs::camera::scene_extent;
use splatrs::dataset::{self, camera_from_colmap};
use splatrs::model::Model;
use splatrs::render::{render, save_png};
use splatrs::train::{evaluate, train, Progress, TrainConfig};
use splatrs::{colmap, ply, Real};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "splatrs",
    version,
    about = "3D Gaussian splatting, CPU reference implementation"
)]
struct Cli {
    /// Number of worker threads (default: all cores).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
struct DataArgs {
    /// COLMAP project root containing `sparse/0` (or `sparse`) and `images`.
    #[arg(long)]
    data: PathBuf,
    /// Sparse model directory, overriding the default under `--data`.
    #[arg(long)]
    sparse: Option<PathBuf>,
    /// Images directory, overriding the default under `--data`.
    #[arg(long)]
    images: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Summarise a COLMAP reconstruction and check it is usable for training.
    Info {
        #[command(flatten)]
        data: DataArgs,
    },
    /// Train a Gaussian splat from a COLMAP reconstruction.
    Train {
        #[command(flatten)]
        data: DataArgs,
        /// Output directory.
        #[arg(long)]
        out: PathBuf,
        /// Number of iterations (at least 1).
        #[arg(long, default_value_t = 30_000, value_parser = clap::value_parser!(u32).range(1..))]
        iterations: u32,
        /// Integer factor by which to downscale the training images (at least 1).
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        downscale: u32,
        /// Hold out every 8th image for a final PSNR report.
        #[arg(long)]
        eval: bool,
        /// Train on a white background instead of black.
        #[arg(long)]
        white_bg: bool,
        /// Mip-Splatting style antialiasing (opacity compensation for the 2D dilation).
        #[arg(long)]
        antialias: bool,
        /// AbsGS densification criterion (absolute per-pixel gradients).
        #[arg(long)]
        absgrad: bool,
        /// Highest spherical harmonics degree (0 to 3).
        #[arg(long, default_value_t = 3)]
        sh_degree: usize,
        /// Densification gradient threshold (0.0002, or 0.0008 with --absgrad).
        #[arg(long)]
        grad_threshold: Option<Real>,
        /// Disable densification entirely.
        #[arg(long)]
        no_densify: bool,
        /// Also prune Gaussians whose screen radius exceeded this many pixels (off by
        /// default, see DensifyConfig::max_screen_size).
        #[arg(long)]
        max_screen_size: Option<u32>,
        /// Random seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Extra iterations at which to save a checkpoint `.ply`.
        #[arg(long, value_delimiter = ',')]
        save_at: Vec<u32>,
        /// Iterations between progress lines.
        #[arg(long, default_value_t = 100)]
        log_interval: u32,
    },
    /// Render a trained `.ply` from the cameras of a COLMAP reconstruction.
    Render {
        #[command(flatten)]
        data: DataArgs,
        /// The trained splat.
        #[arg(long)]
        ply: PathBuf,
        /// Output directory for PNG files.
        #[arg(long)]
        out: PathBuf,
        /// Integer factor by which to downscale the render resolution (at least 1).
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        downscale: u32,
        /// Render on a white background.
        #[arg(long)]
        white_bg: bool,
        /// Opacity compensation, must match what the splat was trained with.
        #[arg(long)]
        antialias: bool,
        /// Only render the view with this image name.
        #[arg(long)]
        view: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .context("thread pool")?;
    }
    match cli.command {
        Command::Info { data } => info(&data),
        Command::Train {
            data,
            out,
            iterations,
            downscale,
            eval,
            white_bg,
            antialias,
            absgrad,
            sh_degree,
            grad_threshold,
            no_densify,
            max_screen_size,
            seed,
            save_at,
            log_interval,
        } => {
            if sh_degree > 3 {
                bail!("--sh-degree must be 0 to 3");
            }
            let mut cfg = TrainConfig {
                iterations,
                white_background: white_bg,
                antialias,
                absgrad,
                sh_degree,
                densify: !no_densify,
                max_screen_size,
                seed,
                log_interval,
                ..TrainConfig::default()
            };
            // Absolute per-pixel gradients are never smaller than the signed norm, so the
            // AbsGS criterion needs its own threshold (0.0008 in the paper and in gsplat).
            cfg.densify_grad_threshold = match (grad_threshold, absgrad) {
                (Some(t), _) => t,
                (None, true) => 0.0008,
                (None, false) => cfg.densify_grad_threshold,
            };
            cfg.save_iterations.extend(save_at);
            run_train(&data, &out, downscale, eval, cfg)
        }
        Command::Render {
            data,
            ply,
            out,
            downscale,
            white_bg,
            antialias,
            view,
        } => run_render(
            &data,
            &ply,
            &out,
            downscale,
            white_bg,
            antialias,
            view.as_deref(),
        ),
    }
}

fn info(data: &DataArgs) -> anyhow::Result<()> {
    let sparse = dataset::locate_sparse(&data.data, data.sparse.as_deref())?;
    let model = colmap::read_model(&sparse)?;
    println!("sparse model: {}", sparse.display());
    match dataset::locate(&data.data, data.sparse.as_deref(), data.images.as_deref()) {
        Ok((_, images)) => println!("images dir:   {}", images.display()),
        Err(e) => println!("images dir:   missing ({e})"),
    }
    let mut ids: Vec<&u32> = model.cameras.keys().collect();
    ids.sort();
    let mut distorted = 0;
    for id in ids {
        let c = &model.cameras[id];
        let flag = if c.model.is_pinhole() {
            ""
        } else {
            "  <- distortion model, undistort first"
        };
        if !c.model.is_pinhole() {
            distorted += 1;
        }
        println!(
            "camera {}: {} {}x{} params {:?}{}",
            c.id,
            c.model.name(),
            c.width,
            c.height,
            c.params,
            flag
        );
    }
    println!("registered images: {}", model.images.len());
    println!("3D points: {}", model.points.len());
    if distorted == 0 {
        let cameras: Vec<_> = model
            .images
            .iter()
            .map(|i| camera_from_colmap(&model, i))
            .collect::<Result<_, _>>()?;
        println!("scene extent: {:.4}", scene_extent(&cameras));
        println!("ready for training");
    } else {
        println!("not ready: {distorted} camera(s) have lens distortion");
        println!("run: colmap image_undistorter --image_path <images> --input_path {} --output_path <undistorted> --output_type COLMAP", sparse.display());
    }
    Ok(())
}

fn run_train(
    data: &DataArgs,
    out: &Path,
    downscale: u32,
    eval: bool,
    cfg: TrainConfig,
) -> anyhow::Result<()> {
    let (sparse, images) =
        dataset::locate(&data.data, data.sparse.as_deref(), data.images.as_deref())?;
    eprintln!("loading {} (downscale {downscale})", sparse.display());
    let ds = dataset::load(&sparse, &images, downscale)?;
    let (train_views, test_views) = if eval {
        dataset::split_eval(ds.views, 8)
    } else {
        (ds.views, Vec::new())
    };
    if train_views.is_empty() {
        bail!("no training views");
    }
    // The extent is taken over the training cameras only, as the reference does.
    let train_cameras: Vec<_> = train_views.iter().map(|v| v.camera.clone()).collect();
    let extent = scene_extent(&train_cameras);
    let v0 = &train_views[0].camera;
    eprintln!(
        "{} training views, {} held out, {}x{} px, {} points, extent {:.4}",
        train_views.len(),
        test_views.len(),
        v0.width,
        v0.height,
        ds.points.len(),
        extent
    );
    if ds.points.is_empty() {
        bail!("the reconstruction has no 3D points to initialise from");
    }
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let model = Model::from_points(&ds.points, cfg.sh_degree);

    let on_progress = |p: &Progress| {
        eprintln!(
            "iter {:>6}  loss {:.5}  ema {:.5}  psnr {:5.2}  gaussians {:>8}  {:7.1} ms/iter  +clone {} +split {} -prune {}",
            p.iteration, p.loss, p.ema_loss, p.psnr, p.num_gaussians, p.ms_per_iter, p.densified.0, p.densified.1, p.densified.2
        );
    };
    let on_save = |iteration: u32, m: &Model| -> splatrs::Result<()> {
        let dir = out
            .join("point_cloud")
            .join(format!("iteration_{iteration}"));
        std::fs::create_dir_all(&dir).map_err(|e| splatrs::Error::io(&dir, e))?;
        let path = dir.join("point_cloud.ply");
        ply::write(&path, m)?;
        eprintln!("saved {} ({} gaussians)", path.display(), m.len());
        Ok(())
    };
    let trained = train(model, &train_views, extent, &cfg, on_progress, on_save)?;
    eprintln!(
        "train PSNR (all training views): {:.2} dB",
        evaluate(&trained, &train_views, &cfg)
    );
    if !test_views.is_empty() {
        eprintln!(
            "test PSNR (held-out views):      {:.2} dB",
            evaluate(&trained, &test_views, &cfg)
        );
    }
    Ok(())
}

fn run_render(
    data: &DataArgs,
    ply_path: &Path,
    out: &Path,
    downscale: u32,
    white_bg: bool,
    antialias: bool,
    only: Option<&str>,
) -> anyhow::Result<()> {
    // Rendering needs only the cameras, not the images.
    let sparse = dataset::locate_sparse(&data.data, data.sparse.as_deref())?;
    let model = ply::read(ply_path)?;
    let colmap_model = colmap::read_model(&sparse)?;
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let cfg = TrainConfig {
        white_background: white_bg,
        antialias,
        ..TrainConfig::default()
    };
    let opts = cfg.proj_opts(model.sh_degree);
    let bg = cfg.background();
    let mut rendered = 0;
    for img in &colmap_model.images {
        if let Some(name) = only {
            if img.name != name {
                continue;
            }
        }
        let cam = camera_from_colmap(&colmap_model, img)?;
        let w = ((cam.width as f64 / downscale as f64).round() as u32).max(1);
        let h = ((cam.height as f64 / downscale as f64).round() as u32).max(1);
        let cam = cam.resized(w, h);
        let image = render(&model, &cam, &opts, bg);
        // Keep any subdirectory of the image name so multi-camera rigs do not collide.
        let path = out.join(&img.name).with_extension("png");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        save_png(&path, &image, w, h)?;
        rendered += 1;
    }
    if rendered == 0 {
        bail!("no view rendered (check --view against the image names in images.bin)");
    }
    eprintln!("rendered {rendered} view(s) into {}", out.display());
    Ok(())
}