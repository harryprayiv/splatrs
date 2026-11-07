//! The training loop: the schedule of the reference implementation (learning rates,
//! SH degree ramp, densification window, opacity resets), one random view per iteration.

use crate::adam::{Adam, LearningRates};
use crate::dataset::View;
use crate::densify::{densify_and_prune, reset_opacity, DensifyConfig, DensifyStats};
use crate::error::Result;
use crate::loss::{photometric_loss, psnr};
use crate::model::Model;
use crate::project::{project_all, project_backward_all, ProjOpts};
use crate::rasterize::{bin, rasterize_backward, rasterize_forward, RasterSettings};
use crate::Real;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// All training hyperparameters. Defaults are those of the reference implementation.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Total iterations.
    pub iterations: u32,
    /// Mean learning rate at iteration 0, multiplied by the scene extent.
    pub position_lr_init: Real,
    /// Mean learning rate at `position_lr_max_steps`, multiplied by the scene extent.
    pub position_lr_final: Real,
    /// Length of the exponential decay of the mean learning rate.
    pub position_lr_max_steps: u32,
    /// SH DC learning rate; higher degrees use a twentieth of it.
    pub feature_lr: Real,
    /// Opacity learning rate.
    pub opacity_lr: Real,
    /// Log-scale learning rate.
    pub scaling_lr: Real,
    /// Rotation learning rate.
    pub rotation_lr: Real,
    /// Weight of the D-SSIM term in the loss.
    pub lambda_dssim: Real,
    /// Whether densification runs at all.
    pub densify: bool,
    /// First iteration at which densification may run.
    pub densify_from_iter: u32,
    /// Iteration after which densification stops.
    pub densify_until_iter: u32,
    /// Iterations between densification steps.
    pub densification_interval: u32,
    /// Iterations between opacity resets (0 disables resets).
    pub opacity_reset_interval: u32,
    /// Densification gradient threshold.
    pub densify_grad_threshold: Real,
    /// Clone/split size boundary as a fraction of the scene extent.
    pub percent_dense: Real,
    /// Pruning opacity threshold.
    pub min_opacity: Real,
    /// Screen-radius pruning threshold in pixels (see `DensifyConfig::max_screen_size`).
    pub max_screen_size: Option<u32>,
    /// Use absolute per-pixel gradients for densification (AbsGS).
    pub absgrad: bool,
    /// Mip-Splatting style opacity compensation.
    pub antialias: bool,
    /// White instead of black background.
    pub white_background: bool,
    /// Highest SH degree to train.
    pub sh_degree: usize,
    /// Iterations between SH degree increases.
    pub sh_degree_interval: u32,
    /// Near-plane depth for culling.
    pub near: Real,
    /// Seed for view order and split sampling.
    pub seed: u64,
    /// Iterations at which `on_save` is called (the final iteration always is).
    pub save_iterations: Vec<u32>,
    /// Iterations between progress reports (0 disables).
    pub log_interval: u32,
}

impl Default for TrainConfig {
    fn default() -> TrainConfig {
        TrainConfig {
            iterations: 30_000,
            position_lr_init: 0.000_16,
            position_lr_final: 0.000_001_6,
            position_lr_max_steps: 30_000,
            feature_lr: 0.0025,
            opacity_lr: 0.025,
            scaling_lr: 0.005,
            rotation_lr: 0.001,
            lambda_dssim: 0.2,
            densify: true,
            densify_from_iter: 500,
            densify_until_iter: 15_000,
            densification_interval: 100,
            opacity_reset_interval: 3000,
            densify_grad_threshold: 0.0002,
            percent_dense: 0.01,
            min_opacity: 0.005,
            max_screen_size: None,
            absgrad: false,
            antialias: false,
            white_background: false,
            sh_degree: 3,
            sh_degree_interval: 1000,
            near: 0.2,
            seed: 0,
            save_iterations: vec![7000],
            log_interval: 100,
        }
    }
}

impl TrainConfig {
    /// Background colour implied by `white_background`.
    pub fn background(&self) -> [Real; 3] {
        if self.white_background {
            [1.0; 3]
        } else {
            [0.0; 3]
        }
    }

    /// Projection options for the current model state.
    pub fn proj_opts(&self, sh_degree: usize) -> ProjOpts {
        ProjOpts {
            near: self.near,
            eps2d: 0.3,
            antialias: self.antialias,
            sh_degree,
        }
    }

    /// Mean learning rate at a given iteration (exponential decay, times the extent).
    pub fn position_lr(&self, iteration: u32, extent: Real) -> Real {
        let t = (iteration as Real / self.position_lr_max_steps.max(1) as Real).clamp(0.0, 1.0);
        let init = self.position_lr_init * extent;
        let fin = self.position_lr_final * extent;
        (init.ln() * (1.0 - t) + fin.ln() * t).exp()
    }
}

/// Progress report passed to the callback.
#[derive(Clone, Debug)]
pub struct Progress {
    /// Iteration just finished (1-based).
    pub iteration: u32,
    /// Loss of this iteration.
    pub loss: Real,
    /// Exponential moving average of the loss.
    pub ema_loss: Real,
    /// PSNR of this iteration's render against its ground truth.
    pub psnr: Real,
    /// Current number of Gaussians.
    pub num_gaussians: usize,
    /// Mean wall time per iteration since the previous report, in milliseconds.
    pub ms_per_iter: f64,
    /// Densification events since the previous report: (cloned, split, pruned).
    pub densified: (usize, usize, usize),
}

/// Train `model` on `views`. `on_progress` is called every `log_interval` iterations and
/// `on_save` at every iteration in `save_iterations` and at the end.
pub fn train(
    mut model: Model,
    views: &[View],
    extent: Real,
    cfg: &TrainConfig,
    mut on_progress: impl FnMut(&Progress),
    mut on_save: impl FnMut(u32, &Model) -> Result<()>,
) -> Result<Model> {
    if views.is_empty() {
        return Err(crate::Error::Invalid("no training views".to_string()));
    }
    if cfg.iterations == 0 {
        return Err(crate::Error::Invalid(
            "iterations must be at least 1".to_string(),
        ));
    }
    if cfg.sh_degree > crate::sh::MAX_DEGREE {
        return Err(crate::Error::Invalid(format!(
            "sh_degree {} exceeds the maximum of {}",
            cfg.sh_degree,
            crate::sh::MAX_DEGREE
        )));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut adam = Adam::new(model.len());
    let mut stats = DensifyStats::new(model.len());
    let background = cfg.background();
    model.max_sh_degree = cfg.sh_degree;
    model.sh_degree = 0;

    let mut order: Vec<usize> = Vec::new();
    let mut ema_loss: Real = 0.0;
    let mut timer = Instant::now();
    let mut densified = (0usize, 0usize, 0usize);

    for iteration in 1..=cfg.iterations {
        if cfg.sh_degree_interval > 0
            && iteration % cfg.sh_degree_interval == 0
            && model.sh_degree < cfg.sh_degree
        {
            model.sh_degree += 1;
        }
        if order.is_empty() {
            order = (0..views.len()).collect();
            order.shuffle(&mut rng);
        }
        let view = &views[order.pop().expect("order is non-empty")];
        let cam = &view.camera;
        let (w, h) = (cam.width, cam.height);

        // Forward.
        let opts = cfg.proj_opts(model.sh_degree);
        let projs = project_all(&model, cam, &opts);
        let binning = bin(&projs, w, h);
        let settings = RasterSettings {
            width: w,
            height: h,
            background,
        };
        let out = rasterize_forward(&projs, &binning, &settings);
        let gt = view.to_real();
        let (loss, dimage) =
            photometric_loss(&out.image, &gt, w as usize, h as usize, cfg.lambda_dssim);

        // Backward.
        let (pgrads, abs_grads) = rasterize_backward(&projs, &binning, &settings, &out, &dimage);
        let grads = project_backward_all(&model, cam, &opts, &pgrads);
        if cfg.densify && iteration < cfg.densify_until_iter {
            let abs = if cfg.absgrad {
                Some(abs_grads.as_slice())
            } else {
                None
            };
            stats.accumulate(&projs, &pgrads, abs, w, h);
        }

        // Optimiser step.
        let lrs = LearningRates {
            mean: cfg.position_lr(iteration, extent),
            log_scale: cfg.scaling_lr,
            quat: cfg.rotation_lr,
            opacity: cfg.opacity_lr,
            sh_dc: cfg.feature_lr,
            sh_rest: cfg.feature_lr / 20.0,
        };
        adam.step(&mut model.gaussians, &grads, &lrs);

        // Density control.
        if cfg.densify && iteration < cfg.densify_until_iter {
            if iteration > cfg.densify_from_iter && iteration % cfg.densification_interval == 0 {
                let dcfg = DensifyConfig {
                    grad_threshold: cfg.densify_grad_threshold,
                    percent_dense: cfg.percent_dense,
                    min_opacity: cfg.min_opacity,
                    prune_big_world: cfg.opacity_reset_interval > 0
                        && iteration > cfg.opacity_reset_interval,
                    max_screen_size: cfg.max_screen_size,
                    ..DensifyConfig::default()
                };
                let r =
                    densify_and_prune(&mut model, &mut adam, &mut stats, &dcfg, extent, &mut rng);
                densified.0 += r.cloned;
                densified.1 += r.split;
                densified.2 += r.pruned;
            }
            let reset = cfg.opacity_reset_interval > 0
                && (iteration % cfg.opacity_reset_interval == 0
                    || (cfg.white_background && iteration == cfg.densify_from_iter));
            if reset {
                reset_opacity(&mut model, &mut adam, 0.01);
            }
        }

        ema_loss = if iteration == 1 {
            loss
        } else {
            0.4 * loss + 0.6 * ema_loss
        };
        if cfg.log_interval > 0
            && (iteration % cfg.log_interval == 0 || iteration == cfg.iterations)
        {
            let elapsed = timer.elapsed().as_secs_f64() * 1000.0;
            let since = if iteration % cfg.log_interval == 0 {
                cfg.log_interval
            } else {
                iteration % cfg.log_interval
            };
            on_progress(&Progress {
                iteration,
                loss,
                ema_loss,
                psnr: psnr(&out.image, &gt),
                num_gaussians: model.len(),
                ms_per_iter: elapsed / since.max(1) as f64,
                densified,
            });
            densified = (0, 0, 0);
            timer = Instant::now();
        }
        if cfg.save_iterations.contains(&iteration) && iteration != cfg.iterations {
            on_save(iteration, &model)?;
        }
    }
    on_save(cfg.iterations, &model)?;
    Ok(model)
}

/// Mean PSNR of the model over a set of views.
pub fn evaluate(model: &Model, views: &[View], cfg: &TrainConfig) -> Real {
    if views.is_empty() {
        return Real::NAN;
    }
    let opts = cfg.proj_opts(model.sh_degree);
    let bg = cfg.background();
    let mut total = 0.0;
    for v in views {
        let img = crate::render::render(model, &v.camera, &opts, bg);
        total += psnr(&img, &v.to_real());
    }
    total / views.len() as Real
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::math::V3;

    fn one_view() -> View {
        View {
            camera: Camera::look_at(
                "v.png",
                16,
                16,
                16.0,
                16.0,
                V3::new(0.0, 0.0, -3.0),
                V3::ZERO,
                V3::new(0.0, -1.0, 0.0),
            ),
            rgb: vec![0; 16 * 16 * 3],
        }
    }

    fn empty_model() -> Model {
        Model {
            gaussians: Vec::new(),
            sh_degree: 0,
            max_sh_degree: 0,
        }
    }

    #[test]
    fn zero_iterations_is_rejected_without_saving() {
        let cfg = TrainConfig {
            iterations: 0,
            sh_degree: 0,
            log_interval: 0,
            ..TrainConfig::default()
        };
        let mut saved = false;
        let result = train(
            empty_model(),
            &[one_view()],
            1.0,
            &cfg,
            |_| {},
            |_, _| {
                saved = true;
                Ok(())
            },
        );
        assert!(matches!(result, Err(crate::Error::Invalid(_))));
        assert!(!saved, "a rejected run must not write a checkpoint");
    }
}