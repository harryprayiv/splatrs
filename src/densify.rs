//! Adaptive density control: clone small under-reconstructed Gaussians, split large ones,
//! prune transparent or oversized ones. The rules and constants are those of the reference
//! implementation; the "absgrad" option (Ye et al. 2024, AbsGS) swaps the accumulated
//! screen-space gradient for the accumulated absolute per-pixel contributions.

use crate::adam::Adam;
use crate::math::V3;
use crate::model::{retain_by_mask, Gaussian, Model};
use crate::project::{Proj, ProjGrad};
use crate::Real;
use rand::Rng;

/// Per-Gaussian statistics accumulated between densification steps.
#[derive(Clone, Debug, Default)]
pub struct DensifyStats {
    /// Sum over iterations of the norm of the NDC-space positional gradient.
    pub grad_accum: Vec<Real>,
    /// Number of iterations in which the Gaussian was visible.
    pub denom: Vec<u32>,
    /// Largest screen-space radius seen.
    pub max_radii: Vec<u32>,
}

impl DensifyStats {
    /// Zero statistics for `n` Gaussians.
    pub fn new(n: usize) -> DensifyStats {
        DensifyStats {
            grad_accum: vec![0.0; n],
            denom: vec![0; n],
            max_radii: vec![0; n],
        }
    }

    /// Record one iteration. The pixel-space gradient is converted to NDC units by scaling
    /// with half the image size, which is the convention the thresholds were tuned for.
    pub fn accumulate(
        &mut self,
        projs: &[Proj],
        grads: &[ProjGrad],
        abs_grads: Option<&[[Real; 2]]>,
        width: u32,
        height: u32,
    ) {
        let sx = width as Real * 0.5;
        let sy = height as Real * 0.5;
        for i in 0..projs.len() {
            if !projs[i].visible() {
                continue;
            }
            let g = match abs_grads {
                Some(a) => a[i],
                None => grads[i].mean2d,
            };
            let gx = g[0] * sx;
            let gy = g[1] * sy;
            self.grad_accum[i] += (gx * gx + gy * gy).sqrt();
            self.denom[i] += 1;
            self.max_radii[i] = self.max_radii[i].max(projs[i].radius);
        }
    }

    fn reset(&mut self, n: usize) {
        *self = DensifyStats::new(n);
    }
}

/// Densification thresholds.
#[derive(Clone, Copy, Debug)]
pub struct DensifyConfig {
    /// Average NDC gradient norm above which a Gaussian is cloned or split.
    pub grad_threshold: Real,
    /// Gaussians whose largest scale exceeds this fraction of the scene extent are split
    /// rather than cloned.
    pub percent_dense: Real,
    /// Gaussians less opaque than this are pruned.
    pub min_opacity: Real,
    /// Whether to prune Gaussians whose largest scale exceeds `max_world_size_frac` times
    /// the scene extent. The reference enables this after its first opacity reset.
    pub prune_big_world: bool,
    /// Fraction of the scene extent above which `prune_big_world` applies.
    pub max_world_size_frac: Real,
    /// Prune Gaussians whose screen radius exceeded this many pixels since the previous
    /// densification step. Off by default: the reference implementation resets its radius
    /// statistics before testing them, so its screen-size pruning never fires, and gsplat's
    /// default strategy leaves the equivalent check disabled as well.
    pub max_screen_size: Option<u32>,
    /// Number of children a split produces.
    pub split_children: usize,
}

impl Default for DensifyConfig {
    fn default() -> DensifyConfig {
        DensifyConfig {
            grad_threshold: 0.0002,
            percent_dense: 0.01,
            min_opacity: 0.005,
            prune_big_world: false,
            max_world_size_frac: 0.1,
            max_screen_size: None,
            split_children: 2,
        }
    }
}

/// What one densification step did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DensifyReport {
    /// Gaussians cloned.
    pub cloned: usize,
    /// Gaussians split (each replaced by `split_children`).
    pub split: usize,
    /// Gaussians pruned afterwards.
    pub pruned: usize,
}

fn normal_sample<R: Rng>(rng: &mut R) -> Real {
    // Box-Muller; u1 is kept away from zero so the log is finite.
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as Real
}

/// Clone, split and prune. The model, optimiser state and statistics are kept in step; the
/// statistics are reset to zero afterwards.
pub fn densify_and_prune<R: Rng>(
    model: &mut Model,
    adam: &mut Adam,
    stats: &mut DensifyStats,
    cfg: &DensifyConfig,
    extent: Real,
    rng: &mut R,
) -> DensifyReport {
    let n0 = model.len();
    assert_eq!(stats.grad_accum.len(), n0);
    let grads: Vec<Real> = (0..n0)
        .map(|i| {
            if stats.denom[i] > 0 {
                stats.grad_accum[i] / stats.denom[i] as Real
            } else {
                0.0
            }
        })
        .collect();
    let size_limit = cfg.percent_dense * extent;
    let mut report = DensifyReport::default();
    // Index into `stats` for each Gaussian of the current list; new Gaussians have none.
    let mut origin: Vec<Option<usize>> = (0..n0).map(Some).collect();

    // Clone: high gradient, small.
    let clones: Vec<Gaussian> = (0..n0)
        .filter(|&i| {
            grads[i] >= cfg.grad_threshold && model.gaussians[i].scales().max_elem() <= size_limit
        })
        .map(|i| model.gaussians[i])
        .collect();
    report.cloned = clones.len();
    adam.append_zeros(clones.len());
    origin.extend(std::iter::repeat_n(None, clones.len()));
    model.gaussians.extend(clones);

    // Split: high gradient, large. Only the original Gaussians are candidates.
    let split_idx: Vec<usize> = (0..n0)
        .filter(|&i| {
            grads[i] >= cfg.grad_threshold && model.gaussians[i].scales().max_elem() > size_limit
        })
        .collect();
    report.split = split_idx.len();
    let mut children = Vec::with_capacity(split_idx.len() * cfg.split_children);
    let shrink = (0.8 * cfg.split_children as Real).ln();
    for &i in &split_idx {
        let g = model.gaussians[i];
        let s = g.scales();
        let rot = g.rotation().to_mat();
        for _ in 0..cfg.split_children {
            let local = V3::new(
                normal_sample(rng) * s.x,
                normal_sample(rng) * s.y,
                normal_sample(rng) * s.z,
            );
            let mean = g.mean_v() + rot.mul_v(local);
            let mut child = g;
            child.mean = mean.to_array();
            for k in 0..3 {
                child.log_scale[k] = g.log_scale[k] - shrink;
            }
            children.push(child);
        }
    }
    adam.append_zeros(children.len());
    origin.extend(std::iter::repeat_n(None, children.len()));
    model.gaussians.extend(children);
    let mut keep = vec![true; model.len()];
    for &i in &split_idx {
        keep[i] = false;
    }
    model.retain_mask(&keep);
    adam.retain_mask(&keep);
    retain_by_mask(&mut origin, &keep);

    // Prune: transparent, or too large on screen or in the world. Screen radii only exist
    // for the original Gaussians; new ones are never screen-size-pruned in the step that
    // made them.
    let n2 = model.len();
    let mut keep = vec![true; n2];
    for i in 0..n2 {
        let g = &model.gaussians[i];
        let mut prune = g.alpha() < cfg.min_opacity;
        if cfg.prune_big_world {
            prune |= g.scales().max_elem() > cfg.max_world_size_frac * extent;
        }
        if let (Some(max_screen), Some(old)) = (cfg.max_screen_size, origin[i]) {
            prune |= stats.max_radii[old] > max_screen;
        }
        keep[i] = !prune;
    }
    report.pruned = keep.iter().filter(|&&k| !k).count();
    model.retain_mask(&keep);
    adam.retain_mask(&keep);
    stats.reset(model.len());
    report
}

/// Reset every opacity to at most `max_alpha`, zeroing the optimiser's opacity moments.
pub fn reset_opacity(model: &mut Model, adam: &mut Adam, max_alpha: Real) {
    let limit = crate::math::inverse_sigmoid(max_alpha);
    for g in model.gaussians.iter_mut() {
        g.opacity = g.opacity.min(limit);
    }
    adam.reset_opacity_state();
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn gaussian(mean: Real, log_scale: Real, opacity: Real) -> Gaussian {
        Gaussian {
            mean: [mean; 3],
            log_scale: [log_scale; 3],
            quat: [1.0, 0.0, 0.0, 0.0],
            opacity,
            sh: [[0.0; 3]; 16],
        }
    }

    #[test]
    fn clone_split_prune_bookkeeping() {
        // Four Gaussians: small+high grad (clone), large+high grad (split),
        // transparent (prune), ordinary (kept).
        let extent: Real = 10.0;
        let mut model = Model {
            gaussians: vec![
                gaussian(0.0, (0.001 as Real).ln(), 2.0),
                gaussian(1.0, (0.5 as Real).ln(), 2.0),
                gaussian(2.0, (0.001 as Real).ln(), -10.0),
                gaussian(3.0, (0.001 as Real).ln(), 2.0),
            ],
            sh_degree: 0,
            max_sh_degree: 3,
        };
        let mut adam = Adam::new(4);
        let mut stats = DensifyStats::new(4);
        stats.grad_accum = vec![1.0, 1.0, 0.0, 0.0];
        stats.denom = vec![1, 1, 1, 1];
        let cfg = DensifyConfig::default();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let report = densify_and_prune(&mut model, &mut adam, &mut stats, &cfg, extent, &mut rng);
        assert_eq!(
            report,
            DensifyReport {
                cloned: 1,
                split: 1,
                pruned: 1
            }
        );
        // Survivors: original 0, original 3, clone of 0, two children of 1.
        assert_eq!(model.len(), 5);
        assert_eq!(adam.m.len(), 5);
        assert_eq!(stats.denom.len(), 5);
        assert_eq!(model.gaussians[0].mean, [0.0; 3]);
        assert_eq!(model.gaussians[1].mean, [3.0; 3]);
        assert_eq!(model.gaussians[2].mean, [0.0; 3]);
        for child in &model.gaussians[3..5] {
            let expect = (0.5 as Real).ln() - (1.6 as Real).ln();
            assert!((child.log_scale[0] - expect).abs() < 1e-5);
        }
    }
}