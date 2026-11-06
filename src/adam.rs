//! Adam optimiser over the Gaussian parameter records, with one learning rate per
//! parameter group, matching the reference implementation's optimiser setup.

use crate::model::{retain_by_mask, Gaussian};
use crate::sh::MAX_COEFFS;
use crate::Real;
use rayon::prelude::*;

/// Learning rate of each parameter group.
#[derive(Clone, Copy, Debug)]
pub struct LearningRates {
    /// Means (scheduled by the trainer).
    pub mean: Real,
    /// Log scales.
    pub log_scale: Real,
    /// Rotations.
    pub quat: Real,
    /// Logit opacity.
    pub opacity: Real,
    /// SH degree-0 coefficients.
    pub sh_dc: Real,
    /// SH higher-degree coefficients.
    pub sh_rest: Real,
}

/// Adam state: first and second moments shaped like the parameters, plus the step count.
#[derive(Clone, Debug)]
pub struct Adam {
    /// First moments.
    pub m: Vec<Gaussian>,
    /// Second moments.
    pub v: Vec<Gaussian>,
    /// Number of steps taken.
    pub step: u64,
    /// Exponential decay of the first moment.
    pub beta1: Real,
    /// Exponential decay of the second moment.
    pub beta2: Real,
    /// Added to the denominator (1e-15 in the reference).
    pub eps: Real,
}

#[inline]
fn update(
    p: &mut Real,
    g: Real,
    m: &mut Real,
    v: &mut Real,
    lr: Real,
    b1: Real,
    b2: Real,
    c1: Real,
    c2: Real,
    eps: Real,
) {
    *m = b1 * *m + (1.0 - b1) * g;
    *v = b2 * *v + (1.0 - b2) * g * g;
    let m_hat = *m / c1;
    let v_hat = *v / c2;
    *p -= lr * m_hat / (v_hat.sqrt() + eps);
}

impl Adam {
    /// Zero state for `n` Gaussians.
    pub fn new(n: usize) -> Adam {
        Adam {
            m: vec![Gaussian::default(); n],
            v: vec![Gaussian::default(); n],
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-15,
        }
    }

    /// One optimisation step.
    pub fn step(&mut self, params: &mut [Gaussian], grads: &[Gaussian], lr: &LearningRates) {
        assert_eq!(params.len(), grads.len());
        assert_eq!(params.len(), self.m.len());
        self.step += 1;
        let (b1, b2, eps) = (self.beta1, self.beta2, self.eps);
        let c1 = 1.0 - b1.powi(self.step as i32);
        let c2 = 1.0 - b2.powi(self.step as i32);
        let lr = *lr;
        params
            .par_iter_mut()
            .zip(grads.par_iter())
            .zip(self.m.par_iter_mut())
            .zip(self.v.par_iter_mut())
            .for_each(|(((p, g), m), v)| {
                for i in 0..3 {
                    update(
                        &mut p.mean[i],
                        g.mean[i],
                        &mut m.mean[i],
                        &mut v.mean[i],
                        lr.mean,
                        b1,
                        b2,
                        c1,
                        c2,
                        eps,
                    );
                    update(
                        &mut p.log_scale[i],
                        g.log_scale[i],
                        &mut m.log_scale[i],
                        &mut v.log_scale[i],
                        lr.log_scale,
                        b1,
                        b2,
                        c1,
                        c2,
                        eps,
                    );
                }
                for i in 0..4 {
                    update(
                        &mut p.quat[i],
                        g.quat[i],
                        &mut m.quat[i],
                        &mut v.quat[i],
                        lr.quat,
                        b1,
                        b2,
                        c1,
                        c2,
                        eps,
                    );
                }
                update(
                    &mut p.opacity,
                    g.opacity,
                    &mut m.opacity,
                    &mut v.opacity,
                    lr.opacity,
                    b1,
                    b2,
                    c1,
                    c2,
                    eps,
                );
                for k in 0..MAX_COEFFS {
                    let rate = if k == 0 { lr.sh_dc } else { lr.sh_rest };
                    for c in 0..3 {
                        update(
                            &mut p.sh[k][c],
                            g.sh[k][c],
                            &mut m.sh[k][c],
                            &mut v.sh[k][c],
                            rate,
                            b1,
                            b2,
                            c1,
                            c2,
                            eps,
                        );
                    }
                }
            });
    }

    /// Drop the state of removed Gaussians.
    pub fn retain_mask(&mut self, keep: &[bool]) {
        retain_by_mask(&mut self.m, keep);
        retain_by_mask(&mut self.v, keep);
    }

    /// Append zero state for `k` new Gaussians.
    pub fn append_zeros(&mut self, k: usize) {
        self.m.extend(std::iter::repeat_n(Gaussian::default(), k));
        self.v.extend(std::iter::repeat_n(Gaussian::default(), k));
    }

    /// Zero the opacity moments, as the reference does when it resets opacities.
    pub fn reset_opacity_state(&mut self) {
        for m in self.m.iter_mut() {
            m.opacity = 0.0;
        }
        for v in self.v.iter_mut() {
            v.opacity = 0.0;
        }
    }
}