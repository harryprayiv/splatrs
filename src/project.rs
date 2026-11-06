//! Projection of 3D Gaussians to screen space (EWA splatting, Zwicker et al. 2001, as used
//! by 3DGS) with a hand-derived backward pass.
//!
//! Forward, per Gaussian: camera-space mean `t = W p + c`, 3D covariance
//! `Sigma = R S S^T R^T`, affine approximation of the projection `J` at `t`, 2D covariance
//! `Sigma' = J W Sigma W^T J^T` dilated by `eps2d` on the diagonal, its inverse (the conic),
//! a conservative pixel radius, the SH colour for the view direction, and the opacity
//! (optionally compensated for the dilation, following Mip-Splatting).

use crate::camera::Camera;
use crate::math::{normalize_backward, quat_to_mat_backward, sigmoid, Quat, M3, V3};
use crate::model::{Gaussian, Model};
use crate::sh::{self, MAX_COEFFS};
use crate::Real;
use rayon::prelude::*;

/// Tile edge in pixels.
pub const TILE: u32 = 16;

/// Options of the projection stage.
#[derive(Clone, Copy, Debug)]
pub struct ProjOpts {
    /// Camera-space depth at or below which a Gaussian is culled.
    pub near: Real,
    /// Dilation added to the diagonal of the 2D covariance (0.3 in the reference).
    pub eps2d: Real,
    /// Scale opacity by `sqrt(det(Sigma') / det(Sigma' + eps2d I))` so the dilation does not
    /// brighten small Gaussians (Mip-Splatting, Yu et al. 2024).
    pub antialias: bool,
    /// Active spherical harmonics degree.
    pub sh_degree: usize,
}

impl Default for ProjOpts {
    fn default() -> ProjOpts {
        ProjOpts {
            near: 0.2,
            eps2d: 0.3,
            antialias: false,
            sh_degree: 3,
        }
    }
}

/// Screen-space description of one Gaussian for one camera. `radius == 0` means culled.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Proj {
    /// Projected mean in continuous pixel coordinates.
    pub mean2d: [Real; 2],
    /// Inverse 2D covariance `(A, B, C)`; the exponent is `-0.5 (A dx^2 + 2 B dx dy + C dy^2)`.
    pub conic: [Real; 3],
    /// Camera-space depth, used for sorting.
    pub depth: Real,
    /// Conservative radius in pixels; zero when the Gaussian is not drawn.
    pub radius: u32,
    /// Effective opacity (after the sigmoid and any antialiasing compensation).
    pub opacity: Real,
    /// RGB colour for this view direction.
    pub color: [Real; 3],
    /// First tile touched (x, y), inclusive.
    pub tile_min: [u32; 2],
    /// Last tile touched (x, y), exclusive.
    pub tile_max: [u32; 2],
}

impl Proj {
    /// Whether the Gaussian is drawn at all.
    pub fn visible(&self) -> bool {
        self.radius > 0
    }

    /// Number of tiles whose lists this Gaussian is added to.
    pub fn num_tiles(&self) -> u32 {
        if self.radius == 0 {
            0
        } else {
            (self.tile_max[0] - self.tile_min[0]) * (self.tile_max[1] - self.tile_min[1])
        }
    }
}

/// Gradient of the loss with respect to the screen-space quantities of one Gaussian.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProjGrad {
    /// `dL/d mean2d` in pixel units.
    pub mean2d: [Real; 2],
    /// `dL/d (A, B, C)`.
    pub conic: [Real; 3],
    /// `dL/d opacity` (effective opacity).
    pub opacity: Real,
    /// `dL/d colour`.
    pub color: [Real; 3],
}

/// Everything the forward pass computes, kept so the backward pass can reuse it.
struct Core {
    t: V3,
    txc: Real,
    tyc: Real,
    clamped_x: bool,
    clamped_y: bool,
    s: V3,
    q: Quat,
    rq: M3,
    m: M3,
    sigma: M3,
    tmat: [[Real; 3]; 2],
    a0: Real,
    b0: Real,
    c0: Real,
    det0: Real,
    a: Real,
    b: Real,
    c: Real,
    det: Real,
    conic: [Real; 3],
    comp: Real,
    mean2d: [Real; 2],
    radius: Real,
    o_raw: Real,
    dir: V3,
    dist: Real,
    basis: [Real; MAX_COEFFS],
    color_raw: [Real; 3],
}

fn clamp_flag(v: Real, lo: Real, hi: Real) -> (Real, bool) {
    if v < lo {
        (lo, true)
    } else if v > hi {
        (hi, true)
    } else {
        (v, false)
    }
}

fn core(g: &Gaussian, cam: &Camera, opts: &ProjOpts) -> Option<Core> {
    let p = g.mean_v();
    let t = cam.world_to_cam(p);
    if t.z.is_nan() || t.z <= opts.near {
        return None;
    }
    let s = g.scales();
    let q = g.rotation();
    let rq = q.to_mat();
    let m = rq.scale_cols(s);
    let sigma = m.mul_m(&m.transpose());

    // Affine approximation of the projection, evaluated at a point clamped to a slightly
    // enlarged frustum so that Gaussians far outside the view do not blow up the Jacobian.
    let tan_x = 0.5 * cam.width as Real / cam.fx;
    let tan_y = 0.5 * cam.height as Real / cam.fy;
    let lim_x_pos = (cam.width as Real - cam.cx) / cam.fx + 0.3 * tan_x;
    let lim_x_neg = cam.cx / cam.fx + 0.3 * tan_x;
    let lim_y_pos = (cam.height as Real - cam.cy) / cam.fy + 0.3 * tan_y;
    let lim_y_neg = cam.cy / cam.fy + 0.3 * tan_y;
    let rz = 1.0 / t.z;
    let rx = t.x * rz;
    let ry = t.y * rz;
    let (rxc, clamped_x) = clamp_flag(rx, -lim_x_neg, lim_x_pos);
    let (ryc, clamped_y) = clamp_flag(ry, -lim_y_neg, lim_y_pos);
    let txc = rxc * t.z;
    let tyc = ryc * t.z;
    let rz2 = rz * rz;
    let j = [
        [cam.fx * rz, 0.0, -cam.fx * txc * rz2],
        [0.0, cam.fy * rz, -cam.fy * tyc * rz2],
    ];

    // T = J W, Sigma' = T Sigma T^T.
    let mut tmat = [[0.0; 3]; 2];
    for r in 0..2 {
        for c in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += j[r][k] * cam.rot.m[k][c];
            }
            tmat[r][c] = acc;
        }
    }
    let mut ts = [[0.0; 3]; 2];
    for r in 0..2 {
        for c in 0..3 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += tmat[r][k] * sigma.m[k][c];
            }
            ts[r][c] = acc;
        }
    }
    let mut a0 = 0.0;
    let mut b0 = 0.0;
    let mut c0 = 0.0;
    for c in 0..3 {
        a0 += ts[0][c] * tmat[0][c];
        b0 += ts[0][c] * tmat[1][c];
        c0 += ts[1][c] * tmat[1][c];
    }
    let det0 = a0 * c0 - b0 * b0;
    let a = a0 + opts.eps2d;
    let b = b0;
    let c = c0 + opts.eps2d;
    let det = a * c - b * b;
    if det.is_nan() || det <= 0.0 {
        return None;
    }
    let inv_det = 1.0 / det;
    let conic = [c * inv_det, -b * inv_det, a * inv_det];
    let comp = if opts.antialias {
        if det0 > 0.0 {
            (det0 * inv_det).sqrt()
        } else {
            0.0
        }
    } else {
        1.0
    };

    // Conservative radius: three standard deviations along the major axis.
    let mid = 0.5 * (a + c);
    let lambda1 = mid + (mid * mid - det).max(0.1).sqrt();
    let radius = (3.0 * lambda1.sqrt()).ceil();

    let mean2d = [cam.fx * rx + cam.cx, cam.fy * ry + cam.cy];
    let o_raw = sigmoid(g.opacity);

    let d = p - cam.center();
    let dist = d.norm();
    let dir = d * (1.0 / dist);
    let basis = sh::basis(opts.sh_degree, dir);
    let n = sh::num_coeffs(opts.sh_degree);
    let mut color_raw = [0.5; 3];
    for k in 0..n {
        for ch in 0..3 {
            color_raw[ch] += basis[k] * g.sh[k][ch];
        }
    }

    Some(Core {
        t,
        txc,
        tyc,
        clamped_x,
        clamped_y,
        s,
        q,
        rq,
        m,
        sigma,
        tmat,
        a0,
        b0,
        c0,
        det0,
        a,
        b,
        c,
        det,
        conic,
        comp,
        mean2d,
        radius,
        o_raw,
        dir,
        dist,
        basis,
        color_raw,
    })
}

/// Number of tiles along x and y for an image size.
pub fn tile_grid(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(TILE), height.div_ceil(TILE))
}

/// Project one Gaussian. Returns a default (culled) `Proj` when it is not drawn.
pub fn project_one(g: &Gaussian, cam: &Camera, opts: &ProjOpts) -> Proj {
    let Some(k) = core(g, cam, opts) else {
        return Proj::default();
    };
    let (tiles_x, tiles_y) = tile_grid(cam.width, cam.height);
    let r = k.radius;
    if r.is_nan() || r <= 0.0 {
        return Proj::default();
    }
    let tile = TILE as Real;
    let clamp_tile = |v: Real, n: u32| -> u32 {
        if v <= 0.0 {
            0
        } else if v >= n as Real {
            n
        } else {
            v as u32
        }
    };
    let tx0 = clamp_tile(((k.mean2d[0] - r) / tile).floor(), tiles_x);
    let tx1 = clamp_tile(((k.mean2d[0] + r) / tile).ceil(), tiles_x);
    let ty0 = clamp_tile(((k.mean2d[1] - r) / tile).floor(), tiles_y);
    let ty1 = clamp_tile(((k.mean2d[1] + r) / tile).ceil(), tiles_y);
    if tx0 >= tx1 || ty0 >= ty1 {
        return Proj::default();
    }
    let mut color = k.color_raw;
    for ch in 0..3 {
        color[ch] = color[ch].max(0.0);
    }
    Proj {
        mean2d: k.mean2d,
        conic: k.conic,
        depth: k.t.z,
        radius: r as u32,
        opacity: k.o_raw * k.comp,
        color,
        tile_min: [tx0, ty0],
        tile_max: [tx1, ty1],
    }
}

/// Project every Gaussian of a model for one camera.
pub fn project_all(model: &Model, cam: &Camera, opts: &ProjOpts) -> Vec<Proj> {
    let opts = ProjOpts {
        sh_degree: model.sh_degree,
        ..*opts
    };
    model
        .gaussians
        .par_iter()
        .map(|g| project_one(g, cam, &opts))
        .collect()
}

/// Backward pass of [`project_one`]: given `dL/d(screen quantities)` return `dL/d(parameters)`.
/// A culled Gaussian gets a zero gradient. The caller passes the same `opts` as the forward
/// pass (including the active SH degree).
pub fn project_backward_one(
    g: &Gaussian,
    cam: &Camera,
    opts: &ProjOpts,
    dp: &ProjGrad,
) -> Gaussian {
    let mut out = Gaussian::default();
    let Some(k) = core(g, cam, opts) else {
        return out;
    };

    // Colour: dL/dsh and dL/ddir, through the clamp at zero.
    let n = sh::num_coeffs(opts.sh_degree);
    let bgrad = sh::basis_grad(opts.sh_degree, k.dir);
    let mut ddir = V3::ZERO;
    for ch in 0..3 {
        if k.color_raw[ch] > 0.0 {
            let dc = dp.color[ch];
            for kk in 0..n {
                out.sh[kk][ch] = dc * k.basis[kk];
                ddir += bgrad[kk] * (dc * g.sh[kk][ch]);
            }
        }
    }
    // dir = d / |d| with d = p - camera centre: d(dir)/d(p) = (I - dir dir^T) / |d|.
    let dmean_color = (ddir - k.dir * k.dir.dot(ddir)) * (1.0 / k.dist);

    // Opacity: effective = sigmoid(logit) * comp.
    let do_raw = dp.opacity * k.comp;
    let dcomp = dp.opacity * k.o_raw;
    out.opacity = do_raw * k.o_raw * (1.0 - k.o_raw);

    // Conic (A, B, C) = (c, -b, a) / det with det = a c - b^2.
    let (a, b, c, det) = (k.a, k.b, k.c, k.det);
    let (da_, db_, dc_) = (dp.conic[0], dp.conic[1], dp.conic[2]);
    let inv_det2 = 1.0 / (det * det);
    let mut da = (-c * c * da_ + b * c * db_ - b * b * dc_) * inv_det2;
    let mut db = (2.0 * b * c * da_ - (a * c + b * b) * db_ + 2.0 * a * b * dc_) * inv_det2;
    let mut dc = (-b * b * da_ + a * b * db_ - a * a * dc_) * inv_det2;

    // Antialiasing compensation comp = sqrt(det0 / det).
    if opts.antialias && k.det0 > 0.0 {
        let ddet0 = dcomp * 0.5 * k.comp / k.det0;
        let ddet = -dcomp * 0.5 * k.comp / det;
        da += ddet0 * k.c0 + ddet * c;
        dc += ddet0 * k.a0 + ddet * a;
        db += ddet0 * (-2.0 * k.b0) + ddet * (-2.0 * b);
    }

    // Sigma' = T Sigma T^T with the symmetric gradient G = [[da, db/2], [db/2, dc]]:
    // dSigma = T^T G T, dT = 2 G T Sigma.
    let gmat = [[da, 0.5 * db], [0.5 * db, dc]];
    let tm = k.tmat;
    let mut gt = [[0.0; 3]; 2];
    for r in 0..2 {
        for col in 0..3 {
            gt[r][col] = gmat[r][0] * tm[0][col] + gmat[r][1] * tm[1][col];
        }
    }
    let mut dsigma = M3::ZERO;
    for i in 0..3 {
        for jj in 0..3 {
            dsigma.m[i][jj] = tm[0][i] * gt[0][jj] + tm[1][i] * gt[1][jj];
        }
    }
    let mut dt = [[0.0; 3]; 2];
    for r in 0..2 {
        for col in 0..3 {
            let mut acc = 0.0;
            for kk in 0..3 {
                acc += gt[r][kk] * k.sigma.m[kk][col];
            }
            dt[r][col] = 2.0 * acc;
        }
    }

    // T = J W: dJ = dT W^T.
    let mut dj = [[0.0; 3]; 2];
    for r in 0..2 {
        for col in 0..3 {
            let mut acc = 0.0;
            for kk in 0..3 {
                acc += dt[r][kk] * cam.rot.m[col][kk];
            }
            dj[r][col] = acc;
        }
    }

    // J = [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]] at the clamped point.
    let (fx, fy) = (cam.fx, cam.fy);
    let rz = 1.0 / k.t.z;
    let rz2 = rz * rz;
    let rz3 = rz2 * rz;
    let dtxc = dj[0][2] * (-fx * rz2);
    let dtyc = dj[1][2] * (-fy * rz2);
    let mut dtx = 0.0;
    let mut dty = 0.0;
    let mut dtz = dj[0][0] * (-fx * rz2)
        + dj[1][1] * (-fy * rz2)
        + dj[0][2] * (2.0 * fx * k.txc * rz3)
        + dj[1][2] * (2.0 * fy * k.tyc * rz3);
    // When clamped, the clamped coordinate is z times a constant, so it moves with z only.
    if k.clamped_x {
        dtz += dtxc * k.txc * rz;
    } else {
        dtx += dtxc;
    }
    if k.clamped_y {
        dtz += dtyc * k.tyc * rz;
    } else {
        dty += dtyc;
    }

    // mean2d = (fx x / z + cx, fy y / z + cy), unclamped.
    dtx += dp.mean2d[0] * fx * rz;
    dtz += dp.mean2d[0] * (-fx * k.t.x * rz2);
    dty += dp.mean2d[1] * fy * rz;
    dtz += dp.mean2d[1] * (-fy * k.t.y * rz2);

    // t = W p + c.
    let dmean = cam.rot.transpose().mul_v(V3::new(dtx, dty, dtz)) + dmean_color;
    out.mean = dmean.to_array();

    // Sigma = M M^T: dM = 2 dSigma M.  M = R diag(s).
    let dm = dsigma.mul_m(&k.m).scaled(2.0);
    let mut ds = [0.0; 3];
    let mut dr = M3::ZERO;
    for r in 0..3 {
        for col in 0..3 {
            ds[col] += dm.m[r][col] * k.rq.m[r][col];
            dr.m[r][col] = dm.m[r][col] * k.s.get(col);
        }
    }
    for i in 0..3 {
        out.log_scale[i] = ds[i] * k.s.get(i);
    }
    let dq_unit = quat_to_mat_backward(k.q, &dr);
    out.quat = normalize_backward(g.quat, dq_unit);
    out
}

/// Backward pass for every Gaussian of a model.
pub fn project_backward_all(
    model: &Model,
    cam: &Camera,
    opts: &ProjOpts,
    grads: &[ProjGrad],
) -> Vec<Gaussian> {
    let opts = ProjOpts {
        sh_degree: model.sh_degree,
        ..*opts
    };
    model
        .gaussians
        .par_iter()
        .zip(grads.par_iter())
        .map(|(g, dp)| project_backward_one(g, cam, &opts, dp))
        .collect()
}