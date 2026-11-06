//! Tile-based, depth-sorted alpha compositing of projected Gaussians: forward and backward.
//!
//! Every visible Gaussian is added to the list of each 16x16 tile it overlaps (one "entry"
//! per tile), the entries are sorted by `(tile, depth)`, and each pixel composites its
//! tile's list front to back until the transmittance drops below `1e-4`. The backward pass
//! walks the same lists back to front, exactly as the reference CUDA rasterizer does, and
//! accumulates per-entry gradients that are then summed per Gaussian.

use crate::project::{tile_grid, Proj, ProjGrad, TILE};
use crate::{atomic_load, atomic_store, AtomicReal, Real};
use rayon::prelude::*;

/// Image size and background colour.
#[derive(Clone, Copy, Debug)]
pub struct RasterSettings {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Background composited behind the Gaussians.
    pub background: [Real; 3],
}

/// Sorted tile lists for one view.
#[derive(Clone, Debug)]
pub struct Binning {
    /// Tiles along x.
    pub tiles_x: u32,
    /// Tiles along y.
    pub tiles_y: u32,
    /// Gaussian index of each entry.
    pub entry_gaussian: Vec<u32>,
    /// First entry index of each Gaussian; `len() == num_gaussians + 1`.
    pub entry_offset: Vec<u32>,
    /// `(key, entry)` pairs sorted by key, key being `tile << 32 | depth bits`.
    pub sorted: Vec<(u64, u32)>,
    /// `[start, end)` into `sorted` for each tile.
    pub tile_range: Vec<(u32, u32)>,
}

/// Result of the forward pass, kept for the backward pass.
#[derive(Clone, Debug)]
pub struct RasterOutput {
    /// RGB image, row-major, `width * height * 3`.
    pub image: Vec<Real>,
    /// Transmittance left after the last composited Gaussian, per pixel.
    pub final_t: Vec<Real>,
    /// Number of list entries up to and including the last contributing one, per pixel.
    pub n_contrib: Vec<u32>,
}

/// Lowest opacity contribution that is composited.
const ALPHA_MIN: Real = 1.0 / 255.0;
/// Opacities are clamped to this value.
const ALPHA_MAX: Real = 0.99;
/// Transmittance at which a pixel stops compositing.
const T_MIN: Real = 1e-4;

/// Build the sorted tile lists for a set of projections.
pub fn bin(projs: &[Proj], width: u32, height: u32) -> Binning {
    let (tiles_x, tiles_y) = tile_grid(width, height);
    let mut entry_offset = Vec::with_capacity(projs.len() + 1);
    let mut total: usize = 0;
    for p in projs {
        assert!(
            !p.visible() || (p.tile_max[0] <= tiles_x && p.tile_max[1] <= tiles_y),
            "projection tile rectangle exceeds the {width}x{height} tile grid; the \
             projections were made for a different image size"
        );
        entry_offset.push(total as u32);
        total += p.num_tiles() as usize;
    }
    entry_offset.push(total as u32);
    assert!(total < u32::MAX as usize, "too many tile entries");

    // One (key, gaussian) pair per entry, in Gaussian order so entries of a Gaussian are
    // contiguous; flat_map_iter + collect preserves that order.
    let entries: Vec<(u64, u32)> = projs
        .par_iter()
        .enumerate()
        .flat_map_iter(|(gi, p)| {
            let depth_bits = (p.depth as f32).to_bits() as u64;
            let visible = p.visible();
            let (x0, y0, x1, y1) = if visible {
                (p.tile_min[0], p.tile_min[1], p.tile_max[0], p.tile_max[1])
            } else {
                (0, 0, 0, 0)
            };
            (y0..y1).flat_map(move |ty| {
                (x0..x1).map(move |tx| {
                    let tile = (ty * tiles_x + tx) as u64;
                    ((tile << 32) | depth_bits, gi as u32)
                })
            })
        })
        .collect();
    debug_assert_eq!(entries.len(), total);

    let entry_gaussian: Vec<u32> = entries.iter().map(|e| e.1).collect();
    let mut sorted: Vec<(u64, u32)> = entries
        .iter()
        .enumerate()
        .map(|(e, k)| (k.0, e as u32))
        .collect();
    drop(entries);
    sorted.par_sort_unstable();

    let num_tiles = (tiles_x * tiles_y) as usize;
    let mut tile_range = vec![(0u32, 0u32); num_tiles];
    let mut s = 0usize;
    while s < sorted.len() {
        let tile = (sorted[s].0 >> 32) as usize;
        let mut e = s + 1;
        while e < sorted.len() && (sorted[e].0 >> 32) as usize == tile {
            e += 1;
        }
        tile_range[tile] = (s as u32, e as u32);
        s = e;
    }
    Binning {
        tiles_x,
        tiles_y,
        entry_gaussian,
        entry_offset,
        sorted,
        tile_range,
    }
}

/// Per-entry screen data gathered for a tile's inner loops.
#[derive(Clone, Copy)]
struct TileGauss {
    mean: [Real; 2],
    conic: [Real; 3],
    opacity: Real,
    color: [Real; 3],
    /// Largest exponent at which `opacity * exp(-sigma)` still reaches `ALPHA_MIN`.
    sigma_max: Real,
    /// Tile-local pixel rectangle `[x0, x1) x [y0, y1)` that can contain pixels with
    /// `sigma <= sigma_max`; outside it the Gaussian is skipped without evaluation.
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
}

impl TileGauss {
    /// Build the entry for a tile whose top-left pixel is `(tx0, ty0)` and size `w x h`.
    fn new(p: &Proj, tx0: u32, ty0: u32, w: u32, h: u32) -> TileGauss {
        let sigma_max = (p.opacity / ALPHA_MIN).ln();
        let [a, b, c] = p.conic;
        let det = a * c - b * b;
        let (x0, x1, y0, y1) = if sigma_max > 0.0 && det > 0.0 {
            // Bounding box of the ellipse 0.5 d^T M d <= sigma_max, with a margin against
            // rounding; pixel i has its centre at i + 0.5.
            let half_x = (2.0 * sigma_max * c / det).sqrt() + 1e-3;
            let half_y = (2.0 * sigma_max * a / det).sqrt() + 1e-3;
            let clip = |lo: Real, hi: Real, offset: u32, n: u32| -> (u32, u32) {
                let lo = (lo - 0.5).ceil() - offset as Real;
                let hi = (hi - 0.5).floor() - offset as Real + 1.0;
                let lo = lo.clamp(0.0, n as Real) as u32;
                let hi = hi.clamp(0.0, n as Real) as u32;
                (lo, hi)
            };
            let (x0, x1) = clip(p.mean2d[0] - half_x, p.mean2d[0] + half_x, tx0, w);
            let (y0, y1) = clip(p.mean2d[1] - half_y, p.mean2d[1] + half_y, ty0, h);
            (x0, x1, y0, y1)
        } else {
            (0, 0, 0, 0)
        };
        TileGauss {
            mean: p.mean2d,
            conic: p.conic,
            opacity: p.opacity,
            color: p.color,
            sigma_max,
            x0,
            x1,
            y0,
            y1,
        }
    }

    fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }
}

/// Geometry of one tile.
#[derive(Clone, Copy)]
struct TileGeom {
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
}

fn tile_geom(binning: &Binning, settings: &RasterSettings, tile: usize) -> TileGeom {
    let tx = tile as u32 % binning.tiles_x;
    let ty = tile as u32 / binning.tiles_x;
    let x0 = tx * TILE;
    let y0 = ty * TILE;
    TileGeom {
        x0,
        y0,
        w: (settings.width - x0).min(TILE),
        h: (settings.height - y0).min(TILE),
    }
}

fn gather(projs: &[Proj], binning: &Binning, tile: usize, geom: &TileGeom) -> Vec<TileGauss> {
    let (s, e) = binning.tile_range[tile];
    (s..e)
        .map(|i| {
            let entry = binning.sorted[i as usize].1 as usize;
            let p = &projs[binning.entry_gaussian[entry] as usize];
            TileGauss::new(p, geom.x0, geom.y0, geom.w, geom.h)
        })
        .collect()
}

struct TileOut {
    geom: TileGeom,
    color: Vec<Real>,
    final_t: Vec<Real>,
    n_contrib: Vec<u32>,
}

/// Evaluate one Gaussian at one pixel: returns `(G, alpha)` or `None` when skipped.
#[inline]
fn evaluate(g: &TileGauss, px: Real, py: Real) -> Option<(Real, Real)> {
    let dx = px - g.mean[0];
    let dy = py - g.mean[1];
    let sigma = 0.5 * (g.conic[0] * dx * dx + g.conic[2] * dy * dy) + g.conic[1] * dx * dy;
    if sigma < 0.0 || sigma > g.sigma_max {
        return None;
    }
    let gauss = (-sigma).exp();
    let og = g.opacity * gauss;
    if og < ALPHA_MIN {
        return None;
    }
    Some((gauss, og.min(ALPHA_MAX)))
}

/// Composite one tile. Entries are visited in depth order and each only touches the pixels
/// of its own bounding rectangle; per-pixel state carries the front-to-back compositing,
/// so the result equals a per-pixel walk over the whole list.
fn forward_tile(
    projs: &[Proj],
    binning: &Binning,
    settings: &RasterSettings,
    tile: usize,
) -> TileOut {
    let geom = tile_geom(binning, settings, tile);
    let list = gather(projs, binning, tile, &geom);
    let n = (geom.w * geom.h) as usize;
    let mut color = vec![0.0; n * 3];
    let mut trans = vec![1.0; n];
    let mut n_contrib = vec![0u32; n];
    let mut done = vec![false; n];
    for (k, g) in list.iter().enumerate() {
        if g.is_empty() {
            continue;
        }
        for j in g.y0..g.y1 {
            let py = (geom.y0 + j) as Real + 0.5;
            for i in g.x0..g.x1 {
                let pix = (j * geom.w + i) as usize;
                if done[pix] {
                    continue;
                }
                let px = (geom.x0 + i) as Real + 0.5;
                let Some((_, alpha)) = evaluate(g, px, py) else {
                    continue;
                };
                let t = trans[pix];
                let test_t = t * (1.0 - alpha);
                if test_t < T_MIN {
                    done[pix] = true;
                    continue;
                }
                for ch in 0..3 {
                    color[pix * 3 + ch] += alpha * t * g.color[ch];
                }
                trans[pix] = test_t;
                n_contrib[pix] = k as u32 + 1;
            }
        }
    }
    for pix in 0..n {
        for ch in 0..3 {
            color[pix * 3 + ch] += trans[pix] * settings.background[ch];
        }
    }
    TileOut {
        geom,
        color,
        final_t: trans,
        n_contrib,
    }
}

/// Render the projected Gaussians.
pub fn rasterize_forward(
    projs: &[Proj],
    binning: &Binning,
    settings: &RasterSettings,
) -> RasterOutput {
    let num_tiles = (binning.tiles_x * binning.tiles_y) as usize;
    let tiles: Vec<TileOut> = (0..num_tiles)
        .into_par_iter()
        .map(|tile| forward_tile(projs, binning, settings, tile))
        .collect();
    let npix = (settings.width * settings.height) as usize;
    let mut image = vec![0.0; npix * 3];
    let mut final_t = vec![1.0; npix];
    let mut n_contrib = vec![0u32; npix];
    let width = settings.width as usize;
    for t in &tiles {
        let g = t.geom;
        for j in 0..g.h as usize {
            for i in 0..g.w as usize {
                let src = j * g.w as usize + i;
                let dst = (g.y0 as usize + j) * width + g.x0 as usize + i;
                image[dst * 3..dst * 3 + 3].copy_from_slice(&t.color[src * 3..src * 3 + 3]);
                final_t[dst] = t.final_t[src];
                n_contrib[dst] = t.n_contrib[src];
            }
        }
    }
    RasterOutput {
        image,
        final_t,
        n_contrib,
    }
}

/// Number of gradient slots stored per entry: mean2d (2), conic (3), opacity, colour (3),
/// plus the absolute-value mean2d accumulators (2) used by AbsGS-style densification.
const SLOTS: usize = 11;

/// Per-pixel state of the back-to-front walk.
#[derive(Clone, Copy)]
struct PixelBack {
    t: Real,
    accum: [Real; 3],
    last_alpha: Real,
    last_color: [Real; 3],
}

/// Backward pass of one tile: entries in reverse depth order, each over its own pixel
/// rectangle, with per-pixel state so the walk equals a per-pixel reverse walk.
fn backward_tile(
    projs: &[Proj],
    binning: &Binning,
    settings: &RasterSettings,
    out: &RasterOutput,
    dimage: &[Real],
    tile: usize,
    slots: &[AtomicReal],
) {
    let geom = tile_geom(binning, settings, tile);
    let (start, _) = binning.tile_range[tile];
    let list = gather(projs, binning, tile, &geom);
    if list.is_empty() {
        return;
    }
    let n = (geom.w * geom.h) as usize;
    let width = settings.width as usize;
    let bg = settings.background;
    let mut state = Vec::with_capacity(n);
    let mut dl = Vec::with_capacity(n);
    let mut bg_dot = Vec::with_capacity(n);
    let mut last = Vec::with_capacity(n);
    let mut t_final = Vec::with_capacity(n);
    for j in 0..geom.h as usize {
        for i in 0..geom.w as usize {
            let pix = (geom.y0 as usize + j) * width + geom.x0 as usize + i;
            let d = [dimage[pix * 3], dimage[pix * 3 + 1], dimage[pix * 3 + 2]];
            dl.push(d);
            bg_dot.push(bg[0] * d[0] + bg[1] * d[1] + bg[2] * d[2]);
            last.push(out.n_contrib[pix] as usize);
            t_final.push(out.final_t[pix]);
            state.push(PixelBack {
                t: out.final_t[pix],
                accum: [0.0; 3],
                last_alpha: 0.0,
                last_color: [0.0; 3],
            });
        }
    }
    let mut acc = vec![[0.0; SLOTS]; list.len()];
    for k in (0..list.len()).rev() {
        let g = &list[k];
        if g.is_empty() {
            continue;
        }
        let a = &mut acc[k];
        for j in g.y0..g.y1 {
            let py = (geom.y0 + j) as Real + 0.5;
            for i in g.x0..g.x1 {
                let pix = (j * geom.w + i) as usize;
                if k >= last[pix] {
                    continue;
                }
                let px = (geom.x0 + i) as Real + 0.5;
                let Some((gauss, alpha)) = evaluate(g, px, py) else {
                    continue;
                };
                let s = &mut state[pix];
                let d = dl[pix];
                // Transmittance in front of this Gaussian.
                s.t /= 1.0 - alpha;
                let t = s.t;
                let mut dalpha = 0.0;
                for ch in 0..3 {
                    s.accum[ch] =
                        s.last_alpha * s.last_color[ch] + (1.0 - s.last_alpha) * s.accum[ch];
                    dalpha += (g.color[ch] - s.accum[ch]) * d[ch];
                    a[6 + ch] += alpha * t * d[ch];
                }
                dalpha *= t;
                s.last_color = g.color;
                s.last_alpha = alpha;
                dalpha += -t_final[pix] / (1.0 - alpha) * bg_dot[pix];

                // alpha = opacity * G unless clamped at ALPHA_MAX, where it is constant.
                if g.opacity * gauss < ALPHA_MAX {
                    a[5] += gauss * dalpha;
                    let dgauss = g.opacity * dalpha;
                    // G = exp(-sigma), sigma = 0.5 (A dx^2 + 2 B dx dy + C dy^2), d = pixel - mean.
                    let dsigma = -dgauss * gauss;
                    let dx = px - g.mean[0];
                    let dy = py - g.mean[1];
                    let ddx = dsigma * (g.conic[0] * dx + g.conic[1] * dy);
                    let ddy = dsigma * (g.conic[1] * dx + g.conic[2] * dy);
                    a[0] -= ddx;
                    a[1] -= ddy;
                    a[2] += dsigma * 0.5 * dx * dx;
                    a[3] += dsigma * dx * dy;
                    a[4] += dsigma * 0.5 * dy * dy;
                    a[9] += ddx.abs();
                    a[10] += ddy.abs();
                }
            }
        }
    }
    for (k, a) in acc.iter().enumerate() {
        let entry = binning.sorted[start as usize + k].1 as usize;
        for s in 0..SLOTS {
            atomic_store(&slots[entry * SLOTS + s], a[s]);
        }
    }
}

/// Backward pass: given `dL/d image`, return `dL/d(screen quantities)` per Gaussian and
/// the per-Gaussian sum of absolute per-pixel `dL/d mean2d` contributions (AbsGS).
pub fn rasterize_backward(
    projs: &[Proj],
    binning: &Binning,
    settings: &RasterSettings,
    out: &RasterOutput,
    dimage: &[Real],
) -> (Vec<ProjGrad>, Vec<[Real; 2]>) {
    assert_eq!(
        dimage.len(),
        (settings.width * settings.height * 3) as usize
    );
    let num_entries = binning.sorted.len();
    let slots: Vec<AtomicReal> = (0..num_entries * SLOTS)
        .map(|_| AtomicReal::new(0))
        .collect();
    let num_tiles = (binning.tiles_x * binning.tiles_y) as usize;
    (0..num_tiles)
        .into_par_iter()
        .for_each(|tile| backward_tile(projs, binning, settings, out, dimage, tile, &slots));

    let n = projs.len();
    let reduced: Vec<(ProjGrad, [Real; 2])> = (0..n)
        .into_par_iter()
        .map(|g| {
            let mut sum = [0.0; SLOTS];
            let (s, e) = (
                binning.entry_offset[g] as usize,
                binning.entry_offset[g + 1] as usize,
            );
            for entry in s..e {
                for k in 0..SLOTS {
                    sum[k] += atomic_load(&slots[entry * SLOTS + k]);
                }
            }
            (
                ProjGrad {
                    mean2d: [sum[0], sum[1]],
                    conic: [sum[2], sum[3], sum[4]],
                    opacity: sum[5],
                    color: [sum[6], sum[7], sum[8]],
                },
                [sum[9], sum[10]],
            )
        })
        .collect();
    reduced.into_iter().unzip()
}