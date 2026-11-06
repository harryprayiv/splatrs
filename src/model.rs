//! The Gaussian parameter record and the collection of them that makes up a scene.
//!
//! Parameters are stored in their unconstrained form, exactly as the reference
//! implementation stores them: log scales, logit opacity and an unnormalised quaternion.
//! The same record type is reused for gradients and for the Adam moments, since all three
//! have the shape of the parameters.

use crate::math::{inverse_sigmoid, sigmoid, Quat, V3};
use crate::sh::{self, MAX_COEFFS};
use crate::Real;
use kiddo::{KdTree, SquaredEuclidean};
use rayon::prelude::*;
use std::collections::HashMap;

/// One Gaussian's parameters (or a parameter-shaped record such as a gradient).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gaussian {
    /// World-space mean.
    pub mean: [Real; 3],
    /// Natural log of the three axis scales.
    pub log_scale: [Real; 3],
    /// Rotation as an unnormalised quaternion `(w, x, y, z)`.
    pub quat: [Real; 4],
    /// Opacity before the sigmoid.
    pub opacity: Real,
    /// Spherical harmonic coefficients: `sh[k][channel]`, `k = 0` is the DC term.
    pub sh: [[Real; 3]; MAX_COEFFS],
}

impl Gaussian {
    /// Mean as a vector.
    pub fn mean_v(&self) -> V3 {
        V3::from_array(self.mean)
    }

    /// Axis scales after the exp activation.
    pub fn scales(&self) -> V3 {
        V3::from_array(self.log_scale).exp()
    }

    /// Unit rotation quaternion.
    pub fn rotation(&self) -> Quat {
        Quat::from_array(self.quat).normalized()
    }

    /// Opacity after the sigmoid activation.
    pub fn alpha(&self) -> Real {
        sigmoid(self.opacity)
    }
}

/// A scene: a list of Gaussians plus the currently active SH degree.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// The Gaussians.
    pub gaussians: Vec<Gaussian>,
    /// Degree used when evaluating colour; raised during training.
    pub sh_degree: usize,
    /// Highest degree the model carries coefficients for.
    pub max_sh_degree: usize,
}

impl Model {
    /// Number of Gaussians.
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    /// Whether the model has no Gaussians.
    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }

    /// Initialise from a sparse point cloud the way the reference implementation does:
    /// scale from the mean squared distance to the three nearest neighbours, identity
    /// rotation, opacity 0.1, colour in the DC term.
    pub fn from_points(points: &[(V3, [u8; 3])], max_sh_degree: usize) -> Model {
        let dist2 = mean_nn_dist2(points);
        let opacity = inverse_sigmoid(0.1);
        let gaussians = points
            .par_iter()
            .zip(dist2.par_iter())
            .map(|(&(p, rgb), &d2)| {
                let d2: Real = (d2 as Real).max(1e-7);
                let log_s = d2.sqrt().ln();
                let mut sh = [[0.0; 3]; MAX_COEFFS];
                for c in 0..3 {
                    sh[0][c] = sh::rgb_to_sh(rgb[c] as Real / 255.0);
                }
                Gaussian {
                    mean: p.to_array(),
                    log_scale: [log_s; 3],
                    quat: [1.0, 0.0, 0.0, 0.0],
                    opacity,
                    sh,
                }
            })
            .collect();
        Model {
            gaussians,
            sh_degree: 0,
            max_sh_degree,
        }
    }

    /// Keep only the Gaussians whose mask entry is true.
    pub fn retain_mask(&mut self, keep: &[bool]) {
        retain_by_mask(&mut self.gaussians, keep);
    }
}

/// Remove the entries of `v` whose `keep` entry is false, preserving order.
pub fn retain_by_mask<T>(v: &mut Vec<T>, keep: &[bool]) {
    debug_assert_eq!(v.len(), keep.len());
    let mut i = 0;
    v.retain(|_| {
        let k = keep[i];
        i += 1;
        k
    });
}

/// Mean squared distance from each point to its three nearest other points, in f64.
/// Points with fewer than three neighbours use whatever neighbours exist; a lone point
/// gets 0 (the caller clamps). Coincident points are collapsed to one tree entry with a
/// multiplicity, which keeps the tree well-formed for any input and counts duplicates as
/// zero-distance neighbours.
pub fn mean_nn_dist2(points: &[(V3, [u8; 3])]) -> Vec<f64> {
    if points.len() < 2 {
        return vec![0.0; points.len()];
    }
    // Unique positions, keyed by their exact bit pattern.
    let mut index: HashMap<[u64; 3], u32> = HashMap::new();
    let mut unique: Vec<[f64; 3]> = Vec::new();
    let mut count: Vec<u32> = Vec::new();
    let mut of_point: Vec<u32> = Vec::with_capacity(points.len());
    for (p, _) in points {
        let q = [p.x as f64, p.y as f64, p.z as f64];
        let key = [q[0].to_bits(), q[1].to_bits(), q[2].to_bits()];
        let u = *index.entry(key).or_insert_with(|| {
            unique.push(q);
            count.push(0);
            (unique.len() - 1) as u32
        });
        count[u as usize] += 1;
        of_point.push(u);
    }
    let mut tree: KdTree<f64, 3> = KdTree::with_capacity(unique.len());
    for (u, q) in unique.iter().enumerate() {
        tree.add(q, u as u64);
    }
    let k = unique.len().min(4);
    of_point
        .par_iter()
        .map(|&u| {
            let mut hits = tree.nearest_n::<SquaredEuclidean>(&unique[u as usize], k);
            hits.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // Walk outwards, expanding multiplicities, until three other points are counted.
            let mut sum = 0.0;
            let mut n = 0u32;
            for h in &hits {
                let item = h.item as usize;
                let m = if item == u as usize {
                    count[item] - 1
                } else {
                    count[item]
                };
                let take = m.min(3 - n);
                sum += h.distance * take as f64;
                n += take;
                if n == 3 {
                    break;
                }
            }
            if n > 0 {
                sum / n as f64
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_distances_match_brute_force() {
        let mut pts = Vec::new();
        let mut s: u64 = 12345;
        for _ in 0..200 {
            let mut r = || {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64) as Real
            };
            pts.push((V3::new(r(), r(), r()), [0u8; 3]));
        }
        let got = mean_nn_dist2(&pts);
        for i in 0..pts.len() {
            let mut d: Vec<f64> = (0..pts.len())
                .filter(|&j| j != i)
                .map(|j| {
                    let (a, b) = (pts[i].0, pts[j].0);
                    let dx = a.x as f64 - b.x as f64;
                    let dy = a.y as f64 - b.y as f64;
                    let dz = a.z as f64 - b.z as f64;
                    dx * dx + dy * dy + dz * dz
                })
                .collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let want = (d[0] + d[1] + d[2]) / 3.0;
            assert!(
                (got[i] - want).abs() < 1e-5 * want.max(1e-6),
                "point {i}: {} vs {}",
                got[i],
                want
            );
        }
    }

    #[test]
    fn nn_distances_handle_many_coincident_points() {
        // 40 points at the origin plus a few others: the kd-tree bucket limit is 32.
        let mut pts: Vec<(V3, [u8; 3])> = (0..40).map(|_| (V3::ZERO, [0u8; 3])).collect();
        pts.push((V3::new(1.0, 0.0, 0.0), [0u8; 3]));
        pts.push((V3::new(0.0, 2.0, 0.0), [0u8; 3]));
        let got = mean_nn_dist2(&pts);
        for &d in &got[..40] {
            assert_eq!(d, 0.0);
        }
        // (1,0,0): three nearest others are all at the origin, squared distance 1.
        assert!((got[40] - 1.0).abs() < 1e-12);
        // (0,2,0): nearest others are at the origin, squared distance 4.
        assert!((got[41] - 4.0).abs() < 1e-12);
        // Two points only: each sees the other once.
        let two = vec![(V3::ZERO, [0u8; 3]), (V3::new(0.0, 0.0, 3.0), [0u8; 3])];
        let got = mean_nn_dist2(&two);
        assert!((got[0] - 9.0).abs() < 1e-12 && (got[1] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn retain_mask_keeps_order() {
        let mut v = vec![1, 2, 3, 4, 5];
        retain_by_mask(&mut v, &[true, false, true, false, true]);
        assert_eq!(v, vec![1, 3, 5]);
    }
}