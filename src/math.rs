//! Small vector, matrix and quaternion types over `Real`, plus the activation functions
//! used by the parameterisation.

use crate::Real;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};

/// A 3-vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3 {
    /// x component.
    pub x: Real,
    /// y component.
    pub y: Real,
    /// z component.
    pub z: Real,
}

impl V3 {
    /// The zero vector.
    pub const ZERO: V3 = V3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Construct from components.
    pub const fn new(x: Real, y: Real, z: Real) -> V3 {
        V3 { x, y, z }
    }

    /// Construct from an array.
    pub const fn from_array(a: [Real; 3]) -> V3 {
        V3 {
            x: a[0],
            y: a[1],
            z: a[2],
        }
    }

    /// Convert to an array.
    pub const fn to_array(self) -> [Real; 3] {
        [self.x, self.y, self.z]
    }

    /// Dot product.
    pub fn dot(self, o: V3) -> Real {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// Cross product.
    pub fn cross(self, o: V3) -> V3 {
        V3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// Squared Euclidean norm.
    pub fn norm2(self) -> Real {
        self.dot(self)
    }

    /// Euclidean norm.
    pub fn norm(self) -> Real {
        self.norm2().sqrt()
    }

    /// Unit vector in the same direction. The zero vector is returned unchanged.
    pub fn normalized(self) -> V3 {
        let n = self.norm();
        if n > 0.0 {
            self * (1.0 / n)
        } else {
            self
        }
    }

    /// Elementwise product.
    pub fn hadamard(self, o: V3) -> V3 {
        V3::new(self.x * o.x, self.y * o.y, self.z * o.z)
    }

    /// Elementwise exp.
    pub fn exp(self) -> V3 {
        V3::new(self.x.exp(), self.y.exp(), self.z.exp())
    }

    /// Largest component.
    pub fn max_elem(self) -> Real {
        self.x.max(self.y).max(self.z)
    }

    /// Component by index (0, 1, 2).
    pub fn get(self, i: usize) -> Real {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

impl Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        V3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl AddAssign for V3 {
    fn add_assign(&mut self, o: V3) {
        self.x += o.x;
        self.y += o.y;
        self.z += o.z;
    }
}

impl Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        V3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul<Real> for V3 {
    type Output = V3;
    fn mul(self, s: Real) -> V3 {
        V3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Neg for V3 {
    type Output = V3;
    fn neg(self) -> V3 {
        V3::new(-self.x, -self.y, -self.z)
    }
}

/// A 3x3 matrix stored row-major: `m[row][col]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct M3 {
    /// Rows.
    pub m: [[Real; 3]; 3],
}

impl M3 {
    /// The identity matrix.
    pub const IDENTITY: M3 = M3 {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    /// The zero matrix.
    pub const ZERO: M3 = M3 { m: [[0.0; 3]; 3] };

    /// Build from three row vectors.
    pub fn from_rows(r0: V3, r1: V3, r2: V3) -> M3 {
        M3 {
            m: [r0.to_array(), r1.to_array(), r2.to_array()],
        }
    }

    /// Row `i` as a vector.
    pub fn row(&self, i: usize) -> V3 {
        V3::from_array(self.m[i])
    }

    /// Column `j` as a vector.
    pub fn col(&self, j: usize) -> V3 {
        V3::new(self.m[0][j], self.m[1][j], self.m[2][j])
    }

    /// Matrix times vector.
    pub fn mul_v(&self, v: V3) -> V3 {
        V3::new(self.row(0).dot(v), self.row(1).dot(v), self.row(2).dot(v))
    }

    /// Matrix times matrix.
    pub fn mul_m(&self, o: &M3) -> M3 {
        let mut r = M3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += self.m[i][k] * o.m[k][j];
                }
                r.m[i][j] = s;
            }
        }
        r
    }

    /// Transpose.
    pub fn transpose(&self) -> M3 {
        let mut r = M3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[j][i];
            }
        }
        r
    }

    /// `self * diag(s)`: scales column `j` by `s[j]`.
    pub fn scale_cols(&self, s: V3) -> M3 {
        let mut r = *self;
        for i in 0..3 {
            r.m[i][0] *= s.x;
            r.m[i][1] *= s.y;
            r.m[i][2] *= s.z;
        }
        r
    }

    /// Multiply every entry by a scalar.
    pub fn scaled(&self, s: Real) -> M3 {
        let mut r = *self;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] *= s;
            }
        }
        r
    }

    /// Frobenius inner product: sum of elementwise products.
    pub fn frob_dot(&self, o: &M3) -> Real {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                s += self.m[i][j] * o.m[i][j];
            }
        }
        s
    }
}

impl Add for M3 {
    type Output = M3;
    fn add(self, o: M3) -> M3 {
        let mut r = self;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] += o.m[i][j];
            }
        }
        r
    }
}

/// A quaternion stored as (w, x, y, z), w being the scalar part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    /// Scalar part.
    pub w: Real,
    /// x of the vector part.
    pub x: Real,
    /// y of the vector part.
    pub y: Real,
    /// z of the vector part.
    pub z: Real,
}

impl Quat {
    /// The identity rotation.
    pub const IDENTITY: Quat = Quat {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Construct from an array in (w, x, y, z) order.
    pub const fn from_array(a: [Real; 4]) -> Quat {
        Quat {
            w: a[0],
            x: a[1],
            y: a[2],
            z: a[3],
        }
    }

    /// Convert to an array in (w, x, y, z) order.
    pub const fn to_array(self) -> [Real; 4] {
        [self.w, self.x, self.y, self.z]
    }

    /// Euclidean norm of the four components.
    pub fn norm(self) -> Real {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Unit quaternion in the same direction. A zero quaternion becomes the identity.
    pub fn normalized(self) -> Quat {
        let n = self.norm();
        if n > 0.0 {
            Quat {
                w: self.w / n,
                x: self.x / n,
                y: self.y / n,
                z: self.z / n,
            }
        } else {
            Quat::IDENTITY
        }
    }

    /// Rotation matrix of a unit quaternion (the caller normalises first).
    pub fn to_mat(self) -> M3 {
        let Quat { w, x, y, z } = self;
        M3 {
            m: [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
        }
    }
}

/// Given `dL/dR` for `R = rotation(q)` with `q` a unit quaternion, return `dL/dq`
/// treating the four components as independent (the normalisation is chained separately
/// by [`normalize_backward`]).
pub fn quat_to_mat_backward(q: Quat, dr: &M3) -> [Real; 4] {
    let Quat { w, x, y, z } = q;
    let d_dw = M3 {
        m: [
            [0.0, -2.0 * z, 2.0 * y],
            [2.0 * z, 0.0, -2.0 * x],
            [-2.0 * y, 2.0 * x, 0.0],
        ],
    };
    let d_dx = M3 {
        m: [
            [0.0, 2.0 * y, 2.0 * z],
            [2.0 * y, -4.0 * x, -2.0 * w],
            [2.0 * z, 2.0 * w, -4.0 * x],
        ],
    };
    let d_dy = M3 {
        m: [
            [-4.0 * y, 2.0 * x, 2.0 * w],
            [2.0 * x, 0.0, 2.0 * z],
            [-2.0 * w, 2.0 * z, -4.0 * y],
        ],
    };
    let d_dz = M3 {
        m: [
            [-4.0 * z, -2.0 * w, 2.0 * x],
            [2.0 * w, -4.0 * z, 2.0 * y],
            [2.0 * x, 2.0 * y, 0.0],
        ],
    };
    [
        dr.frob_dot(&d_dw),
        dr.frob_dot(&d_dx),
        dr.frob_dot(&d_dy),
        dr.frob_dot(&d_dz),
    ]
}

/// Chain `dL/d(unit q)` back through `unit q = raw / |raw|` to get `dL/d(raw)`.
pub fn normalize_backward(raw: [Real; 4], d_unit: [Real; 4]) -> [Real; 4] {
    let n = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3]).sqrt();
    if n <= 0.0 {
        return [0.0; 4];
    }
    let u = [raw[0] / n, raw[1] / n, raw[2] / n, raw[3] / n];
    let dot = u[0] * d_unit[0] + u[1] * d_unit[1] + u[2] * d_unit[2] + u[3] * d_unit[3];
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = (d_unit[i] - u[i] * dot) / n;
    }
    out
}

/// Logistic sigmoid.
pub fn sigmoid(x: Real) -> Real {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse of the logistic sigmoid; `x` must lie strictly inside (0, 1).
pub fn inverse_sigmoid(x: Real) -> Real {
    (x / (1.0 - x)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_is_orthonormal() {
        let q = Quat {
            w: 0.3,
            x: -0.5,
            y: 0.7,
            z: 0.2,
        }
        .normalized();
        let r = q.to_mat();
        let rtr = r.transpose().mul_m(&r);
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((rtr.m[i][j] - expect).abs() < 1e-5, "R^T R not identity");
            }
        }
        // Right-handed: det must be +1, check via the cross product of the first two columns.
        let c0 = r.col(0);
        let c1 = r.col(1);
        let c2 = r.col(2);
        assert!((c0.cross(c1) - c2).norm() < 1e-5);
    }

    #[test]
    fn sigmoid_roundtrip() {
        for &v in &[0.01, 0.1, 0.5, 0.9, 0.99] {
            let x: Real = v;
            assert!((sigmoid(inverse_sigmoid(x)) - x).abs() < 1e-5);
        }
    }
}