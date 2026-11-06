//! The de facto standard 3DGS `.ply` layout (binary little-endian, float32 properties),
//! as written by the reference implementation and read by Blender 5.3, SuperSplat and the
//! rest of the ecosystem.
//!
//! Properties per vertex, in order: `x y z nx ny nz f_dc_0..2 f_rest_0..(3*(K-1)-1)
//! opacity scale_0..2 rot_0..3`, where `K = (degree+1)^2`. `f_rest` is stored
//! channel-major: `f_rest[c * (K-1) + (k-1)]` holds coefficient `k` of channel `c`.
//! Opacity is the logit, scales are logs, the rotation is the unnormalised quaternion
//! `(w, x, y, z)`. Normals are written as zeros and ignored when reading.

use crate::error::{Error, Result};
use crate::model::{Gaussian, Model};
use crate::sh::{num_coeffs, MAX_DEGREE};
use crate::Real;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write a model. Coefficients above `model.max_sh_degree` are not written.
pub fn write(path: &Path, model: &Model) -> Result<()> {
    let degree = model.max_sh_degree.min(MAX_DEGREE);
    let k = num_coeffs(degree);
    let n_rest = 3 * (k - 1);
    let file = std::fs::File::create(path).map_err(|e| Error::io(path, e))?;
    let mut w = BufWriter::new(file);
    let io = |e: std::io::Error| Error::io(path, e);

    let mut header = String::new();
    header.push_str("ply\nformat binary_little_endian 1.0\n");
    header.push_str(&format!("element vertex {}\n", model.len()));
    for name in [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    ] {
        header.push_str(&format!("property float {name}\n"));
    }
    for i in 0..n_rest {
        header.push_str(&format!("property float f_rest_{i}\n"));
    }
    header.push_str("property float opacity\n");
    for i in 0..3 {
        header.push_str(&format!("property float scale_{i}\n"));
    }
    for i in 0..4 {
        header.push_str(&format!("property float rot_{i}\n"));
    }
    header.push_str("end_header\n");
    w.write_all(header.as_bytes()).map_err(io)?;

    let mut row: Vec<f32> = Vec::with_capacity(17 + n_rest);
    for g in &model.gaussians {
        row.clear();
        row.extend(g.mean.iter().map(|&v| v as f32));
        row.extend([0.0f32; 3]);
        row.extend(g.sh[0].iter().map(|&v| v as f32));
        for c in 0..3 {
            for kk in 1..k {
                row.push(g.sh[kk][c] as f32);
            }
        }
        row.push(g.opacity as f32);
        row.extend(g.log_scale.iter().map(|&v| v as f32));
        row.extend(g.quat.iter().map(|&v| v as f32));
        let mut bytes = Vec::with_capacity(row.len() * 4);
        for v in &row {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        w.write_all(&bytes).map_err(io)?;
    }
    w.flush().map_err(io)
}

/// Read a model written in the layout above. Property order is taken from the header, so
/// files from other tools load as long as they use float32 properties with these names.
pub fn read(path: &Path) -> Result<Model> {
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    let fmt = |m: &str| Error::format(path, m);

    // Header.
    let end = find(&bytes, b"end_header\n").ok_or_else(|| fmt("missing end_header"))?;
    let header = std::str::from_utf8(&bytes[..end]).map_err(|_| fmt("header is not UTF-8"))?;
    let body = &bytes[end + "end_header\n".len()..];
    let mut count: Option<usize> = None;
    let mut props: Vec<String> = Vec::new();
    let mut in_vertex = false;
    let mut binary_le = false;
    for line in header.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        match toks.as_slice() {
            ["format", "binary_little_endian", _] => binary_le = true,
            ["format", ..] => return Err(fmt("only binary_little_endian is supported")),
            ["element", "vertex", n] => {
                in_vertex = true;
                count = Some(n.parse().map_err(|_| fmt("bad vertex count"))?);
            }
            ["element", ..] => in_vertex = false,
            ["property", ty, name] if in_vertex => {
                if *ty != "float" && *ty != "float32" {
                    return Err(fmt(&format!(
                        "property {name} has type {ty}; only float is supported"
                    )));
                }
                props.push((*name).to_string());
            }
            _ => {}
        }
    }
    if !binary_le {
        return Err(fmt("missing format line"));
    }
    let count = count.ok_or_else(|| fmt("missing vertex element"))?;
    let stride = props.len() * 4;
    let needed = count
        .checked_mul(stride)
        .ok_or_else(|| fmt("vertex count overflows"))?;
    if body.len() < needed {
        return Err(fmt("file is shorter than the header announces"));
    }
    let index = |name: &str| -> Result<usize> {
        props
            .iter()
            .position(|p| p == name)
            .ok_or_else(|| fmt(&format!("missing property {name}")))
    };
    let ix = [index("x")?, index("y")?, index("z")?];
    let idc = [index("f_dc_0")?, index("f_dc_1")?, index("f_dc_2")?];
    let iop = index("opacity")?;
    let isc = [index("scale_0")?, index("scale_1")?, index("scale_2")?];
    let irot = [
        index("rot_0")?,
        index("rot_1")?,
        index("rot_2")?,
        index("rot_3")?,
    ];
    let n_rest = props.iter().filter(|p| p.starts_with("f_rest_")).count();
    let degree = match n_rest {
        0 => 0,
        9 => 1,
        24 => 2,
        45 => 3,
        other => {
            return Err(fmt(&format!(
                "unsupported number of f_rest properties: {other}"
            )))
        }
    };
    let k = num_coeffs(degree);
    let mut irest = Vec::with_capacity(n_rest);
    for i in 0..n_rest {
        irest.push(index(&format!("f_rest_{i}"))?);
    }

    let mut gaussians = Vec::with_capacity(count);
    let mut row = vec![0.0f32; props.len()];
    for v in 0..count {
        let rec = &body[v * stride..(v + 1) * stride];
        for (i, slot) in row.iter_mut().enumerate() {
            *slot =
                f32::from_le_bytes([rec[i * 4], rec[i * 4 + 1], rec[i * 4 + 2], rec[i * 4 + 3]]);
        }
        let mut g = Gaussian::default();
        for i in 0..3 {
            g.mean[i] = row[ix[i]] as Real;
            g.log_scale[i] = row[isc[i]] as Real;
            g.sh[0][i] = row[idc[i]] as Real;
        }
        for i in 0..4 {
            g.quat[i] = row[irot[i]] as Real;
        }
        g.opacity = row[iop] as Real;
        for c in 0..3 {
            for kk in 1..k {
                g.sh[kk][c] = row[irest[c * (k - 1) + (kk - 1)]] as Real;
            }
        }
        gaussians.push(g);
    }
    Ok(Model {
        gaussians,
        sh_degree: degree,
        max_sh_degree: degree,
    })
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sh::MAX_COEFFS;

    #[test]
    fn roundtrip() {
        let mut gaussians = Vec::new();
        for i in 0..5 {
            let f = i as Real;
            let mut sh = [[0.0; 3]; MAX_COEFFS];
            for k in 0..MAX_COEFFS {
                for c in 0..3 {
                    sh[k][c] = f + k as Real * 0.1 + c as Real * 0.01;
                }
            }
            gaussians.push(Gaussian {
                mean: [f, f + 0.5, f - 0.5],
                log_scale: [-f, -f - 1.0, -f - 2.0],
                quat: [1.0, 0.1 * f, 0.2 * f, 0.3 * f],
                opacity: f * 0.3,
                sh,
            });
        }
        let model = Model {
            gaussians,
            sh_degree: 3,
            max_sh_degree: 3,
        };
        let dir = std::env::temp_dir().join(format!("splatrs-ply-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.ply");
        write(&path, &model).unwrap();
        let back = read(&path).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(back.len(), 5);
        for (a, b) in model.gaussians.iter().zip(&back.gaussians) {
            for i in 0..3 {
                assert!((a.mean[i] - b.mean[i]).abs() < 1e-6);
                assert!((a.log_scale[i] - b.log_scale[i]).abs() < 1e-6);
            }
            for i in 0..4 {
                assert!((a.quat[i] - b.quat[i]).abs() < 1e-6);
            }
            assert!((a.opacity - b.opacity).abs() < 1e-6);
            for k in 0..MAX_COEFFS {
                for c in 0..3 {
                    assert!((a.sh[k][c] - b.sh[k][c]).abs() < 1e-6, "sh[{k}][{c}]");
                }
            }
        }
    }
}