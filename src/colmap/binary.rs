//! Binary COLMAP format, following `colmap/src/colmap/scene/reconstruction_io.cc`. All
//! integers and doubles are little-endian.

use super::{CameraModel, ColmapCamera, ColmapImage, ColmapModel, ColmapPoint};
use crate::error::{Error, Result};
use std::path::Path;

/// Little-endian cursor over a byte buffer.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    path: &'a Path,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(Error::format(self.path, "unexpected end of file"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn i32(&mut self) -> Result<i32> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn f64(&mut self) -> Result<f64> {
        let b = self.take(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn cstring(&mut self) -> Result<String> {
        let start = self.pos;
        while self.pos < self.buf.len() && self.buf[self.pos] != 0 {
            self.pos += 1;
        }
        if self.pos >= self.buf.len() {
            return Err(Error::format(self.path, "unterminated string"));
        }
        let s = String::from_utf8_lossy(&self.buf[start..self.pos]).into_owned();
        self.pos += 1;
        Ok(s)
    }
    fn count(&mut self, what: &str) -> Result<usize> {
        let n = self.u64()?;
        // Each record is at least 8 bytes, so a count beyond the remaining bytes is corrupt.
        if n > (self.buf.len() - self.pos) as u64 {
            return Err(Error::format(
                self.path,
                format!("implausible {what} count {n}"),
            ));
        }
        Ok(n as usize)
    }
}

fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| Error::io(path, e))
}

pub(super) fn read_cameras(path: &Path) -> Result<Vec<ColmapCamera>> {
    let bytes = read_bytes(path)?;
    let mut r = Reader {
        buf: &bytes,
        pos: 0,
        path,
    };
    let n = r.count("camera")?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let id = r.u32()?;
        let model_id = r.i32()?;
        let model = CameraModel::from_id(model_id)
            .ok_or_else(|| Error::format(path, format!("unknown camera model id {model_id}")))?;
        let width = r.u64()?;
        let height = r.u64()?;
        let mut params = Vec::with_capacity(model.num_params());
        for _ in 0..model.num_params() {
            params.push(r.f64()?);
        }
        out.push(ColmapCamera {
            id,
            model,
            width,
            height,
            params,
        });
    }
    Ok(out)
}

pub(super) fn read_images(path: &Path) -> Result<Vec<ColmapImage>> {
    let bytes = read_bytes(path)?;
    let mut r = Reader {
        buf: &bytes,
        pos: 0,
        path,
    };
    let n = r.count("image")?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let id = r.u32()?;
        let qvec = [r.f64()?, r.f64()?, r.f64()?, r.f64()?];
        let tvec = [r.f64()?, r.f64()?, r.f64()?];
        let camera_id = r.u32()?;
        let name = r.cstring()?;
        let n2d = r.count("2D point")?;
        // Each 2D observation is x, y (f64) and a point3D id (u64); not kept.
        r.take(n2d * 24)?;
        out.push(ColmapImage {
            id,
            qvec,
            tvec,
            camera_id,
            name,
        });
    }
    Ok(out)
}

pub(super) fn read_points(path: &Path) -> Result<Vec<ColmapPoint>> {
    let bytes = read_bytes(path)?;
    let mut r = Reader {
        buf: &bytes,
        pos: 0,
        path,
    };
    let n = r.count("3D point")?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let id = r.u64()?;
        let xyz = [r.f64()?, r.f64()?, r.f64()?];
        let rgb = [r.u8()?, r.u8()?, r.u8()?];
        let error = r.f64()?;
        let track_len = r.count("track element")?;
        // Each track element is an image id (u32) and a 2D point index (u32); not kept.
        r.take(track_len * 8)?;
        out.push(ColmapPoint {
            id,
            xyz,
            rgb,
            error,
        });
    }
    Ok(out)
}

/// Write a sparse model in the binary format (`cameras.bin`, `images.bin`, `points3D.bin`).
/// Images are written without 2D observations and points without tracks.
pub fn write_model_bin(dir: &Path, m: &ColmapModel) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    let mut ids: Vec<u32> = m.cameras.keys().copied().collect();
    ids.sort_unstable();

    let mut cams = Vec::new();
    cams.extend((m.cameras.len() as u64).to_le_bytes());
    for id in &ids {
        let c = &m.cameras[id];
        cams.extend(c.id.to_le_bytes());
        cams.extend(c.model.id().to_le_bytes());
        cams.extend(c.width.to_le_bytes());
        cams.extend(c.height.to_le_bytes());
        if c.params.len() != c.model.num_params() {
            return Err(Error::Invalid(format!(
                "camera {}: model {} expects {} parameters, found {}",
                c.id,
                c.model.name(),
                c.model.num_params(),
                c.params.len()
            )));
        }
        for p in &c.params {
            cams.extend(p.to_le_bytes());
        }
    }
    let path = dir.join("cameras.bin");
    std::fs::write(&path, cams).map_err(|e| Error::io(&path, e))?;

    let mut imgs = Vec::new();
    imgs.extend((m.images.len() as u64).to_le_bytes());
    for im in &m.images {
        imgs.extend(im.id.to_le_bytes());
        for q in im.qvec {
            imgs.extend(q.to_le_bytes());
        }
        for t in im.tvec {
            imgs.extend(t.to_le_bytes());
        }
        imgs.extend(im.camera_id.to_le_bytes());
        imgs.extend(im.name.as_bytes());
        imgs.push(0);
        imgs.extend(0u64.to_le_bytes());
    }
    let path = dir.join("images.bin");
    std::fs::write(&path, imgs).map_err(|e| Error::io(&path, e))?;

    let mut pts = Vec::new();
    pts.extend((m.points.len() as u64).to_le_bytes());
    for p in &m.points {
        pts.extend(p.id.to_le_bytes());
        for v in p.xyz {
            pts.extend(v.to_le_bytes());
        }
        pts.extend(p.rgb);
        pts.extend(p.error.to_le_bytes());
        pts.extend(0u64.to_le_bytes());
    }
    let path = dir.join("points3D.bin");
    std::fs::write(&path, pts).map_err(|e| Error::io(&path, e))
}