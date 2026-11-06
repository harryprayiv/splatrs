//! Text COLMAP format (`cameras.txt`, `images.txt`, `points3D.txt`).

use super::{CameraModel, ColmapCamera, ColmapImage, ColmapModel, ColmapPoint};
use crate::error::{Error, Result};
use std::path::Path;

fn read_text(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| Error::io(path, e))
}

/// Lines of a COLMAP text file with comments and blank lines removed.
fn content_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
}

fn parse<T: std::str::FromStr>(path: &Path, tok: Option<&str>, what: &str) -> Result<T> {
    let tok = tok.ok_or_else(|| Error::format(path, format!("missing {what}")))?;
    tok.parse::<T>()
        .map_err(|_| Error::format(path, format!("bad {what}: {tok:?}")))
}

pub(super) fn read_cameras(path: &Path) -> Result<Vec<ColmapCamera>> {
    let text = read_text(path)?;
    let mut out = Vec::new();
    for line in content_lines(&text) {
        let mut t = line.split_whitespace();
        let id: u32 = parse(path, t.next(), "camera id")?;
        let model_name = t
            .next()
            .ok_or_else(|| Error::format(path, "missing camera model"))?;
        let model = CameraModel::from_name(model_name)
            .ok_or_else(|| Error::format(path, format!("unknown camera model {model_name}")))?;
        let width: u64 = parse(path, t.next(), "width")?;
        let height: u64 = parse(path, t.next(), "height")?;
        let mut params = Vec::new();
        for tok in t {
            params.push(parse::<f64>(path, Some(tok), "camera parameter")?);
        }
        if params.len() != model.num_params() {
            return Err(Error::format(
                path,
                format!(
                    "camera {id}: model {} expects {} parameters, found {}",
                    model.name(),
                    model.num_params(),
                    params.len()
                ),
            ));
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

/// `images.txt` holds two lines per image: a header
/// `IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME` and a line of 2D observations, which
/// COLMAP writes even when it is empty and which some other writers omit. A line is treated
/// as a header when it has at least ten tokens, the first nine are numeric and the tenth is
/// not (a file name). Every other line is an observation line and is skipped.
pub(super) fn read_images(path: &Path) -> Result<Vec<ColmapImage>> {
    let text = read_text(path)?;
    let mut out = Vec::new();
    for line in content_lines(&text) {
        let toks: Vec<&str> = line.split_whitespace().collect();
        let is_header = toks.len() >= 10
            && toks[0].parse::<u32>().is_ok()
            && toks[1..9].iter().all(|t| t.parse::<f64>().is_ok())
            && toks[9].parse::<f64>().is_err();
        if !is_header {
            continue;
        }
        let id: u32 = parse(path, Some(toks[0]), "image id")?;
        let qvec = [
            parse(path, Some(toks[1]), "qw")?,
            parse(path, Some(toks[2]), "qx")?,
            parse(path, Some(toks[3]), "qy")?,
            parse(path, Some(toks[4]), "qz")?,
        ];
        let tvec = [
            parse(path, Some(toks[5]), "tx")?,
            parse(path, Some(toks[6]), "ty")?,
            parse(path, Some(toks[7]), "tz")?,
        ];
        let camera_id: u32 = parse(path, Some(toks[8]), "camera id")?;
        let name = toks[9].to_string();
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
    let text = read_text(path)?;
    let mut out = Vec::new();
    for line in content_lines(&text) {
        let mut t = line.split_whitespace();
        let id: u64 = parse(path, t.next(), "point id")?;
        let xyz = [
            parse(path, t.next(), "x")?,
            parse(path, t.next(), "y")?,
            parse(path, t.next(), "z")?,
        ];
        let rgb = [
            parse(path, t.next(), "r")?,
            parse(path, t.next(), "g")?,
            parse(path, t.next(), "b")?,
        ];
        let error: f64 = parse(path, t.next(), "error")?;
        out.push(ColmapPoint {
            id,
            xyz,
            rgb,
            error,
        });
    }
    Ok(out)
}

/// Write a sparse model in the text format (`cameras.txt`, `images.txt`, `points3D.txt`),
/// with empty observation lines and empty tracks.
pub fn write_model_txt(dir: &Path, m: &ColmapModel) -> Result<()> {
    use std::fmt::Write as _;
    std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    let mut ids: Vec<u32> = m.cameras.keys().copied().collect();
    ids.sort_unstable();

    let mut cams = String::from("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n");
    for id in &ids {
        let c = &m.cameras[id];
        let params: Vec<String> = c.params.iter().map(|p| format!("{p}")).collect();
        let _ = writeln!(
            cams,
            "{} {} {} {} {}",
            c.id,
            c.model.name(),
            c.width,
            c.height,
            params.join(" ")
        );
    }
    let path = dir.join("cameras.txt");
    std::fs::write(&path, cams).map_err(|e| Error::io(&path, e))?;

    let mut imgs = String::from(
        "# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
    );
    for im in &m.images {
        let _ = writeln!(
            imgs,
            "{} {} {} {} {} {} {} {} {} {}\n",
            im.id,
            im.qvec[0],
            im.qvec[1],
            im.qvec[2],
            im.qvec[3],
            im.tvec[0],
            im.tvec[1],
            im.tvec[2],
            im.camera_id,
            im.name
        );
    }
    let path = dir.join("images.txt");
    std::fs::write(&path, imgs).map_err(|e| Error::io(&path, e))?;

    let mut pts = String::from(
        "# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
    );
    for p in &m.points {
        let _ = writeln!(
            pts,
            "{} {} {} {} {} {} {} {}",
            p.id, p.xyz[0], p.xyz[1], p.xyz[2], p.rgb[0], p.rgb[1], p.rgb[2], p.error
        );
    }
    let path = dir.join("points3D.txt");
    std::fs::write(&path, pts).map_err(|e| Error::io(&path, e))
}