//! Error type shared by the library.

use std::path::PathBuf;

/// Everything that can go wrong while loading data, training or writing results.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A file could not be read or written.
    #[error("io error at {path}: {source}")]
    Io {
        /// The path involved.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },
    /// A file did not have the expected structure.
    #[error("format error in {path}: {message}")]
    Format {
        /// The path involved.
        path: PathBuf,
        /// What was wrong.
        message: String,
    },
    /// An image could not be decoded or encoded.
    #[error("image error at {path}: {source}")]
    Image {
        /// The path involved.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: image::ImageError,
    },
    /// COLMAP produced a camera model with lens distortion. Training needs undistorted
    /// PINHOLE or SIMPLE_PINHOLE cameras.
    #[error(
        "camera {id} uses model {model}, which has lens distortion; run \
         `colmap image_undistorter` to produce PINHOLE cameras and train on its output"
    )]
    UnsupportedCameraModel {
        /// COLMAP camera id.
        id: u32,
        /// COLMAP model name.
        model: String,
    },
    /// An image on disk does not match the size recorded by COLMAP.
    #[error("image {path} is {actual_w}x{actual_h} but COLMAP recorded {expected_w}x{expected_h}")]
    ImageSizeMismatch {
        /// The image path.
        path: PathBuf,
        /// COLMAP width.
        expected_w: u32,
        /// COLMAP height.
        expected_h: u32,
        /// Width on disk.
        actual_w: u32,
        /// Height on disk.
        actual_h: u32,
    },
    /// Any other invalid input.
    #[error("{0}")]
    Invalid(String),
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Build an io error carrying the path.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Error {
        Error::Io {
            path: path.into(),
            source,
        }
    }

    /// Build a format error carrying the path.
    pub fn format(path: impl Into<PathBuf>, message: impl Into<String>) -> Error {
        Error::Format {
            path: path.into(),
            message: message.into(),
        }
    }
}