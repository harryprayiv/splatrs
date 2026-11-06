# splatrs

splatrs trains and renders 3D Gaussian splats (Kerbl et al. 2023) from photographs. It is a from-scratch Rust implementation whose every differentiable stage has a hand-derived backward pass checked against finite differences, and it runs on the CPU, so it works on machines without a discrete GPU. It writes the standard 3DGS `.ply` layout, which Blender 5.3 imports and renders natively and which every splat viewer reads.

## What it is and what it is not

It is a reference implementation: correct, verified, readable, and structured so that a GPU backend can be added underneath the same training loop. It is not fast. On a laptop CPU a 30000 iteration run takes between one and several hours depending on image resolution and scene size. For a fast, GPU-accelerated Rust trainer that already exists, see Brush (ArthurBrussee/brush, Apache-2.0), which runs on Intel, AMD and NVIDIA GPUs through wgpu.

The pipeline is the one from the paper: COLMAP poses and sparse points in, Gaussians initialised from the points, each iteration renders one view through a tile-sorted differentiable rasterizer, the loss is 0.8 L1 + 0.2 D-SSIM, Adam updates the parameters, and adaptive density control clones, splits and prunes Gaussians during the first half of training. Two later improvements are available as flags: Mip-Splatting style opacity compensation (`--antialias`) and the AbsGS densification criterion (`--absgrad`).

## Requirements

The flake provides a dev shell with `cargo`, `rustc`, `clippy`, `rustfmt`, `rust-analyzer` and `colmap`. Without Nix, any Rust 1.85 or newer toolchain builds the crate; COLMAP is only needed to process real captures.

## Building and testing

```sh
nix develop
cargo build --release
cargo test
cargo test --features f64
cargo clippy --all-targets -- -D warnings
```

`cargo test` runs the unit tests, the COLMAP and PLY round trips, the synthetic end-to-end training tests, the command-line smoke test, and the finite-difference checks whose gradients single precision can resolve. `cargo test --features f64` builds every scalar as `f64` and runs the full set of gradient checks (spherical harmonics, projection with and without antialiasing, the rasterizer, SSIM, the photometric loss, and the whole parameters-to-loss chain) at a relative tolerance of 1e-5.

`nix build` produces the `splatrs` binary; it needs `Cargo.lock` to exist, so run `cargo generate-lockfile` once in the dev shell first.

## Quick start without a camera

The `synthetic` example renders a random scene with splatrs itself and writes it out as a COLMAP project, so the whole pipeline can be exercised and timed before COLMAP is involved.

```sh
cargo run --release --example synthetic -- --out /tmp/synth --gaussians 3000 --views 24 --width 480 --height 320
./target/release/splatrs info --data /tmp/synth
./target/release/splatrs train --data /tmp/synth --out /tmp/synth-out --iterations 1500 --sh-degree 1 --eval
./target/release/splatrs render --data /tmp/synth --ply /tmp/synth-out/point_cloud/iteration_1500/point_cloud.ply --out /tmp/synth-renders
```

On two cores this run reaches about 35 dB PSNR on the training views and 28.6 dB on the held-out views in 1500 iterations, growing from 1500 to about 13000 Gaussians.

## Real captures

Shoot 50 to 300 sharp, well-exposed photographs of a static scene with large overlap between neighbouring shots, moving the camera around the subject rather than rotating on the spot. Fixed focal length, fixed exposure and no motion blur matter more than resolution. Put the images in a directory `images` under a project directory and run COLMAP from the dev shell. The `--SiftExtraction.use_gpu 0` and `--SiftMatching.use_gpu 0` flags keep COLMAP on the CPU, which avoids OpenGL problems on headless or integrated-graphics machines.

```sh
colmap feature_extractor --database_path proj/database.db --image_path proj/images --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 --SiftExtraction.use_gpu 0
colmap exhaustive_matcher --database_path proj/database.db --SiftMatching.use_gpu 0
colmap mapper --database_path proj/database.db --image_path proj/images --output_path proj/sparse
colmap image_undistorter --image_path proj/images --input_path proj/sparse/0 --output_path proj/undistorted --output_type COLMAP
```

For captures with more than a few hundred images, `sequential_matcher` is much faster than `exhaustive_matcher` when the images were taken in order. The undistortion step is mandatory: splatrs trains only on PINHOLE and SIMPLE_PINHOLE cameras, and `image_undistorter` writes exactly those, together with undistorted copies of the images, into `proj/undistorted/images` and `proj/undistorted/sparse`. `splatrs info` reports whether a reconstruction is ready and prints the undistortion command when it is not.

```sh
./target/release/splatrs info --data proj/undistorted
./target/release/splatrs train --data proj/undistorted --out proj/splat --downscale 2 --eval
./target/release/splatrs render --data proj/undistorted --ply proj/splat/point_cloud/iteration_30000/point_cloud.ply --out proj/renders --downscale 2
```

`--data` expects the COLMAP layout: `sparse/0` (or `sparse`) with `cameras`, `images` and `points3D` as `.bin` or `.txt`, and `images` next to it. `--sparse` and `--images` override either location. The trainer writes `point_cloud/iteration_7000/point_cloud.ply` and `point_cloud/iteration_<last>/point_cloud.ply` under `--out`, the directory layout the reference implementation uses for its splats (the reference's `cameras.json` and `cfg_args` are not written, so SIBR will not open the directory as a model).

## Using the result in Blender

Blender 5.3 (final release expected 10 November 2026) imports `.ply` Gaussian splats natively and renders them in Workbench, EEVEE and Cycles; the file written by splatrs is the standard layout that importer reads. Earlier Blender versions need an add-on such as KIRI Engine's 3DGS Render or BlendSplat, which read the same file. The scene is stored in COLMAP's world frame, which has no fixed up direction, so expect to rotate the imported object. SuperSplat and other web viewers open the file directly as well.

## Command reference

`splatrs info --data DIR` prints the cameras, the number of registered images and 3D points, the scene extent, and whether the cameras are distortion-free.

`splatrs train --data DIR --out DIR` trains. Options: `--iterations N` (30000), `--downscale K` (integer image downscale, 1), `--eval` (hold out every 8th image and report its PSNR at the end), `--white-bg`, `--antialias`, `--absgrad`, `--sh-degree D` (3), `--grad-threshold T` (0.0002, or 0.0008 when `--absgrad` is set), `--no-densify`, `--max-screen-size PX` (screen-radius pruning, off by default), `--seed S`, `--save-at A,B,C` (extra checkpoint iterations; 7000 and the last iteration are always saved), `--log-interval N` (100), and the global `--threads N`.

`splatrs render --data DIR --ply FILE --out DIR` renders every camera of the reconstruction to a PNG named after its image, keeping any subdirectory of the image name. Options: `--downscale K`, `--white-bg`, `--antialias` (must match training), `--view NAME` (one image only).

## Performance

Cost is proportional to the number of Gaussian-pixel pairs that pass the alpha threshold, at roughly 15 ns per pair per core for the forward pass and 25 ns for the backward pass. A 1 megapixel view with 20 contributing Gaussians per pixel therefore costs about 0.8 core-seconds per iteration, so a 30000 iteration run on 8 cores is on the order of an hour at `--downscale 2` for a 2 megapixel capture and several hours at full resolution. Train with `--downscale 2` or `4` first, and keep the full-resolution run for the final result. Memory is about 240 bytes per Gaussian for the parameters and the same again for the gradients and each of the two Adam moments, plus 44 bytes per tile entry during an iteration.

## Verification

The crate builds with `f32` scalars by default and with `f64` scalars under the `f64` feature; the same code runs in both. Every backward pass is checked against central finite differences under `f64`: the spherical harmonics basis and its derivatives, the projection (means, scales, rotations, opacity and all 48 SH coefficients, with and without antialiasing, with and without Jacobian clamping), the rasterizer (means, conics, opacities and colours of overlapping Gaussians), SSIM, the photometric loss, and the whole chain from parameters to loss. The tiled rasterizer is also compared against a plain per-pixel compositor on a random scene, the COLMAP reader was validated on files written by COLMAP's own `read_write_model.py`, and the PLY layout follows the reference `save_ply` attribute order and `f_rest` channel-major ordering.

## Roadmap

A wgpu compute backend for the projection and rasterization kernels, so the same training loop runs on Intel integrated GPUs. SIMD inner loops for the CPU backend. 3DGS-MCMC relocation as an alternative to clone-and-split densification. Mip-Splatting's 3D filter, exposure compensation, depth regularisation and image masks from the 2024 reference update. Nerfstudio `transforms.json` input. `.spz` export. Checkpointing with optimiser state. An interactive viewer.

## License

AGPL-3.0-or-later. The license text is not included in this tree; fetch it with `curl -o LICENSE https://www.gnu.org/licenses/agpl-3.0.txt`.
