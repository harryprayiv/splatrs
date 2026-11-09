# Building and running splatrs on macOS

This guide takes a Mac from nothing to a trained Gaussian splat. It covers Apple silicon (M1 and later) and Intel Macs on a currently supported macOS release. splatrs is pure Rust with no system libraries, so the only things a Mac needs are a Rust toolchain and, for processing your own photographs, COLMAP. There are two ways to get them: Nix, which reproduces exactly the environment the project is developed in, and Homebrew plus rustup, which is what most Mac developers already have. Pick one; do not mix them in the same shell.

All commands below are typed in Terminal (zsh, the macOS default). Lines that start with `cd` assume you cloned the repository to `~/git/splatrs`; adjust the path if you put it elsewhere.

## 1. Check the machine

Find out which processor you have, because it decides which binaries you get:

```sh
uname -m
sysctl -n machdep.cpu.brand_string
```

`arm64` means Apple silicon and `x86_64` means Intel. On Apple silicon, make sure Terminal is not running under Rosetta: if `uname -m` prints `x86_64` on an M-series Mac, open Terminal's Get Info panel in Finder, untick "Open using Rosetta" and start a new window. A toolchain installed under Rosetta builds Intel binaries, which run about half as fast through translation.

Install Apple's command line developer tools, which provide the system linker that Rust uses on macOS. If they are already installed, the command says so and does nothing.

```sh
xcode-select --install
```

A full Xcode installation is not needed.

## 2a. Route one: Nix (recommended, reproducible)

The project's `flake.nix` defines a development shell with `cargo`, `rustc`, `clippy`, `rustfmt`, `rust-analyzer` and `colmap`, and builds for `aarch64-darwin` and `x86_64-darwin` as well as Linux. Using it gives you the same compiler and the same COLMAP version as every other contributor.

Install Nix with the Determinate Systems installer, which enables flakes and survives macOS upgrades better than the original installer:

```sh
curl -fsSL https://install.determinate.systems/nix | sh -s -- install
```

Open a new Terminal window so the shell picks up Nix, then check it:

```sh
nix --version
```

If you installed Nix with the official installer from nixos.org instead, enable flakes once:

```sh
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

Clone the repository and enter the development shell:

```sh
mkdir -p ~/git
cd ~/git
git clone https://github.com/harryprayiv/splatrs
cd splatrs
nix develop
```

The first `nix develop` downloads the toolchain and COLMAP from the Nix binary cache. If the cache has no prebuilt COLMAP for your exact platform, Nix compiles it from source, which can take a long time on a laptop; this happens once. Inside the shell, `which cargo colmap` should print paths under `/nix/store`.

Continue with section 3, staying inside `nix develop`.

A known macOS quirk: a major macOS update can overwrite `/etc/zshrc` and remove the lines that put Nix on your PATH. If `nix` stops being found after an update, rerun the installer, which repairs it, or follow the repair instructions it prints.

## 2b. Route two: Homebrew and rustup

Install Homebrew from https://brew.sh if you do not have it. Then install Rust with rustup, the official toolchain installer, rather than Homebrew's `rust` formula, because rustup keeps the compiler current and the crate requires Rust 1.85 or newer:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Accept the default installation, open a new Terminal window, and check the toolchain targets your processor:

```sh
rustc --version
rustc -vV | grep host
```

The host line should read `aarch64-apple-darwin` on Apple silicon or `x86_64-apple-darwin` on Intel. Add the linting and formatting components used by the test instructions:

```sh
rustup component add clippy rustfmt
```

Install COLMAP from Homebrew:

```sh
brew install colmap
colmap -h | head -n 3
```

Note which COLMAP version that prints. COLMAP 3.13 renamed its GPU options, and section 5 uses the new names; on 3.12 or older replace `--FeatureExtraction.use_gpu` with `--SiftExtraction.use_gpu` and `--FeatureMatching.use_gpu` with `--SiftMatching.use_gpu`.

Clone the repository:

```sh
mkdir -p ~/git
cd ~/git
git clone https://github.com/harryprayiv/splatrs
cd splatrs
```

## 3. Build and test

These commands are the same on both routes.

```sh
cargo generate-lockfile
cargo build --release
cargo test
cargo test --features f64
cargo clippy --all-targets -- -D warnings
```

`cargo test` runs the unit tests, the file format round trips, the synthetic training tests and a command line smoke test. `cargo test --features f64` rebuilds every scalar in double precision and runs the full set of finite difference gradient checks. Both must report no failures, and clippy must print nothing. Confirm the binary is native:

```sh
file target/release/splatrs
```

It should say `arm64` on Apple silicon or `x86_64` on Intel.

`nix build` produces the same binary as a Nix package, but only sees files that git tracks, so `Cargo.lock` has to be committed (or at least added with `git add`) before `nix build` works. `cargo` does not have that requirement.

## 4. First run on a synthetic scene

The `synthetic` example renders a random scene with splatrs itself and writes it as a COLMAP project, so the whole pipeline can be checked before COLMAP or a camera is involved:

```sh
cargo run --release --example synthetic -- --out /tmp/synth --gaussians 3000 --views 24 --width 480 --height 320
./target/release/splatrs info --data /tmp/synth
./target/release/splatrs train --data /tmp/synth --out /tmp/synth-out --iterations 1500 --sh-degree 1 --eval
./target/release/splatrs render --data /tmp/synth --ply /tmp/synth-out/point_cloud/iteration_1500/point_cloud.ply --out /tmp/synth-renders
open /tmp/synth-renders
```

`info` ends with `ready for training`. Training ends with a train PSNR in the mid 30s and a test PSNR in the high 20s. The last command opens the rendered views in Finder.

## 5. A real benchmark scene

The 3DGS authors publish the Tanks and Temples and Deep Blending scenes already processed by COLMAP. Download the archive (about 650 MB; its licence allows non-commercial research use) and train on the Truck scene. Its images are already half resolution, so train at `--downscale 1`:

```sh
mkdir -p ~/splat-data
cd ~/splat-data
curl -L -o tandt_db.zip https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip
cd ~/git/splatrs
./target/release/splatrs info --data ~/splat-data/tandt/truck
./target/release/splatrs train --data ~/splat-data/tandt/truck --out ~/splat-data/out/truck-7k --downscale 1 --iterations 7000 --eval
```

The first log line reads `225 training views, 26 held out, 979x546 px, 136029 points`.

For your own photographs, put them in `proj/images` and run COLMAP on the CPU. The GPU flags matter on a Mac: COLMAP's GPU feature code needs CUDA, which macOS does not have, and its OpenGL fallback is unreliable on current macOS.

```sh
colmap feature_extractor --database_path proj/database.db --image_path proj/images --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 0
colmap exhaustive_matcher --database_path proj/database.db --FeatureMatching.use_gpu 0
colmap mapper --database_path proj/database.db --image_path proj/images --output_path proj/sparse
colmap image_undistorter --image_path proj/images --input_path proj/sparse/0 --output_path proj/undistorted --output_type COLMAP
./target/release/splatrs info --data proj/undistorted
./target/release/splatrs train --data proj/undistorted --out proj/splat --downscale 2 --eval
```

The README covers how to photograph a scene so that this works, and how to read the results.

Photos from an iPhone are HEIC by default, and splatrs and COLMAP read JPEG and PNG. Either set the camera to "Most Compatible" under Settings, Camera, Formats before shooting, or convert a folder with the built-in `sips` tool, which writes the JPEGs next to the originals:

```sh
cd proj/images
sips -s format jpeg -s formatOptions 95 IMG_0001.HEIC --out IMG_0001.jpg
```

Run `sips` once per file (Shortcuts or Automator can batch it), then move the HEIC originals out of `proj/images` so COLMAP sees only the JPEGs. Apple ProRAW DNGs should be converted with one fixed set of settings in a raw developer, as the README describes.

## 6. Performance on a Mac

splatrs runs on the CPU. GPU acceleration through Metal is not implemented; the planned wgpu backend would run on Apple GPUs. On Apple silicon the CPU path is still usable for benchmark scenes at reduced resolution, and the unified memory means large scenes fit as long as the machine has the RAM.

Apple silicon chips combine performance cores and efficiency cores. By default splatrs uses every core; when the work splits evenly, the slower efficiency cores can make each iteration wait for its last tile. Try limiting the worker threads to the number of performance cores and compare the `ms/iter` figure in the training log:

```sh
sysctl -n hw.perflevel0.physicalcpu
./target/release/splatrs --threads 8 train --data ~/splat-data/tandt/truck --out ~/splat-data/out/truck-test --downscale 2 --iterations 300 --eval
```

Replace 8 with the number the first command printed. Keep whichever setting gives the lower `ms/iter`.

Long training runs keep the CPU fully busy. On a MacBook, keep it on power, and prevent sleep for the duration of a run with the built-in `caffeinate` tool, which holds the machine awake until the command it wraps finishes:

```sh
caffeinate -i ./target/release/splatrs train --data ~/splat-data/tandt/truck --out ~/splat-data/out/truck-30k --downscale 1 --eval
```

Fanless MacBook Air models throttle under sustained load, so expect slower iterations after the first few minutes.

## 7. Using the result

The trained file is `point_cloud/iteration_<N>/point_cloud.ply` under the output directory. Blender 5.3 imports it natively on macOS (release expected 10 November 2026); earlier Blender versions need a 3DGS add-on. SuperSplat (https://superspl.at/editor) opens it in Safari or any other browser. The splat is in COLMAP's coordinate frame, so rotate it (roughly minus 90 degrees about X) and scale it after importing.

## 8. Troubleshooting

`xcrun: error: invalid active developer path` or `linker cc not found`: the command line tools are missing or were removed by a macOS update. Run `xcode-select --install` again.

`error: package requires rustc 1.85` or newer: the Homebrew route has an old compiler. Run `rustup update stable`. If `which rustc` points into `/opt/homebrew` or `/usr/local`, a Homebrew `rust` formula is shadowing rustup; remove it with `brew uninstall rust`.

`Unrecognized command line option: --SiftExtraction.use_gpu` or `--FeatureExtraction.use_gpu`: COLMAP renamed these options in version 3.13. Use the pair that matches the version `colmap -h` reports, as described in section 2b.

`error: no example target named synthetic`: the example files must be in `examples/` at the root of the repository, next to `src/`, not inside `src/`.

`Error: image ... is WxH but COLMAP recorded ..., and the two sizes do not have the same aspect ratio`: an image was cropped or rotated after COLMAP ran. splatrs accepts uniformly rescaled images but nothing else. Run COLMAP again on the images you intend to train on.

`Killed: 9` during training: macOS ran out of memory. Use a larger `--downscale`, a higher `--grad-threshold` so densification grows fewer Gaussians, or close other applications.

A run on Apple silicon is much slower than expected: check that `file target/release/splatrs` says `arm64`. An `x86_64` binary on Apple silicon runs under Rosetta; reinstall the toolchain from a native Terminal as described in section 1.
