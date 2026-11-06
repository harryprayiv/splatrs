# splatrs: Gaussian splatting from photographs, start to finish

splatrs trains and renders 3D Gaussian splats (Kerbl, Kopanas, Leimkuhler and Drettakis, SIGGRAPH 2023) from photographs. It is a from-scratch Rust implementation whose every differentiable stage has a hand-derived backward pass checked against finite differences, and it runs on the CPU, so it works on machines without a discrete GPU. It writes the standard 3DGS `.ply` layout, which Blender 5.3 imports and renders natively and which every splat viewer reads.

This document is the complete guide: what Gaussian splatting is and what it needs, how to photograph or film a scene so that it reconstructs well, how to turn the images into camera poses with COLMAP, how to train and render with splatrs, how to use the result in Blender and elsewhere, and what the technique cannot do. The mathematics and the verification strategy are in DESIGN.md.

## 1. What Gaussian splatting is

A Gaussian splat represents a scene as a few hundred thousand to a few million small, soft, coloured, semi-transparent ellipsoids called 3D Gaussians. Each one has a position, a shape (three axis lengths and a rotation), an opacity, and a colour that varies with viewing direction through spherical harmonics. To render a view, every Gaussian is projected onto the image plane as a 2D ellipse, the ellipses are sorted by depth, and each pixel blends them front to back the way layers of tinted glass combine. Because this rendering is differentiable, the scene can be fitted to photographs by gradient descent: render, compare with the photograph, adjust every Gaussian a little, repeat thirty thousand times, while new Gaussians are added where the error is high and useless ones are removed.

Compared with photogrammetry meshes, splats reproduce fuzzy, thin and semi-transparent things (hair, foliage, glass edges, fabric) and view-dependent shading (sheen, soft reflections) far better, and they look like photographs rather than textured geometry. Compared with neural radiance fields (NeRF), they train in minutes on a GPU instead of hours and render in real time. The price is that a splat is not geometry: it has no surfaces, no normals, cannot be relit, and lighting and shadows are baked into it as they were on the day of the capture.

## 2. What the technique needs, and the registration question

Gaussian splatting itself derives nothing about the cameras. Its inputs are a set of images, a calibrated camera (focal length and principal point) and a pose (position and orientation) for every image, and a sparse point cloud to start from. It fits the scene to those poses and never moves them. Where the poses come from is a separate problem called structure from motion (SfM), and in the standard pipeline COLMAP solves it: it finds distinctive feature points in every image, matches them between images, estimates relative camera motion from the matches, triangulates 3D points, and refines everything jointly with bundle adjustment. The output is the camera calibration, a pose per image and the sparse point cloud.

This answers the question of stabilization and registration directly. Nothing has to be stabilized, ordered, or registered beforehand: SfM registers unordered still photographs on its own, as long as they share enough recognisable detail. Video is accepted only as a source of still frames; the frames are extracted and then treated as an unordered photo set. Electronic image stabilization in the camera is actively harmful, because it crops and warps every frame differently, which breaks the assumption that all frames share one fixed calibration. Optical stabilization in the lens shifts the optical centre slightly from frame to frame; it is tolerable on phones but should be turned off where the camera allows it. A gimbal is useful only for what it was built for, which is reducing motion blur; it contributes nothing to registration.

Methods that estimate poses and scene together, without SfM, exist and are improving quickly: COLMAP-free 3DGS, Nope-NeRF, and the feed-forward networks DUSt3R, MASt3R, VGGT and MapAnything that predict geometry and cameras from raw image pairs or sets. They tolerate texture-poor scenes and sparse views that make COLMAP fail, at the cost of lower pose accuracy and, for the learned ones, large pretrained models. The best results today still come from running COLMAP (sometimes initialised from one of those networks) and then splatting; that is the path this guide follows.

## 3. What reconstructs well and what does not

The technique works on static, opaque, textured scenes under constant lighting, seen from many overlapping viewpoints. Objects on a table, rooms, building facades, gardens, statues, vehicles, rock faces, tool collections and machinery all reconstruct well. The following break or degrade it, roughly in order of how often they cause trouble.

Anything that moves during the capture (people, pets, leaves in wind, water, clouds, curtains, shadows moving as the sun moves) is seen in different places by different cameras; SfM may still register the scene from the static parts, but the moving things become smeared ghosts and floating fragments, and if they dominate the images SfM fails outright.

Changing exposure, white balance or tone mapping between shots makes the same surface a different colour in different images, which the model cannot explain with geometry, so it compensates with floaters and muddy colour. Lock everything.

Glossy and mirror surfaces (polished metal, glass, screens, wet floors, glazed ceramics) reflect a different image from each direction. Spherical harmonics absorb soft highlights, but a sharp reflection gets reconstructed as a phantom object behind the surface, and transparent objects become partial fog. A polarising filter helps with glass and water. Expect to accept imperfections here.

Textureless surfaces (white walls, uniform painted metal, clear sky) give SfM nothing to match. The scene can still register from its textured parts, and the splat will fill flat areas with large soft Gaussians, but a room of bare white walls will fail to register. Temporary texture (tape, printed markers, a patterned cloth) that stays in place for the whole capture fixes this where it is acceptable.

Thin structures (wires, railings, chair legs, grass) need many close views, and far backgrounds and sky get few consistent observations and produce floaters in front of the camera. Both improve with more images at more distances.

Scale is arbitrary: SfM recovers the scene up to an unknown global scale, so a splat has no metric units unless something of known size is measured in the result or the poses come from a device that knows the scale.

## 4. Capturing: cameras, settings and formats

Any camera works, including a phone, as long as its automatic behaviour can be switched off. A mirrorless or SLR camera with a prime lens gives the cleanest results; a phone in its "Pro" or manual mode is a close second; a phone in its default mode with computational HDR is the worst option because every frame gets a different multi-frame merge and tone curve.

Use one camera and one lens for the whole capture and do not touch the zoom. COLMAP can share one calibration across all images when that holds (`--ImageReader.single_camera 1`), which makes registration much more robust. A moderate wide angle (24 to 35 mm full-frame equivalent) gives generous overlap between shots and little distortion; ultra-wide and fisheye lenses capture more per shot but need COLMAP's fisheye models and undistortion, and they put more pixels into distorted corners. Zoom lenses are fine if the focal length is taped.

Set the camera to full manual. Fix the focus (manual focus, or autofocus once and switch to manual) at a distance that keeps the subject sharp, and use an aperture around f/5.6 to f/8 for depth of field; a blurry background is a defect here, not a style. Fix the shutter speed fast enough to freeze your own hand motion, 1/250 s or faster hand-held, and raise ISO as far as needed to get there; moderate noise is far less harmful than motion blur, because blur destroys the features SfM matches and smears every edge the splat learns. Fix white balance to a preset. Fix exposure, and do not change it even when the camera turns towards a window; an occasional clipped highlight is better than inconsistent exposure. Turn off every automatic enhancement: HDR, "smart" modes, night modes, electronic stabilization, lens shading correction if it is optional, and any per-shot sharpening or noise reduction that varies. Shoot in constant light: an overcast day, or indoors with all lights on and the sun not moving across the scene, and finish the capture in one session.

Shoot RAW if the camera offers it (DNG, CR3, NEF, ARW) and convert every file with identical settings to 8-bit JPEG at quality 95 or above, or PNG; splatrs and COLMAP read JPEG and PNG, and 8-bit sRGB is what the loss expects. If the camera can only produce JPEG, use its highest quality setting. Keep every image at the same resolution and orientation; do not crop, rotate or correct individual frames, and do not run any automatic per-image adjustment. Resolution of 12 to 24 megapixels is plenty; training downsamples, and COLMAP works at full resolution for features. Keep the original files; nothing is gained by compressing them further.

Phones: use a camera application with manual control, shoot DNG or the highest JPEG quality, disable HDR and stabilization, lock exposure and focus (most apps let you long-press to lock), and avoid portrait and night modes entirely.

## 5. Capturing: how to move the camera

The goal is that every surface is seen sharply from many directions, with each photograph overlapping its neighbours by about two thirds, and with the camera actually moving between shots rather than rotating in place. Rotating on the spot produces a panorama, which has no parallax and gives SfM no depth; the camera position has to change.

For an object or a small subject, walk at least three full orbits around it at different heights (below, level, above), taking a photograph every 10 to 15 degrees, so 24 to 36 per orbit, then add close-ups of detailed areas and a few shots from directly above. 100 to 200 photographs is typical. Keep the whole object in frame or overlapping between shots, and keep the background static. A turntable does not work with a fixed camera, because the background then disagrees with the object; if a turntable is unavoidable, the background must be featureless and masked, which this tool does not do.

For a room, walk slowly along the walls and through the middle, photographing outward and inward, at two heights, with every wall, the floor and the ceiling seen from several positions. Doorways and corners need extra coverage, because they are where registration chains break. 200 to 500 photographs is typical.

For a building or an outdoor space, walk several loops at different distances, photograph each facade from many angles including oblique ones, and include ground and sky only incidentally; the sky itself will not reconstruct and the splat will show it as a faint shell. Drones are a good source of oblique views when the law allows them.

Avoid pure forward motion along a corridor without side views, avoid long sequences where consecutive frames are nearly identical, and avoid gaps where two parts of the scene share no photographs. If you move from one room to another, take a dense run of photographs through the doorway so the two rooms link.

Video is acceptable when stills are impractical, with these rules: shoot the highest resolution and bitrate the camera has (4K, high bitrate, an all-intra or high-quality codec if available), set shutter speed manually to 1/250 s or faster (ignore the 180 degree rule, this is not cinematography), lock exposure, focus and white balance, disable stabilization, move slowly, and extract two to four frames per second rather than every frame. Nearly identical consecutive frames add compute and no information, and rolling shutter skews fast pans, which COLMAP's camera models do not describe (3DGUT, a 2025 method, does, but it is not in this tool). Expect video frames to be softer and noisier than stills from the same camera.

## 6. From images to camera poses with COLMAP

Put the images in `proj/images`. From the project's Nix dev shell (`nix develop`), which provides `colmap`, run the following. The GPU flags keep COLMAP on the CPU, which avoids OpenGL problems on headless and integrated-graphics machines; on a machine with a working discrete GPU they can be dropped.

```sh
colmap feature_extractor --database_path proj/database.db --image_path proj/images --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 --SiftExtraction.use_gpu 0
colmap exhaustive_matcher --database_path proj/database.db --SiftMatching.use_gpu 0
colmap mapper --database_path proj/database.db --image_path proj/images --output_path proj/sparse
colmap image_undistorter --image_path proj/images --input_path proj/sparse/0 --output_path proj/undistorted --output_type COLMAP
```

The first command detects SIFT features in every image and records the camera model; `SIMPLE_RADIAL` is the right default for an unknown lens, `OPENCV` when the lens has visible distortion, `OPENCV_FISHEYE` for fisheye lenses. The second matches features between every pair of images, which is thorough but quadratic in the number of images; above a few hundred images use `colmap sequential_matcher` when the images were taken in order (it also loops back to detect revisits) or `colmap vocab_tree_matcher` with a downloaded vocabulary tree when they were not. The third builds the reconstruction incrementally and writes one or more models under `proj/sparse`; `proj/sparse/0` is normally the largest, and if there are several it means the capture split into disconnected pieces, which more linking photographs would fix. The fourth is mandatory for splatrs: it resamples every image to a distortion-free pinhole camera and writes the images and the model to `proj/undistorted`, since the splat renderer models an ideal pinhole and would otherwise fit the lens distortion with wrong geometry.

Check the result before training:

```sh
./target/release/splatrs info --data proj/undistorted
```

It prints the cameras, the number of registered images and 3D points, and `ready for training` when every camera is pinhole. If far fewer images registered than you took, the usual causes are motion blur, exposure changes, too little overlap, or texture-poor surfaces; the fix is almost always more and better photographs, not different COLMAP settings. If several models came out of the mapper, the capture had gaps.

Frames from video are extracted with ffmpeg; two frames per second from a slow walk is a good starting density:

```sh
nix shell nixpkgs#ffmpeg --command ffmpeg -i proj/capture.mp4 -vf fps=2 -q:v 1 proj/images/frame_%05d.jpg
```

Delete visibly blurred frames before running COLMAP; a few bad frames cost more than they add.

## 7. Training with splatrs

Build once with `cargo build --release` inside `nix develop`. Then:

```sh
./target/release/splatrs train --data proj/undistorted --out proj/splat --downscale 2 --eval
./target/release/splatrs render --data proj/undistorted --ply proj/splat/point_cloud/iteration_30000/point_cloud.ply --out proj/renders --downscale 2
```

`--data` expects the COLMAP layout: `sparse/0` (or `sparse`) with `cameras`, `images` and `points3D` as `.bin` or `.txt`, and `images` next to it; `--sparse` and `--images` override either location. Training writes `point_cloud/iteration_7000/point_cloud.ply` and `point_cloud/iteration_<last>/point_cloud.ply` under `--out`, the directory layout the reference implementation uses for its splats (the reference's `cameras.json` and `cfg_args` are not written, so SIBR will not open the directory as a model).

`--downscale K` trains on images reduced by an integer factor; 2 or 4 for a first run, then 1 or 2 for the final one. `--eval` holds out every eighth image and reports its PSNR at the end, which is the only honest measure of quality since training views can be memorised. `--iterations N` defaults to 30000; results are usable from about 7000 and improve slowly after 15000, when densification stops. `--white-bg` for captures on white backgrounds. `--antialias` enables Mip-Splatting style opacity compensation, which reduces aliasing when the result will be viewed at very different resolutions from the training images. `--absgrad` switches the densification criterion to AbsGS, which grows finer detail at the cost of more Gaussians; it uses a threshold of 0.0008 unless `--grad-threshold` is given. `--sh-degree D` (0 to 3) controls view dependence; 3 is the default and costs memory, 1 or 2 is enough for matte scenes. `--no-densify` keeps the Gaussian count fixed, which is only useful for experiments. `--max-screen-size PX` enables screen-size pruning, which is off by default (see DESIGN.md). `--seed`, `--save-at A,B,C`, `--log-interval N` and the global `--threads N` do what their names say.

The progress line shows the loss, its moving average, the PSNR of the current view, the number of Gaussians, the time per iteration and the densification counts since the previous line. Loss should fall steadily and PSNR rise; a PSNR that stalls around 20 dB on a well-captured scene means the poses are wrong (check `info` and the number of registered images), a Gaussian count that explodes past a few million means the gradient threshold is too low for the image resolution or the scene has lots of moving or glossy content.

## 8. Judging and improving results

Render the training views and compare them with the photographs; `render` writes a PNG per image. Look for: floaters (small clouds in empty space, from sky, reflections or inconsistent exposure), blurred regions (too few views of that area, or motion blur in the sources), missing thin structures (more close views), and colour casts (exposure or white balance drift). Held-out PSNR above 25 dB is decent and above 30 dB is good for real captures; synthetic scenes reach higher.

Most defects are fixed by capturing again with the rules in sections 4 and 5. Floaters that remain can be removed in an editor such as SuperSplat (crop boxes, opacity thresholds, selection by position); Blender 5.3 exposes the splat as point cloud data in Geometry Nodes, which can cull by position and opacity. Training for longer rarely helps; training at full resolution helps sharpness; `--antialias` helps when the viewer zooms far in or out.

## 9. Using the result in Blender and other tools

Blender 5.3 (final release expected 10 November 2026) imports `.ply` Gaussian splats natively and renders them in Workbench, EEVEE and Cycles; the file written by splatrs is the standard layout that importer reads. Earlier Blender versions need an add-on such as KIRI Engine's 3DGS Render or BlendSplat, which read the same file. The splat is stored in COLMAP's world frame, which is right-handed with no fixed up direction: cameras look down +z with y down. Blender's world is z up, so expect to rotate the imported object, typically by about minus 90 degrees around X as a starting point and then by eye, and to scale it, because the units are arbitrary. A splat renders as it was lit during the capture; lights in Blender do not affect it, it casts no true shadows, and objects placed inside it do not receive its light, so compositing works best when the splat is a background or a hero object under lighting that matches the capture. SuperSplat (web), PlayCanvas, Nerfstudio's viewer and many others open the same file. For delivery on the web, convert to a compressed format: SPZ (Niantic) or SOG (PlayCanvas) reduce a file by five to ten times; the Khronos glTF extension KHR_gaussian_splatting is the standardisation effort.

## 10. Things that are easy to overlook

Compute: on a laptop CPU a 30000 iteration run at one megapixel takes around an hour and a full-resolution run several hours; GPU implementations do the same in minutes. Memory: about a kilobyte per Gaussian during training (parameters, gradients, two Adam moments) plus the images; a million Gaussians need a few gigabytes. Storage: a 30000 iteration splat with degree 3 harmonics is about 250 bytes per Gaussian on disk, so 100 to 500 MB uncompressed for a scene.

Privacy and rights: a capture records everything in view, including faces, licence plates, documents and screens, in a form that can be inspected from new angles; blur or exclude what should not be shared, and remember that photographing other people's property for a 3D model can be restricted.

Dynamic scenes need the 4D variants (4DGS and successors) and synchronised multi-camera rigs; a single moving camera cannot capture motion. Relighting needs the inverse-rendering variants (GS-IR and others) and is still far from production quality. Meshes can be extracted from splats (SuGaR, and 2D Gaussian splatting which fits surfels instead of ellipsoids) when geometry rather than appearance is the goal; a plain splat has no usable surface.

Exposure compensation, depth supervision from LiDAR or monocular depth, and image masks are in the 2024 update of the reference implementation and in gsplat, and are on this tool's roadmap. The GPU-accelerated Rust alternative is Brush (ArthurBrussee/brush), which runs on Intel, AMD and NVIDIA GPUs through wgpu, trains with MCMC densification, and reads the same COLMAP input.

## 11. Building, testing and the code

The flake provides a dev shell with `cargo`, `rustc`, `clippy`, `rustfmt`, `rust-analyzer` and `colmap`. Without Nix, any Rust 1.85 or newer toolchain builds the crate.

```sh
nix develop
cargo generate-lockfile
cargo build --release
cargo test
cargo test --features f64
cargo clippy --all-targets -- -D warnings
```

`cargo test` runs the unit tests, the COLMAP and PLY round trips, the synthetic end-to-end training tests, the command-line smoke test, and the finite-difference checks whose gradients single precision can resolve. `cargo test --features f64` builds every scalar as `f64` and runs the full set of gradient checks (spherical harmonics, projection with and without antialiasing, the rasterizer, SSIM, the photometric loss, and the whole parameters-to-loss chain) at a relative tolerance of 1e-5. `nix build` produces the binary and needs `Cargo.lock` to exist.

The `synthetic` example renders a random scene with splatrs itself and writes it out as a COLMAP project, so the whole pipeline can be exercised and timed before any photograph is taken:

```sh
cargo run --release --example synthetic -- --out /tmp/synth --gaussians 3000 --views 24 --width 480 --height 320
./target/release/splatrs info --data /tmp/synth
./target/release/splatrs train --data /tmp/synth --out /tmp/synth-out --iterations 1500 --sh-degree 1 --eval
./target/release/splatrs render --data /tmp/synth --ply /tmp/synth-out/point_cloud/iteration_1500/point_cloud.ply --out /tmp/synth-renders
```

On two cores this run reaches about 35 dB PSNR on the training views and 28.6 dB on the held-out views in 1500 iterations, growing from 1500 to about 13000 Gaussians.

The pipeline implemented is the one from the paper: COLMAP poses and sparse points in, Gaussians initialised from the points, each iteration renders one view through a tile-sorted differentiable rasterizer, the loss is 0.8 L1 + 0.2 D-SSIM, Adam updates the parameters, and adaptive density control clones, splits and prunes Gaussians during the first half of training. Mip-Splatting opacity compensation and the AbsGS densification criterion are available as flags. Performance is proportional to the number of Gaussian-pixel pairs that pass the alpha threshold, at roughly 15 ns per pair per core forward and 25 ns backward. The crate builds with `f32` scalars by default and with `f64` under the `f64` feature; the same code runs in both. Every backward pass is checked against central finite differences under `f64`, the tiled rasterizer is compared against a plain per-pixel compositor, the COLMAP reader was validated on files written by COLMAP's own `read_write_model.py`, and the PLY layout follows the reference attribute order. DESIGN.md records the conventions, the derivations, the deliberate differences from the reference and the known limitations.

Roadmap: a wgpu compute backend so the same training loop runs on Intel integrated GPUs, SIMD inner loops for the CPU backend, 3DGS-MCMC relocation, Mip-Splatting's 3D filter, exposure compensation, depth regularisation and masks, Nerfstudio `transforms.json` input, SPZ export, checkpointing with optimiser state, and an interactive viewer.

## License

AGPL-3.0-or-later. The license text is not included in this tree; fetch it with `curl -o LICENSE https://www.gnu.org/licenses/agpl-3.0.txt`.
