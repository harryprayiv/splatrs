# splatrs design notes

This document records the conventions, the mathematics of each differentiable stage, the places where splatrs deliberately differs from the reference implementation, and how each part is verified. The source modules are named in the order the data flows through them.

## Scalar type

`splatrs::Real` is `f32` by default and `f64` under the `f64` cargo feature. All geometry, parameters, gradients and images use it; only file formats fix a width (COLMAP stores `f64`, the PLY stores `f32`, PNG stores `u8`). The feature exists so the gradient checks can run at double precision against the same code that trains at single precision. Nothing in the crate is specialised on the precision.

## Conventions

COLMAP conventions are used throughout. A camera transforms a world point by `x_cam = R x_world + t`; the camera looks down +z with x to the right and y down. The pixel with integer index `(i, j)` has its centre at continuous coordinates `(i + 0.5, j + 0.5)`, and a camera-space point projects to `u = fx x / z + cx`, `v = fy y / z + cy`. The principal point is used as recorded; the reference implementation assumes it sits at the image centre, so splatrs is exact on undistorted COLMAP output where the principal point is usually off-centre by a few pixels.

Rotations are stored as unnormalised quaternions `(w, x, y, z)` and normalised before use. Scales are stored as natural logarithms, opacity as a logit. These are exactly the stored quantities of the reference, which is why the PLY round-trips with every other tool.

Spherical harmonics use the real basis and constants of the reference, in its coefficient order, up to degree 3 (16 coefficients per channel). A colour channel is `max(0, sum_k basis_k(d) c_k + 0.5)` with `d` the unit vector from the camera centre to the Gaussian mean.

## Data layout

`model::Gaussian` is one plain record of 59 scalars: mean (3), log scale (3), quaternion (4), opacity (1) and SH coefficients (48). The same record type holds a gradient and each Adam moment, so the optimiser, densification and pruning move whole rows and cannot desynchronise the parameter, gradient and moment arrays. `model::Model` is the vector of records plus the active SH degree.

`project::Proj` is the screen-space description of one Gaussian for one camera: 2D mean, conic, depth, pixel radius, effective opacity, colour and the rectangle of 16x16 tiles it touches. `project::ProjGrad` carries the loss gradient with respect to the same screen quantities back from the rasterizer.

`rasterize::Binning` holds one entry per (Gaussian, tile) pair. Entries are generated in Gaussian order so each Gaussian's entries are contiguous, then a `(tile << 32 | depth bits, entry)` key array is sorted, and per-tile ranges are recorded. The backward pass writes per-entry gradient slots through relaxed atomic stores (each slot is written by exactly one tile task) and reduces them per Gaussian over the contiguous entry range, which makes the reduction deterministic without locks or unsafe code.

## Projection (`project.rs`)

Forward, for one Gaussian: `t = R p + c`; cull if `t.z <= near` (0.2 by default, as in the reference). `Sigma = M M^T` with `M = R_q diag(s)`. The affine approximation of the projection is `J = [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]]` evaluated at `x` and `y` clamped to a frustum enlarged by 30 percent on each side, following gsplat's rule, which handles an off-centre principal point. `T = J R`, `Sigma' = T Sigma T^T`, dilated by `eps2d = 0.3` on the diagonal. The conic is the inverse of the dilated `Sigma'`, the radius is `ceil(3 sqrt(lambda_max))`, the tile rectangle is the bounding box of the circle of that radius clamped to the grid, and the effective opacity is `sigmoid(o)` times the Mip-Splatting compensation `sqrt(det(Sigma') / det(Sigma' + eps2d I))` when antialiasing is on, 1 otherwise.

Backward: the conic path uses the derivative of a 2x2 inverse, written out for the three independent entries. The symmetric gradient `G` with respect to `Sigma'` gives `dSigma = T^T G T` and `dT = 2 G T Sigma`; `dJ = dT R^T`; the Jacobian entries give `dt` including the extra `z` term when the clamp is active; the 2D mean adds its own `dt`; `dp = R^T dt` plus the view-direction term of the colour. `Sigma = M M^T` gives `dM = 2 dSigma M`, which splits into the scale and rotation gradients; the quaternion gradient goes through the explicit derivative of the rotation matrix and then through the normalisation. All of this is in `project_backward_one`, which recomputes the forward intermediates rather than storing them, so the per-Gaussian memory is the record itself.

## Rasterization (`rasterize.rs`)

Each pixel composites its tile's entries front to back: `alpha = min(0.99, o exp(-sigma))` with `sigma = 0.5 (A dx^2 + 2 B dx dy + C dy^2)`, entries with `alpha < 1/255` skipped, and compositing stopping when the transmittance would drop below 1e-4. The forward pass records the final transmittance and the index of the last contributing entry per pixel; the backward pass walks the contributing entries in reverse, recovering the transmittance in front of each entry by division, and uses the reference's running `accum_rec` recurrence for the colour behind each entry. The gradient with respect to the conic entries is the true gradient with respect to `(A, B, C)`; the reference stores half of the `B` gradient and compensates later, which is a convention, not a difference in result.

The tile loops are written entry-outer rather than pixel-outer: each entry visits only the pixels of its own bounding rectangle, derived from the exponent at which it can no longer reach the 1/255 threshold, and per-pixel state carries the compositing. This is an acceleration structure only; a test compares the result with a plain per-pixel compositor and the two agree to within floating-point noise. Pixels outside the rectangle are exactly the pixels the per-pixel walk would skip.

AbsGS support is the extra pair of per-entry slots that accumulate the absolute value of each pixel's contribution to the 2D mean gradient, summed per Gaussian alongside the signed gradient. The trainer scales either quantity by half the image size, which converts the pixel-space gradient to the NDC-space gradient the reference's threshold of 0.0002 was tuned for.

## Loss (`loss.rs`)

The loss is `(1 - lambda) L1 + lambda (1 - SSIM)` with `lambda = 0.2`. SSIM uses the reference's 11x11 Gaussian window with sigma 1.5 as a separable, zero-padded, same-size convolution, constants `C1 = 0.01^2` and `C2 = 0.03^2`, averaged over all pixels and channels. Its gradient with respect to the rendered image is obtained by differentiating the per-pixel SSIM with respect to the three blurred statistics that depend on the render (`mu_x`, `E[x^2]`, `E[xy]`) and applying the blur again, which is its own adjoint because the window is symmetric.

## Optimiser and schedule (`adam.rs`, `train.rs`)

Adam with `beta = (0.9, 0.999)` and `eps = 1e-15`, one learning rate per parameter group: means `1.6e-4` times the scene extent decaying exponentially to `1.6e-6` times the extent over 30000 iterations, log scales `5e-3`, rotations `1e-3`, opacity `2.5e-2`, SH DC `2.5e-3`, higher SH `2.5e-3 / 20`. The SH degree rises by one every 1000 iterations. One random view per iteration, with the views reshuffled every epoch. The scene extent is 1.1 times the largest distance of any camera centre from the mean camera centre.

Densification runs every 100 iterations between iterations 500 and 15000 using the per-Gaussian mean of the accumulated NDC gradient norm over the iterations in which the Gaussian was visible. Gaussians above the threshold are cloned when their largest scale is at most 1 percent of the extent and split into two otherwise, the children sampled from the parent's own distribution and scaled down by 1.6. Afterwards Gaussians with opacity below 0.005 are pruned, and once the first opacity reset has happened, so are Gaussians whose largest scale exceeds 10 percent of the extent. Opacities are reset to at most 0.01 every 3000 iterations, and also at iteration 500 on a white background; the reset zeroes the optimiser's opacity moments, as the reference does. New Gaussians enter the optimiser with zero moments and share the global step count, which is what the reference's optimiser surgery produces.

## Deliberate differences from the reference

The optimiser step happens before densification within an iteration, as in gsplat. The reference replaces every parameter tensor during densification, which leaves it without gradients for that iteration, so the reference effectively skips the step on densification iterations; that is an artefact, not a design, and it is not reproduced.

Screen-radius pruning is off by default. The reference computes `max_radii2D` for it but resets that array to zero in `densification_postfix` before the prune mask is built, so the check never fires there; gsplat's default strategy also leaves its equivalent disabled. The `--max-screen-size` flag turns on the check using the radii accumulated since the previous densification step, for anyone who wants to experiment with it.

The principal point is honoured and the Jacobian clamp limits are gsplat's, which account for it. The reference ignores `cx` and `cy`.

The tile rectangle of a Gaussian is the reference's circle-of-radius bounding box, so the set of candidate pixels is the same; the per-entry pixel rectangle inside a tile is an optimisation that cannot change the result.

Everything else (initialisation from the point cloud, learning rates, thresholds, intervals, SH constants and ordering, loss, PLY layout, hold-out rule) follows the reference.

## Verification

`tests/gradcheck.rs` checks the SH basis derivatives and the projection backward pass against central finite differences, in both precisions. `tests/gradcheck_f64.rs` (compiled only with the `f64` feature) adds the rasterizer backward pass over nine overlapping Gaussians, the SSIM and photometric loss gradients, and the full chain from parameters to loss for six overlapping Gaussians, each at a relative tolerance of 1e-5. The chain check uses a pure D-SSIM loss because the L1 term has kinks wherever the render crosses the target, which make central differences unreliable at that tolerance; the L1 gradient is checked on its own.

`tests/pipeline.rs` trains a synthetic scene rendered by the renderer itself, with and without densification, and asserts a PSNR improvement of at least 6 dB and a final PSNR above 24 dB; writes a COLMAP model in both file formats and reads it back; loads a dataset at two scales and checks the intrinsics follow the resampling; runs the `info`, `train` and `render` subcommands on a synthetic project; and compares the tiled rasterizer against a per-pixel compositor. The COLMAP binary reader was additionally run on files produced by COLMAP's own `scripts/python/read_write_model.py` (version 3.11.1), including non-empty 2D observations and tracks, and the resulting scene extent matched the value computed in Python to all printed digits. The camera model table was checked against `src/colmap/sensor/models.h` of COLMAP 3.12, which defines `RAD_TAN_THIN_PRISM_FISHEYE` as model 11 with 16 parameters.

`cargo clippy --all-targets -- -D warnings` passes in both precisions, with three lints allowed crate-wide: `needless_range_loop` and `too_many_arguments` because the numeric code is written with explicit indices by choice, and `unnecessary_cast` because casts between `Real` and the fixed-width file types are identity casts in one of the two builds.

## Known limitations

CPU only; the per-pair cost is about 15 ns forward and 25 ns backward per core. No checkpointing of the optimiser state (a saved PLY can be rendered but not resumed). No exposure compensation, depth supervision or masks. The initial point cloud must come from COLMAP; there is no random initialisation. Images must match the size COLMAP recorded for their camera. Only `SIMPLE_PINHOLE` and `PINHOLE` cameras train; other models must be undistorted by COLMAP first.
