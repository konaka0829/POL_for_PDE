# AGENTS.md — POL_for_PDE Stage A: ONE integration (no external browsing required)

## Goal
Implement Stage A: add ONE (Optical Neural Engine) baselines so we can compare ONE vs existing FNO scripts under identical data pipelines and metrics.

**Hard rule:** Do not change behavior of existing scripts (fourier_2d.py, fourier_2d_time.py). Prefer adding new files.

## About reference code access (IMPORTANT)
Codex may NOT be able to browse external GitHub repositories.
You MUST follow this priority order:

1) If `third_party/ONE_PDE_public/` exists, you may read it as reference.
2) Else if `third_party/ONE_PDE_public-main.zip` exists, unzip it into `third_party/ONE_PDE_public_ref/` and read that as reference.
3) Else: implement from the paper-grounded spec below (no reference code required).

**Do NOT copy-paste large code blocks from third-party repos.**
Use them only to understand structure and re-implement cleanly with attribution comments.

## Paper-grounded spec for ONE (enough to implement from scratch)
ONE uses:
- Diffractive Optical Neural Networks (DONN) for Fourier-space processing
- Optical XBAR (matrix-vector multiplication) for real-space linear mixing
- Nonlinearity: `tanh` (electronics-side)

DONN diffraction model (Fresnel / convolution via FFT):
- Impulse response:
  h(x,y) = exp(i k z) / (i λ z) * exp( i k/(2z) * (x^2 + y^2) )
  where k = 2π/λ, z is propagation distance, λ is wavelength.
- Propagation can be computed by FFT:
  F(field_next) = F(field) * H
  where H is FFT2(h) or an equivalent Fresnel transfer function in frequency domain.
- Phase-only modulation: multiply by exp(i * φ(x,y)), with φ constrained to [0, 2π].

Recommended implementation in PyTorch:
- Represent optical field as complex tensor (torch.cfloat).
- Precompute transfer function H for each spatial size (S,S) and device/dtype.
- Use:
  field = ifft2( fft2(field) * H )
  field = field * exp(1j * phase_mask)

Typical optical defaults (overridable via CLI):
- wavelength = 532e-9
- pixel_size = 36e-6
- distance = 0.254

XBAR model:
- treat as matrix multiplication (dense linear mixing).
- optional Gaussian noise added to outputs to emulate measurement noise.

## Stage A model scope (minimal)
We only need a ONE variant that can run on the same datasets as the repo’s FNO baselines:
- Darcy (2D steady): match fourier_2d.py preprocessing/normalization/metric.
- Navier–Stokes (2D time): match fourier_2d_time.py preprocessing/rollout/metric.

We do NOT need to implement the full multi-branch ONE (physics-parameter branch etc.) in Stage A.
Instead, implement an FNO-like backbone where the Fourier integral operator is replaced with a DONN-based operator.

## Comparability requirements (must match existing scripts)
### Darcy: one_2d.py
- Data pipeline must match fourier_2d.py exactly:
  - same r, same s computation, same mat keys (`coeff`, `sol`)
  - same single_split vs separate_files behavior
  - same UnitGaussianNormalizer usage and y decoding before LpLoss
- Metric: utilities3.LpLoss (relative L2 style used in scripts)

### Navier–Stokes: one_2d_time.py
- Data pipeline must match fourier_2d_time.py:
  - same data-mode behavior and split args
  - same slicing: input first T_in frames, predict next T frames
  - same autoregressive rollout identical (step loop)
- No normalization (keep consistent with FNO time script style)
- Report step-wise and full-trajectory losses

## Deliverables (files to add)
- Package: `one/`
  - `one/__init__.py`
  - `one/donn_layers.py` (Fresnel propagation + phase-only mask)
  - `one/one_models.py` (DONN operator + FNO-like network for Darcy + time model for NS)
  - `one/one_config.py` (defaults + argparse helpers)
- Scripts in repo root:
  - `one_2d.py` (Darcy)
  - `one_2d_time.py` (NS)
- README.md update: add usage examples for ONE scripts.

## CLI requirements
- Reuse existing cli_utils helpers:
  - add_data_mode_args / validate_data_mode_args
  - add_split_args where applicable
- Add `--smoke-test` for each ONE script:
  - do not load dataset
  - run forward+backward on synthetic tensors
  - exit successfully

## Engineering constraints
- CPU/GPU compatible (no hardcoded .cuda()).
- Do not assume fixed batch_size in reshape (handle last batch).
- Keep dependencies minimal (torch/numpy only; reuse utilities3).

## Post-work checks
- python -m compileall .
- python one_2d.py --help
- python one_2d_time.py --help
- python one_2d.py --smoke-test
- python one_2d_time.py --smoke-test
