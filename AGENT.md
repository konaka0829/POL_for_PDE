# AGENT.md

## Purpose of this repository

This repository should be a slim implementation of the current research code for **time-scaled PDE surrogate operator learning** on 1D periodic Burgers-type problems.

The active code should center on:

- 1D periodic Burgers target dynamics.
- Surrogate PDE families: Burgers, reaction--diffusion, and Kuramoto--Sivashinsky style reservoirs.
- Model 1 / Model 2 / Model 3 experiments.
- The time-scaled Model 1 error decomposition based on `Delta_scale`.

The repository previously contained legacy Fourier Neural Operator, LowRank, Darcy, Navier--Stokes, image, and old reservoir/RFM code. Those are not part of the current research implementation and should not remain in the active tree after the slimming refactor.

---

## Non-negotiable theoretical convention

### Do not use the old raw `Delta_time` formulation as the main theory

The current theory treats the surrogate readout time `Ttilde` as a **time scaling**, not as a separate time-mismatch error.

For target final time `T` and surrogate readout time `Ttilde`, define

```text
alpha = Ttilde / T
```

The rescaled surrogate trajectory is

```text
r_{theta,alpha}(s; u0) = Gtilde_{theta, alpha*s}(E u0),    s in [0,T].
```

The time-scaled generator residual is

```text
R_scale_{theta,alpha}(r) = F(Q r) - alpha * Q Ftilde_theta(r).
```

The main Model 1 bound is

```text
D1(theta, alpha*T)
  <= exp(beta*T) * Delta_init(theta)
     + c_beta_T * Delta_scale(theta, alpha; T),
```

where

```text
c_beta_T = sqrt((exp(2*beta*T)-1)/(2*beta))    if beta != 0,
         = sqrt(T)                             if beta == 0.
```

In the current fully observed 1D implementation, `E = I` and `Q = I`, so `Delta_init = 0` unless a non-identity encoder/decoder is explicitly introduced.

### Consequence for code

For `Ttilde != T`, never compute the primary defect as an unscaled residual on native surrogate times `0..T`.

Incorrect pattern:

```python
defects = generator_defect(surrogate_states[: step_T + 1], cfg)
```

Correct pattern:

```python
alpha = Ttilde / cfg.T
s_grid = torch.arange(step_T + 1) * cfg.dt
r_alpha = surrogate_state_at_native_times(alpha * s_grid)
defects = scaled_generator_defect(r_alpha, cfg, alpha=alpha)
```

Prefer linear interpolation when `alpha*s_grid` is not exactly aligned with the native surrogate `dt` grid.

---

## Active core files to preserve

The following files or their slim equivalents are core and should remain:

```text
pol/burgers_spectral_1d.py
pol/reservoir_1d.py
pol/ridge.py
pol/elm.py
pol/features_1d.py
pol/io_mat.py                 # create from the useful part of utilities3.py
pol/cli.py                    # create from the useful part of cli_utils.py
pol/plotting.py               # create from the useful 1D part of viz_utils.py
pol/model123_1d/__init__.py
pol/model123_1d/initial_conditions.py
pol/model123_1d/datasets.py
pol/model123_1d/metrics.py
pol/model123_1d/predictors.py
pol/model123_1d/error_decomposition.py
model123_burgers_1d.py
model1_error_decomposition_1d.py
scripts/run_model123_param_sweep.py
scripts/generate_burgers_1d.py
README.md
requirements.txt
pyproject.toml
LICENSE
.gitignore
```

It is acceptable to keep the two root entry points `model123_burgers_1d.py` and `model1_error_decomposition_1d.py` for backward compatibility, even if future cleanup moves them under `scripts/`.

---

## Files and directories to remove from the active tree

Remove legacy code rather than leaving a large in-repo archive. If a historical note is needed, add a short `docs/legacy_removed.md` listing removed groups.

### FNO / LowRank legacy

```text
fourier_1d.py
fourier_2d.py
fourier_2d_time.py
fourier_3d.py
lowrank_operators/
scripts/eval.py
scripts/fourier_2d_tuned.py
scripts/fourier_3d_time.py
scripts/fourier_on_images.py
scripts/super_resolution.py
```

### Darcy / Navier--Stokes / MATLAB legacy data generation

```text
data_generation/darcy/
data_generation/navier_stokes/
data_generation/burgers/GRF1.m
data_generation/burgers/burgers1.m
data_generation/burgers/gen_burgers1.m
```

### Deprecated Model123 modules

```text
pol/model123_1d/models.py
pol/model123_1d/observations.py
pol/model123_1d/solvers.py
```

### Old reservoir/RFM experiments

```text
reservoir_burgers_1d.py
rfm_burgers_1d.py
pol/encoder_1d.py
scripts/hparam_search_reservoir_burgers.py
scripts/sweep_burgers_nu_feature_times.py
```

### Redundant sweep/plot scripts after consolidation

```text
scripts/run_model123_nu_sweep.py
scripts/run_model123_ttilde_sweep.py
scripts/plot_model123_beta5_profiles.py
scripts/plot_model123_ks_profiles.py
```

### Helper modules to consolidate, then delete

```text
utilities3.py   -> move needed MatReader functionality to pol/io_mat.py
cli_utils.py    -> move needed CLI helpers to pol/cli.py
viz_utils.py    -> move useful 1D plotting and save_figure_all_formats to pol/plotting.py
```

After migration, update all imports and delete the old helper modules.

---

## Time-scaled residual formulas

Use the generator conventions from `pol/reservoir_1d.py`.

### Target Burgers generator

```text
F(z) = target_nu * z_xx - z * z_x
```

### Burgers surrogate

```text
Ftilde(z) = res_burgers_nu * z_xx - res_burgers_b * z * z_x
```

Therefore

```text
R_scale(z)
  = (target_nu - alpha*res_burgers_nu) * z_xx
    + (alpha*res_burgers_b - 1.0) * z * z_x.
```

### Reaction--diffusion surrogate

The implemented RD surrogate is

```text
Ftilde(z) = rd_nu * z_xx + rd_alpha * z - rd_beta * z^3.
```

Therefore

```text
R_scale(z)
  = (target_nu - alpha*rd_nu) * z_xx
    - z * z_x
    - alpha*rd_alpha * z
    + alpha*rd_beta * z^3.
```

### Kuramoto--Sivashinsky surrogate

The implemented KS surrogate is

```text
Ftilde(z) = -ks_b * z * z_x - ks_eta * z_xx - ks_kappa * z_xxxx.
```

Therefore

```text
R_scale(z)
  = (target_nu + alpha*ks_eta) * z_xx
    + (alpha*ks_b - 1.0) * z * z_x
    + alpha*ks_kappa * z_xxxx.
```

### Required implementation names

Prefer explicit names:

```python
scaled_defect_burgers_reservoir(..., alpha: float, ...)
scaled_defect_reaction_diffusion_reservoir(..., alpha: float, ...)
scaled_defect_ks_reservoir(..., alpha: float, ...)
scaled_generator_defect(z: torch.Tensor, cfg: ErrorDecompositionConfig, *, alpha: float) -> torch.Tensor
```

`generator_defect` may remain only as a deprecated compatibility alias for `alpha=1.0`.

---

## Error decomposition implementation requirements

`pol/model123_1d/error_decomposition.py` should expose a clear, fully discrete full-state special case.

### Required discrete quantities

Use the 1D uniform-grid norm:

```text
h = 1 / nx
||v||_{L_h^2} = sqrt(h * sum_j v_j^2)
```

Use shared metrics from `pol/model123_1d/metrics.py` whenever possible.

### Time grid

Use target-time grid `s_n` over `[0,T]`:

```text
s_n = n * dt, n = 0,...,N_T.
```

Use trapezoidal time quadrature by default.

### Rescaled surrogate states

For each `Ttilde`:

```text
alpha = Ttilde / T
r_alpha[n] = surrogate state at native time alpha*s_n.
```

Use interpolation if `alpha*s_n` is not exactly an integer multiple of `dt`.

### Per-sample quantities

For each sample `i`:

```text
D1_i = ||u_i(T) - r_{alpha,i}(T)||_{L_h^2}
Delta_init_i = 0      # full-state identity encode/decode case
Delta_scale_i^2 = sum_n w_n ||R_scale(r_{alpha,i}(s_n))||_{L_h^2}^2
rhs_beta0_i = Delta_init_i + sqrt(T) * Delta_scale_i
rhs_beta_i = exp(beta*T)*Delta_init_i + c_beta_T * Delta_scale_i
```

### Aggregate quantities

Use empirical RMS over samples:

```text
D1 = sqrt(mean_i D1_i^2)
Delta_init = sqrt(mean_i Delta_init_i^2)
Delta_scale = sqrt(mean_i Delta_scale_i^2)
rhs_beta0 = Delta_init + sqrt(T) * Delta_scale
rhs_beta = exp(beta*T) * Delta_init + c_beta_T * Delta_scale
```

Do not define `rhs_beta` as `sqrt(mean_i rhs_beta_i^2)` unless that value is clearly named as a diagnostic, for example `rhs_beta_pathwise_rms`.

---

## Beta calibration requirements

`beta` is a one-sided Lipschitz constant for the **target Burgers generator**.

For a fixed `alpha`, calibrate beta using the target states `u(s)` and the rescaled surrogate states `r_alpha(s)`, not the native surrogate states `r(s)` unless `alpha=1`.

Supported modes:

```text
zero
fixed
analytic_safe
analytic_safe_poincare
empirical_pairwise
```

### analytic_safe

Use

```text
M_K_hat = max ||d_x z||_infty
beta = 0.5 * M_K_hat
```

over the target and rescaled surrogate states used for calibration.

### analytic_safe_poincare

If samplewise means match, use

```text
beta = 0.5 * M_K_hat - target_nu * (2*pi)^2.
```

Raise a clear error if the means do not match within tolerance.

### empirical_pairwise

For state pairs `a,b`, compute

```text
q(a,b) = <F(a)-F(b), a-b>_h / ||a-b||_{L_h^2}^2
```

where `F` is the target Burgers generator with `target_nu`. Use the max plus optional margin.

### Multiple Ttilde values

Prefer per-alpha beta calibration and store results under something like:

```python
result["beta_details_by_ttilde"] = {
    "0.8": {...},
    "1.0": {...},
    "1.2": {...},
}
```

Rows and summary rows should carry their own `beta_value`.

---

## Output schema requirements

### Per-sample rows

Include at least:

```text
sample_index
T
Ttilde
alpha
D1_abs_l2h
Delta_init_abs_l2h
Delta_scale_abs_l2h
rhs_beta0_pathwise_abs_l2h
rhs_beta_pathwise_abs_l2h
beta_mode
beta_value
c_beta_T
```

Optional compatibility aliases:

```text
Delta_dyn_abs_l2h = Delta_scale_abs_l2h
rhs_beta_abs_l2h = rhs_beta_pathwise_abs_l2h
```

Do not expose `Delta_time_abs_l2h` as a primary field. If retained, it must be explicitly named as legacy, e.g. `legacy_Delta_time_abs_l2h`.

### Summary rows

Include at least:

```text
Ttilde
alpha
num_samples
D1
Delta_init
Delta_scale
rhs_beta0
rhs_beta
beta_mode
beta_value
c_beta_T
```

Optional compatibility aliases:

```text
Delta_dyn = Delta_scale
```

### Plots

Use time-scaled names and labels:

```text
delta_scale_vs_ttilde
scaled_bound_vs_ttilde
scaled_bound_scatter
```

Avoid "time mismatch" plot names for primary outputs.

---

## Metrics convention

Use `pol/model123_1d/metrics.py` as the single source of truth.

It should provide:

```python
discrete_l2h_norm
per_sample_abs_l2h_error
dataset_abs_l2h_error
per_sample_rel_l2h_error
dataset_rel_l2h_mean
rms_l2  # backwards-compatible alias for dataset_abs_l2h_error
```

The main metric is absolute discrete `L_h^2`. Relative error is secondary and should still be written for comparison.

`model123_burgers_1d.py`, `pol/model123_1d/experiments.py`, and sweep scripts must not compute their own incompatible metric formulas.

---

## Model123 and sweep requirements

### `model123_burgers_1d.py`

- Use shared metrics.
- Write both absolute and relative metrics to `run_config.json`:

```text
main_metric = "abs_l2h"
train_absL2h
test_absL2h
train_relL2
test_relL2
```

- Print absolute metrics first, relative metrics second.

### `scripts/run_model123_param_sweep.py`

This should be the single active sweep script.

Required behavior:

- Read `train_absL2h`, `test_absL2h`, `train_relL2`, and `test_relL2` from each run's `run_config.json`.
- Use `test_absL2h` as the default ranking and plotting metric.
- Keep relative metrics in CSV for compatibility.
- Support sweeps over:

```text
Ttilde
res_burgers_nu
res_burgers_b
rd_nu
rd_alpha
rd_beta
ks_b
ks_eta
ks_kappa
dt
K
J
```

- Validate pure argument errors, such as `max_workers <= 0`, before data-file existence checks.

Remove separate `nu` and `Ttilde` sweep scripts after the parameter sweep fully covers them.

---

## Data generation requirements

Keep one active Python Burgers data generator for Model123, preferably:

```text
scripts/generate_burgers_1d.py
```

It should produce `.mat` or `.pt` outputs compatible with `model123_burgers_1d.py` and include useful metadata when possible:

```text
T
dt
nu
nx
num_samples
```

Remove redundant old generators after consolidation.

---

## Packaging requirements

Add or update `pyproject.toml` so imports and tests work without setting `PYTHONPATH=.` manually.

The slim repository should not depend on `torchvision`; remove it from `requirements.txt` because image/FNO scripts are removed.

Keep dependencies minimal, for example:

```text
torch
numpy
scipy
h5py
matplotlib
pytest
```

Add a pytest `slow` marker and mark expensive smoke/integration tests accordingly.

---

## README requirements

Rewrite `README.md`. It must not describe the old Fourier Neural Operator repository as the main project.

Recommended structure:

```text
# Time-Scaled PDE Surrogate Operator Learning

## Overview
## Theory-to-code map
## Model 1 / Model 2 / Model 3
## Burgers target and surrogate PDEs
## Time-scaled residual and Delta_scale
## Installation
## Data generation
## Running Model123
## Running time-scaled error decomposition
## Running parameter sweeps
## Output schema
## Tests
## Removed legacy code
```

Include short example commands for:

```bash
python scripts/generate_burgers_1d.py ...
python model123_burgers_1d.py --model model1 ...
python model1_error_decomposition_1d.py --ttilde-values 0.8,1.0,1.2 ...
python scripts/run_model123_param_sweep.py --sweep Ttilde=0.8,1.0,1.2 ...
pytest -q
```

---

## Tests to add or update

### Time-scaled defect formula tests

Create/update tests that verify:

1. Burgers scaled defect at `alpha=1` equals the old unscaled formula.
2. Burgers scaled defect is zero for arbitrary states when effective coefficients match, for example:

```text
target_nu = 0.05
res_burgers_nu = 0.025
res_burgers_b = 0.5
alpha = 2.0
```

because `alpha*res_burgers_nu = target_nu` and `alpha*res_burgers_b = 1`.

3. RD and KS scaled defects match explicit `F_target - alpha*F_surrogate` formulas.

### Time-scaled error decomposition tests

Verify:

- Summary rows include `alpha`, `Delta_scale`, and theorem-consistent `rhs_beta`.
- Primary summary rows do not include `Delta_time` unless it is clearly named `legacy_*`.
- For `beta_mode=zero` and full-state identity setting:

```text
rhs_beta = sqrt(T) * Delta_scale
```

- For `Ttilde != T`, the defect is evaluated along `r(alpha*s)`, not native `r(s)`.

### Import side-effect tests

Core modules should import without launching training, loading data, or running experiments.

Test imports for:

```text
pol.burgers_spectral_1d
pol.reservoir_1d
pol.ridge
pol.elm
pol.features_1d
pol.model123_1d.metrics
pol.model123_1d.predictors
pol.model123_1d.error_decomposition
model123_burgers_1d
model1_error_decomposition_1d
scripts.run_model123_param_sweep
```

### Sweep tests

Update tests to target `scripts/run_model123_param_sweep.py`, not deleted `run_model123_nu_sweep.py` or `run_model123_ttilde_sweep.py`.

### Removed module tests

Delete tests that only cover removed legacy modules, such as tests for `pol.encoder_1d`, unless the module is intentionally retained.

---

## Development rules

1. Do not add hidden training or data-loading side effects at import time.
2. Prefer explicit names over ambiguous legacy names.
3. Preserve backward-compatible aliases only when cheap and clearly documented.
4. Use absolute discrete `L_h^2` as the main metric.
5. Use relative error only as a secondary metric.
6. Keep the code CPU-friendly for tests.
7. Mark expensive tests as `slow`.
8. Do not expand the project with unrelated dependencies.
9. Do not implement general finite-dimensional `Q_J` or `Delta_obs^(J)` in this refactor.
10. The final code should be understandable as a direct implementation of the time-scaled TeX theory.

---

## Final verification checklist

Before finishing, check:

```bash
pytest -q
```

If slow tests are marked and the full suite is expensive, also check:

```bash
pytest -q -m "not slow"
```

Also manually inspect that:

- `error_decomposition.py` has `scaled_generator_defect` and uses `Delta_scale` as primary.
- `model1_error_decomposition_1d.py` prints `alpha` and `Delta_scale`.
- `README.md` no longer describes FNO as the main project.
- `requirements.txt` no longer includes `torchvision`.
- Legacy source files listed above are not present in the active tree.
- `run_model123_param_sweep.py` records absolute and relative metrics.
