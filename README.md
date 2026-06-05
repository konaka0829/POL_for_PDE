# Time-Scaled PDE Surrogate Operator Learning

## Overview

This repository contains a slim research implementation for time-scaled surrogate PDE operator learning on 1D periodic Burgers-type problems. The active code focuses on Burgers target dynamics, Burgers/reaction-diffusion/Kuramoto-Sivashinsky surrogate reservoirs, Model 1/2/3 experiments, and the Model 1 time-scaled error decomposition.

## Theory-to-Code Map

For target final time `T` and surrogate readout time `Ttilde`, the code uses

```text
alpha = Ttilde / T
r_alpha(s) = Gtilde_{alpha*s}(u0),  s in [0,T]
R_scale(r) = F(r) - alpha * Ftilde(r)
```

The primary bound reported by `pol/model123_1d/error_decomposition.py` is

```text
D1(theta, alpha*T) <= exp(beta*T) Delta_init + c_beta_T Delta_scale
```

In the current full-state 1D setting, `Delta_init = 0`.

## Model 1 / Model 2 / Model 3

- Model 1 directly reads the surrogate state at `Ttilde`.
- Model 2 fits a ridge readout from finite-dimensional trajectory observations.
- Model 3 adds fixed random ELM features before the ridge readout.

The main runner is:

```bash
python model123_burgers_1d.py --model model1 --data-file data/burgers_model123.mat --T 1.0 --Ttilde 1.0
```

## Burgers Target and Surrogate PDEs

The target generator is

```text
F(z) = target_nu * z_xx - z * z_x
```

Supported surrogate families are:

- Burgers: `res_burgers_nu * z_xx - res_burgers_b * z*z_x`
- Reaction-diffusion: `rd_nu * z_xx + rd_alpha*z - rd_beta*z^3`
- Kuramoto-Sivashinsky style: `-ks_b*z*z_x - ks_eta*z_xx - ks_kappa*z_xxxx`

## Time-Scaled Generator Defect and Delta_scale

For `Ttilde != T`, the scaled generator defect is evaluated along `r_alpha(s)`, not along native surrogate times `s`. Linear interpolation is used when `alpha*s` does not land exactly on the native `dt` grid.

Primary numerical-study diagnostics use the sample-wise integrated quantity `delta_scale_pathwise_abs_l2h`; `Delta_scale_abs_l2h` and `Delta_dyn` may appear as compatibility aliases.
The alpha/parameter heatmap correlations use this per-sample integrated defect, not instantaneous residual fields. For Model 1 the error variable is `D1`; for Model 2/3 it is the readout prediction error, so the correlation is a diagnostic against the underlying surrogate PDE defect rather than a direct theorem bound.

## Installation

```bash
python -m pip install -e .[test]
```

or:

```bash
python -m pip install -r requirements.txt
```

## Data Generation

Generate a MATLAB file compatible with `model123_burgers_1d.py`:

```bash
python scripts/generate_burgers_1d.py --out-file data/burgers_model123.mat --num-samples 1200 --grid-size 256 --nu 0.05 --T 1.0 --dt 0.001
```

The `.mat` output includes `a`, `u`, `T`, `dt`, `nu`, `nx`, and `num_samples`.
For `.pt` output, the saved bundle contains exactly the requested split:
`u0_train/y_train` have `ntrain` samples and `u0_test/y_test` have `ntest`
samples. If no split is specified, `--num-samples N --format pt` writes
train `N` / test `0`.

## Running Model123

```bash
python model123_burgers_1d.py --model model1 --data-file data/burgers_model123.mat --T 1.0 --Ttilde 1.0 --reservoir burgers
python model123_burgers_1d.py --model model2 --data-file data/burgers_model123.mat --obs points --J 32
python model123_burgers_1d.py --model model3 --data-file data/burgers_model123.mat --elm-h 512
```

Runs write `run_config.json` with `main_metric = "abs_l2h"`, `train_absL2h`, `test_absL2h`, `train_relL2`, and `test_relL2`.
The Model123 runner and feature extraction require `Ttilde`, `Tr`, and any
explicit `--feature-times` to lie on the `dt` grid. Non-grid values such as
`--Ttilde 0.055 --dt 0.01` raise `ValueError` instead of silently rounding.
This grid-alignment rule is only for predictors/features; the time-scaled error
decomposition still uses interpolation for `r_alpha(s)` as required by the
theory.

For synthetic exact-inclusion and smoke studies, use:

```bash
python scripts/run_model123_synthetic_study.py --total-samples 120 --ntrain 100 --ntest 20
```

The old root command `python model123_error_study.py ...` remains as a backward
compatible wrapper. The main real-data runner is `model123_burgers_1d.py`.

## Running Time-Scaled Error Decomposition

```bash
python model1_error_decomposition_1d.py --ttilde-values 0.8,1.0,1.2 --T 1.0 --reservoir burgers --beta-mode zero
```

Printed summaries include `Ttilde`, `alpha`, `D1`, `Delta_scale`, `rhs_beta`, `beta`, and `mode`.

## Running Parameter Sweeps

```bash
python scripts/run_model123_param_sweep.py --sweep Ttilde=0.8,1.0,1.2 --models model1,model2,model3
```

The unified sweep supports `alpha`, `Ttilde`, `res_burgers_nu`, `res_burgers_b`, `rd_nu`, `rd_alpha`, `rd_beta`, `ks_b`, `ks_eta`, `ks_kappa`, `dt`, `K`, and `J`. It ranks and plots by `test_absL2h` while retaining relative metrics in CSV/JSON.
When `--feature-times` is not provided, `K` selects `K` positive observation times evenly over `(0, Ttilde]`; for example `K=4` uses approximately `Ttilde/4, Ttilde/2, 3*Ttilde/4, Ttilde`.
Its default data file is `data/burgers_model123.mat`, matching the generation
example above. Default `dt` is `1e-2` and default Burgers inner `fine_dt` is
`1e-4` for practical smoke and sweep startup runs; override them for higher
accuracy studies.

Alpha sweep with integrated defect diagnostics:

```bash
python scripts/run_model123_param_sweep.py \
  --models model1,model2,model3 \
  --reservoir burgers \
  --burgers-dealias 0 \
  --sweep alpha=0.6,0.8,1.0,1.2 \
  --T 1.0 \
  --compute-time-scaled-defect \
  --data-file data/burgers_model123.mat
```

Parameter sweep at fixed alpha:

```bash
python scripts/run_model123_param_sweep.py \
  --models model1,model2,model3 \
  --reservoir burgers \
  --burgers-dealias 0 \
  --T 1.0 \
  --Ttilde 1.0 \
  --sweep res_burgers_nu=0.02,0.04,0.05,0.06 \
  --compute-time-scaled-defect \
  --data-file data/burgers_model123.mat
```

Full alpha-parameter defect study:

```bash
python scripts/run_model123_alpha_param_defect_study.py \
  --models model1,model2,model3 \
  --reservoir burgers \
  --burgers-dealias 0 \
  --parameter res_burgers_nu \
  --parameter-values 0.02,0.04,0.05,0.06 \
  --alpha-values 0.6,0.8,1.0,1.2 \
  --T 1.0 \
  --data-file data/burgers_model123.mat \
  --out-root outputs/alpha_param_defect_study
```

The correlation heatmaps use `corr_i(model_error_abs_l2h_i, delta_scale_pathwise_abs_l2h_i)` across test samples at each grid cell. They are correlations with the integrated generator defect, not instantaneous field values.
For theory-consistent Burgers coefficient checks, the examples set `--burgers-dealias 0`; using `--burgers-dealias 1` is valid for dealiased numerical diagnostics, but the analytic defect magnitude need not be exactly zero even when coefficients match.

## Output Schema

Error decomposition per-sample rows include:

```text
sample_index, T, Ttilde, alpha, D1_abs_l2h,
Delta_init_abs_l2h, delta_scale_pathwise_abs_l2h,
Delta_scale_abs_l2h,
rhs_beta0_pathwise_abs_l2h, rhs_beta_pathwise_abs_l2h,
beta_mode, beta_value, c_beta_T
```

Summary rows include:

```text
Ttilde, alpha, num_samples, D1, Delta_init,
delta_scale_rms_abs_l2h, delta_scale_mean_abs_l2h,
delta_scale_std_abs_l2h, Delta_scale, rhs_beta0, rhs_beta,
beta_mode, beta_value, c_beta_T
```

Model123 runs with `--compute-time-scaled-defect` additionally write
`time_scaled_defect_metrics.json`, `time_scaled_defect_per_sample.csv/json`,
and `error_vs_defect_scatter.{png,pdf,svg}`. The metrics include
`corr_error_delta_scale_pearson`, `corr_error_delta_scale_spearman`, and for
Model 1 `max_abs_difference_model1_D1`, which checks that the Model 1 prediction
error and `D1_model1_abs_l2h` were computed from the same surrogate trajectory.

## Tests

```bash
pytest -q -m "not slow"
pytest -q
```

`pytest -q -m "not slow"` runs the lightweight unit checks. Full `pytest -q`
also runs subprocess and dataset-generation smoke tests, which are marked
`slow`, have explicit timeouts, and pin BLAS/Torch thread counts to one.

## Removed Legacy Code

Legacy FNO, LowRank, Darcy, Navier-Stokes, image, MATLAB generator, and old reservoir/RFM experiment files are not part of the active implementation. See `docs/legacy_removed.md` for the removed groups.
