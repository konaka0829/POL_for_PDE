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

## Time-Scaled Residual and Delta_scale

For `Ttilde != T`, the residual is evaluated along `r_alpha(s)`, not along native surrogate times `s`. Linear interpolation is used when `alpha*s` does not land exactly on the native `dt` grid.

Primary output fields use `Delta_scale`. `Delta_dyn` may appear only as a compatibility alias for `Delta_scale`.

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

## Running Model123

```bash
python model123_burgers_1d.py --model model1 --data-file data/burgers_model123.mat --T 1.0 --Ttilde 1.0 --reservoir burgers
python model123_burgers_1d.py --model model2 --data-file data/burgers_model123.mat --obs points --J 32
python model123_burgers_1d.py --model model3 --data-file data/burgers_model123.mat --elm-h 512
```

Runs write `run_config.json` with `main_metric = "abs_l2h"`, `train_absL2h`, `test_absL2h`, `train_relL2`, and `test_relL2`.

## Running Time-Scaled Error Decomposition

```bash
python model1_error_decomposition_1d.py --ttilde-values 0.8,1.0,1.2 --T 1.0 --reservoir burgers --beta-mode zero
```

Printed summaries include `Ttilde`, `alpha`, `D1`, `Delta_scale`, `rhs_beta`, `beta`, and `mode`.

## Running Parameter Sweeps

```bash
python scripts/run_model123_param_sweep.py --sweep Ttilde=0.8,1.0,1.2 --data-file data/burgers_model123.mat --models model1,model2,model3
```

The unified sweep supports `Ttilde`, `res_burgers_nu`, `res_burgers_b`, `rd_nu`, `rd_alpha`, `rd_beta`, `ks_b`, `ks_eta`, `ks_kappa`, `dt`, `K`, and `J`. It ranks and plots by `test_absL2h` while retaining relative metrics in CSV/JSON.

## Output Schema

Error decomposition per-sample rows include:

```text
sample_index, T, Ttilde, alpha, D1_abs_l2h,
Delta_init_abs_l2h, Delta_scale_abs_l2h,
rhs_beta0_pathwise_abs_l2h, rhs_beta_pathwise_abs_l2h,
beta_mode, beta_value, c_beta_T
```

Summary rows include:

```text
Ttilde, alpha, num_samples, D1, Delta_init,
Delta_scale, rhs_beta0, rhs_beta, beta_mode, beta_value, c_beta_T
```

## Tests

```bash
pytest -q
pytest -q -m "not slow"
```

## Removed Legacy Code

Legacy FNO, LowRank, Darcy, Navier-Stokes, image, MATLAB generator, and old reservoir/RFM experiment files are not part of the active implementation. See `docs/legacy_removed.md` for the removed groups.
