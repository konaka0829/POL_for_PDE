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

## Paper 1 E0 acceptance gate

The current Paper 1 E0 is a fail-fast validation of Fourier conventions,
reference convergence, the finite-input interface, and the fixed Model 1
decoder. Run the short wiring check with:

```bash
python scripts/paper1/run_e0.py \
  --config configs/paper1_e0_smoke.json \
  --output-dir outputs/paper1_e0_smoke \
  --overwrite
```

Run the scientific calibration separately with
`configs/paper1_e0_main.json`. The smoke tolerances only validate wiring and
are not the production accuracy criterion. Exit status zero means every
required check passed; status one means at least one acceptance check failed.
An existing output is rejected before computation unless `--overwrite` is
given.

The output contains `e0_summary.json`, convergence CSV/JSON,
`resampling_checks.json`, `input_interface_checks.json`,
`model1_identity.json`, the hash-validated `master_initial_conditions.pt` and
`master_manifest.json`, the resolved config, and environment metadata. A
passing E0 also writes `accepted_production_config.json`, containing only the
joint-validated reference resolution and time-integration settings; a failed
E0 never writes this file. The convergence artifact records spatial,
temporal, and selected-pair joint checks plus per-run solver-cache statistics.

Reuse exactly the same master fields for production data generation with:

```bash
python scripts/paper1/generate_master_dataset.py \
  --config outputs/paper1_e0_main/accepted_production_config.json \
  --master-initial-conditions outputs/paper1_e0_main/master_initial_conditions.pt \
  --output-dir outputs/paper1_master \
  --overwrite
```

The generator validates sample IDs/count, domain, seed, GRF parameters,
dtype, maximum resolution, schema, and tensor hash before use. If the selected
production `reference_nx` is smaller than the archive resolution, it derives
the field by spectral low-pass resampling rather than regenerating the GRF.

`scripts/run_e0_smoke_suite.py` is a legacy suite for static Model 2/3 ridge,
headroom, and generator-defect experiments. It does not implement or validate
the current Paper 1 E0.

## Paper 1 E1 heat calibration

Run the prerequisite E0 and the short heat-to-heat calibration with:

```bash
python scripts/paper1/run_e0.py --config configs/paper1_e0_for_e1_smoke.json --output-dir outputs/paper1_e0_for_e1_smoke --overwrite
python scripts/paper1/run_e1.py --config configs/paper1_e1_smoke.json --e0-dir outputs/paper1_e0_for_e1_smoke --output-dir outputs/paper1_e1_smoke --overwrite --torch-threads 1
```

E1 records validation-only ridge selection, stable/unstable multiplier and
readout diagnostics, identifiability, noise repeats, prerequisite hashes,
machine-readable checks, plots, and a SHA-256 artifact manifest.

E1 is a heat-to-heat Model 2 calibration using the exact Fourier semigroup
(`spectral_exact`).  Stable means `nu*T - nu_tilde*T_tilde > 0`; unstable
means the inverse-diffusive readout multiplier grows with wavenumber.  The
only input reaching the surrogate is the finite path `n_ref -> n_tar ->
n_sur`. Ridge candidates use training and validation data only; the test split
is evaluated once after selecting zeta. Mode-wise training variance records
identifiability. A singular unregularized covariance is represented by a null
regularized condition number plus an explicit infinite boolean, not JSON
infinity.

The zero-ridge candidate uses an explicit thin-SVD minimum-norm solve with
cutoff `eps(dtype) * max(N,J) * sigma_max` unless `ridge_svd_rcond` is set.
This avoids a LAPACK-driver-dependent rank-deficient `lstsq` solution. Ridge
selection sees only train and validation arrays; candidates within
`ridge_tie_tolerance` use the configured deterministic `largest_zeta` tie
break. Test data is evaluated only after the model has been fixed.

Every profile evaluates the same scientific acceptance metrics: the
identifiable nonconstant-mode fraction, maximum identifiable diagonal
relative error, identifiable off-diagonal relative norm, stable/unstable
operator-norm direction and monotonicity, and the clean-field error divided by
the Fourier representation floor. Thresholds are explicit in the E1 config;
`main` uses the paper acceptance thresholds while `smoke` uses looser wiring
thresholds without skipping the calculations.

Before exit zero, E1 rereads every CSV, JSON, and selected-model tensor,
rejects non-finite values, verifies exact Cartesian key sets and plot files,
checks the expected artifact set, and verifies final byte sizes and SHA-256
records. The E0 prerequisite gate cross-checks all E0 schemas and nested
statuses, selected convergence settings, accepted production config, master
metadata, and tensor hashes. Exit code zero therefore means both numerical
and saved-artifact checks passed; failures retain best-effort summary,
environment, and failure records and return nonzero.

Production reuses the Paper 1 E0 master (it does not create an E1-specific
master):

```bash
python3 scripts/paper1/run_e0.py \
  --config configs/paper1_e0_main.json \
  --output-dir outputs/paper1_e0_main \
  --overwrite

python3 scripts/paper1/run_e1.py \
  --config configs/paper1_e1_main.json \
  --e0-dir outputs/paper1_e0_main \
  --output-dir outputs/paper1_e1_main \
  --overwrite \
  --torch-threads 1
```

For a numeric-only smoke, add `--skip-plots`. Its `plot_manifest.json` has
status `skipped`, and the output must contain no image files.

The main heat products are deliberately mild (`0.01*0.01` target and
`0.005*0.01`/`0.015*0.01` surrogates): at q=65 the extreme multipliers are
about 0.132 and 7.55, avoiding the former overflow/underflow regime. The run
writes the six CSV tables, selected models, E0 prerequisite report, data and
resolved-config metadata, environment/failure/plot summaries, E1 summary,
and a read-back-verified artifact manifest. The checked-in smoke is a wiring
and reduced scientific integration test; it is not a production result.
The principal artifacts are the six CSV tables, `selected_models.pt`,
`data_manifest.json` (split, source, config, prerequisite, and canonical model
hashes), `e0_prerequisite.json`, `plot_manifest.json`, `e1_summary.json`, and
`artifact_manifest.json`. Production continues to reuse the master generated
by `configs/paper1_e0_main.json`; no separate E1 production master is used.

### E1 resolution sweep

The three E1 resolutions are independent except for the actual sampling and
Fourier-band requirements. They satisfy
`1 <= n_tar <= n_ref`, `1 <= J <= n_sur`, and full observation means
`J = n_sur`. For every requested `q = 2K+1`, exact calibration additionally
requires `K < n_tar/2` and `K < J/2`. There is no general ordering constraint
between `n_tar` and `J`.

The v2 sweep spec uses Cartesian `grid` axes plus optional `fixed` dimensions.
It deduplicates `(n_tar,n_sur,J)` while retaining `experiment_names` as a JSON
array string in aggregate CSVs. The default experiments cover full-observation
`n_tar` by `n_sur`, fixed `n_sur=512` `n_tar` by `J`, and `n_sur` sweeps at
fixed `n_tar=256` for `J=65` and `J=96`.

Run, inspect, or regenerate plots with:

```bash
python3 scripts/paper1/run_e1_sweep.py --jobs 2 --torch-threads 1
python3 scripts/paper1/run_e1_sweep.py --dry-run
python3 scripts/paper1/run_e1_sweep.py --plot-only
```

Re-running the normal command resumes from every run whose
`e1_summary.json` has `status: "pass"`. Per-run E1 figures are omitted by
default; add `--with-per-run-plots` to create them. Aggregate CSVs are rebuilt
from successful run artifacts and the 14 aggregate plot families are generated
automatically in PNG and PDF.

## Paper 1 E2 surrogate parameters and readout time

E2 compares Models 1--3 on identical finite Burgers inputs, sample IDs,
splits, surrogate states, and fixed-\(J\) observations while sweeping
surrogate viscosity and readout time for Burgers and reaction--diffusion
families. E0 and a hash-validated Burgers master dataset are direct
prerequisites; E1 output is not a runtime dependency.

All ridge, Model 3 candidate, viscosity, time, final sweep resolution, and
shared representative choices use validation/non-test data only.
`model_specific_optima.json` records each
model's independent coordinate path separately from the representative-model choice in
`shared_representatives.json`. The enforced order is train/validation
selection, shared representative, non-test-only convergence, any finer-pilot
reruns, atomic selection and complete evaluation-plan freeze with read-back
verification, and only then one test evaluation from the loaded plan. A
rejected pilot never generates a test state or feature. If no resolution has a
strictly finer confirmation, E2 stops before freeze/test and publishes only
validation, convergence, and structured failure evidence. The test evaluator
is reconstructed from the read-back `frozen_evaluation_plan.pt`; in-memory
selection objects are not an inference input. Model 3 candidates are selected
as `(width, weight scale, bias scale, zeta)` by the mean across selection
seeds; the selected zeta is common to every selection/evaluation seed.
Disjoint evaluation seeds provide the reported Student-t seed confidence
interval.

Convergence sample IDs are checked against the actual shuffled dataset split,
not against a numeric ID range. Selection state/features contain only
train/validation samples; test surrogate states are first generated after the
selection record has been atomically written and read back. The reaction--
diffusion `two_thirds` convention filters only the sampled cubic spectrum, not
the linear reaction term, and is not described as exact cubic de-aliasing.

After shared points are selected, E2 compares terminal states on a common
spectral grid, fixed-\(J\) features, and predictions from a finest-resolution
frozen readout. Passing family bases and their global maximum are written to
`e2_handoff.json` for E3 only for a passing run; scientific or procedural
failure has nonzero exit status and no handoff. Content-addressed state and
feature caches are separate and shared across Models 1--3. `--resume` accepts
only hash-verified complete output/cache artifacts bound to the currently
supplied canonical config, E0 summary/config/archive, dataset
manifest/payload, dataset/split and sample IDs, protocol version, and plot
policy. Relocating identical content is allowed; missing or changed inputs are
rejected. Partial resume uses only validated cache units. `--overwrite` and
`--resume` are mutually exclusive.
Cache keys use canonical JSON and the v2 cache schema; state and fixed-\(J\)
feature units are committed atomically and partial/tampered units are rejected
under `--resume`.

The frozen-plan content hash covers tensor hashes, shapes and dtypes as well
as decoder conventions, physical coordinates, ridge diagnostics, Model 3
activation/seeds/candidate metadata, final pilot, and the complete input hash
chain. Every test row, summary, and passing E3 handoff carries that hash.
Formal Model 3 selection-seed statistics and ensemble-prediction diagnostics
use separate column prefixes.

Final artifacts are built in an attempt-staging directory, validated against
an explicit pass/fail and plot-policy contract, manifested once, and then
published. A pre-test failure therefore has no frozen plan, test CSV,
test-model archive, test plot, or `e2_handoff.json`. Unknown top-level files,
directories, stale plots, test tables, or handoffs cause resume validation to
fail rather than being incorporated into the manifest.

Smoke:

```bash
python3 scripts/paper1/run_e0.py \
  --config configs/paper1_e0_smoke.json \
  --output-dir outputs/paper1_e0_smoke --overwrite
python3 scripts/paper1/generate_master_dataset.py \
  --config outputs/paper1_e0_smoke/accepted_production_config.json \
  --master-initial-conditions outputs/paper1_e0_smoke/master_initial_conditions.pt \
  --output-dir outputs/paper1_master_burgers_smoke --overwrite
python3 scripts/paper1/run_e2.py \
  --config configs/paper1_e2_smoke.json \
  --e0-dir outputs/paper1_e0_smoke \
  --dataset-dir outputs/paper1_master_burgers_smoke \
  --output-dir outputs/paper1_e2_smoke --overwrite --torch-threads 1 \
  --batch-size 64
```

Future production uses the same commands with `paper1_e0_main.json`,
`paper1_e2_main.json`, and production output directories. The checked-in main
parameter grids are initial candidates and must be validated; this repository
does not treat them as pre-established optima. Inspect structural cost without
running the production sweep:

```bash
python3 scripts/paper1/run_e2.py \
  --config configs/paper1_e2_main.json --dry-run-cost
```

Use the same three-stage E0, master-dataset, E2 sequence for the eventual
production run. `--skip-plots` is a debugging option only and must not be used
for the final paper run. Principal E2 artifacts are the
validation/test tables, Model 3 per-seed and aggregate tables, frozen selection
record, model-specific/shared selections, convergence tables, E3 handoff,
plots, environment, summary, and final SHA-256 manifest. The legacy root
Model123 calibration suite is a separate compatibility experiment and does not
implement this Paper 1 prerequisite, freeze, convergence, or handoff protocol.

## Data Generation

Generate a smoke `.pt` dataset from the checked-in B0 config:

```bash
python scripts/generate_burgers_1d.py \
  --config configs/B0_smoke.json \
  --format pt \
  --output-dir outputs/smoke_data
```

Generate a larger GRF Burgers dataset from the B1 config:

```bash
python scripts/generate_burgers_1d.py \
  --config configs/B1_burgers_grf.json \
  --format pt \
  --out-file data/burgers_B1_grf.pt
```

The runner supports both `.mat` datasets with `a/u` arrays and `.pt`
datasets containing either raw `a/u` tensors or pre-split
`u0_train/y_train`, `u0_val/y_val`, and `u0_test/y_test`.
Dataset metadata is validated by default against the resolved CLI/config
settings. Mismatches in `T`, `dt`, `nx`, `target_nu`, `ic_type`,
`domain_length`, solver/time-integrator metadata, or dealias metadata raise an
error. Missing metadata is a warning by default for legacy compatibility; add
`--require-complete-metadata` for B1/paper runs to make missing fields an
error. Use `--allow-metadata-mismatch` only for intentional legacy runs; the
warning and validation result are written to `run_config.json`.

For spatial subsampling, metadata `nx` means the raw dataset resolution before
`--sub`. The model uses `effective_nx = dataset_nx / sub` after subsampling.
`run_config.json` records `grid.dataset_nx`, `grid.raw_nx`,
`grid.effective_nx`, `grid.sub`, `grid.domain_length`, and `grid.dx`;
metadata validation compares `nx` against the raw dataset resolution, not the
effective model resolution. The spatial convention is
`dx = domain_length / effective_nx` for discrete L2h quantities and
`kappa_k = 2*pi*k/domain_length` for Fourier pseudo-spectral derivatives.
For `domain_length != 1`, the current GRF and Fourier initial-condition
samplers are specified on the normalized periodic coordinate `xi = x/L`.
Equivalently, sampled physical profiles are interpreted as
`u(x) = u_tilde(x/L)`. PDE solvers and reservoirs use physical Fourier
wavenumbers `kappa_k = 2*pi*k/domain_length`, but the random IC distribution is
stored in normalized-coordinate mode via metadata
`ic_coordinate_convention = normalized_periodic_coordinate_x_over_L`.
This field is written to `.pt` metadata and `.mat` files; legacy datasets may
omit it, so missing values remain a metadata warning unless a future workflow
chooses to make the field mandatory. B1 currently uses `domain_length=1.0`, so
the normalized-coordinate and physical-domain interpretations coincide there.
A future physical-domain GRF sampler should use a distinct metadata value and
define correlations directly with physical wavenumbers
`kappa_m = 2*pi*m/domain_length`; unlike the current convention, changing `L`
would then change the physical correlation-length interpretation rather than
stretching a fixed profile distribution.

Generate a MATLAB file compatible with `model123_burgers_1d.py`:

```bash
python scripts/generate_burgers_1d.py --out-file data/burgers_model123.mat --num-samples 1200 --grid-size 256 --nu 0.05 --T 1.0 --dt 0.001
```

The `.mat` output includes `a`, `u`, `T`, `dt`, `nu`/`target_nu`, `nx`,
`domain_length`, string metadata such as `ic_type` and `solver`, and
`num_samples`. Numeric PDE fields and metadata are read through separate paths,
so `.mat` string metadata is not cast to float. For `.pt` output, the saved
bundle contains exactly the requested train/validation/test split plus
normalized metadata when available. Burgers split-step and ETDRK4 solvers use
`domain_length` in the Fourier wavenumbers; B0/B1 use `domain_length=1.0`.
The same Fourier convention is used by surrogate reservoir families, including
heat, advection, Burgers, reaction-diffusion, and KS reservoirs.

## Running Model123

```bash
python model123_burgers_1d.py --model model1 --data-file data/burgers_model123.mat --T 1.0 --Ttilde 1.0 --reservoir burgers
python model123_burgers_1d.py --model model2 --data-file data/burgers_model123.mat --obs points --J 32
python model123_burgers_1d.py --model model3 --data-file data/burgers_model123.mat --elm-h 512
```

Runs write `run_config.json` with `main_metric = "abs_l2h"`, `train_absL2h`, `test_absL2h`, `train_relL2`, and `test_relL2`.
Current runs also include `val_absL2h` when `--nval > 0`,
`relL2_mean`/`relL2_agg`, split hashes, metadata validation results, config
hashes, git/runtime metadata, command line, and dtype information. Absolute
L2h metrics, ridge diagnostics, zeta-path objectives, learning-curve metrics,
and headroom diagnostics all use `dx = domain_length/effective_nx`. Ridge
metadata records the legacy equivalent parameter as `N*zeta/dx`, not
`N*zeta*effective_nx` unless `domain_length=1`. Use
`--data-dtype {preserve,float32,float64}`, `--sim-dtype {float32,float64}`,
and `--ridge-dtype {float32,float64}` to control data tensors, simulation
features, and ridge solves separately.
`--sim-dtype float64` is used by the surrogate feature generation in the main
runner, zeta-path, and learning-curve scripts; the feature cache key includes
the simulation dtype, `domain_length`, `effective_nx`, `dx`, and `sub`, so
float32/float64 and L=1/L!=1 cached features do not collide.
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

The unified sweep supports `alpha`, `Ttilde`, `res_burgers_nu`,
`res_burgers_b`, `rd_nu`, `rd_alpha`, `rd_beta`, `ks_b`, `ks_eta`,
`ks_kappa`, `heat_nu`, `advection_c`, `dt`, `K`, and `J`. Reservoir choices
include `burgers`, `reaction_diffusion`, `ks`, `static`, `heat`, and
`advection`; Burgers reservoirs can use `--burgers-scheme etdrk4`.
If `--Ttilde` is not specified, sweeps resolve `Ttilde = T`. For `alpha`
sweeps, each child run uses `Ttilde = alpha*T`; the old implicit `Ttilde=1.0`
default is no longer used.

When validation data is available, hyperparameter and sweep-setting selection
uses `val_absL2h`. `test_absL2h` is retained for final evaluation after
selection and must not be used as the formal selection metric. Legacy runs
without validation are marked with `test_absL2h_legacy_fallback`. Sweep
`summary.csv/json` and `best_runs.csv/json` store the row-level
`selection_metric`.

Ridge regularization is reported as `ridge_zeta` with convention
`normalized_empirical_l2h_unweighted_frobenius`; legacy `--ridge-lambda` is
accepted as an alias and recorded separately in `run_config.json`.
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
Defect diagnostics resolve target viscosity in this order:
`--defect-target-nu`, resolved `--target-nu`/config value, dataset metadata
`target_nu` or `nu`, then a warning fallback of `0.05`. Both
`time_scaled_defect_metrics.json` and `run_config.json` record
`target_nu_source`.
When target and surrogate coefficients match exactly, integrated defect values
can be identically zero. Alpha-parameter defect plots fall back to a linear
y-axis for all-zero/non-positive finite values to avoid log-scale warnings;
positive finite values keep the previous log-scale behavior.

Both `run_model123_param_sweep.py` and
`run_model123_alpha_param_defect_study.py` support strict existing-result
audits:

```bash
python scripts/run_model123_param_sweep.py ... --check-existing
python scripts/run_model123_alpha_param_defect_study.py ... --check-existing
python scripts/run_model123_param_sweep.py ... --skip-existing --reuse-report summary
```

The audit compares the stored `run_config.json` against the expected data
hash, config hash, split seeds, model, reservoir parameters, ridge settings,
dtype settings, and solver settings. `--check-existing` is audit-only: it does
not launch jobs and does not overwrite normal sweep outputs such as
`summary.csv/json`, `best_runs.csv/json`, profile plots, or comparison plots.
It writes only audit outputs:
`existing_audit.csv`, `existing_audit.json`,
`existing_audit_summary.csv`, and `existing_audit_summary.json`, and returns
exit code `2` for missing or mismatched results. `--skip-existing
--reuse-report summary` remains a normal sweep/reuse path and may write normal
summary files.

## Validation-Selected Zeta, Learning Curves, and Headroom

Run a zeta path with cached features:

```bash
python scripts/run_zeta_path.py \
  --config configs/B0_smoke.json \
  --data-file outputs/smoke_data/burgers_model123.pt \
  --model model2 \
  --reservoir static \
  --zeta-grid 1e-8,1e-6,1e-4 \
  --use-feature-cache \
  --output-dir outputs/smoke_zeta_path
```

The best zeta is selected by `val_absL2h`; cache metadata includes dataset,
split, surrogate, observation, feature shape, feature hash, `domain_length`,
`effective_nx`, `dx`, and `sub` information.
On cache hits the script reads `metadata.json` back into `run_config.json`,
including feature paths and dtype. zeta-path, learning-curve, and headroom use
the same dataset loading and metadata validation policy as the main runner.
`scripts/run_learning_curve.py` reuses the same zeta-path machinery while
fixing validation/test splits and varying only the training subset size.

Fourier diagonal linear headroom is available via:

```bash
python scripts/run_headroom_burgers.py \
  --config configs/B0_smoke.json \
  --data-file outputs/smoke_data/burgers_model123.pt \
  --zeta-grid 1e-8,1e-6,1e-4 \
  --output-dir outputs/smoke_headroom
```

This also selects zeta by validation error and reports `headroom_H` and
`linear_explained_variance`.

## E0-E3 Suite Entry Points

E0 smoke suite runs a compact B0 workflow: dataset generation/reuse, static
Model 2/3 validation-selected zeta paths, Fourier diagonal headroom, and one
matching-coefficient Model 1 Burgers defect check.

```bash
python scripts/run_e0_smoke_suite.py \
  --config configs/B0_smoke.json \
  --output-dir outputs/e0_smoke \
  --use-feature-cache
```

For E1/E2/E3 suites, use `--max-workers N` to run independent suite rows
concurrently. Each worker launches a child Python process; the suite clamps
BLAS/Torch thread environment variables to 1 to avoid CPU oversubscription.
Avoid duplicate grid entries that would write to the same output directory.

E1 baseline suite compares static, heat, and advection reservoirs under the
same split and validation-only zeta selection:

```bash
python scripts/run_baseline_suite.py \
  --config configs/B0_smoke.json \
  --data-file outputs/e0_smoke/data/burgers_model123.pt \
  --models model2,model3 \
  --reservoirs static,heat,advection \
  --zeta-grid 1e-8,1e-6,1e-4 \
  --output-dir outputs/baseline_suite \
  --use-feature-cache
```

E2 Burgers calibration suite records effective coefficients
`effective_nu = alpha*res_burgers_nu` and
`effective_b = alpha*res_burgers_b`, plus mismatch columns against the target
Burgers generator:

```bash
python scripts/run_burgers_calibration_suite.py \
  --config configs/B0_smoke.json \
  --data-file outputs/e0_smoke/data/burgers_model123.pt \
  --models model1,model2,model3 \
  --alpha-values 0.8,1.0,1.2 \
  --res-burgers-nu-values 0.005,0.01,0.02 \
  --res-burgers-b-values 0.8,1.0,1.2 \
  --zeta-grid 1e-8,1e-6,1e-4 \
  --output-dir outputs/burgers_calibration \
  --compute-time-scaled-defect \
  --burgers-scheme etdrk4 \
  --burgers-dealias 0
```

E3 nonlinear surrogate suite first runs Fourier diagonal linear headroom and
then compares static/heat/advection controls with Burgers/RD/KS reservoirs.
It reports `D_lin_abs_l2h`, `improvement_over_dlin_abs`,
`ratio_to_dlin`, and `beats_dlin`. Selection within each model/reservoir group
uses validation error only; test error is reported after selection.

```bash
python scripts/run_nonlinear_surrogate_suite.py \
  --config configs/B0_smoke.json \
  --data-file outputs/e0_smoke/data/burgers_model123.pt \
  --models model2,model3 \
  --reservoirs static,heat,advection,burgers,reaction_diffusion,ks \
  --alpha-values 0.8,1.0,1.2 \
  --zeta-grid 1e-8,1e-6,1e-4 \
  --output-dir outputs/nonlinear_surrogate_suite \
  --use-feature-cache
```

## ETDRK4 Solver Check

`pol/spectral_etdrk4_1d.py` implements Cox--Matthews ETDRK4 for periodic
Burgers with nonlinear-term dealiasing. The coefficient implementation has
unit tests for the `L=0` RK4 limit and the `N=0` exact exponential limit.
Time-scaled defect diagnostics also support `--burgers-scheme etdrk4`; target
and Burgers surrogate trajectories use the same physical `domain_length`, `dt`,
dealiasing, dtype/device, and viscosity conventions.
Run the small convergence smoke check with:

```bash
python scripts/check_solver_convergence.py \
  --config configs/B0_smoke.json \
  --output-dir outputs/smoke_solver_check
```

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
rhs_beta_theorem_components, rhs_beta_pathwise_rms,
rhs_beta_legacy_alias_of,
beta_mode, beta_value, c_beta_T
```

For `beta_mode=analytic_safe_poincare`, the Poincare correction uses the first
nonzero periodic physical wavenumber on `[0,L]`:

```text
poincare_shift = -target_nu * (2*pi/domain_length)^2
```

Model123 runs with `--compute-time-scaled-defect` additionally write
`time_scaled_defect_metrics.json`, `time_scaled_defect_per_sample.csv/json`,
and `error_vs_defect_scatter.{png,pdf,svg}`. The metrics include
`corr_error_delta_scale_pearson`, `corr_error_delta_scale_spearman`, and for
Model 1 `max_abs_difference_model1_D1`, which checks that the Model 1 prediction
error and `D1_model1_abs_l2h` were computed from the same surrogate trajectory.
`time_scaled_defect_metrics.json` is self-describing for the L2h convention: it
records `domain_length`, `effective_nx`, `dx`, and
`l2h_convention = dx=domain_length/effective_nx`.

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
