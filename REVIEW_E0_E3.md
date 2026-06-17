# E0-E3 Code Review

## Verdict

FAIL: CRITICAL ISSUES FOUND

The core time-scaled residual convention is preserved, and the E1/E2/E3 suite selection paths use validation metrics after the small fixes applied during review. However, E3 exposes `--compute-spectral-error` without actually producing the requested spectral diagnostics, so the E0-E3 implementation is not complete.

Small fixes applied during this review:

- `scripts/suite_common.py`: `best_by_validation` now raises if `val_absL2h` is missing instead of falling back to test error; zeta suite rows now recover `data_hash` from feature cache metadata when `run_config.json` lacks it.
- `scripts/run_burgers_calibration_suite.py`: E2 coefficient columns now honor `--defect-target-nu`.
- `scripts/run_nonlinear_surrogate_suite.py`: E3 defect reruns are limited to reservoirs supported by the defect implementation (`burgers`, `reaction_diffusion`, `ks`), while static/heat/advection rows are marked `defect_status=not_applicable`.
- `tests/test_suite_scripts.py`: added coverage that suite best-selection requires a validation metric.

## Critical Issues

### 1. E3 spectral error CLI is a stub, not an implementation

- Severity: high
- File and lines: `scripts/run_nonlinear_surrogate_suite.py:93-105`, `scripts/run_nonlinear_surrogate_suite.py:364-370`
- What is wrong: `--compute-spectral-error` only writes `spectral_error_manifest.json` saying the helper is available and predictions are not persisted. It does not compute spectral errors for selected best rows, and it does not write `spectral_error_<model>_<reservoir>.csv` or plots.
- Why it matters: E3 is supposed to compare nonlinear surrogate errors against linear headroom and optionally diagnose which Fourier modes improve or fail. The current flag gives users a successful exit code but no requested spectral diagnostic artifact, which is experimentally misleading.
- Concrete fix: Persist predictions/targets for selected rows, or rerun selected model/reservoir settings in-process to obtain predictions; then call `spectral_error_rows`, write per-best-row CSV/JSON, and save png/pdf/svg plots. If that is too much for this phase, remove or hide `--compute-spectral-error` rather than advertising a nonfunctional option.

### 2. Defect metadata mislabels the target trajectory solver for dataset-based diagnostics

- Severity: medium
- File and lines: `model123_burgers_1d.py:823-841`
- What is wrong: `time_scaled_defect_metrics.json` records `"target_trajectory_solver": args.burgers_scheme`, but `compute_time_scaled_defect_for_dataset` uses the dataset-provided final target `target_T`; it does not regenerate the target trajectory with `args.burgers_scheme`. If the dataset was generated with ETDRK4 and the user runs defect diagnostics with a different surrogate Burgers scheme, this field can be wrong.
- Why it matters: E0 specifically requires solver-choice metadata to make ETDRK4 defect diagnostics auditable. Incorrect target solver metadata can make later aggregated defect files look like they used a different target integrator than the dataset actually used.
- Concrete fix: Pass `dataset_meta` into `compute_and_save_defect_outputs` and record target solver from dataset metadata (`solver`, `time_integrator`, or `burgers_scheme`) separately from the surrogate trajectory solver. Keep `burgers_scheme` as the surrogate Burgers integrator when applicable.

### 3. `--max-workers` is exposed but ignored by suite scripts

- Severity: medium
- File and lines: `scripts/run_baseline_suite.py:47`, `scripts/run_burgers_calibration_suite.py:61`, `scripts/run_nonlinear_surrogate_suite.py:60`
- What is wrong: E1/E2/E3 parsers expose `--max-workers`, but all jobs are launched sequentially in plain loops.
- Why it matters: This is brittle CLI behavior. Users will reasonably expect bounded parallelism for grid sweeps, especially B1-scale E2/E3 runs. Silent no-op performance flags make experiment planning unreliable.
- Concrete fix: Either implement bounded `ThreadPoolExecutor`/`ProcessPoolExecutor` subprocess scheduling with deterministic result collection, or remove the option from CLI and README until it is implemented.

### 4. Plotting failures can still fail completed suite runs

- Severity: medium
- File and lines: `scripts/run_burgers_calibration_suite.py:250-252`, `scripts/run_nonlinear_surrogate_suite.py:372-373`
- What is wrong: CSV/JSON outputs are written before plotting, but plot generation is not isolated. Any matplotlib/runtime plotting error exits the suite nonzero after successful numerical runs.
- Why it matters: The implementation requirements say plotting failures should not corrupt or invalidate machine-readable outputs. On headless or restricted environments this can turn a valid experiment into a failed suite.
- Concrete fix: Wrap optional plotting calls in a small helper that records `plot_warnings.json` and preserves CSV/JSON outputs and exit status unless the user requests strict plotting.

## Test Leakage Audit

Validation/test separation is correct for the reviewed E1/E2/E3 suite selection paths after the small review patch. `run_zeta_path.py` selects `best_by_val` using `val_absL2h`, `run_headroom_burgers.py` selects zeta by validation error, and `scripts/suite_common.py:409-418` now raises when `val_absL2h` is missing instead of falling back to `test_absL2h`.

Residual legacy fallback labels remain in `model123_burgers_1d.py` for single-run reporting when no validation split exists, but the new suite best-selection helper no longer uses that as a selection fallback.

## Time-Scaled Defect Audit

The alpha-scaled residual convention is preserved. `rescaled_surrogate_states` evaluates the surrogate at `alpha*s` (`pol/model123_1d/error_decomposition.py:463-470`), and the residual formulas use `F - alpha*Ftilde` (`pol/model123_1d/error_decomposition.py:278-316`). The deprecated `generator_defect` alias is still present but explicitly maps to `alpha=1` and is not used as the primary diagnostic path.

ETDRK4 support is wired into target trajectory generation for full synthetic error decomposition (`pol/model123_1d/error_decomposition.py:160-198`) and into Burgers reservoir simulation. The requested E2 matching-coefficient ETDRK4 smoke produced `delta_scale_rms_abs_l2h=0.0` for both model1 and model2 rows.

## domain_length / dx Audit

The main numerical conventions are in good shape:

- `dx = domain_length / effective_nx` is used by L2h metrics and ridge diagnostics.
- Fourier derivatives and reservoir multipliers use physical wavenumbers `2*pi*k/domain_length`.
- New tests cover heat multipliers, advection shifts, L2h scaling, spectral derivative amplitude, and feature-cache key separation.
- `time_scaled_defect_metrics.json` now records `domain_length`, `effective_nx`, `dx`, and `l2h_convention`.

The remaining metadata caveat is the target solver label in dataset-based defect metrics, described above.

## E1 Baseline Suite Audit

The baseline suite correctly builds the model/reservoir/alpha Cartesian product and delegates model2/model3 training to `run_zeta_path.py`, which selects by validation error. The requested smoke with `model2,model3` and `static,heat` passed and wrote two-model/two-reservoir outputs.

Issues: `--max-workers` is a no-op, and plotting/child-run failure handling is basic but serviceable for B0-scale runs.

## E2 Burgers Calibration Audit

The coefficient formulas are correct:

- `effective_nu = alpha * res_burgers_nu`
- `effective_b = alpha * res_burgers_b`
- `mismatch_nu = target_nu - effective_nu`
- `mismatch_b = 1.0 - effective_b`

The residual sign convention is consistent because the actual nonlinear residual term uses `effective_b - 1.0` in `scaled_defect_burgers_reservoir`. After the review patch, `--defect-target-nu` also affects the coefficient/mismatch columns.

The requested E2 smoke passed with matching coefficients: `effective_nu=0.01`, `effective_b=1.0`, `mismatch_nu=0.0`, `mismatch_b=0.0`, and `delta_scale_rms_abs_l2h=0.0`.

## E3 Nonlinear Surrogate Suite Audit

Headroom parsing and D_lin comparison are correct in the tested path. `normalize_headroom` refuses missing or nonpositive `test_Dlin2`, and `add_dlin_columns` computes:

- `improvement_over_dlin_abs = D_lin_abs_l2h - test_absL2h`
- `ratio_to_dlin = test_absL2h / D_lin_abs_l2h`
- `beats_dlin = test_absL2h < D_lin_abs_l2h`

The requested E3 smoke passed with `static` and `reaction_diffusion`. An additional defect-enabled E3 smoke also passed after limiting defect diagnostics to supported nonlinear reservoirs.

The blocking E3 gap is spectral error: the CLI flag succeeds but does not produce the required CSV/plot diagnostics.

## Performance and Maintainability

- Suite scripts are mostly subprocess wrappers, which is pragmatic, but repeated selected-run reruns for defect diagnostics recompute features after zeta selection.
- `--max-workers` is currently misleading because it is not used.
- The common suite helper reduces duplication, but plotting and failure handling should be centralized more cleanly.
- The broad `except Exception` blocks preserve summary output, but important child-run failures can be easy to miss unless users inspect `failed_runs.json`.

## Tests Run

- `python3 -m pip install -e '.[test]'`
  - Failed in sandbox first due network-restricted build dependency lookup.
  - Re-run with network escalation also failed because the project build backend lacks the PEP 660 `build_editable` hook and there is no `setup.py`/`setup.cfg`.
- `python3 -m compileall -q .`
  - Passed.
- `python3 -m pytest -q -m 'not slow'`
  - Passed: `122 passed, 1 skipped, 12 deselected, 2 warnings in 15.69s`.
- `python3 -m pytest -q`
  - Passed: `134 passed, 1 skipped, 2 warnings in 55.58s`.
- `python3 scripts/run_e0_smoke_suite.py --config configs/B0_smoke.json --output-dir outputs/review_e0_smoke --use-feature-cache --python python3`
  - Passed.
- `python3 scripts/run_baseline_suite.py --config configs/B0_smoke.json --data-file outputs/review_e0_smoke/data/burgers_model123.pt --models model2,model3 --reservoirs static,heat --zeta-grid 1e-8,1e-6 --output-dir outputs/review_baseline --use-feature-cache --python python3`
  - Passed.
- `python3 scripts/run_burgers_calibration_suite.py --config configs/B0_smoke.json --data-file outputs/review_e0_smoke/data/burgers_model123.pt --models model1,model2 --alpha-values 1.0 --res-burgers-nu-values 0.01 --res-burgers-b-values 1.0 --zeta-grid 1e-8,1e-6 --output-dir outputs/review_calibration --compute-time-scaled-defect --burgers-scheme etdrk4 --burgers-dealias 0 --python python3`
  - Passed.
- `python3 scripts/run_nonlinear_surrogate_suite.py --config configs/B0_smoke.json --data-file outputs/review_e0_smoke/data/burgers_model123.pt --models model2 --reservoirs static,reaction_diffusion --alpha-values 1.0 --zeta-grid 1e-8,1e-6 --output-dir outputs/review_nonlinear --use-feature-cache --python python3`
  - Passed.
- Extra review check: `python3 scripts/run_nonlinear_surrogate_suite.py --config configs/B0_smoke.json --data-file outputs/review_e0_smoke/data/burgers_model123.pt --models model2 --reservoirs static,reaction_diffusion --alpha-values 1.0 --zeta-grid 1e-8 --output-dir outputs/review_nonlinear_defect --use-feature-cache --compute-time-scaled-defect --python python3`
  - Passed.
- Extra review check: `python3 scripts/run_nonlinear_surrogate_suite.py --config configs/B0_smoke.json --data-file outputs/review_e0_smoke/data/burgers_model123.pt --models model2 --reservoirs static --alpha-values 1.0 --zeta-grid 1e-8 --output-dir outputs/review_nonlinear_spectral --use-feature-cache --compute-spectral-error --python python3`
  - Passed, but only wrote `spectral_error_manifest.json`; no spectral CSV or plots were produced.

## Recommended Fixes

1. Implement the E3 spectral error diagnostic end-to-end, or remove `--compute-spectral-error` until it can write the promised artifacts.
2. Correct dataset-based defect metadata so `target_trajectory_solver` comes from dataset metadata, not from the surrogate Burgers scheme CLI.
3. Implement or remove `--max-workers` in E1/E2/E3 suite scripts.
4. Wrap optional plotting in non-fatal warning capture.
5. Add an integration test for `--compute-spectral-error` that asserts the expected `spectral_error_<model>_<reservoir>.csv` and plots exist.
6. Add a metadata regression test where the dataset target solver differs from the requested surrogate Burgers scheme and verify `time_scaled_defect_metrics.json` labels both correctly.
