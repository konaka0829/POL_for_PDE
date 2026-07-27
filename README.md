# Surrogate-PDE Feature Maps for PDE Operator Learning

This repository is the active Paper 1 codebase for PDE solution-operator
approximation with a fixed surrogate PDE as a dynamic feature generator. It
contains E0 numerical/finite-input validation, E1 heat-to-heat calibration,
and E2 validation-only surrogate selection followed by frozen test evaluation.
E3–E7 are not implemented here.

## Install and test

Python 3.10 or newer is required. The canonical development installation is:

```bash
python3 -m pip install -e '.[test]'
python3 -m compileall -q pol tests
python3 -m pytest --collect-only -q
python3 -m pytest -q
```

The `test` extra includes `build>=1.2`, which is required by the wheel/sdist
tests.

## Run Paper 1

The unified runner is the normal entry point:

```bash
pol run configs/runs/paper1_e0_smoke.json
pol run configs/runs/paper1_e1_smoke.json
pol run configs/runs/paper1_e2_smoke.json
pol run configs/runs/paper1_e2_main.json --plan
```

Production profiles are not run implicitly.

## Compute and plot requests

Compute identity contains canonical scientific configuration identities,
scientific dependency identities, protocol versions, and compute-affecting
options. Raw source paths, source-file byte layout, run-spec whitespace, and
plot settings are provenance only.

Plot requests are independent:

```bash
pol run configs/runs/paper1_e1_smoke.json --plots-only
```

Changing DPI, format, or another valid plot setting reuses verified compute
artifacts. Plot failure is recorded as a plot-request failure and does not
change a valid compute status. A byte or semantic compute-artifact failure
marks compute invalid and prevents plotting.

Normal reuse and `--plots-only` perform artifact-only validation and never
rerun E0 or experiment solvers. Explicit verification is available as:

```bash
pol verify /path/to/run
pol verify /path/to/run --deep
```

Only `--deep` reruns E0 numerical checks.

## Architecture and invariants

Scientific recipes live below `pol.paper1`; orchestration and publication live
in `pol.runtime`, `pol.workflow`, and `pol.plots`. Reusable Burgers, ETDRK4,
and GRF routines live in the neutral, import-light `pol.numerics` package.

`n_ref`, `n_tar`, `n_sur`, `J`, and `q` are independent. The relevant
representability conditions are `J <= n_sur` and `q <= n_tar`; there is no
general `n_tar <= J` rule. Surrogate inputs are constructed only from the
finite `n_tar` input. Test labels do not participate in selection, seeds,
representatives, convergence, or plan freeze.

## Refactoring status

Phases 1–6 are complete:

1. unified runner;
2. common infrastructure;
3. E2 responsibility split;
4. generic matrix and artifact-only plots;
5. compatibility-script removal;
6. neutral numerics extraction and Model123/time-scaled separation.

The removed Model123/time-scaled research is recoverable from Git tag
`pre-phase6-model123`. See
[`docs/phase6_legacy_migration.md`](docs/phase6_legacy_migration.md).
