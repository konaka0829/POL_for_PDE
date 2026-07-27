# Repository guidance

## Active Paper 1 research

The active project studies PDE solution-operator approximation using a fixed
surrogate PDE as a dynamic feature generator. The surrogate need not equal the
target PDE. Finite observations of its state feed three readouts:

- Model 1: fixed, untrained decoding;
- Model 2: affine ridge prediction of target Fourier coefficients;
- Model 3: fixed random nonlinear features with a learned affine output.

Paper 1 currently contains E0 (fail-fast Fourier, resampling, reference,
finite-input, leakage, and Model 1 checks), E1 (heat-to-heat calibration), and
E2 (validation-only surrogate parameter/time selection, convergence,
freeze/read-back, then test evaluation). E3--E7 will be added in later phases.

## Scientific invariants

- Treat `n_ref`, `n_tar`, `n_sur`, `J`, and `q` independently.
- Never introduce a general `n_tar <= J` assumption.
- Enforce only the relevant representability conditions, including
  `J <= n_sur` and `q <= n_tar`.
- Build surrogate inputs from the finite `n_tar` input. Never recover or access
  discarded reference-grid high frequencies.
- Preserve train/validation/test separation. Test data must not select ridge
  values, surrogate parameters, Model 3 candidates, seeds, or representatives.
- In E2, durably save and read back the selection record and frozen evaluation
  plan before the first test state solve or test metric.
- Reject unknown scientific configuration keys with their JSON path.
- Before refactoring scientific core code, fix a validated regression baseline
  independently of the refactored execution path.

The Model123 and time-scaled/generator-defect experiments were separated in
Phase 6 and are recoverable from the archive anchor documented in
`docs/phase6_legacy_migration.md`. They are not active package dependencies.

## Change discipline

Keep scientific computation below import-safe recipes, with the unified runner
above them. Do not restore the Paper 1 compatibility scripts removed in Phase
5. Do not mix algorithm changes, artifact protocol
changes, workflow refactors, and legacy cleanup in one phase. Production
profiles must not be run unless explicitly requested.

Use `python3 -m pip install -e '.[test]'` for a clean development/test
installation; the test extra includes the packaging build dependency.

## Refactoring phase names

The implementation phase source of truth is: Phase 1 unified runner; Phase 2
common infrastructure; Phase 3 E2 responsibility split; Phase 4 generic
matrix and artifact-only plot commonization; Phase 5 compatibility-script
removal; Phase 6 legacy Model123 separation (complete). The former “Phase 3A/3B”
matrix/plot labels mean Phase 4. Phases 5 and 6 are complete.
