# Removed and retained legacy code

The active first-paper entry point is `pol run`. Phase 5 removed the six
Paper 1 compatibility scripts formerly under `scripts/paper1`; direct recipes
remain internal/test APIs.

The Model123 and time-scaled/generator-defect tree is still present for
compatibility pending Phase 6. Phase 6 has not been performed.

Removed groups:

- Fourier Neural Operator scripts (`fourier_*.py`, image/FNO scripts).
- LowRank operator implementations.
- Darcy and Navier-Stokes data generation.
- MATLAB Burgers generators.
- Deprecated Model123 modules (`models.py`, `observations.py`, `solvers.py`).
- Old reservoir/RFM experiment entry points.
- Redundant Model123 nu/Ttilde sweep and profile plotting scripts.
- Root helper modules after migration to `pol.io_mat`, `pol.cli`, and `pol.plotting`.
