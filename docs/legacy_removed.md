# Removed Legacy Code

This repository was slimmed to the current 1D time-scaled PDE surrogate operator learning implementation.

Removed groups:

- Fourier Neural Operator scripts (`fourier_*.py`, image/FNO scripts).
- LowRank operator implementations.
- Darcy and Navier-Stokes data generation.
- MATLAB Burgers generators.
- Deprecated Model123 modules (`models.py`, `observations.py`, `solvers.py`).
- Old reservoir/RFM experiment entry points.
- Redundant Model123 nu/Ttilde sweep and profile plotting scripts.
- Root helper modules after migration to `pol.io_mat`, `pol.cli`, and `pol.plotting`.
