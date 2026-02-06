# Project instructions for Codex (Phase 0: digital-only)

## Goal (Phase 0)
Implement a *digital-only* variant of this repository that follows the math of:
- Optical Fourier Reservoir Neural Operator (OF-RNO) style reservoir features
- Readout-only learning
- (Optional but required for this task) Physics-informed, label-free readout fitting for Darcy via PDE residual (batch ridge + online RLS)

**Deliverable**: a new runnable script `rno_2d.py` that mirrors `fourier_2d.py`:
- Darcy data loading (same CLI: --data-mode, --data-file/--train-file/--test-file, --ntrain/--ntest, --r, --grid-size)
- Training / fitting
- Inference
- Visualization outputs (learning curve, sample comparisons, error histogram) to `visualizations/rno_2d/`

## Do NOT do
- No optical hardware code, no SLM/camera interfaces.
- No large refactors of existing scripts.
- Do not change `fourier_2d.py` behavior.
- Do not add heavy dependencies (stick to torch/numpy/scipy/matplotlib already in requirements).

## Must reuse existing helpers
- Use `cli_utils.add_data_mode_args` and `cli_utils.validate_data_mode_args`.
- Use `utilities3.MatReader`, `UnitGaussianNormalizer`, and `LpLoss` if needed.
- Use `viz_utils.py` functions (plot_learning_curve, plot_2d_comparison, plot_error_histogram, rel_l2).

## Mathematical spec to implement (digital OF-RNO)
We replace learnable FNO spectral layers with a fixed random "optical Fourier reservoir" block.
All reservoir parameters are fixed after initialization (buffers), only readout is learned.

Let v_l ∈ R^{B×C×s×s}.
For each layer l:
1) Pre-mix:    ṽ = B_l · v_l   (pointwise channel mixing with fixed random matrix B_l ∈ R^{C×C})
2) Optical map per-channel:
   For each channel c:
   - complex field: E = IFFT2( Π_K( M_{l,c} ⊙ FFT2(ṽ_c) ) )
   - intensity:     I = |E|^2   (real, nonnegative)
   Stack channels: O_l(ṽ) = I   ∈ R^{B×C×s×s}
   Π_K is a low-frequency mask with |kx|<=K and |ky|<=K (keep four low-freq corners).
   M_{l,c} are fixed random complex masks (only supported where Π_K=1; elsewhere 0).
3) Post-mix:   u = A_l · O_l(ṽ)  (fixed random A_l ∈ R^{C×C})
4) Update:     v_{l+1} = σ( α v_l + β u + b_l )  (α,β scalars; b_l is fixed per-channel bias)
σ is a configurable nonlinearity (tanh by default).

Input lifting:
- Build grid like `fourier_2d.py` and concatenate (a, x, y) => 3 channels.
- Apply a fixed random pointwise linear map to C channels + nonlinearity to get v_0.

Readout:
- Predict û(x,y) from v_L via a learned pointwise linear readout:
  û = w^T v_L + b    (w ∈ R^C, b ∈ R). Same w,b for all spatial points.

## Readout learning modes (must implement)
Add CLI flag `--fit-mode` with:
1) `supervised_sgd`:
   - Learn w,b by minimizing rel L2 (or L2) against ground truth y (like fourier_2d loop),
     but only optimize readout parameters (reservoir frozen).
2) `supervised_ridge`:
   - Fit w,b by ridge regression using all training points (efficient accumulation of normal equations).
3) `pde_ridge` (label-free):
   - Fit w,b from Darcy PDE residual without using y labels.
4) `pde_rls` (label-free online):
   - Fit w,b using online RLS updates from PDE residual equations.

## Darcy PDE (for label-free fitting)
Assume standard Darcy (as in FNO benchmark):
-∇·(a(x)∇u(x)) = 1 on (0,1)^2, with u=0 on boundary.

Finite difference on s×s grid, spacing h = 1/(s-1):
For interior (i,j):
(L_a u)_{i,j} = (1/h^2) * [
  a_{i+1/2,j}(u_{i,j}-u_{i+1,j}) +
  a_{i-1/2,j}(u_{i,j}-u_{i-1,j}) +
  a_{i,j+1/2}(u_{i,j}-u_{i,j+1}) +
  a_{i,j-1/2}(u_{i,j}-u_{i,j-1})
]
where a_{i+1/2,j} = 0.5(a_{i,j}+a_{i+1,j}) etc.

Residual equations:
(L_a û)_{i,j} ≈ 1  for interior points,
û = 0 for boundary points.

Because û = w^T v + b is linear in (w,b), these constraints become linear least squares.

Implementation requirements:
- Provide a vectorized Torch implementation of L_a applied to a batch of fields (B×C×s×s).
- Support sampling M collocation points per sample for PDE fitting (CLI: --pde-samples, default ~2048).
- Always include boundary equations (all boundary points).
- Add ridge regularization lambda (CLI: --ridge-lam).
- For RLS, add forgetting factor rho and initial delta (CLI: --rls-rho, --rls-delta).

## Outputs and QA
- Save visualizations to `visualizations/rno_2d/` in png/pdf/svg.
- Add a README section describing `rno_2d.py` usage & args.
- Add a `--smoke` option that generates tiny random tensors if dataset files are missing, so the script can be sanity-checked without data.
- Provide at least one smoke command in comments at the bottom of rno_2d.py.
