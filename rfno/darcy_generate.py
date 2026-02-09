from __future__ import annotations

import numpy as np
import scipy.io
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve
import torch

from data_generation.navier_stokes.random_fields import GaussianRF


def _harmonic_mean(a: float, b: float, eps: float = 1e-12) -> float:
    return 2.0 * a * b / (a + b + eps)


def solve_darcy_2d_fd(
    coeff: np.ndarray,
    forcing: float | np.ndarray = 1.0,
    solver: str = "spsolve",
    cg_tol: float = 1e-10,
    cg_maxiter: int = 2000,
) -> np.ndarray:
    """Solve -div(a grad u)=f with u|boundary=0 using flux-form 5-point FD."""
    s = coeff.shape[0]
    if coeff.shape != (s, s):
        raise ValueError("coeff must be square (S, S).")

    if np.isscalar(forcing):
        f = np.full((s, s), float(forcing), dtype=np.float64)
    else:
        f = np.asarray(forcing, dtype=np.float64)
        if f.shape != (s, s):
            raise ValueError("forcing must have shape (S, S).")

    n_in = s - 2
    n_unknown = n_in * n_in
    h = 1.0 / (s - 1)

    rows = []
    cols = []
    vals = []
    rhs = np.zeros(n_unknown, dtype=np.float64)

    def lin_idx(i: int, j: int) -> int:
        return (i - 1) * n_in + (j - 1)

    for i in range(1, s - 1):
        for j in range(1, s - 1):
            p = lin_idx(i, j)
            a_c = float(coeff[i, j])
            a_e = _harmonic_mean(a_c, float(coeff[i + 1, j]))
            a_w = _harmonic_mean(a_c, float(coeff[i - 1, j]))
            a_n = _harmonic_mean(a_c, float(coeff[i, j + 1]))
            a_s = _harmonic_mean(a_c, float(coeff[i, j - 1]))

            diag = a_e + a_w + a_n + a_s
            rows.append(p)
            cols.append(p)
            vals.append(diag)

            if i + 1 <= s - 2:
                rows.append(p)
                cols.append(lin_idx(i + 1, j))
                vals.append(-a_e)
            if i - 1 >= 1:
                rows.append(p)
                cols.append(lin_idx(i - 1, j))
                vals.append(-a_w)
            if j + 1 <= s - 2:
                rows.append(p)
                cols.append(lin_idx(i, j + 1))
                vals.append(-a_n)
            if j - 1 >= 1:
                rows.append(p)
                cols.append(lin_idx(i, j - 1))
                vals.append(-a_s)

            rhs[p] = (h * h) * f[i, j]

    a_mat = coo_matrix((vals, (rows, cols)), shape=(n_unknown, n_unknown)).tocsr()

    if solver == "spsolve":
        u_interior = spsolve(a_mat, rhs)
    elif solver == "cg":
        u_interior, info = cg(a_mat, rhs, rtol=cg_tol, atol=0.0, maxiter=cg_maxiter)
        if info != 0:
            raise RuntimeError(f"cg failed with info={info}")
    else:
        raise ValueError("solver must be 'spsolve' or 'cg'.")

    u = np.zeros((s, s), dtype=np.float64)
    u[1:-1, 1:-1] = u_interior.reshape(n_in, n_in)
    return u


def generate_darcy_dataset(
    n_samples: int,
    s: int,
    map_type: str = "piecewise",
    a_pos: float = 12.0,
    a_neg: float = 3.0,
    alpha: float = 2.0,
    tau: float = 3.0,
    forcing: float = 1.0,
    grf_device: str = "cpu",
    solver: str = "spsolve",
    cg_tol: float = 1e-10,
    cg_maxiter: int = 2000,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    grf = GaussianRF(dim=2, size=s, alpha=alpha, tau=tau, device=grf_device)
    z = grf.sample(n_samples).detach().cpu().numpy()

    if map_type == "piecewise":
        coeff = np.where(z >= 0.0, a_pos, a_neg)
    elif map_type == "exp":
        coeff = np.exp(z)
    else:
        raise ValueError("map_type must be 'piecewise' or 'exp'.")

    coeff = coeff.astype(np.float64)
    sol = np.zeros_like(coeff, dtype=np.float64)

    for i in range(n_samples):
        sol[i] = solve_darcy_2d_fd(
            coeff=coeff[i],
            forcing=forcing,
            solver=solver,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
        )

    return torch.from_numpy(coeff.astype(np.float32)), torch.from_numpy(sol.astype(np.float32))


def save_dataset_to_mat(path: str, coeff: torch.Tensor, sol: torch.Tensor) -> None:
    scipy.io.savemat(
        path,
        {
            "coeff": coeff.detach().cpu().numpy().astype(np.float32),
            "sol": sol.detach().cpu().numpy().astype(np.float32),
        },
    )
