from __future__ import annotations

import torch


def _build_phi_t_y(
    phi: torch.Tensor,
    y: torch.Tensor,
    y_chunk_size: int | None,
) -> torch.Tensor:
    if y_chunk_size is None or y.shape[1] <= y_chunk_size:
        return phi.T @ y

    m = phi.shape[1]
    j = y.shape[1]
    out = torch.empty(m, j, device=phi.device, dtype=phi.dtype)
    for start in range(0, j, y_chunk_size):
        end = min(start + y_chunk_size, j)
        out[:, start:end] = phi.T @ y[:, start:end]
    return out


def solve_ridge(
    phi: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    jitter: float = 1e-10,
    method: str = "cholesky",
    y_chunk_size: int | None = 1024,
) -> torch.Tensor:
    """Solve multi-output ridge in primal form.

    Args:
        phi: (N, M)
        y: (N, J)
        lam: positive ridge parameter
        jitter: initial jitter for Cholesky fallback ladder
        method: "cholesky" or "solve"
        y_chunk_size: chunk width for building Phi^T Y

    Returns:
        W_T: (M, J)
    """
    if phi.ndim != 2 or y.ndim != 2:
        raise ValueError("phi and y must be rank-2")
    if phi.shape[0] != y.shape[0]:
        raise ValueError("phi and y must have the same number of rows")
    if lam <= 0:
        raise ValueError("lam must be > 0")

    n, m = phi.shape
    _ = n

    eye = torch.eye(m, device=phi.device, dtype=phi.dtype)
    a = phi.T @ phi + lam * eye
    b = _build_phi_t_y(phi, y, y_chunk_size=y_chunk_size)

    method = method.lower()
    if method == "solve":
        return torch.linalg.solve(a, b)
    if method != "cholesky":
        raise ValueError("method must be one of {'cholesky', 'solve'}")

    jitter_values = [0.0, jitter, jitter * 1e2, jitter * 1e4]
    for jtr in jitter_values:
        try:
            a_reg = a if jtr == 0.0 else (a + jtr * eye)
            chol = torch.linalg.cholesky(a_reg)
            return torch.cholesky_solve(b, chol)
        except RuntimeError:
            continue

    # Final fallback
    return torch.linalg.solve(a + (jitter * 1e6) * eye, b)
