import torch


def _phi_t_y_chunked(Phi: torch.Tensor, Y: torch.Tensor, chunk_size: int | None) -> torch.Tensor:
    if chunk_size is None or chunk_size <= 0 or chunk_size >= Y.shape[1]:
        return Phi.t() @ Y

    M = Phi.shape[1]
    J = Y.shape[1]
    out = torch.empty(M, J, device=Phi.device, dtype=Phi.dtype)
    for start in range(0, J, chunk_size):
        end = min(start + chunk_size, J)
        out[:, start:end] = Phi.t() @ Y[:, start:end]
    return out


def solve_ridge(
    Phi: torch.Tensor,
    Y: torch.Tensor,
    lam: float,
    jitter: float = 1e-10,
    method: str = "cholesky",
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Solve multi-output ridge: W_T = (Phi^T Phi + lam I)^(-1) Phi^T Y.

    Args:
        Phi: (N, M)
        Y: (N, J)
        lam: positive regularization
        jitter: initial extra diagonal term
        method: "cholesky" or "solve"
        chunk_size: column chunk size for Phi.T @ Y
    Returns:
        W_T: (M, J)
    """
    if Phi.ndim != 2 or Y.ndim != 2:
        raise ValueError("Phi and Y must be 2D tensors")
    if Phi.shape[0] != Y.shape[0]:
        raise ValueError(f"Batch mismatch: Phi {Phi.shape}, Y {Y.shape}")
    if lam <= 0.0:
        raise ValueError("lam must be > 0 for SPD ridge system")

    N, M = Phi.shape
    eye = torch.eye(M, device=Phi.device, dtype=Phi.dtype)

    A = Phi.t() @ Phi
    A = A + lam * eye

    B = _phi_t_y_chunked(Phi, Y, chunk_size=chunk_size)

    method = method.lower()
    if method == "solve":
        return torch.linalg.solve(A, B)

    if method != "cholesky":
        raise ValueError(f"Unknown method: {method}")

    jitter_val = jitter
    for _ in range(6):
        try:
            L = torch.linalg.cholesky(A + jitter_val * eye)
            return torch.cholesky_solve(B, L)
        except RuntimeError:
            jitter_val *= 100.0

    # fallback
    return torch.linalg.solve(A + jitter_val * eye, B)
