import torch


class GridBasis:
    def __init__(self, output_shape: tuple[int, ...]) -> None:
        self.output_shape = output_shape
        self.J = 1
        for s in output_shape:
            self.J *= s

    def fit(self, Y_train_flat: torch.Tensor) -> "GridBasis":
        _ = Y_train_flat
        return self

    def encode(self, Y: torch.Tensor) -> torch.Tensor:
        return Y.reshape(Y.shape[0], -1)

    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        return coeffs.reshape(coeffs.shape[0], *self.output_shape)


class PODBasis:
    def __init__(self, basis_dim: int, center: bool = True) -> None:
        self.basis_dim = basis_dim
        self.center = center
        self.mean: torch.Tensor | None = None
        self.U: torch.Tensor | None = None

    def fit(self, Y_train_flat: torch.Tensor) -> "PODBasis":
        if Y_train_flat.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tuple(Y_train_flat.shape)}")

        X = Y_train_flat
        if self.center:
            self.mean = X.mean(dim=0, keepdim=True)
            Xc = X - self.mean
        else:
            self.mean = torch.zeros(1, X.shape[1], device=X.device, dtype=X.dtype)
            Xc = X

        k = min(self.basis_dim, X.shape[0], X.shape[1])
        if k <= 0:
            raise ValueError("basis_dim must be positive")

        try:
            # pca_lowrank returns V with shape (D, q), columns are components.
            _, _, V = torch.pca_lowrank(Xc, q=k, center=False)
            self.U = V[:, :k]
        except Exception:
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            self.U = Vh[:k, :].t().contiguous()

        return self

    def encode(self, Y_flat: torch.Tensor) -> torch.Tensor:
        if self.U is None or self.mean is None:
            raise RuntimeError("PODBasis must be fit before encode")
        X = Y_flat - self.mean if self.center else Y_flat
        return X @ self.U

    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        if self.U is None or self.mean is None:
            raise RuntimeError("PODBasis must be fit before decode")
        Y = coeffs @ self.U.t()
        if self.center:
            Y = Y + self.mean
        return Y
