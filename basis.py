from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GridBasis:
    spatial_shape: tuple[int, ...]

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim < 2:
            raise ValueError("Expected y shape (N, ...)")
        return y.reshape(y.shape[0], -1)

    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.ndim != 2:
            raise ValueError("Expected coeffs shape (N, J)")
        return coeffs.reshape(coeffs.shape[0], *self.spatial_shape)


class PODBasis:
    def __init__(self, components: torch.Tensor, mean: torch.Tensor):
        if components.ndim != 2:
            raise ValueError("components must be (J, K)")
        if mean.ndim != 1:
            raise ValueError("mean must be (J,)")
        if components.shape[0] != mean.shape[0]:
            raise ValueError("components and mean mismatch")
        self.components = components
        self.mean = mean

    @classmethod
    def fit(
        cls,
        y_train_flat: torch.Tensor,
        basis_dim: int,
        center: bool = True,
    ) -> "PODBasis":
        if y_train_flat.ndim != 2:
            raise ValueError("y_train_flat must be rank-2")

        n, j = y_train_flat.shape
        k = min(int(basis_dim), n, j)
        if k <= 0:
            raise ValueError("basis_dim must be positive")

        if center:
            mean = y_train_flat.mean(dim=0)
            yc = y_train_flat - mean
        else:
            mean = torch.zeros(j, device=y_train_flat.device, dtype=y_train_flat.dtype)
            yc = y_train_flat

        use_pca_lowrank = hasattr(torch, "pca_lowrank")
        if use_pca_lowrank:
            try:
                _, _, v = torch.pca_lowrank(yc, q=k, center=False)
                components = v[:, :k]
                return cls(components=components, mean=mean)
            except RuntimeError:
                pass

        # Fallback: SVD
        _, _, vh = torch.linalg.svd(yc, full_matrices=False)
        components = vh[:k, :].T
        return cls(components=components, mean=mean)

    def encode(self, y_flat: torch.Tensor) -> torch.Tensor:
        if y_flat.ndim != 2:
            raise ValueError("y_flat must be (N, J)")
        return (y_flat - self.mean) @ self.components

    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.ndim != 2:
            raise ValueError("coeffs must be (N, K)")
        return coeffs @ self.components.T + self.mean
