from __future__ import annotations

import torch


class RidgeReadout2D:
    """Closed-form ridge readout for shared spatial linear head u = w^T phi + b."""

    def __init__(self, in_channels: int, ridge_lambda: float = 1e-6, stats_device: str = "cpu"):
        self.in_channels = in_channels
        self.ridge_lambda = float(ridge_lambda)
        self.stats_device = torch.device(stats_device)
        self._dtype = torch.float64

        self.reset()

    def reset(self) -> None:
        c = self.in_channels
        self.sum_phi = torch.zeros(c, device=self.stats_device, dtype=self._dtype)
        self.sum_y = torch.zeros((), device=self.stats_device, dtype=self._dtype)
        self.sum_phiphi = torch.zeros(c, c, device=self.stats_device, dtype=self._dtype)
        self.sum_phiy = torch.zeros(c, device=self.stats_device, dtype=self._dtype)
        self.count = 0
        self.w = None
        self.b = None

    @torch.no_grad()
    def update(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate sufficient statistics.

        Args:
            features: (B, C, S, S)
            targets:  (B, S, S)
        """
        bsz, channels, sx, sy = features.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")

        n = sx * sy
        f = features.reshape(bsz, channels, n).to(self.stats_device, dtype=self._dtype)
        y = targets.reshape(bsz, n).to(self.stats_device, dtype=self._dtype)

        self.sum_phi += f.sum(dim=(0, 2))
        self.sum_y += y.sum()
        self.sum_phiphi += torch.einsum("bcn,bdn->cd", f, f)
        self.sum_phiy += torch.einsum("bcn,bn->c", f, y)
        self.count += bsz * n

    @torch.no_grad()
    def solve(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.solve_with_lambda(self.ridge_lambda)

    @torch.no_grad()
    def solve_with_lambda(self, ridge_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count <= 0:
            raise RuntimeError("No samples accumulated. Call update() first.")

        m = float(self.count)
        mu_phi = self.sum_phi / m
        mu_y = self.sum_y / m

        s = self.sum_phiphi - m * torch.outer(mu_phi, mu_phi)
        t = self.sum_phiy - m * mu_phi * mu_y

        eye = torch.eye(self.in_channels, device=self.stats_device, dtype=self._dtype)
        a = s + float(ridge_lambda) * eye

        w = torch.linalg.solve(a, t)
        b = mu_y - torch.dot(mu_phi, w)

        self.w = w
        self.b = b
        return w, b

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self.w is None or self.b is None:
            raise RuntimeError("Readout is not solved. Call solve() first.")

        w = self.w.to(features.device, dtype=features.dtype)
        b = self.b.to(features.device, dtype=features.dtype)
        return torch.einsum("c,bcxy->bxy", w, features) + b
