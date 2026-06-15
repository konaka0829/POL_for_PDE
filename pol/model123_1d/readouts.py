from __future__ import annotations

from dataclasses import dataclass

import torch

from .metrics import dataset_abs_l2h_rmse, dataset_rel_l2h_aggregate, dataset_rel_l2h_mean


@dataclass
class FourierDiagonalReadout:
    zeta: float
    domain_length: float = 1.0
    mean_x: torch.Tensor | None = None
    mean_y: torch.Tensor | None = None
    weights: torch.Tensor | None = None

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "FourierDiagonalReadout":
        xh = torch.fft.rfft(x, dim=-1)
        yh = torch.fft.rfft(y, dim=-1)
        self.mean_x = xh.mean(dim=0, keepdim=True)
        self.mean_y = yh.mean(dim=0, keepdim=True)
        xc = xh - self.mean_x
        yc = yh - self.mean_y
        var = torch.mean(torch.abs(xc) ** 2, dim=0)
        cov = torch.mean(torch.conj(xc) * yc, dim=0)
        self.weights = cov / (var + float(self.zeta))
        return self

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_x is None or self.mean_y is None or self.weights is None:
            raise RuntimeError("FourierDiagonalReadout.fit must be called before predict")
        xh = torch.fft.rfft(x, dim=-1)
        yh = self.mean_y + (xh - self.mean_x) * self.weights.unsqueeze(0)
        return torch.fft.irfft(yh, n=x.shape[-1], dim=-1)


def evaluate_readout(pred: torch.Tensor, target: torch.Tensor, *, domain_length: float = 1.0) -> dict[str, float]:
    return {
        "absL2h": dataset_abs_l2h_rmse(pred, target, domain_length=domain_length),
        "relL2_mean": dataset_rel_l2h_mean(pred, target, domain_length=domain_length),
        "relL2_agg": dataset_rel_l2h_aggregate(pred, target, domain_length=domain_length),
    }


def target_variance_l2h(y: torch.Tensor, *, domain_length: float = 1.0) -> float:
    centered = y - y.mean(dim=0, keepdim=True)
    return dataset_abs_l2h_rmse(centered, torch.zeros_like(centered), domain_length=domain_length) ** 2
