from __future__ import annotations

import torch


def rms_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    err = pred - target
    return torch.sqrt(torch.mean(torch.mean(err * err, dim=-1))).item()
