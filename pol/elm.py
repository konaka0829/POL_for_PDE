from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class FixedRandomELM:
    """Fixed random projection + nonlinearity for ELM-style features."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        activation: str = "tanh",
        seed: int = 0,
        weight_scale: float = 0.0,
        bias_scale: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if in_dim <= 0 or hidden_dim <= 0:
            raise ValueError("in_dim and hidden_dim must be positive")
        if activation not in {"tanh", "relu"}:
            raise ValueError("activation must be tanh or relu")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        if weight_scale <= 0.0:
            weight_scale = 1.0 / math.sqrt(float(in_dim))

        self.weight = weight_scale * torch.randn(hidden_dim, in_dim, generator=gen, dtype=dtype)
        self.bias = bias_scale * torch.randn(hidden_dim, generator=gen, dtype=dtype)

        if device is not None:
            self.weight = self.weight.to(device)
            self.bias = self.bias.to(device)

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "FixedRandomELM":
        self.weight = self.weight.to(device=device, dtype=dtype or self.weight.dtype)
        self.bias = self.bias.to(device=device, dtype=dtype or self.bias.dtype)
        return self

    @torch.no_grad()
    def __call__(self, phi: torch.Tensor) -> torch.Tensor:
        h = phi @ self.weight.t() + self.bias
        if self.activation == "tanh":
            return torch.tanh(h)
        return F.relu(h)
