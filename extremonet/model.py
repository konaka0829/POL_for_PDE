from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class ExtremeLearning(nn.Module):
    def __init__(
        self,
        indim: int,
        outdim: int,
        *,
        c: int = 1,
        s: float = 1.0,
        activation: nn.Module | None = None,
        norm_mean: torch.Tensor | np.ndarray | float = 0.0,
        norm_std: torch.Tensor | np.ndarray | float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.indim = int(indim)
        self.outdim = int(outdim)
        self.c = int(max(1, c))
        self.s = float(s)
        self.activation = activation if activation is not None else nn.Tanh()
        self.seed = int(seed)

        norm_mean_t = torch.as_tensor(norm_mean, dtype=torch.float32).reshape(1, -1)
        norm_std_t = torch.as_tensor(norm_std, dtype=torch.float32).reshape(1, -1)
        if norm_mean_t.shape[1] == 1:
            norm_mean_t = norm_mean_t.expand(1, self.indim)
        if norm_std_t.shape[1] == 1:
            norm_std_t = norm_std_t.expand(1, self.indim)
        if norm_mean_t.shape[1] != self.indim or norm_std_t.shape[1] != self.indim:
            raise ValueError("norm_mean/norm_std must be scalar or match indim")

        self.register_buffer("norm_mean", norm_mean_t.clone())
        self.register_buffer("norm_std", norm_std_t.clone())

        self.register_buffer("R", self._init_sparse_random())
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + 12345)
        b = torch.rand(1, self.outdim, generator=gen, dtype=torch.float32) * 2.0 - 1.0
        self.register_buffer("b", b)

    def _init_sparse_random(self) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        r = torch.zeros(self.indim, self.outdim, dtype=torch.float32)
        for j in range(self.outdim):
            k = min(self.c, self.indim)
            idx = torch.randperm(self.indim, generator=gen)[:k]
            vals = torch.rand(k, generator=gen, dtype=torch.float32) * (2.0 / np.sqrt(max(k, 1))) - 1.0
            r[idx, j] = vals
        return r

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.norm_mean) / (self.norm_std + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.scale(x)
        y = y @ self.R + self.b
        return self.activation(self.s * y)


class ExtremONet(nn.Module):
    def __init__(
        self,
        trunk: ExtremeLearning,
        branch: ExtremeLearning,
        *,
        outdim: int = 1,
    ) -> None:
        super().__init__()
        if trunk.outdim != branch.outdim:
            raise ValueError("trunk.outdim and branch.outdim must match")
        self.trunk = trunk
        self.branch = branch
        self.psize = int(trunk.outdim)
        self.outdim = int(outdim)

        self.register_buffer("A", torch.zeros(self.psize, self.outdim, dtype=torch.float32))
        self.register_buffer("B", torch.zeros(1, self.outdim, dtype=torch.float32))

    def forward_features(self, x_query: torch.Tensor, u_sensors: torch.Tensor) -> torch.Tensor:
        b = self.branch(u_sensors)

        if x_query.ndim == 2:
            t = self.trunk(x_query)
            return b.unsqueeze(1) * t.unsqueeze(0)
        if x_query.ndim == 3:
            bsz, nq, d = x_query.shape
            t = self.trunk(x_query.reshape(bsz * nq, d)).reshape(bsz, nq, self.psize)
            return b.unsqueeze(1) * t
        raise ValueError(f"x_query must be 2D or 3D, got shape {tuple(x_query.shape)}")

    def predict_tensor(self, x_query: torch.Tensor, u_sensors: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x_query, u_sensors)
        return torch.einsum("bnp,po->bno", h, self.A) + self.B.view(1, 1, self.outdim)

    def set_readout(self, A: torch.Tensor, B: torch.Tensor) -> None:
        if A.shape != self.A.shape:
            raise ValueError(f"A shape mismatch: expected {tuple(self.A.shape)}, got {tuple(A.shape)}")
        if B.shape != self.B.shape:
            raise ValueError(f"B shape mismatch: expected {tuple(self.B.shape)}, got {tuple(B.shape)}")
        self.A.copy_(A)
        self.B.copy_(B)

    @staticmethod
    def normalized_mse(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mse = torch.mean((y_true - y_pred) ** 2)
        denom = torch.var(y_true) + eps
        return mse / denom

    def get_config(self) -> dict[str, Any]:
        return {
            "trunk": {
                "indim": self.trunk.indim,
                "outdim": self.trunk.outdim,
                "c": self.trunk.c,
                "s": self.trunk.s,
                "seed": self.trunk.seed,
                "norm_mean": self.trunk.norm_mean.detach().cpu(),
                "norm_std": self.trunk.norm_std.detach().cpu(),
                "activation": self.trunk.activation.__class__.__name__,
            },
            "branch": {
                "indim": self.branch.indim,
                "outdim": self.branch.outdim,
                "c": self.branch.c,
                "s": self.branch.s,
                "seed": self.branch.seed,
                "norm_mean": self.branch.norm_mean.detach().cpu(),
                "norm_std": self.branch.norm_std.detach().cpu(),
                "activation": self.branch.activation.__class__.__name__,
            },
            "outdim": self.outdim,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExtremONet":
        def _act(name: str) -> nn.Module:
            if name.lower() == "relu":
                return nn.ReLU()
            return nn.Tanh()

        tcfg = config["trunk"]
        bcfg = config["branch"]
        trunk = ExtremeLearning(
            tcfg["indim"],
            tcfg["outdim"],
            c=tcfg["c"],
            s=tcfg["s"],
            activation=_act(tcfg.get("activation", "Tanh")),
            norm_mean=tcfg.get("norm_mean", 0.0),
            norm_std=tcfg.get("norm_std", 1.0),
            seed=tcfg.get("seed", 0),
        )
        branch = ExtremeLearning(
            bcfg["indim"],
            bcfg["outdim"],
            c=bcfg["c"],
            s=bcfg["s"],
            activation=_act(bcfg.get("activation", "Tanh")),
            norm_mean=bcfg.get("norm_mean", 0.0),
            norm_std=bcfg.get("norm_std", 1.0),
            seed=bcfg.get("seed", 0),
        )
        return cls(trunk=trunk, branch=branch, outdim=config.get("outdim", 1))
