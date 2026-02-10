from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from extremonet.model import ExtremONet


@dataclass
class TrainResult:
    best_log10_lambda: float
    train_nmse: float
    val_nmse: float
    history_log10_lambda: list[float]
    history_train_nmse: list[float]
    history_val_nmse: list[float]


def _iter_batches(indices: np.ndarray, batch_size: int):
    if batch_size <= 0:
        yield indices
        return
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


def _to_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(np.asarray(x), device=device, dtype=torch.float32)


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    # EON keeps most tensors as buffers; parameters may be empty.
    first_param = next(model.parameters(), None)
    if first_param is not None:
        return first_param.device
    first_buffer = next(model.buffers(), None)
    if first_buffer is not None:
        return first_buffer.device
    return torch.device("cpu")


def _batch_features(model: ExtremONet, x_query: torch.Tensor, u_sensors: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    u_b = u_sensors[idx]
    if x_query.ndim == 3:
        x_b = x_query[idx]
    else:
        x_b = x_query
    return model.forward_features(x_b, u_b)


def _ridge_stats(
    model: ExtremONet,
    x_query: torch.Tensor,
    u_sensors: torch.Tensor,
    y_query: torch.Tensor,
    train_idx: np.ndarray,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = model.psize
    outdim = model.outdim
    device = y_query.device

    hth = torch.zeros(p + 1, p + 1, device=device)
    hty = torch.zeros(p + 1, outdim, device=device)

    for batch_idx in _iter_batches(train_idx, batch_size):
        h = _batch_features(model, x_query, u_sensors, batch_idx)
        y = y_query[batch_idx]

        h2 = h.reshape(-1, p)
        y2 = y.reshape(-1, outdim)
        n = h2.shape[0]

        hth[:p, :p] += h2.T @ h2
        hth[:p, p] += h2.sum(dim=0)
        hth[p, :p] += h2.sum(dim=0)
        hth[p, p] += float(n)

        hty[:p] += h2.T @ y2
        hty[p] += y2.sum(dim=0)

    return hth, hty


def _predict_with_aug(h: torch.Tensor, a_aug: torch.Tensor) -> torch.Tensor:
    p = h.shape[-1]
    a = a_aug[:p]
    b = a_aug[p : p + 1]
    return torch.einsum("bnp,po->bno", h, a) + b.view(1, 1, -1)


def _eval_nmse(
    model: ExtremONet,
    x_query: torch.Tensor,
    u_sensors: torch.Tensor,
    y_query: torch.Tensor,
    idx: np.ndarray,
    a_aug: torch.Tensor,
    batch_size: int,
) -> float:
    numer = 0.0
    denom = 0.0
    for batch_idx in _iter_batches(idx, batch_size):
        h = _batch_features(model, x_query, u_sensors, batch_idx)
        y = y_query[batch_idx]
        yp = _predict_with_aug(h, a_aug)

        y2 = y.reshape(-1)
        yp2 = yp.reshape(-1)
        numer += float(torch.sum((y2 - yp2) ** 2).item())
        denom += float(torch.sum((y2 - torch.mean(y2)) ** 2).item())
    return numer / max(denom, 1e-12)


def train_ridge(
    model: ExtremONet,
    x_query: Any,
    u_sensors: Any,
    y_query: Any,
    *,
    bounds: tuple[float, float] = (-10.0, 10.0),
    iters: int = 200,
    val_frac: float = 0.2,
    seed: int = 0,
    batch_size: int = 64,
) -> TrainResult:
    device = _infer_model_device(model)
    x_t = _to_tensor(x_query, device)
    u_t = _to_tensor(u_sensors, device)
    y_t = _to_tensor(y_query, device)

    if y_t.ndim == 2:
        y_t = y_t.unsqueeze(-1)

    n = u_t.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(max(1, round(n * val_frac)))
    n_val = min(n_val, n - 1) if n > 1 else 1
    val_idx = idx[:n_val]
    train_idx = idx[n_val:] if n > 1 else idx

    hth, hty = _ridge_stats(model, x_t, u_t, y_t, train_idx, batch_size)

    tr_hist: list[float] = []
    va_hist: list[float] = []
    l_hist: list[float] = []
    a_hist: list[torch.Tensor] = []

    def objective(log10_lambda: float) -> float:
        lam = 10.0 ** float(log10_lambda)
        reg = lam * torch.eye(hth.shape[0], device=hth.device, dtype=hth.dtype)
        a_aug = torch.linalg.solve(hth + reg, hty)

        tr_nmse = _eval_nmse(model, x_t, u_t, y_t, train_idx, a_aug, batch_size)
        va_nmse = _eval_nmse(model, x_t, u_t, y_t, val_idx, a_aug, batch_size)

        l_hist.append(float(log10_lambda))
        tr_hist.append(float(tr_nmse))
        va_hist.append(float(va_nmse))
        a_hist.append(a_aug.detach().clone())
        return va_nmse

    minimize_scalar(
        objective,
        bounds=(float(bounds[0]), float(bounds[1])),
        method="bounded",
        options={"maxiter": int(iters), "xatol": 1e-2},
    )

    best_i = int(np.argmin(va_hist))
    best_aug = a_hist[best_i]
    p = model.psize

    with torch.no_grad():
        model.set_readout(best_aug[:p], best_aug[p : p + 1])

    return TrainResult(
        best_log10_lambda=l_hist[best_i],
        train_nmse=tr_hist[best_i],
        val_nmse=va_hist[best_i],
        history_log10_lambda=l_hist,
        history_train_nmse=tr_hist,
        history_val_nmse=va_hist,
    )
