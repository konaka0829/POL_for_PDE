from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ArrayLike = Union[np.ndarray, "torch.Tensor"]

COLOR_INPUT = "tab:blue"
COLOR_GT = "tab:orange"
COLOR_PRED = "tab:green"
COLOR_RESERVOIR_EVOLUTION = "tab:red"


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if "torch" in str(type(x)):
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _strip_known_ext(path: str) -> str:
    base, ext = os.path.splitext(path)
    if ext.lower() in {".png", ".pdf", ".svg"}:
        return base
    return path


def save_figure_all_formats(
    fig: plt.Figure,
    out_path_no_ext: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
    pad_inches: float = 0.05,
) -> Tuple[str, str, str]:
    out_path_no_ext = _strip_known_ext(out_path_no_ext)
    out_dir = os.path.dirname(out_path_no_ext) or "."
    ensure_dir(out_dir)
    paths = (out_path_no_ext + ".png", out_path_no_ext + ".pdf", out_path_no_ext + ".svg")
    fig.patch.set_facecolor("white")
    fig.savefig(paths[0], dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    fig.savefig(paths[1], bbox_inches=bbox_inches, pad_inches=pad_inches)
    fig.savefig(paths[2], bbox_inches=bbox_inches, pad_inches=pad_inches)
    return paths


def rel_l2(pred: ArrayLike, gt: ArrayLike, eps: float = 1e-12) -> float:
    p = _to_numpy(pred).reshape(-1)
    y = _to_numpy(gt).reshape(-1)
    return float(np.linalg.norm(p - y) / (np.linalg.norm(y) + eps))


def rmse(pred: ArrayLike, gt: ArrayLike) -> float:
    p = _to_numpy(pred)
    y = _to_numpy(gt)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def plot_error_histogram(
    errors: Sequence[float],
    out_path_no_ext: str,
    bins: int = 30,
    title: Optional[str] = None,
    xlabel: str = "error",
) -> Tuple[str, str, str]:
    errors = np.asarray(list(errors), dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(errors, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    if title is None:
        title = f"error histogram (mean={errors.mean():.3g}, median={np.median(errors):.3g})"
    ax.set_title(title)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)
    return paths


def plot_1d_prediction(
    x: Optional[ArrayLike],
    gt: ArrayLike,
    pred: ArrayLike,
    out_path_no_ext: str,
    input_u0: Optional[ArrayLike] = None,
    title_prefix: str = "",
) -> Tuple[str, str, str]:
    y = _to_numpy(gt).reshape(-1)
    p = _to_numpy(pred).reshape(-1)
    if x is None:
        x_np = np.linspace(0.0, 1.0, num=y.shape[0])
    else:
        x_np = _to_numpy(x).reshape(-1)

    e_rel = rel_l2(p, y)
    e_rmse = rmse(p, y)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if input_u0 is not None:
        u0 = _to_numpy(input_u0).reshape(-1)
        ax.plot(x_np, u0, label="input (u0)", linewidth=1.0, alpha=0.7, color=COLOR_INPUT)
    ax.plot(x_np, y, label="GT", linewidth=2.0, color=COLOR_GT)
    ax.plot(x_np, p, label="Pred", linewidth=2.0, linestyle="--", color=COLOR_PRED)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"{title_prefix} relL2={e_rel:.3g}  RMSE={e_rmse:.3g}".strip())
    fig.tight_layout()
    paths = save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)
    return paths


def plot_1d_reservoir_evolution(
    x: Optional[ArrayLike],
    states: Sequence[ArrayLike],
    times: Sequence[float],
    out_path_no_ext: str,
    input_u0: Optional[ArrayLike] = None,
    gt: Optional[ArrayLike] = None,
    pred: Optional[ArrayLike] = None,
    title_prefix: str = "",
    max_curves: int = 8,
) -> Tuple[str, str, str]:
    if len(states) == 0:
        raise ValueError("states must be non-empty")
    if len(states) != len(times):
        raise ValueError("states/times length mismatch")
    y0 = _to_numpy(states[0]).reshape(-1)
    x_np = np.linspace(0.0, 1.0, num=y0.shape[0]) if x is None else _to_numpy(x).reshape(-1)
    keep = np.arange(len(states)) if len(states) <= max_curves else np.unique(np.linspace(0, len(states) - 1, max_curves, dtype=int))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if input_u0 is not None:
        ax.plot(x_np, _to_numpy(input_u0).reshape(-1), label="input (u0)", linewidth=1.0, alpha=0.7)
    for idx in keep:
        ax.plot(x_np, _to_numpy(states[idx]).reshape(-1), linewidth=1.0, alpha=0.7, label=f"r t={times[idx]:.3g}")
    if gt is not None:
        ax.plot(x_np, _to_numpy(gt).reshape(-1), label="GT", linewidth=2.0, color=COLOR_GT)
    if pred is not None:
        ax.plot(x_np, _to_numpy(pred).reshape(-1), label="Pred", linewidth=2.0, linestyle="--", color=COLOR_PRED)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    if title_prefix:
        ax.set_title(title_prefix.strip())
    fig.tight_layout()
    paths = save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)
    return paths

