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


def _default_x_for_y(x: Optional[ArrayLike], y: np.ndarray) -> np.ndarray:
    if x is None:
        return np.linspace(0.0, 1.0, num=y.shape[0], endpoint=False)
    x_np = _to_numpy(x).reshape(-1)
    if x_np.shape[0] != y.shape[0]:
        raise ValueError(f"x/y length mismatch: x has {x_np.shape[0]} points, y has {y.shape[0]}")
    return x_np


def plot_single_waveform(
    x: Optional[ArrayLike],
    y: ArrayLike,
    out_path_no_ext: str,
    *,
    label: Optional[str] = None,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "u",
    ylim: Optional[tuple[float, float]] = None,
    linewidth: float = 2.0,
) -> Tuple[str, str, str]:
    y_np = _to_numpy(y).reshape(-1)
    x_np = _default_x_for_y(x, y_np)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(x_np, y_np, label=label, linewidth=linewidth, color=COLOR_INPUT)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    if label:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)
    return paths


def plot_waveform_overlay(
    x: Optional[ArrayLike],
    curves: Sequence[dict],
    out_path_no_ext: str,
    *,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "u",
    ylim: Optional[tuple[float, float]] = None,
) -> Tuple[str, str, str]:
    if not curves:
        raise ValueError("curves must be non-empty")
    first = _to_numpy(curves[0]["y"]).reshape(-1)
    x_np = _default_x_for_y(x, first)
    colors = [COLOR_GT, COLOR_PRED, COLOR_INPUT]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for idx, curve in enumerate(curves):
        y_np = _to_numpy(curve["y"]).reshape(-1)
        if y_np.shape != first.shape:
            raise ValueError(f"curve {idx} has shape {y_np.shape}, expected {first.shape}")
        kwargs = {
            "label": curve.get("label"),
            "linestyle": curve.get("linestyle", "-"),
            "linewidth": curve.get("linewidth", 2.0),
            "alpha": curve.get("alpha", 1.0),
            "color": curve.get("color", colors[idx % len(colors)]),
        }
        ax.plot(x_np, y_np, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)
    return paths


def plot_feature_vector(
    values: ArrayLike,
    out_path_no_ext: str,
    *,
    title: str = "",
    xlabel: str = "feature index",
    ylabel: str = "value",
    max_points: int = 512,
) -> Tuple[str, str, str]:
    vec = _to_numpy(values).reshape(-1)
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    truncated = vec.shape[0] > max_points
    plot_vec = vec[:max_points]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(np.arange(plot_vec.shape[0]), plot_vec, linewidth=1.2, color=COLOR_RESERVOIR_EVOLUTION)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    shown_title = title
    if truncated:
        suffix = f"first {max_points} of {vec.shape[0]} features"
        shown_title = f"{title} ({suffix})" if title else suffix
    if shown_title:
        ax.set_title(shown_title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)
    return paths


def plot_spacetime(
    x: ArrayLike,
    t: ArrayLike,
    u_xt: ArrayLike,
    out_path_no_ext: str,
    *,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "t",
    cbar_label: str = "u",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    symmetric_colorlim: bool = False,
) -> Tuple[str, str, str]:
    x_np = _to_numpy(x).reshape(-1)
    t_np = _to_numpy(t).reshape(-1)
    u_np = _to_numpy(u_xt)
    if u_np.ndim == 3 and u_np.shape[0] == 1:
        u_np = u_np[0]
    if u_np.ndim != 2:
        raise ValueError(f"u_xt must have shape (Nt, Nx) or (1, Nt, Nx), got {tuple(u_np.shape)}")
    if u_np.shape != (t_np.shape[0], x_np.shape[0]):
        raise ValueError(
            "u_xt shape mismatch: got %s, expected (%d, %d)"
            % (tuple(u_np.shape), t_np.shape[0], x_np.shape[0])
        )
    if symmetric_colorlim:
        finite = u_np[np.isfinite(u_np)]
        if finite.size:
            lim = float(np.max(np.abs(finite)))
            vmin = -lim if vmin is None else vmin
            vmax = lim if vmax is None else vmax
    if x_np.shape[0] > 1:
        dx = float(np.median(np.diff(x_np)))
        x0 = float(x_np[0] - 0.5 * dx)
        x1 = float(x_np[-1] + 0.5 * dx)
    else:
        x0, x1 = float(x_np[0]) - 0.5, float(x_np[0]) + 0.5
    if t_np.shape[0] > 1:
        dt = float(np.median(np.diff(t_np)))
        t0 = float(t_np[0] - 0.5 * dt)
        t1 = float(t_np[-1] + 0.5 * dt)
    else:
        t0, t1 = float(t_np[0]) - 0.5, float(t_np[0]) + 0.5
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(
        u_np,
        origin="lower",
        aspect="auto",
        extent=(x0, x1, t0, t1),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
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
