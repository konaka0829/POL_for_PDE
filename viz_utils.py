"""viz_utils.py

Reusable visualization utilities for operator-learning / PDE regression scripts.

Design goals:
  - Provide *consistent* qualitative visual checks across 1D/2D/space-time tasks.
  - Always save figures in **png + pdf + svg** (as requested).
  - Keep dependencies minimal (matplotlib + numpy + torch optional).
  - Work in headless environments (sets matplotlib backend to 'Agg').

Typical outputs
  - learning_curve.* : train/test scalar metrics vs epoch
  - sample_*.*       : GT vs Pred vs Error for a few samples
  - error_hist.*     : histogram of per-sample relative L2
  - error_time.*     : relative L2 vs time index (for space-time outputs)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

import matplotlib

# Headless backend for servers / clusters.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert torch tensor / numpy array to numpy array on CPU."""
    if "torch" in str(type(x)):
        # torch.Tensor
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _strip_known_ext(path: str) -> str:
    """If user passes 'foo.png' etc, strip the extension to get a base path."""
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
    """Save a matplotlib figure to PNG/PDF/SVG.

    Args:
        fig: Matplotlib Figure
        out_path_no_ext: output path without extension (directory will be created)
        dpi: dpi for raster outputs (PNG)
        bbox_inches/pad_inches: forwarded to fig.savefig

    Returns:
        (png_path, pdf_path, svg_path)
    """
    out_path_no_ext = _strip_known_ext(out_path_no_ext)
    out_dir = os.path.dirname(out_path_no_ext) or "."
    ensure_dir(out_dir)

    png_path = out_path_no_ext + ".png"
    pdf_path = out_path_no_ext + ".pdf"
    svg_path = out_path_no_ext + ".svg"

    # White background helps when users embed SVG/PDF in slides.
    fig.patch.set_facecolor("white")

    # PNG (raster)
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    # PDF/SVG (vector)
    fig.savefig(pdf_path, bbox_inches=bbox_inches, pad_inches=pad_inches)
    fig.savefig(svg_path, bbox_inches=bbox_inches, pad_inches=pad_inches)

    return png_path, pdf_path, svg_path


def rel_l2(pred: ArrayLike, gt: ArrayLike, eps: float = 1e-12) -> float:
    """Relative L2 error ||pred-gt||_2 / ||gt||_2."""
    p = _to_numpy(pred).reshape(-1)
    y = _to_numpy(gt).reshape(-1)
    num = np.linalg.norm(p - y)
    den = np.linalg.norm(y)
    return float(num / (den + eps))


def rmse(pred: ArrayLike, gt: ArrayLike) -> float:
    p = _to_numpy(pred)
    y = _to_numpy(gt)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(pred: ArrayLike, gt: ArrayLike) -> float:
    p = _to_numpy(pred)
    y = _to_numpy(gt)
    return float(np.mean(np.abs(p - y)))


@dataclass
class LearningCurve:
    epochs: Sequence[int]
    train: Sequence[float]
    test: Sequence[float]
    train_label: str = "train"
    test_label: str = "test"
    metric_name: str = "loss"


def plot_learning_curve(
    curve: LearningCurve,
    out_path_no_ext: str,
    logy: bool = True,
    title: Optional[str] = None,
) -> Tuple[str, str, str]:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(curve.epochs, curve.train, label=curve.train_label)

    # Plot test curve only when it is meaningful.
    if curve.test is not None:
        test_arr = np.asarray(list(curve.test), dtype=float)
        has_any_finite = np.isfinite(test_arr).any()
        if curve.test_label and has_any_finite:
            ax.plot(curve.epochs, test_arr, label=curve.test_label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(curve.metric_name)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)
    return paths


def plot_error_histogram(
    errors: Sequence[float],
    out_path_no_ext: str,
    bins: int = 30,
    title: Optional[str] = None,
) -> Tuple[str, str, str]:
    errors = np.asarray(list(errors), dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(errors, bins=bins)
    ax.set_xlabel("relative L2 error")
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
        ax.plot(x_np, u0, label="input (u0)", linewidth=1.0, alpha=0.7)
    ax.plot(x_np, y, label="GT", linewidth=2.0)
    ax.plot(x_np, p, label="Pred", linewidth=2.0, linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    ax.legend()
    title = f"{title_prefix} relL2={e_rel:.3g}  RMSE={e_rmse:.3g}"
    ax.set_title(title.strip())
    fig.tight_layout()
    paths = save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)
    return paths


def plot_2d_comparison(
    gt: ArrayLike,
    pred: ArrayLike,
    out_path_no_ext: str,
    input_field: Optional[ArrayLike] = None,
    suptitle: Optional[str] = None,
    eps: float = 1e-12,
) -> Tuple[str, str, str]:
    """2D heatmaps: (optional) input, GT, Pred, |error|, and |error|/|GT|.

    The GT and Pred use a shared color scale (vmin/vmax) for fair visual comparison.
    """
    y = _to_numpy(gt)
    p = _to_numpy(pred)
    if y.ndim != 2:
        y = y.squeeze()
    if p.ndim != 2:
        p = p.squeeze()

    err = p - y
    abs_err = np.abs(err)
    rel_err_map = abs_err / (np.abs(y) + eps)

    e_rel = rel_l2(p, y)
    e_rmse = rmse(p, y)
    e_mae = mae(p, y)

    # Shared scale for GT/Pred.
    vmin = float(np.min([y.min(), p.min()]))
    vmax = float(np.max([y.max(), p.max()]))

    panels = []
    titles = []
    if input_field is not None:
        a = _to_numpy(input_field)
        if a.ndim != 2:
            a = a.squeeze()
        panels.append(a)
        titles.append("input")
    panels += [y, p, abs_err, rel_err_map]
    titles += ["GT", "Pred", "|Pred-GT|", "|Pred-GT|/|GT|"]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0))
    if n == 1:
        axes = [axes]

    for i, (ax, data, t) in enumerate(zip(axes, panels, titles)):
        if t in {"GT", "Pred"}:
            im = ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(data, origin="lower")
        ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if suptitle is None:
        suptitle = f"relL2={e_rel:.3g}  RMSE={e_rmse:.3g}  MAE={e_mae:.3g}"
    fig.suptitle(suptitle)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)
    return paths


def plot_2d_time_slices(
    gt: ArrayLike,
    pred: ArrayLike,
    t_indices: Sequence[int],
    out_path_no_ext: str,
    suptitle: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Space-time output (S,S,T): show GT/Pred/Error for selected time indices."""
    y = _to_numpy(gt)
    p = _to_numpy(pred)
    # Accept shapes (S,S,T) or (S,S,T,1)
    y = np.squeeze(y)
    p = np.squeeze(p)
    assert y.ndim == 3 and p.ndim == 3, f"expected (S,S,T), got {y.shape} and {p.shape}"

    T = y.shape[-1]
    t_indices = [int(t) for t in t_indices]
    for t in t_indices:
        if t < 0 or t >= T:
            raise ValueError(f"t index out of range: {t} (T={T})")

    # Shared scale for GT/Pred across selected times.
    y_sel = y[..., t_indices]
    p_sel = p[..., t_indices]
    vmin = float(np.min([y_sel.min(), p_sel.min()]))
    vmax = float(np.max([y_sel.max(), p_sel.max()]))

    ncols = len(t_indices)
    fig, axes = plt.subplots(3, ncols, figsize=(4.0 * ncols, 10.5))
    if ncols == 1:
        axes = np.array(axes).reshape(3, 1)

    for j, t in enumerate(t_indices):
        gt_t = y[..., t]
        pr_t = p[..., t]
        er_t = np.abs(pr_t - gt_t)

        im0 = axes[0, j].imshow(gt_t, origin="lower", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"GT t={t}")
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])
        fig.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.04)

        im1 = axes[1, j].imshow(pr_t, origin="lower", vmin=vmin, vmax=vmax)
        axes[1, j].set_title(f"Pred t={t}")
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])
        fig.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.04)

        im2 = axes[2, j].imshow(er_t, origin="lower")
        axes[2, j].set_title(f"|Err| t={t}")
        axes[2, j].set_xticks([])
        axes[2, j].set_yticks([])
        fig.colorbar(im2, ax=axes[2, j], fraction=0.046, pad=0.04)

    if suptitle is None:
        suptitle = f"space-time slices (relL2 over all t = {rel_l2(p, y):.3g})"
    fig.suptitle(suptitle)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)
    return paths


def plot_rel_l2_over_time(
    gt: ArrayLike,
    pred: ArrayLike,
    out_path_no_ext: str,
    eps: float = 1e-12,
    title: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Compute rel L2 per time index for space-time fields (S,S,T)."""
    y = np.squeeze(_to_numpy(gt))
    p = np.squeeze(_to_numpy(pred))
    assert y.ndim == 3 and p.ndim == 3, f"expected (S,S,T), got {y.shape} and {p.shape}"
    T = y.shape[-1]

    errs = []
    for t in range(T):
        y_t = y[..., t].reshape(-1)
        p_t = p[..., t].reshape(-1)
        num = np.linalg.norm(p_t - y_t)
        den = np.linalg.norm(y_t)
        errs.append(float(num / (den + eps)))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(range(T), errs)
    ax.set_xlabel("time index")
    ax.set_ylabel("relative L2")
    ax.grid(True, alpha=0.3)
    if title is None:
        title = f"relL2 over time (mean={np.mean(errs):.3g}, max={np.max(errs):.3g})"
    ax.set_title(title)
    fig.tight_layout()
    paths = save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)
    return paths
