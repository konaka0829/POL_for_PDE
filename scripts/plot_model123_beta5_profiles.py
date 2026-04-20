#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


MODELS = ("model1", "model2", "model3")
MODEL_LABELS = {
    "model1": "Model 1",
    "model2": "Model 2",
    "model3": "Model 3",
}
MODEL_MARKERS = {
    "model1": "o",
    "model2": "s",
    "model3": "^",
}


def load_rows(summary_csv: Path, beta: float) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with summary_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            rd_beta = float(row["rd_beta"])
            if not np.isclose(rd_beta, beta):
                continue
            rows.append(
                {
                    "model": row["model"],
                    "rd_nu": float(row["rd_nu"]),
                    "rd_beta": rd_beta,
                    "Ttilde": float(row["Ttilde"]),
                    "test_relL2": float(row["test_relL2"]),
                }
            )
    if not rows:
        raise ValueError(f"No valid rows found in {summary_csv} for beta={beta}")
    return rows


def best_row(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    return min(rows, key=lambda row: float(row["test_relL2"]))


def save_figure(fig: plt.Figure, out_no_ext: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_no_ext.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ttilde_vs_error(
    model_rows: dict[str, list[dict[str, float | str]]],
    best_rows: dict[str, dict[str, float | str]],
    out_no_ext: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for model in MODELS:
        nu_star = float(best_rows[model]["rd_nu"])
        rows = [
            row for row in model_rows[model] if np.isclose(float(row["rd_nu"]), nu_star)
        ]
        rows.sort(key=lambda row: float(row["Ttilde"]))
        ax.plot(
            [float(row["Ttilde"]) for row in rows],
            [float(row["test_relL2"]) for row in rows],
            marker=MODEL_MARKERS[model],
            linewidth=1.8,
            label=f"{MODEL_LABELS[model]} (rd_nu*={nu_star:g})",
        )

    ax.set_xlabel("T_tilde")
    ax.set_ylabel("error (test relL2)")
    ax.set_title("beta = 5: T_tilde vs. error with optimal rd_nu")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_no_ext)


def plot_nu_vs_error(
    model_rows: dict[str, list[dict[str, float | str]]],
    best_rows: dict[str, dict[str, float | str]],
    out_no_ext: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for model in MODELS:
        ttilde_star = float(best_rows[model]["Ttilde"])
        rows = [
            row for row in model_rows[model] if np.isclose(float(row["Ttilde"]), ttilde_star)
        ]
        rows.sort(key=lambda row: float(row["rd_nu"]))
        ax.plot(
            [float(row["rd_nu"]) for row in rows],
            [float(row["test_relL2"]) for row in rows],
            marker=MODEL_MARKERS[model],
            linewidth=1.8,
            label=f"{MODEL_LABELS[model]} (T_tilde*={ttilde_star:g})",
        )

    ax.set_xlabel("rd_nu")
    ax.set_ylabel("error (test relL2)")
    ax.set_title("beta = 5: rd_nu vs. error with optimal T_tilde")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_no_ext)


def plot_ttilde_vs_error_shared_nu(
    model_rows: dict[str, list[dict[str, float | str]]],
    fixed_nu: float,
    out_no_ext: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for model in MODELS:
        rows = [
            row for row in model_rows[model] if np.isclose(float(row["rd_nu"]), fixed_nu)
        ]
        rows.sort(key=lambda row: float(row["Ttilde"]))
        ax.plot(
            [float(row["Ttilde"]) for row in rows],
            [float(row["test_relL2"]) for row in rows],
            marker=MODEL_MARKERS[model],
            linewidth=1.8,
            label=MODEL_LABELS[model],
        )

    ax.set_xlabel("T_tilde")
    ax.set_ylabel("error (test relL2)")
    ax.set_title(f"beta = 5: T_tilde vs. error with shared rd_nu={fixed_nu:g} (Model 3 optimum)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_no_ext)


def plot_nu_vs_error_shared_ttilde(
    model_rows: dict[str, list[dict[str, float | str]]],
    fixed_ttilde: float,
    out_no_ext: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for model in MODELS:
        rows = [
            row for row in model_rows[model] if np.isclose(float(row["Ttilde"]), fixed_ttilde)
        ]
        rows.sort(key=lambda row: float(row["rd_nu"]))
        ax.plot(
            [float(row["rd_nu"]) for row in rows],
            [float(row["test_relL2"]) for row in rows],
            marker=MODEL_MARKERS[model],
            linewidth=1.8,
            label=MODEL_LABELS[model],
        )

    ax.set_xlabel("rd_nu")
    ax.set_ylabel("error (test relL2)")
    ax.set_title(
        f"beta = 5: rd_nu vs. error with shared T_tilde={fixed_ttilde:g} (Model 3 optimum)"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_no_ext)


def plot_3d_summary(
    model_rows: dict[str, list[dict[str, float | str]]],
    best_rows: dict[str, dict[str, float | str]],
    out_no_ext: Path,
) -> None:
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    for model in MODELS:
        rows = model_rows[model]
        x = np.asarray([float(row["Ttilde"]) for row in rows], dtype=float)
        y = np.asarray([float(row["rd_nu"]) for row in rows], dtype=float)
        z = np.asarray([float(row["test_relL2"]) for row in rows], dtype=float)
        ax.scatter(
            x,
            y,
            z,
            s=18,
            alpha=0.55,
            marker=MODEL_MARKERS[model],
            label=MODEL_LABELS[model],
        )

        best = best_rows[model]
        ax.scatter(
            [float(best["Ttilde"])],
            [float(best["rd_nu"])],
            [float(best["test_relL2"])],
            s=90,
            color="red",
            edgecolors="black",
            linewidths=0.6,
            marker="*",
        )

    ax.set_xlabel("T_tilde")
    ax.set_ylabel("rd_nu")
    ax.set_zlabel("error (test relL2)")
    ax.set_title("beta = 5: T_tilde, rd_nu, and error")
    ax.set_yscale("log")
    ax.set_zscale("log")
    ax.view_init(elev=24, azim=-57)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, out_no_ext)


def build_grid(
    rows: list[dict[str, float | str]],
) -> tuple[list[float], list[float], np.ndarray]:
    x_values = sorted({float(row["Ttilde"]) for row in rows})
    y_values = sorted({float(row["rd_nu"]) for row in rows})
    x_to_idx = {value: idx for idx, value in enumerate(x_values)}
    y_to_idx = {value: idx for idx, value in enumerate(y_values)}
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for row in rows:
        xi = x_to_idx[float(row["Ttilde"])]
        yi = y_to_idx[float(row["rd_nu"])]
        grid[yi, xi] = float(row["test_relL2"])
    return x_values, y_values, grid


def plot_error_heatmaps(
    model_rows: dict[str, list[dict[str, float | str]]],
    best_rows: dict[str, dict[str, float | str]],
    out_no_ext: Path,
) -> None:
    grids = {}
    finite_values: list[float] = []
    for model in MODELS:
        x_values, y_values, grid = build_grid(model_rows[model])
        grids[model] = (x_values, y_values, grid)
        finite = grid[np.isfinite(grid)]
        finite_values.extend(finite.tolist())

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        x_values, y_values, grid = grids[model]
        image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            norm=norm,
        )
        xticks = list(range(0, len(x_values), max(1, len(x_values) // 6)))
        yticks = list(range(0, len(y_values), max(1, len(y_values) // 6)))
        ax.set_xticks(xticks, labels=[f"{x_values[i]:g}" for i in xticks], rotation=45, ha="right")
        ax.set_yticks(yticks, labels=[f"{y_values[i]:g}" for i in yticks])
        ax.set_xlabel("T_tilde")
        ax.set_ylabel("rd_nu")
        ax.set_title(MODEL_LABELS[model])

        best = best_rows[model]
        best_x = x_values.index(float(best["Ttilde"]))
        best_y = y_values.index(float(best["rd_nu"]))
        ax.scatter([best_x], [best_y], s=140, marker="*", color="red", edgecolors="white", linewidths=0.8)

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("error (test relL2)")
    fig.suptitle("beta = 5: error heatmaps on the (T_tilde, rd_nu) plane")
    save_figure(fig, out_no_ext)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create beta=5 comparison plots for model1/2/3 from parameter sweep summaries."
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("outputs/model123_param_sweep_20260408_1752"),
        help="Root directory that contains model1/model2/model3 summary.csv files.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=5.0,
        help="Fixed beta value to analyze.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for generated plots. Defaults to <sweep-root>/beta5_plots.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (args.sweep_root / f"beta{args.beta:g}_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows = {
        model: load_rows(args.sweep_root / model / "summary.csv", args.beta) for model in MODELS
    }
    best_rows = {model: best_row(rows) for model, rows in model_rows.items()}
    model3_best = best_rows["model3"]

    plot_ttilde_vs_error(
        model_rows,
        best_rows,
        out_dir / "beta5_ttilde_vs_error_fixed_optimal_nu",
    )
    plot_nu_vs_error(
        model_rows,
        best_rows,
        out_dir / "beta5_nu_vs_error_fixed_optimal_ttilde",
    )
    plot_3d_summary(
        model_rows,
        best_rows,
        out_dir / "beta5_ttilde_nu_error_3d",
    )
    plot_ttilde_vs_error_shared_nu(
        model_rows,
        float(model3_best["rd_nu"]),
        out_dir / "beta5_ttilde_vs_error_fixed_model3_optimal_nu",
    )
    plot_nu_vs_error_shared_ttilde(
        model_rows,
        float(model3_best["Ttilde"]),
        out_dir / "beta5_nu_vs_error_fixed_model3_optimal_ttilde",
    )
    plot_error_heatmaps(
        model_rows,
        best_rows,
        out_dir / "beta5_ttilde_nu_error_heatmaps",
    )

    for model in MODELS:
        best = best_rows[model]
        print(
            f"{model}: best beta={args.beta:g} -> "
            f"rd_nu={float(best['rd_nu']):g}, "
            f"Ttilde={float(best['Ttilde']):g}, "
            f"test_relL2={float(best['test_relL2']):.9g}"
        )
    print(f"saved plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
