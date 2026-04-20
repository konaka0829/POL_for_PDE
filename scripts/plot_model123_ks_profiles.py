#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


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
PARAM_LABELS = {
    "ks_eta": r"$\eta$",
    "ks_kappa": r"$\kappa$",
    "Ttilde": r"$\widetilde{T}$",
}
PARAM_TITLES = {
    "ks_eta": "eta vs. error",
    "ks_kappa": "kappa vs. error",
    "Ttilde": "Ttilde vs. error",
}
PARAM_SCALES = {
    "ks_eta": "log",
    "ks_kappa": "log",
    "Ttilde": "linear",
}


def load_rows(summary_csv: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with summary_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            rows.append(
                {
                    "model": row["model"],
                    "ks_eta": float(row["ks_eta"]),
                    "ks_kappa": float(row["ks_kappa"]),
                    "Ttilde": float(row["Ttilde"]),
                    "test_relL2": float(row["test_relL2"]),
                }
            )
    if not rows:
        raise ValueError(f"No valid rows found in {summary_csv}")
    return rows


def best_row(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    return min(rows, key=lambda row: float(row["test_relL2"]))


def save_figure(fig: plt.Figure, out_no_ext: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_no_ext.with_suffix(f".{ext}"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def filter_rows(
    rows: list[dict[str, float | str]],
    fixed_params: dict[str, float],
) -> list[dict[str, float | str]]:
    filtered = rows
    for key, value in fixed_params.items():
        filtered = [
            row for row in filtered if np.isclose(float(row[key]), value, rtol=0.0, atol=1e-12)
        ]
    return filtered


def make_profile_plot(
    model_rows: dict[str, list[dict[str, float | str]]],
    target_param: str,
    fixed_params_by_model: dict[str, dict[str, float]],
    out_no_ext: Path,
    title_suffix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    all_errors: list[float] = []

    for model in MODELS:
        rows = filter_rows(model_rows[model], fixed_params_by_model[model])
        rows.sort(key=lambda row: float(row[target_param]))
        x = [float(row[target_param]) for row in rows]
        y = [float(row["test_relL2"]) for row in rows]
        all_errors.extend(y)
        fixed_text = ", ".join(
            f"{PARAM_LABELS[key]}={value:g}" for key, value in fixed_params_by_model[model].items()
        )
        ax.plot(
            x,
            y,
            marker=MODEL_MARKERS[model],
            linewidth=1.8,
            label=f"{MODEL_LABELS[model]} ({fixed_text})",
        )

    ax.set_xlabel(PARAM_LABELS[target_param])
    ax.set_ylabel("Test relative $L^2$ error")
    ax.set_title(f"{PARAM_TITLES[target_param]} ({title_suffix})")
    ax.set_xscale(PARAM_SCALES[target_param])
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    if all_errors:
        ymin = min(all_errors)
        ymax = max(all_errors)
        ax.set_ylim(ymin / 1.15, ymax * 1.15)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_no_ext)


def build_grid(
    rows: list[dict[str, float | str]],
    x_param: str,
    y_param: str,
    fixed_params: dict[str, float],
) -> tuple[list[float], list[float], np.ndarray]:
    filtered = filter_rows(rows, fixed_params)
    x_values = sorted({float(row[x_param]) for row in filtered})
    y_values = sorted({float(row[y_param]) for row in filtered})
    x_to_idx = {value: idx for idx, value in enumerate(x_values)}
    y_to_idx = {value: idx for idx, value in enumerate(y_values)}
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for row in filtered:
        grid[y_to_idx[float(row[y_param])], x_to_idx[float(row[x_param])]] = float(
            row["test_relL2"]
        )
    return x_values, y_values, grid


def plot_heatmaps(
    model_rows: dict[str, list[dict[str, float | str]]],
    best_rows: dict[str, dict[str, float | str]],
    out_no_ext: Path,
) -> None:
    finite_values: list[float] = []
    grids: dict[str, tuple[list[float], list[float], np.ndarray]] = {}
    fixed_ttilde: dict[str, float] = {}
    for model in MODELS:
        fixed = {"Ttilde": float(best_rows[model]["Ttilde"])}
        fixed_ttilde[model] = fixed["Ttilde"]
        grid_info = build_grid(model_rows[model], "ks_eta", "ks_kappa", fixed)
        grids[model] = grid_info
        finite = grid_info[2][np.isfinite(grid_info[2])]
        finite_values.extend(finite.tolist())

    norm = mcolors.LogNorm(vmin=min(finite_values), vmax=max(finite_values))
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7), constrained_layout=True)
    image = None
    for ax, model in zip(axes, MODELS):
        x_values, y_values, grid = grids[model]
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", norm=norm)
        xticks = np.arange(len(x_values))
        yticks = np.arange(len(y_values))
        ax.set_xticks(xticks, labels=[f"{v:g}" for v in x_values], rotation=45, ha="right")
        ax.set_yticks(yticks, labels=[f"{v:g}" for v in y_values])
        ax.set_xlabel(PARAM_LABELS["ks_eta"])
        ax.set_ylabel(PARAM_LABELS["ks_kappa"])
        ax.set_title(f"{MODEL_LABELS[model]} ($\\widetilde{{T}}^*={fixed_ttilde[model]:g}$)")
        best = best_rows[model]
        best_x = x_values.index(float(best["ks_eta"]))
        best_y = y_values.index(float(best["ks_kappa"]))
        ax.scatter([best_x], [best_y], s=150, marker="*", color="white", edgecolors="black")

    assert image is not None
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.94)
    cbar.set_label("Test relative $L^2$ error")
    fig.suptitle(r"Error heatmaps on the $(\eta, \kappa)$ plane at each model's optimal $\widetilde{T}$")
    save_figure(fig, out_no_ext)


def write_best_summary(
    best_rows: dict[str, dict[str, float | str]],
    out_dir: Path,
) -> None:
    serializable = {
        model: {
            "ks_eta": float(row["ks_eta"]),
            "ks_kappa": float(row["ks_kappa"]),
            "Ttilde": float(row["Ttilde"]),
            "test_relL2": float(row["test_relL2"]),
        }
        for model, row in best_rows.items()
    }
    (out_dir / "best_runs.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    with (out_dir / "best_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "ks_eta", "ks_kappa", "Ttilde", "test_relL2"]
        )
        writer.writeheader()
        for model in MODELS:
            writer.writerow({"model": model, **serializable[model]})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create publication-style KS parameter comparison plots for model1/2/3."
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("outputs/model123_param_sweep_20260409_KS_refined_unified"),
        help="Root directory that contains model1/model2/model3 summary.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <sweep-root>/paper_plots.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (args.sweep_root / "paper_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows = {model: load_rows(args.sweep_root / model / "summary.csv") for model in MODELS}
    best_rows = {model: best_row(rows) for model, rows in model_rows.items()}
    model3_best = best_rows["model3"]

    fixed_by_own_optimum = {
        model: {
            "ks_kappa": float(best_rows[model]["ks_kappa"]),
            "Ttilde": float(best_rows[model]["Ttilde"]),
        }
        for model in MODELS
    }
    make_profile_plot(
        model_rows,
        "ks_eta",
        fixed_by_own_optimum,
        out_dir / "ks_eta_vs_error_fixed_optimal_kappa_ttilde",
        "fixed at each model's optimal $(\\kappa, \\widetilde{T})$",
    )

    fixed_by_own_optimum = {
        model: {
            "ks_eta": float(best_rows[model]["ks_eta"]),
            "Ttilde": float(best_rows[model]["Ttilde"]),
        }
        for model in MODELS
    }
    make_profile_plot(
        model_rows,
        "ks_kappa",
        fixed_by_own_optimum,
        out_dir / "ks_kappa_vs_error_fixed_optimal_eta_ttilde",
        "fixed at each model's optimal $(\\eta, \\widetilde{T})$",
    )

    fixed_by_own_optimum = {
        model: {
            "ks_eta": float(best_rows[model]["ks_eta"]),
            "ks_kappa": float(best_rows[model]["ks_kappa"]),
        }
        for model in MODELS
    }
    make_profile_plot(
        model_rows,
        "Ttilde",
        fixed_by_own_optimum,
        out_dir / "ttilde_vs_error_fixed_optimal_eta_kappa",
        "fixed at each model's optimal $(\\eta, \\kappa)$",
    )

    shared_model3_eta_kappa = {
        model: {
            "ks_eta": float(model3_best["ks_eta"]),
            "ks_kappa": float(model3_best["ks_kappa"]),
        }
        for model in MODELS
    }
    make_profile_plot(
        model_rows,
        "Ttilde",
        shared_model3_eta_kappa,
        out_dir / "ttilde_vs_error_fixed_model3_optimal_eta_kappa",
        "fixed at Model 3 optimum $(\\eta, \\kappa)$",
    )

    shared_model3_kappa_ttilde = {
        model: {
            "ks_kappa": float(model3_best["ks_kappa"]),
            "Ttilde": float(model3_best["Ttilde"]),
        }
        for model in MODELS
    }
    make_profile_plot(
        model_rows,
        "ks_eta",
        shared_model3_kappa_ttilde,
        out_dir / "ks_eta_vs_error_fixed_model3_optimal_kappa_ttilde",
        "fixed at Model 3 optimum $(\\kappa, \\widetilde{T})$",
    )

    shared_model3_eta_ttilde = {
        model: {
            "ks_eta": float(model3_best["ks_eta"]),
            "Ttilde": float(model3_best["Ttilde"]),
        }
        for model in MODELS
    }
    make_profile_plot(
        model_rows,
        "ks_kappa",
        shared_model3_eta_ttilde,
        out_dir / "ks_kappa_vs_error_fixed_model3_optimal_eta_ttilde",
        "fixed at Model 3 optimum $(\\eta, \\widetilde{T})$",
    )

    plot_heatmaps(
        model_rows,
        best_rows,
        out_dir / "ks_eta_kappa_error_heatmaps",
    )
    write_best_summary(best_rows, out_dir)

    for model in MODELS:
        best = best_rows[model]
        print(
            f"{model}: "
            f"eta={float(best['ks_eta']):g}, "
            f"kappa={float(best['ks_kappa']):g}, "
            f"Ttilde={float(best['Ttilde']):g}, "
            f"test_relL2={float(best['test_relL2']):.9g}"
        )
    print(f"saved plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
