#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VALID_MODELS = ("model1", "model2", "model3")
MODEL_LABELS = {
    "model1": "Model 1",
    "model2": "Model 2",
    "model3": "Model 3",
}
MODEL_COLORS = {
    "model1": "#0072B2",
    "model2": "#E69F00",
    "model3": "#009E73",
}


def read_ok_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            try:
                row = dict(row)
                for key in ("rd_alpha", "rd_beta", "alpha", "rd_nu", "test_absL2h"):
                    row[key] = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(row["test_absL2h"]):
                rows.append(row)
        return rows


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def filter_fixed_rd_coeffs(
    rows: list[dict[str, Any]],
    *,
    rd_alpha: float,
    rd_beta: float,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if close(row["rd_alpha"], rd_alpha) and close(row["rd_beta"], rd_beta)
    ]


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: row["test_absL2h"])


def slice_series(
    rows: list[dict[str, Any]],
    *,
    x_name: str,
    fixed_values: dict[str, float],
) -> tuple[list[float], list[float]]:
    groups: dict[float, list[float]] = {}
    for row in rows:
        if all(close(row[name], value) for name, value in fixed_values.items()):
            groups.setdefault(row[x_name], []).append(row["test_absL2h"])
    x_values = sorted(groups)
    y_values = [min(groups[x]) for x in x_values]
    return x_values, y_values


def save_plot(
    *,
    model_rows: dict[str, list[dict[str, Any]]],
    x_name: str,
    xlabel: str,
    fixed_values_by_model: dict[str, dict[str, float]],
    out_path: Path,
    log_x: bool = False,
    legend_loc: str | None = None,
    legend_bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plotted = False
    for model in VALID_MODELS:
        rows = model_rows.get(model, [])
        if not rows:
            continue
        x_values, y_values = slice_series(
            rows,
            x_name=x_name,
            fixed_values=fixed_values_by_model[model],
        )
        if not x_values:
            continue
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        raise RuntimeError(f"No rows matched fixed values for {out_path}")

    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Error", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=14, loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/2026_0602_model123_param_sweep_RD"),
        help="Directory containing model1/model2/model3 summary.csv files.",
    )
    parser.add_argument("--rd-alpha", type=float, default=1.0)
    parser.add_argument("--rd-beta", type=float, default=1.0)
    args = parser.parse_args()

    model_rows = {
        model: filter_fixed_rd_coeffs(
            read_ok_rows(args.root / model / "summary.csv"),
            rd_alpha=args.rd_alpha,
            rd_beta=args.rd_beta,
        )
        for model in VALID_MODELS
        if (args.root / model / "summary.csv").exists()
    }
    model_rows = {model: rows for model, rows in model_rows.items() if rows}
    if set(model_rows) != set(VALID_MODELS):
        missing = sorted(set(VALID_MODELS) - set(model_rows))
        raise SystemExit(f"No fixed-coefficient rows found for: {', '.join(missing)}")

    best_by_model = {model: best_row(rows) for model, rows in model_rows.items()}
    model3_best = best_by_model["model3"]

    model_best_nu = {
        model: {"rd_nu": row["rd_nu"]}
        for model, row in best_by_model.items()
    }
    model3_nu = {
        model: {"rd_nu": model3_best["rd_nu"]}
        for model in VALID_MODELS
    }
    model_best_alpha = {
        model: {"alpha": row["alpha"]}
        for model, row in best_by_model.items()
    }
    model3_alpha = {
        model: {"alpha": model3_best["alpha"]}
        for model in VALID_MODELS
    }

    save_plot(
        model_rows=model_rows,
        x_name="alpha",
        xlabel=r"$\alpha$",
        fixed_values_by_model=model_best_nu,
        out_path=args.root / "alpha_vs_error_rd_coeffs_fixed_model_best_nu_all_models",
    )
    save_plot(
        model_rows=model_rows,
        x_name="alpha",
        xlabel=r"$\alpha$",
        fixed_values_by_model=model3_nu,
        out_path=args.root / "alpha_vs_error_rd_coeffs_fixed_model3_nu_all_models",
        legend_loc="center right",
        legend_bbox_to_anchor=(0.985, 0.40),
    )
    save_plot(
        model_rows=model_rows,
        x_name="rd_nu",
        xlabel=r"$\tilde{\nu}$",
        fixed_values_by_model=model_best_alpha,
        out_path=args.root / "nu_vs_error_rd_coeffs_fixed_model_best_alpha_all_models",
        log_x=True,
    )
    save_plot(
        model_rows=model_rows,
        x_name="rd_nu",
        xlabel=r"$\tilde{\nu}$",
        fixed_values_by_model=model3_alpha,
        out_path=args.root / "nu_vs_error_rd_coeffs_fixed_model3_alpha_all_models",
        log_x=True,
    )

    settings = {
        "fixed_rd_coefficients": {
            "rd_alpha": args.rd_alpha,
            "rd_beta": args.rd_beta,
        },
        "metric": "test_absL2h",
        "model_best_fixed_values": {
            model: {
                "alpha": row["alpha"],
                "rd_nu": row["rd_nu"],
                "test_absL2h": row["test_absL2h"],
                "run_dir": row.get("run_dir"),
            }
            for model, row in best_by_model.items()
        },
        "model3_aligned_fixed_values": {
            "source_model": "model3",
            "alpha": model3_best["alpha"],
            "rd_nu": model3_best["rd_nu"],
            "test_absL2h": model3_best["test_absL2h"],
            "run_dir": model3_best.get("run_dir"),
        },
        "plots": {
            "alpha_vs_error_rd_coeffs_fixed_model_best_nu_all_models": (
                "rd_nu is fixed to each model's best row after filtering by rd_alpha and rd_beta."
            ),
            "alpha_vs_error_rd_coeffs_fixed_model3_nu_all_models": (
                "rd_nu is fixed to model3's best row after filtering by rd_alpha and rd_beta."
            ),
            "nu_vs_error_rd_coeffs_fixed_model_best_alpha_all_models": (
                "alpha is fixed to each model's best row after filtering by rd_alpha and rd_beta."
            ),
            "nu_vs_error_rd_coeffs_fixed_model3_alpha_all_models": (
                "alpha is fixed to model3's best row after filtering by rd_alpha and rd_beta."
            ),
        },
    }
    (args.root / "rd_coeffs_fixed_slice_plot_settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
