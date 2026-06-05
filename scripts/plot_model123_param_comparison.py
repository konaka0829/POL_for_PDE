#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_LABELS = {
    "model1": "Model 1",
    "model2": "Model 2",
    "model3": "Model 3",
}

# Okabe-Ito colorblind-safe palette.
MODEL_COLORS = {
    "model1": "#0072B2",
    "model2": "#E69F00",
    "model3": "#009E73",
}


def read_ok_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "ok"]


def profile_min_error(rows: list[dict[str, str]], x_name: str) -> tuple[list[float], list[float]]:
    groups: dict[float, list[float]] = {}
    for row in rows:
        groups.setdefault(float(row[x_name]), []).append(float(row["test_absL2h"]))
    x_values = sorted(groups)
    y_values = [min(groups[x]) for x in x_values]
    return x_values, y_values


def save_plot(
    *,
    model_rows: dict[str, list[dict[str, str]]],
    x_name: str,
    xlabel: str,
    out_path: Path,
    log_x: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for model in ("model1", "model2", "model3"):
        rows = model_rows.get(model, [])
        if not rows:
            continue
        x_values, y_values = profile_min_error(rows, x_name)
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )

    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Error", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=14)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/2026_0520_model123_param_sweep"),
        help="Directory containing model1/model2/model3 summary.csv files.",
    )
    args = parser.parse_args()

    model_rows = {
        model: read_ok_rows(args.root / model / "summary.csv")
        for model in ("model1", "model2", "model3")
        if (args.root / model / "summary.csv").exists()
    }
    if not model_rows:
        raise SystemExit(f"No model summary.csv files found under {args.root}")

    save_plot(
        model_rows=model_rows,
        x_name="res_burgers_nu",
        xlabel=r"$\tilde{\nu}$",
        out_path=args.root / "nu_vs_error_all_models",
        log_x=True,
    )
    save_plot(
        model_rows=model_rows,
        x_name="alpha",
        xlabel=r"$\alpha$",
        out_path=args.root / "alpha_vs_error_all_models",
    )


if __name__ == "__main__":
    main()
