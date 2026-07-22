#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in {row.get('run_id')}")
    return value


def heatmap_matrix(
    rows: list[dict[str, str]], n_tars: list[int], n_surs: list[int], value
) -> np.ndarray:
    matrix = np.full((len(n_tars), len(n_surs)), np.nan)
    lookup = {(int(row["n_tar"]), int(row["n_sur"])): row for row in rows}
    for i, n_tar in enumerate(n_tars):
        for j, n_sur in enumerate(n_surs):
            row = lookup.get((n_tar, n_sur))
            if row is not None:
                matrix[i, j] = value(row)
    return matrix


def draw_heatmap(
    ax, matrix, n_tars, n_surs, title, colorbar_label, *, vmin=None, vmax=None,
    colorbar_format=None,
):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(n_surs)), n_surs)
    ax.set_yticks(range(len(n_tars)), n_tars)
    ax.set_xlabel(r"$n_{\mathrm{sur}}$")
    ax.set_ylabel(r"$n_{\mathrm{tar}}$")
    ax.set_title(title)
    for i in range(len(n_tars)):
        for j in range(len(n_surs)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "failed", ha="center", va="center", color="#555555", fontsize=8)
    colorbar = plt.colorbar(
        image, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04
    )
    if colorbar_format is not None:
        colorbar.formatter = FormatStrFormatter(colorbar_format)
        colorbar.update_ticks()


def save(fig, output_dir: Path, stem: str, outputs: list[str]) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        name = f"{stem}.{suffix}"
        fig.savefig(output_dir / name, dpi=200, bbox_inches="tight")
        outputs.append(name)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Paper 1 E1 sweep summaries")
    parser.add_argument("--sweep-dir", default="outputs_paper1/paper1_e1_sweep")
    parser.add_argument("--q", type=int, default=65)
    parser.add_argument("--j-ntar", type=int, default=256)
    parser.add_argument("--j-nsur", type=int, default=1024)
    parser.add_argument("--noise-level", type=float, default=1e-2)
    args = parser.parse_args()

    output_dir = Path(args.sweep_dir)
    selected = read_csv(output_dir / "sweep_selected_results.csv")
    diagnostics = read_csv(output_dir / "sweep_readout_diagnostics.csv")
    noise = read_csv(output_dir / "sweep_noise_summary.csv")

    selected_q = [row for row in selected if int(row["q"]) == args.q]
    diagnostics_q = [row for row in diagnostics if int(row["q"]) == args.q]
    full_selected = [row for row in selected_q if row["full_observation"].lower() == "true"]
    full_diagnostics = [row for row in diagnostics_q if row["full_observation"].lower() == "true"]
    n_tars = sorted({int(row["n_tar"]) for row in full_selected} | {128, 256, 512, 1024})
    n_surs = sorted({int(row["n_sur"]) for row in full_selected} | {128, 256, 512, 1024})
    regimes = ("stable", "unstable")
    outputs: list[str] = []

    # 4.1 field error heatmap.
    matrices = {
        regime: heatmap_matrix(
            [row for row in full_selected if row["regime"] == regime], n_tars, n_surs,
            lambda row: math.log10(numeric(row, "full_reference_field_relative_l2_mean")),
        ) for regime in regimes
    }
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices.values()])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, regime in zip(axes, regimes):
        draw_heatmap(ax, matrices[regime], n_tars, n_surs, regime.capitalize(),
                     r"$\log_{10} E_{\mathrm{field}}$", vmin=finite.min(), vmax=finite.max(),
                     colorbar_format="%.8f")
    fig.suptitle(f"Full-observation field error (q={args.q}, J=n_sur)")
    save(fig, output_dir, "sweep_field_error_heatmap_q65", outputs)

    # 4.2 field error / representation floor.
    matrices = {
        regime: heatmap_matrix(
            [row for row in full_selected if row["regime"] == regime], n_tars, n_surs,
            lambda row: numeric(row, "field_error_to_representation_floor_ratio"),
        ) for regime in regimes
    }
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices.values()])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, regime in zip(axes, regimes):
        draw_heatmap(ax, matrices[regime], n_tars, n_surs, regime.capitalize(),
                     r"$E_{\mathrm{field}}/E_{\mathrm{repr}}$", vmin=finite.min(), vmax=finite.max(),
                     colorbar_format="%.8g")
    fig.suptitle(f"Field error relative to representation floor (q={args.q}, J=n_sur)")
    save(fig, output_dir, "sweep_field_floor_ratio_heatmap_q65", outputs)

    # 4.3 learned / ideal operator norm.
    matrices = {
        regime: heatmap_matrix(
            [row for row in full_diagnostics if row["regime"] == regime], n_tars, n_surs,
            lambda row: numeric(row, "learned_operator_norm") / numeric(row, "ideal_operator_norm"),
        ) for regime in regimes
    }
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices.values()])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, regime in zip(axes, regimes):
        draw_heatmap(ax, matrices[regime], n_tars, n_surs, regime.capitalize(),
                     r"$\|W_{\mathrm{learned}}\|_{op}/\|W_{\mathrm{ideal}}\|_{op}$",
                     vmin=finite.min(), vmax=finite.max(), colorbar_format="%.8g")
    fig.suptitle(f"Learned / ideal operator norm (q={args.q}, J=n_sur)")
    save(fig, output_dir, "sweep_operator_norm_ratio_heatmap_q65", outputs)

    # 4.4 multiplier recovery diagnostics.
    metrics = (
        ("max_identifiable_diagonal_relative_error", "Maximum diagonal relative error",
         r"$\log_{10} E_{\mathrm{diag,max}}$"),
        ("identifiable_off_diagonal_relative_norm", "Off-diagonal relative norm",
         r"$\log_{10} E_{\mathrm{off}}$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for row_index, (metric, title, label) in enumerate(metrics):
        matrices = {
            regime: heatmap_matrix(
                [row for row in full_diagnostics if row["regime"] == regime], n_tars, n_surs,
                lambda row, metric=metric: math.log10(max(numeric(row, metric), np.finfo(float).tiny)),
            ) for regime in regimes
        }
        finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices.values()])
        for column, regime in enumerate(regimes):
            draw_heatmap(axes[row_index, column], matrices[regime], n_tars, n_surs,
                         f"{regime.capitalize()} — {title}", label,
                         vmin=finite.min(), vmax=finite.max(), colorbar_format="%.4f")
    fig.suptitle(f"Fourier multiplier recovery errors (q={args.q}, J=n_sur)")
    save(fig, output_dir, "sweep_multiplier_recovery_heatmaps_q65", outputs)

    # Dedicated J sweep: fixed n_tar=256, n_sur=1024.
    def j_rows(rows, *, noise_level=None):
        result = [row for row in rows if int(row["q"]) == args.q
                  and int(row["n_tar"]) == args.j_ntar and int(row["n_sur"]) == args.j_nsur]
        if noise_level is not None:
            result = [row for row in result if math.isclose(numeric(row, "noise_level"), noise_level,
                                                             rel_tol=0.0, abs_tol=1e-15)]
        return sorted(result, key=lambda row: (row["regime"], int(row["J"])))

    j_selected, j_diagnostics = j_rows(selected), j_rows(diagnostics)
    j_noise = j_rows(noise, noise_level=args.noise_level)
    expected_js = {256, 512, 1024}
    if {int(row["J"]) for row in j_selected} != expected_js:
        raise ValueError("dedicated J sweep is incomplete")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for regime in regimes:
        rows = [row for row in j_selected if row["regime"] == regime]
        ax.plot([int(row["J"]) for row in rows], [numeric(row, "full_reference_field_relative_l2_mean") for row in rows], "o-", label=f"{regime} learned")
        ax.plot([int(row["J"]) for row in rows], [numeric(row, "output_representation_floor_mean") for row in rows], "--", label=f"{regime} representation floor")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(sorted(expected_js), sorted(expected_js))
    ax.set_xlabel("J"); ax.set_ylabel(r"$E_{\mathrm{field}}$"); ax.grid(True, which="both", alpha=0.3); ax.legend()
    ax.set_title(f"Field error vs J (q={args.q}, n_tar={args.j_ntar}, n_sur={args.j_nsur})")
    save(fig, output_dir, "sweep_field_error_vs_J_q65", outputs)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for regime in regimes:
        rows = [row for row in j_diagnostics if row["regime"] == regime]
        js = [int(row["J"]) for row in rows]
        ax.plot(js, [numeric(row, "learned_operator_norm") for row in rows], "o-", label=f"learned {regime}")
        ax.plot(js, [numeric(row, "ideal_operator_norm") for row in rows], "--", label=f"ideal {regime}")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(sorted(expected_js), sorted(expected_js))
    ax.set_xlabel("J"); ax.set_ylabel("operator norm"); ax.grid(True, which="both", alpha=0.3); ax.legend()
    ax.set_title(f"Operator norm vs J (q={args.q}, n_tar={args.j_ntar}, n_sur={args.j_nsur})")
    save(fig, output_dir, "sweep_operator_norm_vs_J_q65", outputs)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for regime in regimes:
        rows = [row for row in j_noise if row["regime"] == regime]
        ax.plot([int(row["J"]) for row in rows], [numeric(row, "output_perturbation_rms_mean") for row in rows], "o-", label=regime)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(sorted(expected_js), sorted(expected_js))
    ax.set_xlabel("J"); ax.set_ylabel("output perturbation RMS"); ax.grid(True, which="both", alpha=0.3); ax.legend()
    ax.set_title(f"Noise sensitivity vs J (q={args.q}, delta={args.noise_level:g})")
    save(fig, output_dir, "sweep_noise_sensitivity_vs_J_q65_delta1e-2", outputs)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for regime in regimes:
        rows = [row for row in j_diagnostics if row["regime"] == regime]
        ax.plot([int(row["J"]) for row in rows], [numeric(row, "identifiable_nonconstant_fraction") for row in rows], "o-", label=regime)
    ax.set_xscale("log", base=2); ax.set_xticks(sorted(expected_js), sorted(expected_js)); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("J"); ax.set_ylabel("identifiable nonconstant fraction"); ax.grid(True, which="both", alpha=0.3); ax.legend()
    ax.set_title(f"Identifiability vs J (q={args.q}, n_tar={args.j_ntar}, n_sur={args.j_nsur})")
    save(fig, output_dir, "sweep_identifiability_vs_J_q65", outputs)

    manifest = {
        "schema_version": "paper1-e1-sweep-plots-v1",
        "q": args.q,
        "heatmaps": {"selection": "full_observation == True (J == n_sur)", "n_tar": n_tars, "n_sur": n_surs},
        "j_sweep": {"n_tar": args.j_ntar, "n_sur": args.j_nsur, "J": sorted(expected_js), "noise_level": args.noise_level},
        "inputs": {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in ("sweep_selected_results.csv", "sweep_readout_diagnostics.csv", "sweep_noise_summary.csv")
        },
        "outputs": outputs,
    }
    (output_dir / "sweep_plot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
