"""Figures for Paper 1 E2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_formats(
    fig: Any,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            records.append({
                "status": "created",
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "format": fmt,
                **metadata,
            })
        except Exception as exc:
            records.append({
                "status": "fail",
                "relative_path": path.name,
                "format": fmt,
                **metadata,
                "reason": f"{type(exc).__name__}: {exc}",
            })
    return records


def create_e2_plots(
    output_dir: Path,
    result: dict[str, Any],
    config: Any,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 180,
) -> list[dict[str, Any]]:
    e2 = config.e2
    assert e2 is not None
    rows = result["test_sweep"]
    aggregates = result["model3_test_aggregate"]
    representatives = result["shared_representatives"]
    optima = result["model_specific_optima"]
    panels = (("burgers", "nu_tilde"), ("burgers", "T_tilde"),
              ("reaction_diffusion", "nu_tilde"), ("reaction_diffusion", "T_tilde"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    colors = {"model1": "#0072B2", "model2": "#E69F00", "model3": "#009E73"}
    panel_meta = []
    for ax, (family, axis) in zip(axes.ravel(), panels):
        for model in ("model1", "model2", "model3"):
            selected = sorted([r for r in rows if r["family"] == family and r["sweep_axis"] == axis and r["model"] == model],
                              key=lambda r: r["parameter_value"])
            x, y = [r["parameter_value"] for r in selected], [r["field_relative_l2_mean"] for r in selected]
            ax.plot(x, y, "o-", color=colors[model], label=model)
            optimum = optima[family][model][axis]
            ax.axvline(optimum, color=colors[model], linestyle=":", alpha=.5)
            if model == "model3":
                lookup = {(r["nu_tilde"], r["T_tilde"]): r for r in aggregates if r["family"] == family and r["sweep_axis"] == axis}
                lows, highs = [], []
                for row in selected:
                    item = lookup.get((row["nu_tilde"], row["T_tilde"]))
                    lows.append(item["ci95_low"] if item and item["ci95_low"] is not None else row["field_relative_l2_mean"])
                    highs.append(item["ci95_high"] if item and item["ci95_high"] is not None else row["field_relative_l2_mean"])
                ax.fill_between(x, lows, highs, color=colors[model], alpha=.18)
        shared = representatives[family][axis.replace("_tilde", "_star")]
        ax.axvline(shared, color="black", linestyle="--", label="shared" if axis == "nu_tilde" else None)
        if selected:
            ax.axhline(selected[0]["E_repr_q"], color="gray", linestyle="-.", label="E_repr")
        ax.set_yscale("log"); ax.grid(True, which="both", alpha=.25)
        ax.set_xlabel(axis); ax.set_title(f"{family}: {axis}")
        panel_meta.append({"family": family, "axis": axis, "shared": shared,
                           "n_tar": config.spatial.target_data_nx,
                           "actual_final_pilot_n_sur": result["pilot_n_sur"],
                           "J": config.spatial.observation_dim,
                           "q": config.spatial.target_output_dim,
                           "selection_record_hash": result["selection_record_hash"],
                           "frozen_plan_hash": result["frozen_plan_hash"],
                           "curve_identity": {
                               "family": family, "axis": axis,
                               "stage": "initial_or_selected_path"}})
    axes[0, 0].legend(fontsize=8)
    fig.supylabel("test full-reference relative L2")
    fig.tight_layout()
    outputs = []
    try:
        outputs.extend(_save_formats(
            fig, output_dir, "e2_parameter_sweeps", formats, dpi,
            {"kind": "parameter_sweeps", "panels": panel_meta},
        ))
    finally:
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, family in zip(axes, ("burgers", "reaction_diffusion")):
        family_rows = sorted([r for r in result["convergence_results"] if r["family"] == family], key=lambda r: r["n_sur"])
        x = [r["n_sur"] for r in family_rows]
        for key, label in (("terminal_relative_l2_mean", "terminal"), ("feature_relative_l2_mean", "J-feature"),
                           ("prediction_relative_l2_mean", "frozen prediction")):
            values = [r[key] if r["n_sur"] != r["reference_n_sur"] else float("nan")
                      for r in family_rows]
            ax.plot(x, values, "o-", label=label)
        tolerances = e2.convergence.tolerances
        for value, label in ((tolerances.terminal_mean, "terminal threshold"),
                             (tolerances.feature_mean, "feature threshold"),
                             (tolerances.prediction_mean, "prediction threshold")):
            ax.axhline(value, linestyle=":", alpha=.35, label=label)
        base = result["convergence_summary"]["families"][family]["n_sur_base"]
        if base is None:
            ax.text(.03, .04, "no accepted base", transform=ax.transAxes,
                    color="#A00000", fontsize=9)
        else:
            ax.axvline(base, color="black", linestyle="--", label=f"base={base}")
        if x:
            ax.axvline(x[-1], color="gray", linestyle=":",
                       label=f"reference={x[-1]}")
        ax.set_yscale("log"); ax.set_xlabel("n_sur"); ax.set_title(family); ax.grid(True, which="both", alpha=.25)
    axes[0].set_ylabel("relative L2 discrepancy"); axes[0].legend(fontsize=8); fig.tight_layout()
    try:
        outputs.extend(_save_formats(
            fig, output_dir, "e2_nsur_convergence", formats, dpi,
            {
                "kind": "n_sur_convergence",
                "actual_final_pilot_n_sur": result["pilot_n_sur"],
                "selection_record_hash": result["selection_record_hash"],
                "frozen_plan_hash": result["frozen_plan_hash"],
            },
        ))
    finally:
        plt.close(fig)
    return outputs
