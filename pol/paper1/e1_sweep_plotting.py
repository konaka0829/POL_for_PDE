"""Aggregate plotting for Paper 1 E1 sweeps."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_NAMES = ("sweep_selected_results.csv", "sweep_readout_diagnostics.csv", "sweep_noise_summary.csv")
REGIMES = ("stable", "unstable")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def _select(rows: list[dict[str, str]], q: int, **conditions: Any) -> list[dict[str, str]]:
    result = [row for row in rows if int(row["q"]) == q]
    for key, expected in conditions.items():
        if callable(expected):
            result = [row for row in result if expected(row)]
        elif isinstance(expected, bool):
            result = [row for row in result if row[key].lower() == str(expected).lower()]
        else:
            result = [row for row in result if int(row[key]) == int(expected)]
    return result


def _scale(values: list[float], minimum_span: float | None) -> tuple[float, float]:
    low, high = min(values), max(values)
    if minimum_span is not None and high - low < minimum_span:
        middle = (low + high) / 2
        return middle - minimum_span / 2, middle + minimum_span / 2
    if high == low:
        pad = max(abs(high) * .05, .05)
        return low - pad, high + pad
    return low, high


def _heatmap(
    rows: list[dict[str, str]], *, x: str, y: str, metric: str, transform: Callable[[float], float],
    label: str, title: str, minimum_span: float | None = None, panels: tuple[str, ...] = REGIMES,
) -> tuple[plt.Figure, dict[str, Any]]:
    if not rows:
        raise ValueError(f"no rows for {title}")
    xs, ys = sorted({int(r[x]) for r in rows}), sorted({int(r[y]) for r in rows})
    values = [transform(_number(r, metric)) for r in rows]
    vmin, vmax = _scale(values, minimum_span)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#bdbdbd")
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.8), squeeze=False)
    matrices: dict[str, list[list[float | None]]] = {}
    image = None
    for column, regime in enumerate(panels):
        ax = axes[0, column]
        lookup = {(int(r[y]), int(r[x])): transform(_number(r, metric)) for r in rows if r["regime"] == regime}
        matrix = np.full((len(ys), len(xs)), np.nan)
        for iy, yv in enumerate(ys):
            for ix, xv in enumerate(xs):
                if (yv, xv) in lookup:
                    matrix[iy, ix] = lookup[(yv, xv)]
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        for iy in range(len(ys)):
            for ix in range(len(xs)):
                text = f"{matrix[iy, ix]:.2g}" if np.isfinite(matrix[iy, ix]) else "missing"
                ax.text(ix, iy, text, ha="center", va="center", fontsize=6, color="black")
        ax.set_xticks(range(len(xs)), xs, rotation=45)
        ax.set_yticks(range(len(ys)), ys)
        ax.set_xlabel(x.replace("n_sur", r"$n_{\rm sur}$").replace("J", "$J$"))
        ax.set_ylabel(y.replace("n_tar", r"$n_{\rm tar}$"))
        ax.set_title(regime)
        matrices[regime] = [[None if not np.isfinite(v) else float(v) for v in row] for row in matrix]
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), label=label, shrink=.85)
    fig.suptitle(title)
    fig.subplots_adjust(top=.84, bottom=.18, wspace=.25)
    return fig, {"x_values": xs, "y_values": ys, "color_limits": [vmin, vmax], "panel_matrices": matrices}


def _lines(
    groups: list[tuple[str, list[dict[str, str]], str, str]], *, x: str, ylabel: str,
    title: str, ylog: bool = False,
) -> tuple[plt.Figure, dict[str, Any]]:
    if not any(rows for _, rows, _, _ in groups):
        raise ValueError(f"no rows for {title}")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    axis_data: dict[str, Any] = {}
    for label, rows, metric, style in groups:
        rows = sorted(rows, key=lambda r: int(r[x]))
        if not rows:
            continue
        xv, yv = [int(r[x]) for r in rows], [_number(r, metric) for r in rows]
        ax.plot(xv, yv, style, label=label)
        axis_data[label] = {"x": xv, "y": yv}
    if not axis_data:
        raise ValueError(f"slice incomplete for {title}")
    if ylog and all(v > 0 for data in axis_data.values() for v in data["y"]):
        ax.set_yscale("log")
    ax.set_xlabel(x); ax.set_ylabel(ylabel); ax.grid(True, alpha=.3); ax.legend(fontsize=8); ax.set_title(title)
    return fig, {"series": axis_data}


def generate_aggregate_plots(
    output_dir: Path,
    settings: dict[str, Any],
    *,
    input_dir: Path | None = None,
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Generate all aggregate figures and a hash-addressed manifest."""
    q = int(settings["q"])
    noise_level = float(settings.get("noise_level", .01))
    line_slice = settings.get("line_slice", {})
    n_tar_line = int(line_slice["target_data_nx"])
    n_sur_line = int(line_slice["surrogate_internal_nx"])
    target_observation_n_sur = int(settings.get("target_observation_surrogate_nx", 512))
    n_sur_target_n_tar = int(settings.get("surrogate_line_target_data_nx", 256))
    n_sur_Js = tuple(int(value) for value in settings.get("surrogate_line_observation_dims", [65, 96]))
    formats = settings.get("formats", ["png", "pdf"])
    dpi = int(settings.get("dpi", 180))
    source_dir = output_dir if input_dir is None else input_dir
    inputs = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest() for name in CSV_NAMES}
    selected, diagnostics, noise = (_read(source_dir / name) for name in CSV_NAMES)
    full_s = _select(selected, q, full_observation=True)
    full_d = _select(diagnostics, q, full_observation=True)
    grid_s = _select(selected, q, n_sur=target_observation_n_sur)
    grid_d = _select(diagnostics, q, n_sur=target_observation_n_sur)
    j_s = _select(selected, q, n_tar=n_tar_line, n_sur=n_sur_line)
    j_d = _select(diagnostics, q, n_tar=n_tar_line, n_sur=n_sur_line)
    j_n = [r for r in _select(noise, q, n_tar=n_tar_line, n_sur=n_sur_line)
           if math.isclose(_number(r, "noise_level"), noise_level, abs_tol=1e-15)]
    ns_s = [r for r in _select(selected, q, n_tar=n_sur_target_n_tar) if int(r["J"]) in set(n_sur_Js)]
    ns_d = [r for r in _select(diagnostics, q, n_tar=n_sur_target_n_tar) if int(r["J"]) in set(n_sur_Js)]
    jobs: list[tuple[str, str, Callable[[], tuple[plt.Figure, dict[str, Any]]]]] = []

    def heat(name: str, rows: list[dict[str, str]], x: str, y: str, metric: str, label: str,
             title: str, transform=lambda value: value, span=.1) -> None:
        jobs.append((name, "sweep_selected_results.csv" if rows is full_s or rows is grid_s else "sweep_readout_diagnostics.csv",
                     lambda: _heatmap(rows, x=x, y=y, metric=metric, transform=transform, label=label,
                                      title=title, minimum_span=span)))
    log = lambda value: math.log10(value) if value > 0 and math.isfinite(value) else (_ for _ in ()).throw(ValueError("log metric must be positive and finite"))
    for prefix, srows, drows, x, suffix, fixed in (
        ("", full_s, full_d, "n_sur", "", r"$J=n_{\rm sur}$"),
        ("_ntar_J_nsur512", grid_s, grid_d, "J", "_ntar_J_nsur512", rf"$n_{{\rm sur}}={target_observation_n_sur}$"),
    ):
        heat(f"sweep_field_error_heatmap{suffix}_q{q}", srows, x, "n_tar", "full_reference_field_relative_l2_mean",
             "log10(relative L2)", f"Field error, q={q}, {fixed}", log, .25)
        heat(f"sweep_field_floor_ratio_heatmap{suffix}_q{q}", srows, x, "n_tar", "field_error_to_representation_floor_ratio",
             "field error / representation floor", f"Field/floor ratio, q={q}, {fixed}")
        heat(f"sweep_operator_norm_ratio_heatmap{suffix}_q{q}", drows, x, "n_tar", "learned_operator_norm",
             "learned / ideal norm", f"Operator norm ratio, q={q}, {fixed}",
             lambda v, rows=drows: v, .1)
        # Ratio is prepared explicitly so the shared heatmap machinery remains generic.
        for row in drows:
            row["operator_norm_ratio"] = str(_number(row, "learned_operator_norm") / _number(row, "ideal_operator_norm"))
        jobs[-1] = (jobs[-1][0], jobs[-1][1], lambda rows=drows, x=x, title=f"Operator norm ratio, q={q}, {fixed}":
                    _heatmap(rows, x=x, y="n_tar", metric="operator_norm_ratio", transform=lambda v: v,
                             label="learned / ideal norm", title=title, minimum_span=.1))
        jobs.append((f"sweep_multiplier_recovery_heatmaps{suffix}_q{q}", "sweep_readout_diagnostics.csv",
                     lambda rows=drows, x=x, title=f"Multiplier recovery, q={q}, {fixed}": _multiplier(rows, x, title)))

    def add_line(name: str, source: str, groups: list[tuple[str, list[dict[str, str]], str, str]], x: str, ylabel: str, title: str, ylog=False):
        jobs.append((name, source, lambda: _lines(groups, x=x, ylabel=ylabel, title=title, ylog=ylog)))
    add_line(f"sweep_field_error_vs_J_q{q}", "sweep_selected_results.csv",
             [(f"{r} learned", [x for x in j_s if x["regime"] == r], "full_reference_field_relative_l2_mean", "o-") for r in REGIMES] +
             [(f"{r} floor", [x for x in j_s if x["regime"] == r], "output_representation_floor_mean", "--") for r in REGIMES],
             "J", "relative L2", f"Field error vs J, q={q}, n_tar={n_tar_line}, n_sur={n_sur_line}", True)
    add_line(f"sweep_operator_norm_vs_J_q{q}", "sweep_readout_diagnostics.csv",
             [(f"{r} learned", [x for x in j_d if x["regime"] == r], "learned_operator_norm", "o-") for r in REGIMES] +
             [(f"{r} ideal", [x for x in j_d if x["regime"] == r], "ideal_operator_norm", "--") for r in REGIMES],
             "J", "operator norm", f"Operator norm vs J, q={q}", True)
    add_line(f"sweep_noise_sensitivity_vs_J_q{q}_delta1e-2", "sweep_noise_summary.csv",
             [(r, [x for x in j_n if x["regime"] == r], "output_perturbation_rms_mean", "o-") for r in REGIMES],
             "J", "output perturbation RMS", f"Noise sensitivity, q={q}, delta={noise_level:g}", True)
    add_line(f"sweep_identifiability_vs_J_q{q}", "sweep_readout_diagnostics.csv",
             [(r, [x for x in j_d if x["regime"] == r], "identifiable_nonconstant_fraction", "o-") for r in REGIMES],
             "J", "identifiable fraction", f"Identifiability vs J, q={q}")
    add_line(f"sweep_field_error_vs_nsur_q{q}_J65_J96", "sweep_selected_results.csv",
             [(f"{r} J={J} learned", [x for x in ns_s if x["regime"] == r and int(x["J"]) == J],
               "full_reference_field_relative_l2_mean", "o-") for r in REGIMES for J in n_sur_Js] +
             [(f"{r} J={J} floor", [x for x in ns_s if x["regime"] == r and int(x["J"]) == J],
               "output_representation_floor_mean", "--") for r in REGIMES for J in n_sur_Js],
             "n_sur", "relative L2", f"Field error vs n_sur, q={q}, n_tar={n_sur_target_n_tar}", True)
    for row in ns_d:
        row["operator_norm_ratio"] = str(_number(row, "learned_operator_norm") / _number(row, "ideal_operator_norm"))
    add_line(f"sweep_operator_norm_ratio_vs_nsur_q{q}_J65_J96", "sweep_readout_diagnostics.csv",
             [(f"{r} J={J}", [x for x in ns_d if x["regime"] == r and int(x["J"]) == J],
               "operator_norm_ratio", "o-") for r in REGIMES for J in n_sur_Js],
             "n_sur", "learned / ideal norm", f"Operator norm ratio vs n_sur, q={q}, n_tar={n_sur_target_n_tar}")

    entries = []
    for stem, source, factory in jobs:
        try:
            fig, metadata = factory()
            for fmt in formats:
                path = output_dir / f"{stem}.{fmt}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                entries.append({"relative_path": path.name, "format": fmt, "q": q, "noise_level": noise_level,
                                "slice": line_slice, "source_csv": source, "source_file_hash": inputs[source],
                                "status": "pass", **metadata})
            plt.close(fig)
        except Exception as exc:
            entries.append({"relative_path": None, "format": None, "q": q, "noise_level": noise_level,
                            "slice": line_slice, "source_csv": source, "source_file_hash": inputs[source],
                            "status": "skipped", "reason": str(exc), "plot": stem})
    manifest = {"schema_version": "paper1-e1-sweep-plots-v2", "status":
                "pass" if all(x["status"] == "pass" for x in entries) else "fail",
                "q": q, "noise_level": noise_level, "inputs": inputs, "plots": entries}
    if write_manifest:
        (output_dir / "sweep_plot_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def _multiplier(rows: list[dict[str, str]], x: str, title: str) -> tuple[plt.Figure, dict[str, Any]]:
    if not rows:
        raise ValueError(f"no rows for {title}")
    xs, ys = sorted({int(r[x]) for r in rows}), sorted({int(r["n_tar"]) for r in rows})
    metrics = ("max_identifiable_diagonal_relative_error", "identifiable_off_diagonal_relative_norm")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metadata: dict[str, Any] = {"x_values": xs, "y_values": ys, "color_limits": {}}
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#bdbdbd")
    for row_index, metric in enumerate(metrics):
        values = [_number(r, metric) for r in rows]
        vmin, vmax = _scale(values, .1)
        metadata["color_limits"][metric] = [vmin, vmax]
        image = None
        for column, regime in enumerate(REGIMES):
            matrix = np.full((len(ys), len(xs)), np.nan)
            lookup = {(int(r["n_tar"]), int(r[x])): _number(r, metric) for r in rows if r["regime"] == regime}
            for iy, yv in enumerate(ys):
                for ix, xv in enumerate(xs):
                    if (yv, xv) in lookup: matrix[iy, ix] = lookup[(yv, xv)]
                    axes[row_index, column].text(ix, iy, f"{matrix[iy, ix]:.2g}" if np.isfinite(matrix[iy, ix]) else "missing",
                                                 ha="center", va="center", fontsize=6)
            image = axes[row_index, column].imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_index, column].set_xticks(range(len(xs)), xs, rotation=45); axes[row_index, column].set_yticks(range(len(ys)), ys)
            axes[row_index, column].set_title(f"{regime}: {metric}")
        assert image is not None
        fig.colorbar(image, ax=axes[row_index, :].tolist(), shrink=.8)
    fig.suptitle(title); fig.subplots_adjust(top=.9, bottom=.12, wspace=.25, hspace=.35)
    return fig, metadata
