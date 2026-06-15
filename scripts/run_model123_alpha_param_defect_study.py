#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from model123_burgers_1d import pearson_corr_or_none, spearman_corr_or_none
from scripts.run_model123_param_sweep import (
    PARAMETERS,
    SweepParameter,
    append_optional_flag,
    build_job_env,
    build_run_command,
    cast_value,
    canonical_parameter_name,
    dedupe_fieldnames,
    parse_csv_values,
    parse_models,
    require_time_grid_aligned_value,
    safe_tag,
)


def parse_float_csv(raw: str) -> list[float]:
    values = [float(item) for item in parse_csv_values(raw)]
    if not values:
        raise ValueError("Expected at least one value")
    return values


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_all(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path_no_ext.with_suffix("." + ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config", default="")
    parser.add_argument("--models", default="model1,model2,model3")
    parser.add_argument("--reservoir", choices=("burgers", "reaction_diffusion", "ks", "static", "heat", "advection"), default="burgers")
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--parameter-values", required=True)
    parser.add_argument("--alpha-values", required=True)
    parser.add_argument("--data-file", default="data/burgers_model123.mat")
    parser.add_argument("--out-root", default="outputs/alpha_param_defect_study")
    parser.add_argument("--train-split", type=float, default=1000.0 / 1200.0)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--nval", type=int, default=0)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--feature-times", type=str, default="")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--obs", choices=("full", "points", "fourier", "proj"), default="full")
    parser.add_argument("--J", type=int, default=1028)
    parser.add_argument("--sensor-mode", choices=("equispaced", "random"), default="equispaced")
    parser.add_argument("--sensor-seed", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--input-shift", type=float, default=0.0)
    parser.add_argument("--ridge-zeta", type=float, default=None)
    parser.add_argument("--ridge-lambda", type=float, default=None)
    parser.add_argument(
        "--ridge-convention",
        default="normalized_empirical_l2h_unweighted_frobenius",
        choices=("normalized_empirical_l2h_unweighted_frobenius", "legacy_unnormalized_gram"),
    )
    parser.add_argument("--ridge-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--standardize-features", type=int, choices=(0, 1), default=0)
    parser.add_argument("--feature-std-eps", type=float, default=1e-6)
    parser.add_argument("--rd-nu", type=float, default=1e-3)
    parser.add_argument("--rd-alpha", type=float, default=1.0)
    parser.add_argument("--rd-beta", type=float, default=1.0)
    parser.add_argument("--res-burgers-nu", type=float, default=1e-2)
    parser.add_argument("--res-burgers-b", type=float, default=1.0)
    parser.add_argument("--ks-dealias", action="store_true")
    parser.add_argument("--ks-b", type=float, default=1.0)
    parser.add_argument("--ks-eta", type=float, default=1.0)
    parser.add_argument("--ks-kappa", type=float, default=1.0)
    parser.add_argument("--burgers-scheme", choices=("semi_implicit", "split_step"), default="split_step")
    parser.add_argument("--burgers-fine-dt", type=float, default=1e-4)
    parser.add_argument("--burgers-dealias", type=int, choices=(0, 1), default=1)
    parser.add_argument("--elm-h", type=int, default=1024)
    parser.add_argument("--elm-activation", choices=("tanh", "relu", "identity"), default="tanh")
    parser.add_argument("--elm-seed", type=int, default=0)
    parser.add_argument("--elm-weight-scale", type=float, default=0.0)
    parser.add_argument("--elm-bias-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--defect-target-nu", type=float, default=None)
    parser.add_argument("--defect-time-quadrature", choices=("trapezoid", "left"), default="trapezoid")
    parser.add_argument("--defect-beta-mode", choices=("zero", "fixed"), default="zero")
    parser.add_argument("--defect-beta-fixed", type=float, default=0.0)
    parser.add_argument("--defect-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reuse-report", choices=("summary", "verbose", "silent"), default="summary")
    parser.add_argument("--check-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-model", action="store_true")


def safe_numeric_key(value: float, values: list[float]) -> int:
    arr = np.asarray(values, dtype=float)
    matches = np.where(np.isclose(arr, float(value), rtol=1e-9, atol=1e-12))[0]
    if matches.size == 0:
        raise ValueError(f"value {value} not found in numeric grid")
    return int(matches[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Model 1/2/3 alpha-parameter integrated defect study.")
    add_common_args(parser)
    return parser


def validate_args(args: argparse.Namespace) -> tuple[list[str], SweepParameter, list[Any], list[float]]:
    models = parse_models(args.models)
    parameter_name = canonical_parameter_name(args.parameter)
    if parameter_name in {"alpha", "Ttilde"}:
        raise ValueError("--parameter must be a surrogate parameter, not alpha or Ttilde")
    parameter = PARAMETERS[parameter_name]
    if parameter.reservoirs and args.reservoir not in parameter.reservoirs:
        raise ValueError("Parameter %s is not valid for reservoir=%s" % (parameter_name, args.reservoir))
    parameter_values = [cast_value(item, parameter) for item in parse_csv_values(args.parameter_values)]
    if not parameter_values:
        raise ValueError("Expected at least one parameter value")
    alpha_values = parse_float_csv(args.alpha_values)
    if any(alpha <= 0.0 for alpha in alpha_values):
        raise ValueError("alpha values must be positive")
    if parameter.positive and any(value <= 0.0 for value in parameter_values):
        raise ValueError("parameter values must be positive")
    if args.T <= 0.0 or args.dt <= 0.0:
        raise ValueError("--T and --dt must be positive")
    for alpha in alpha_values:
        require_time_grid_aligned_value(alpha * float(args.T), args.dt, "Ttilde", alpha=alpha, T=args.T)
    return models, parameter, parameter_values, alpha_values


def run_command(cmd: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("[dry-run]\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return 0
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log_path.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")
    return proc.returncode


def command_for_model(args: argparse.Namespace, model: str, parameter_name: str, parameter_value: float, alpha: float, run_dir: Path, compute_defect: bool) -> list[str]:
    command_args = argparse.Namespace(**vars(args))
    command_args.Ttilde = float(alpha) * float(args.T)
    command_args.compute_time_scaled_defect = bool(compute_defect)
    command_args.max_workers = 1
    overrides = {parameter_name: parameter_value, "alpha": alpha}
    return build_run_command(command_args, model, overrides, run_dir)


def load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    cfg = load_json(run_dir / "run_config.json")
    return {
        "train_absL2h": float(cfg["train_absL2h"]),
        "test_absL2h": float(cfg["test_absL2h"]),
        "train_relL2": float(cfg["train_relL2"]),
        "test_relL2": float(cfg["test_relL2"]),
    }


def load_error_vectors(run_dir: Path) -> tuple[list[float], list[float]]:
    metrics = load_json(run_dir / "test_error_metrics.json")
    return [float(v) for v in metrics["per_sample_absL2h"]], [float(v) for v in metrics["per_sample_relL2"]]


def find_defect_source(owner_dir: Path, run_dir: Path) -> Path | None:
    for candidate in (run_dir, owner_dir):
        if (candidate / "time_scaled_defect_per_sample.json").exists():
            return candidate
    return None


def summarize_joined(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [float(row["model_error_abs_l2h"]) for row in rows]
    defects = [float(row["delta_scale_pathwise_abs_l2h"]) for row in rows]
    return {
        "delta_scale_rms_abs_l2h": float(np.sqrt(np.mean(np.asarray(defects) ** 2))),
        "delta_scale_mean_abs_l2h": float(np.mean(defects)),
        "delta_scale_std_abs_l2h": float(np.std(defects)),
        "corr_error_delta_scale_pearson": pearson_corr_or_none(errors, defects),
        "corr_error_delta_scale_spearman": spearman_corr_or_none(errors, defects),
        "num_test_samples": len(rows),
    }


def plot_lines(rows: list[dict[str, Any]], x_key: str, fixed_key: str, fixed_value: float, title: str, xlabel: str, out_path: Path) -> None:
    subset = [row for row in rows if row["status"] == "ok" and np.isclose(float(row[fixed_key]), fixed_value)]
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for model in sorted({row["model"] for row in subset}):
        model_rows = sorted([row for row in subset if row["model"] == model], key=lambda row: float(row[x_key]))
        ax.plot([float(row[x_key]) for row in model_rows], [float(row["test_absL2h"]) for row in model_rows], marker="o", label=model)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test absL2h")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_all(fig, out_path)


def matrix_from_rows(rows: list[dict[str, Any]], parameter_values: list[float], alpha_values: list[float], value_key: str, model: str | None = None) -> np.ndarray:
    matrix = np.full((len(parameter_values), len(alpha_values)), np.nan)
    for row in rows:
        if row["status"] != "ok":
            continue
        if model is not None and row["model"] != model:
            continue
        pi = safe_numeric_key(float(row["parameter_value"]), parameter_values)
        ai = safe_numeric_key(float(row["alpha"]), alpha_values)
        value = row.get(value_key)
        matrix[pi, ai] = np.nan if value is None else float(value)
    return matrix


def plot_heatmap(matrix: np.ndarray, parameter: SweepParameter, parameter_values: list[float], alpha_values: list[float], title: str, cbar_label: str, out_path: Path, *, correlation: bool = False) -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    if correlation:
        norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
        cmap = "coolwarm"
        plot_matrix = matrix
    elif np.all(finite > 0.0):
        norm = mcolors.LogNorm(vmin=float(finite.min()), vmax=float(finite.max()))
        cmap = "viridis"
        plot_matrix = matrix
    else:
        norm = None
        cmap = "viridis"
        plot_matrix = matrix
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    image = ax.imshow(plot_matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(alpha_values)), labels=[format(v, ".4g") for v in alpha_values])
    ax.set_yticks(range(len(parameter_values)), labels=[format(v, ".4g") for v in parameter_values])
    ax.set_xlabel("alpha")
    ax.set_ylabel(parameter.name)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)
    if matrix.size <= 64:
        for yi in range(matrix.shape[0]):
            for xi in range(matrix.shape[1]):
                if np.isfinite(matrix[yi, xi]):
                    ax.text(xi, yi, f"{matrix[yi, xi]:.2g}", ha="center", va="center", fontsize=7, color="white")
    fig.tight_layout()
    save_all(fig, out_path)


def plot_grid_scatter(rows: list[dict[str, Any]], model: str, out_path: Path) -> None:
    subset = [row for row in rows if row["status"] == "ok" and row["model"] == model and row.get("delta_scale_rms_abs_l2h") is not None]
    if not subset:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter([float(row["delta_scale_rms_abs_l2h"]) for row in subset], [float(row["test_absL2h"]) for row in subset], s=30)
    ax.set_xlabel("delta_scale_rms_abs_l2h")
    ax.set_ylabel("test absL2h")
    ax.set_title(f"{model}: error vs integrated generator defect")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_all(fig, out_path)


def make_plots(rows: list[dict[str, Any]], models: list[str], parameter: SweepParameter, parameter_values: list[float], alpha_values: list[float], plot_dir: Path) -> None:
    for parameter_value in parameter_values:
        plot_lines(
            rows,
            x_key="alpha",
            fixed_key="parameter_value",
            fixed_value=parameter_value,
            title=f"alpha vs error at {parameter.name}={parameter_value:g}",
            xlabel="alpha",
            out_path=plot_dir / f"alpha_vs_error_all_models__{parameter.name}_{safe_tag(parameter_value)}",
        )
    for alpha in alpha_values:
        plot_lines(
            rows,
            x_key="parameter_value",
            fixed_key="alpha",
            fixed_value=alpha,
            title=f"{parameter.name} vs error at alpha={alpha:g}",
            xlabel=parameter.name,
            out_path=plot_dir / f"parameter_{parameter.name}_vs_error_all_models__alpha_{safe_tag(alpha)}",
        )
    plot_heatmap(
        matrix_from_rows(rows, parameter_values, alpha_values, "delta_scale_rms_abs_l2h"),
        parameter,
        parameter_values,
        alpha_values,
        "Integrated generator defect magnitude",
        "delta_scale_rms_abs_l2h",
        plot_dir / "heatmap_delta_scale_rms",
    )
    for model in models:
        plot_heatmap(
            matrix_from_rows(rows, parameter_values, alpha_values, "test_absL2h", model=model),
            parameter,
            parameter_values,
            alpha_values,
            f"{model}: model error",
            "test absL2h",
            plot_dir / f"heatmap_{model}_error",
        )
        plot_heatmap(
            matrix_from_rows(rows, parameter_values, alpha_values, "corr_error_delta_scale_pearson", model=model),
            parameter,
            parameter_values,
            alpha_values,
            f"{model}: Pearson error/defect correlation",
            "corr(model error, integrated defect)",
            plot_dir / f"heatmap_corr_pearson_{model}_error_delta_scale",
            correlation=True,
        )
        plot_heatmap(
            matrix_from_rows(rows, parameter_values, alpha_values, "corr_error_delta_scale_spearman", model=model),
            parameter,
            parameter_values,
            alpha_values,
            f"{model}: Spearman error/defect correlation",
            "corr(model error, integrated defect)",
            plot_dir / f"heatmap_corr_spearman_{model}_error_delta_scale",
            correlation=True,
        )
        plot_grid_scatter(rows, model, plot_dir / f"scatter_grid_error_vs_delta_scale_{model}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data_seed is None:
        args.data_seed = args.seed
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.ridge_zeta is None and args.ridge_lambda is None:
        args.ridge_zeta = 1e-4
        args.ridge_lambda = 1e-4
    elif args.ridge_zeta is None:
        args.ridge_zeta = float(args.ridge_lambda)
    elif args.ridge_lambda is None:
        args.ridge_lambda = float(args.ridge_zeta)
    models, parameter, parameter_values, alpha_values = validate_args(args)

    out_root = Path(args.out_root)
    runs_root = out_root / "runs"
    plot_dir = out_root / "plots"
    out_root.mkdir(parents=True, exist_ok=True)
    env = build_job_env(os.environ.copy())
    env.setdefault("MPLCONFIGDIR", str(out_root / ".mplconfig"))

    summary_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    had_failure = False
    defect_owner_model = models[0]

    for parameter_value in parameter_values:
        for alpha in alpha_values:
            owner_dir = runs_root / defect_owner_model / f"{parameter.name}_{safe_tag(parameter_value)}__alpha_{safe_tag(alpha)}"
            cell_dirs: dict[str, Path] = {}
            cell_results: dict[str, tuple[str, int]] = {}
            for model in models:
                run_dir = runs_root / model / f"{parameter.name}_{safe_tag(parameter_value)}__alpha_{safe_tag(alpha)}"
                cell_dirs[model] = run_dir
                compute_defect = model == defect_owner_model
                config_exists = (run_dir / "run_config.json").exists()
                defect_exists = (run_dir / "time_scaled_defect_per_sample.json").exists()
                if args.check_existing:
                    return_code = 0
                    if not config_exists:
                        status = "missing"
                    elif compute_defect and not defect_exists:
                        status = "missing_defect"
                    else:
                        status = "ok"
                elif args.skip_existing and config_exists and (not compute_defect or defect_exists):
                    return_code = 0
                    status = "ok"
                else:
                    cmd = command_for_model(args, model, parameter.name, parameter_value, alpha, run_dir, compute_defect)
                    return_code = run_command(cmd, run_dir / "stdout_stderr.log", env, args.dry_run)
                    status = "dry_run" if args.dry_run else ("ok" if return_code == 0 and (run_dir / "run_config.json").exists() else "fail")
                if status == "fail":
                    had_failure = True
                cell_results[model] = (status, return_code)
                if args.reuse_report == "verbose" or (
                    args.reuse_report == "summary" and not args.skip_existing and not args.check_existing
                ):
                    print("[%s] %s=%g alpha=%g -> %s" % (model, parameter.name, parameter_value, alpha, status), flush=True)

            for model in models:
                run_dir = cell_dirs[model]
                row = {
                    "model": model,
                    "reservoir": args.reservoir,
                    "parameter_name": parameter.name,
                    "parameter_value": parameter_value,
                    "T": float(args.T),
                    "Ttilde": float(alpha) * float(args.T),
                    "alpha": alpha,
                    "train_absL2h": None,
                    "test_absL2h": None,
                    "train_relL2": None,
                    "test_relL2": None,
                    "delta_scale_rms_abs_l2h": None,
                    "delta_scale_mean_abs_l2h": None,
                    "delta_scale_std_abs_l2h": None,
                    "corr_error_delta_scale_pearson": None,
                    "corr_error_delta_scale_spearman": None,
                    "num_test_samples": None,
                    "run_dir": str(run_dir),
                    "status": cell_results[model][0],
                    "return_code": cell_results[model][1],
                }
                if row["status"] == "ok" and (run_dir / "run_config.json").exists():
                    row.update(load_run_summary(run_dir))
                    defect_source = find_defect_source(owner_dir, run_dir)
                    if defect_source is not None:
                        defect_rows = load_json(defect_source / "time_scaled_defect_per_sample.json")
                        error_abs, error_rel = load_error_vectors(run_dir)
                        joined = []
                        for idx, defect_row in enumerate(defect_rows):
                            joined_row = {
                                "model": model,
                                "sample_index": int(defect_row["sample_index"]),
                                "reservoir": args.reservoir,
                                "parameter_name": parameter.name,
                                "parameter_value": parameter_value,
                                "T": float(args.T),
                                "Ttilde": float(alpha) * float(args.T),
                                "alpha": alpha,
                                "model_error_abs_l2h": error_abs[idx],
                                "model_error_rel_l2": error_rel[idx],
                                "D1_model1_abs_l2h": float(defect_row["D1_model1_abs_l2h"]),
                                "delta_scale_pathwise_abs_l2h": float(defect_row["delta_scale_pathwise_abs_l2h"]),
                                "Delta_scale_abs_l2h": float(defect_row["Delta_scale_abs_l2h"]),
                            }
                            joined.append(joined_row)
                        per_sample_rows.extend(joined)
                        row.update(summarize_joined(joined))
                summary_rows.append(row)

    summary_fields = [
        "model",
        "reservoir",
        "parameter_name",
        "parameter_value",
        "T",
        "Ttilde",
        "alpha",
        "train_absL2h",
        "test_absL2h",
        "train_relL2",
        "test_relL2",
        "delta_scale_rms_abs_l2h",
        "delta_scale_mean_abs_l2h",
        "delta_scale_std_abs_l2h",
        "corr_error_delta_scale_pearson",
        "corr_error_delta_scale_spearman",
        "num_test_samples",
        "run_dir",
        "status",
        "return_code",
    ]
    per_sample_fields = [
        "model",
        "sample_index",
        "reservoir",
        "parameter_name",
        "parameter_value",
        "T",
        "Ttilde",
        "alpha",
        "model_error_abs_l2h",
        "model_error_rel_l2",
        "D1_model1_abs_l2h",
        "delta_scale_pathwise_abs_l2h",
        "Delta_scale_abs_l2h",
    ]
    write_csv(out_root / "summary.csv", summary_rows, dedupe_fieldnames(summary_fields))
    (out_root / "summary.json").write_text(json.dumps({"config": vars(args), "rows": summary_rows}, indent=2), encoding="utf-8")
    write_csv(out_root / "existing_audit.csv", summary_rows, dedupe_fieldnames(summary_fields))
    (out_root / "existing_audit.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    write_csv(out_root / "per_sample_metrics.csv", per_sample_rows, dedupe_fieldnames(per_sample_fields))
    (out_root / "per_sample_metrics.json").write_text(json.dumps(per_sample_rows, indent=2), encoding="utf-8")
    if args.reuse_report == "summary" and (args.skip_existing or args.check_existing):
        counts: dict[str, int] = {}
        for row in summary_rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        print("[alpha-param] audit summary " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)
    if not args.check_existing:
        make_plots(summary_rows, models, parameter, parameter_values, alpha_values, plot_dir)
    if args.dry_run:
        return 0
    if args.check_existing:
        return 2 if any(row["status"] != "ok" for row in summary_rows) else 0
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
