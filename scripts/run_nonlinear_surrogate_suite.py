#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFECT_RESERVOIRS = {"burgers", "reaction_diffusion", "ks"}

from pol.metadata import file_sha256, get_git_info, get_runtime_info
from scripts.suite_common import (
    alpha_ttilde_pairs,
    best_by_validation,
    load_json,
    model123_command,
    parse_models,
    parse_reservoirs,
    read_config_defaults,
    resolve_T,
    reservoir_grid,
    row_from_zeta_run,
    row_from_model123_run,
    run_recorded_command,
    safe_tag,
    save_all_formats,
    to_jsonable,
    write_csv,
    write_json,
    zeta_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E3 nonlinear surrogate suite against Fourier diagonal headroom.")
    parser.add_argument("--config", default="configs/B0_smoke.json")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", default="outputs/nonlinear_surrogate_suite")
    parser.add_argument("--models", default="model2,model3")
    parser.add_argument("--reservoirs", default="static,heat,advection,burgers,reaction_diffusion,ks")
    parser.add_argument("--alpha-values", default="1.0")
    parser.add_argument("--Ttilde-values", default="")
    parser.add_argument("--zeta-grid", default="1e-8,1e-6,1e-4")
    parser.add_argument("--use-feature-cache", action="store_true")
    parser.add_argument("--refresh-feature-cache", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--require-complete-metadata", action="store_true")
    parser.add_argument("--allow-metadata-mismatch", action="store_true")
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--sim-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--ridge-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--standardize-features", type=int, choices=(0, 1), default=0)
    parser.add_argument("--elm-h", type=int, default=1024)
    parser.add_argument("--elm-activation", choices=("tanh", "relu", "identity"), default="tanh")
    parser.add_argument("--elm-seed", type=int, default=0)
    parser.add_argument("--elm-weight-scale", type=float, default=0.0)
    parser.add_argument("--elm-bias-scale", type=float, default=1.0)
    parser.add_argument("--heat-nu-values", default="0.01")
    parser.add_argument("--advection-c-values", default="1.0")
    parser.add_argument("--res-burgers-nu-values", default="0.01")
    parser.add_argument("--res-burgers-b-values", default="1.0")
    parser.add_argument("--rd-nu-values", default="0.001")
    parser.add_argument("--rd-alpha-values", default="1.0")
    parser.add_argument("--rd-beta-values", default="1.0")
    parser.add_argument("--ks-b-values", default="1.0")
    parser.add_argument("--ks-eta-values", default="1.0")
    parser.add_argument("--ks-kappa-values", default="1.0")
    parser.add_argument("--burgers-scheme", choices=("semi_implicit", "split_step", "etdrk4"), default="split_step")
    parser.add_argument("--burgers-dealias", type=int, choices=(0, 1), default=1)
    parser.add_argument("--ks-dealias", action="store_true")
    parser.add_argument("--compute-time-scaled-defect", action="store_true")
    parser.add_argument("--defect-target-nu", type=float, default=None)
    parser.add_argument("--defect-time-quadrature", choices=("trapezoid", "left"), default="trapezoid")
    parser.add_argument("--defect-beta-mode", choices=("zero", "fixed"), default="zero")
    parser.add_argument("--compute-spectral-error", action="store_true")
    return parser


def spectral_error_rows(pred: torch.Tensor, target: torch.Tensor) -> list[dict[str, float | int]]:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    err_hat = torch.fft.rfft(pred - target, dim=-1, norm="forward")
    target_hat = torch.fft.rfft(target, dim=-1, norm="forward")
    err_energy = torch.mean(torch.abs(err_hat).pow(2), dim=0)
    target_energy = torch.mean(torch.abs(target_hat).pow(2), dim=0)
    return [
        {
            "mode": int(idx),
            "mean_fft_error_sq": float(err_energy[idx].item()),
            "mean_fft_target_sq": float(target_energy[idx].item()),
        }
        for idx in range(err_energy.shape[0])
    ]


def normalize_headroom(summary: dict[str, Any]) -> dict[str, Any]:
    dlin2 = summary.get("test_Dlin2")
    if dlin2 is None or float(dlin2) <= 0.0:
        raise ValueError("headroom summary missing positive test_Dlin2")
    return {
        "D_lin_abs_l2h": math.sqrt(float(dlin2)),
        "headroom_H": summary.get("headroom_H"),
        "linear_explained_variance": summary.get("linear_explained_variance"),
        "headroom_selected_zeta": summary.get("selected_zeta"),
        "domain_length": summary.get("domain_length"),
        "effective_nx": summary.get("effective_nx"),
        "dx": summary.get("dx"),
    }


def add_dlin_columns(row: dict[str, Any], headroom: dict[str, Any]) -> None:
    dlin = float(headroom["D_lin_abs_l2h"])
    test_abs = row.get("test_absL2h")
    if test_abs is None:
        row.update(
            {
                "D_lin_abs_l2h": dlin,
                "headroom_H": headroom.get("headroom_H"),
                "linear_explained_variance": headroom.get("linear_explained_variance"),
                "improvement_over_dlin_abs": None,
                "ratio_to_dlin": None,
                "beats_dlin": None,
            }
        )
        return
    test_abs_f = float(test_abs)
    row.update(
        {
            "D_lin_abs_l2h": dlin,
            "headroom_H": headroom.get("headroom_H"),
            "linear_explained_variance": headroom.get("linear_explained_variance"),
            "improvement_over_dlin_abs": dlin - test_abs_f,
            "ratio_to_dlin": test_abs_f / dlin,
            "beats_dlin": bool(test_abs_f < dlin),
        }
    )


def _plot_bar(rows: list[dict[str, Any]], key: str, out_path: Path, title: str) -> None:
    data = [row for row in rows if row.get(key) is not None and row.get("status") == "ok"]
    if not data:
        return
    labels = [f"{row['model']}/{row['reservoir']}" for row in data]
    values = [float(row[key]) for row in data]
    fig, ax = plt.subplots(figsize=(max(6.0, len(data) * 0.65), 4.2))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_all_formats(fig, out_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    models = parse_models(args.models, allowed={"model2", "model3"})
    reservoirs = parse_reservoirs(args.reservoirs)
    config = read_config_defaults(args.config)
    T = resolve_T(config)
    pairs = alpha_ttilde_pairs(alpha_values=args.alpha_values, ttilde_values=args.Ttilde_values or None, T=T)
    out_dir = Path(args.output_dir)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    headroom_dir = out_dir / "headroom"
    headroom_cmd = [
        str(args.python),
        "scripts/run_headroom_burgers.py",
        "--config",
        str(args.config),
        "--data-file",
        str(args.data_file),
        "--zeta-grid",
        str(args.zeta_grid),
        "--output-dir",
        str(headroom_dir),
        "--sub",
        str(args.sub),
        "--sim-dtype",
        str(args.sim_dtype),
        "--ridge-dtype",
        str(args.ridge_dtype),
    ]
    if args.allow_metadata_mismatch:
        headroom_cmd.append("--allow-metadata-mismatch")
    if args.require_complete_metadata:
        headroom_cmd.append("--require-complete-metadata")
    headroom = {"D_lin_abs_l2h": None, "headroom_H": None, "linear_explained_variance": None}
    try:
        run_recorded_command(
            name="headroom",
            command=headroom_cmd,
            cwd=REPO_ROOT,
            commands=commands,
            dry_run=args.dry_run,
            log_path=headroom_dir / "suite_command.log",
        )
        if not args.dry_run:
            headroom = normalize_headroom(load_json(headroom_dir / "headroom_summary.json"))
            write_json(out_dir / "headroom_summary.json", headroom)
    except Exception as exc:
        failures.append({"phase": "E3", "stage": "headroom", "status": "fail", "reason": str(exc)})

    for model in models:
        for reservoir in reservoirs:
            for params in reservoir_grid(reservoir, args):
                for alpha, Ttilde in pairs:
                    tag_parts = [model, reservoir, f"alpha_{safe_tag(alpha)}"] + [f"{k}_{safe_tag(v)}" for k, v in sorted(params.items())]
                    child_dir = runs_dir / "__".join(tag_parts)
                    selected_dir = child_dir / "selected_run"
                    row = {
                        "phase": "E3",
                        "model": model,
                        "reservoir": reservoir,
                        "alpha": alpha,
                        "Ttilde": Ttilde,
                        "output_dir": str(child_dir),
                        **params,
                    }
                    try:
                        if not args.dry_run and headroom.get("D_lin_abs_l2h") is None:
                            raise ValueError("D_lin_abs_l2h unavailable; headroom failed")
                        cmd = zeta_command(args, model=model, reservoir=reservoir, Ttilde=Ttilde, params=params, out_dir=child_dir)
                        run_recorded_command(
                            name="zeta_path",
                            command=cmd,
                            cwd=REPO_ROOT,
                            commands=commands,
                            dry_run=args.dry_run,
                            log_path=child_dir / "suite_command.log",
                        )
                        if args.dry_run:
                            if args.compute_time_scaled_defect and reservoir in DEFECT_RESERVOIRS:
                                cmd = model123_command(
                                    args,
                                    model=model,
                                    reservoir=reservoir,
                                    Ttilde=Ttilde,
                                    params=params,
                                    out_dir=selected_dir,
                                    zeta=None,
                                    compute_defect=True,
                                )
                                run_recorded_command(
                                    name="model123_selected_defect",
                                    command=cmd,
                                    cwd=REPO_ROOT,
                                    commands=commands,
                                    dry_run=True,
                                    log_path=selected_dir / "suite_command.log",
                                )
                            elif args.compute_time_scaled_defect:
                                row["defect_status"] = "not_applicable"
                            row.update({"status": "dry_run"})
                        else:
                            row.update(row_from_zeta_run(child_dir))
                            if args.compute_time_scaled_defect and reservoir in DEFECT_RESERVOIRS:
                                selected_zeta = row.get("zeta_selected")
                                cmd = model123_command(
                                    args,
                                    model=model,
                                    reservoir=reservoir,
                                    Ttilde=Ttilde,
                                    params=params,
                                    out_dir=selected_dir,
                                    zeta=float(selected_zeta) if selected_zeta is not None else None,
                                    compute_defect=True,
                                )
                                run_recorded_command(
                                    name="model123_selected_defect",
                                    command=cmd,
                                    cwd=REPO_ROOT,
                                    commands=commands,
                                    dry_run=False,
                                    log_path=selected_dir / "suite_command.log",
                                )
                                row.update(row_from_model123_run(selected_dir))
                            elif args.compute_time_scaled_defect:
                                row["defect_status"] = "not_applicable"
                            add_dlin_columns(row, headroom)
                            row.update({"status": "ok"})
                    except Exception as exc:
                        row.update({"status": "fail", "reason": str(exc)})
                        failures.append(row)
                    rows.append(row)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    best_by_group = [] if args.dry_run else best_by_validation(ok_rows, ["model", "reservoir"])
    best_by_model = [] if args.dry_run else best_by_validation(ok_rows, ["model"])
    preferred = [
        "phase",
        "model",
        "reservoir",
        "alpha",
        "Ttilde",
        "zeta_selected",
        "selection_metric_name",
        "selection_metric_value",
        "train_absL2h",
        "val_absL2h",
        "test_absL2h",
        "test_relL2_mean",
        "test_relL2_agg",
        "D_lin_abs_l2h",
        "headroom_H",
        "linear_explained_variance",
        "improvement_over_dlin_abs",
        "ratio_to_dlin",
        "beats_dlin",
        "W_fro_norm",
        "W_l2h_hs_norm",
        "d_eff",
        "cond_zeta",
        "domain_length",
        "effective_nx",
        "dx",
        "output_dir",
        "status",
    ]
    write_csv(out_dir / "nonlinear_summary.csv", rows, preferred)
    write_json(out_dir / "nonlinear_summary.json", {"rows": rows})
    write_csv(out_dir / "best_by_model_reservoir.csv", best_by_group, preferred)
    write_json(out_dir / "best_by_model_reservoir.json", {"rows": best_by_group})
    write_csv(out_dir / "best_overall_by_model.csv", best_by_model, preferred)
    write_json(out_dir / "best_overall_by_model.json", {"rows": best_by_model})
    write_json(out_dir / "commands.json", commands)
    write_json(
        out_dir / "suite_config.json",
        {
            "phase": "E3",
            "args": vars(args),
            "git": get_git_info(REPO_ROOT),
            **get_runtime_info(),
            "config_hash": file_sha256(args.config),
            "data_hash": file_sha256(args.data_file),
            "selection": {"selection_metric": "val_absL2h", "selected_by": "validation"},
            "headroom": headroom,
            "num_rows": len(rows),
            "num_failures": len(failures),
            "spectral_error_normalization": "torch.fft.rfft(..., norm='forward'), mean over samples of squared complex magnitude",
        },
    )
    if failures:
        write_json(out_dir / "failed_runs.json", failures)
    if args.compute_spectral_error:
        write_json(
            out_dir / "spectral_error_manifest.json",
            {
                "status": "helper_available",
                "note": "spectral_error_rows(pred, target) is implemented for saved predictions; suite child scripts do not persist predictions by default.",
            },
        )
    _plot_bar(best_by_group, "test_absL2h", out_dir / "E3_test_absL2h_vs_reservoir", "E3 best test absL2h by model/reservoir")
    _plot_bar(best_by_group, "ratio_to_dlin", out_dir / "E3_ratio_to_dlin_vs_reservoir", "E3 ratio to D_lin by model/reservoir")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
