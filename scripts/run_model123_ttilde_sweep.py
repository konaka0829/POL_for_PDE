#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VALID_MODELS = ("model1", "model2", "model3")


def parse_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_models(raw: str) -> list[str]:
    models = parse_csv_values(raw)
    if not models:
        raise ValueError("At least one model must be provided")
    invalid = [model for model in models if model not in VALID_MODELS]
    if invalid:
        raise ValueError(f"Unsupported model(s): {', '.join(invalid)}")
    return models


def build_ttilde_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("ttilde-step must be positive")
    if stop < start:
        raise ValueError("ttilde-stop must be >= ttilde-start")
    values: list[float] = []
    count = int(round((stop - start) / step))
    for idx in range(count + 1):
        value = start + idx * step
        values.append(round(value, 12))
    if not np.isclose(values[-1], stop):
        values.append(round(stop, 12))
    return values


def parse_ttilde_values(raw: str, start: float, stop: float, step: float) -> list[float]:
    if raw.strip():
        values = [float(item) for item in parse_csv_values(raw)]
    else:
        values = build_ttilde_range(start, stop, step)
    if not values:
        raise ValueError("No Ttilde values were parsed")
    for value in values:
        if value <= 0.0:
            raise ValueError("Ttilde values must be positive")
    return values


def safe_tag(value: float) -> str:
    return format(value, ".12g").replace(".", "p").replace("-", "m")


def clip_for_log(value: float, eps: float) -> float:
    return value if value > 0.0 else eps


def append_optional_flag(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def build_run_command(args: argparse.Namespace, model: str, ttilde: float, out_dir: Path) -> list[str]:
    cmd = [
        args.python,
        "model123_burgers_1d.py",
        "--model",
        model,
        "--data-mode",
        "single_split",
        "--data-file",
        args.data_file,
        "--train-split",
        str(args.train_split),
        "--seed",
        str(args.seed),
        "--ntrain",
        str(args.ntrain),
        "--ntest",
        str(args.ntest),
        "--batch-size",
        str(args.batch_size),
        "--T",
        str(args.T),
        "--Ttilde",
        str(ttilde),
        "--dt",
        str(args.dt),
        "--K",
        str(args.K),
        "--obs",
        args.obs,
        "--J",
        str(args.J),
        "--sensor-mode",
        args.sensor_mode,
        "--sensor-seed",
        str(args.sensor_seed),
        "--ridge-lambda",
        str(args.ridge_lambda),
        "--ridge-dtype",
        args.ridge_dtype,
        "--reservoir",
        args.reservoir,
        "--res-burgers-nu",
        str(args.res_burgers_nu),
        "--res-burgers-b",
        str(args.res_burgers_b),
        "--burgers-scheme",
        args.burgers_scheme,
        "--burgers-fine-dt",
        str(args.burgers_fine_dt),
        "--burgers-dealias",
        str(int(args.burgers_dealias)),
        "--device",
        args.device,
        "--out-dir",
        str(out_dir),
    ]
    append_optional_flag(cmd, "--feature-times", args.feature_times if args.feature_times else None)
    append_optional_flag(cmd, "--sub", args.sub)
    append_optional_flag(cmd, "--input-scale", args.input_scale)
    append_optional_flag(cmd, "--input-shift", args.input_shift)
    append_optional_flag(cmd, "--standardize-features", int(args.standardize_features))
    append_optional_flag(cmd, "--feature-std-eps", args.feature_std_eps)

    if args.shuffle:
        cmd.append("--shuffle")

    if args.reservoir == "reaction_diffusion":
        append_optional_flag(cmd, "--rd-nu", args.rd_nu)
        append_optional_flag(cmd, "--rd-alpha", args.rd_alpha)
        append_optional_flag(cmd, "--rd-beta", args.rd_beta)
    elif args.reservoir == "ks":
        append_optional_flag(cmd, "--ks-b", args.ks_b)
        append_optional_flag(cmd, "--ks-eta", args.ks_eta)
        append_optional_flag(cmd, "--ks-kappa", args.ks_kappa)
        if args.ks_dealias:
            cmd.append("--ks-dealias")

    if model == "model3":
        append_optional_flag(cmd, "--elm-h", args.elm_h)
        append_optional_flag(cmd, "--elm-activation", args.elm_activation)
        append_optional_flag(cmd, "--elm-seed", args.elm_seed)
        append_optional_flag(cmd, "--elm-weight-scale", args.elm_weight_scale)
        append_optional_flag(cmd, "--elm-bias-scale", args.elm_bias_scale)

    if args.save_model:
        cmd.append("--save-model")

    return cmd


def run_one(cmd: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("[dry-run]\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return 0

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log_text = proc.stdout + "\n\n[stderr]\n" + proc.stderr
    log_path.write_text(log_text, encoding="utf-8")
    return proc.returncode


def build_job_env(base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("TORCH_NUM_THREADS", "1")
    return env


def start_job(cmd: list[str], log_path: Path, env: dict[str, str]) -> tuple[subprocess.Popen[str], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write("[command]\n%s\n\n[output]\n" % " ".join(cmd))
    log_file.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc, log_file


def load_run_metrics(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    return {
        "train_relL2": float(payload["train_relL2"]),
        "test_relL2": float(payload["test_relL2"]),
        "resolved_obs": payload["resolved_obs"],
        "resolved_J": int(payload["resolved_J"]),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_model_plot(model: str, rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(
        [row["Ttilde"] for row in rows],
        [row["test_relL2"] for row in rows],
        marker="o",
        linewidth=1.6,
        label=model,
    )
    ax.set_xlabel("Ttilde")
    ax.set_ylabel("test relL2")
    ax.set_title("%s: Ttilde vs error" % model)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def save_combined_plot(
    model_rows: dict[str, list[dict[str, Any]]],
    out_path: Path,
    *,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for model, rows in model_rows.items():
        y_values = [
            row["test_relL2_plot"] if log_y else row["test_relL2"]
            for row in rows
        ]
        ax.plot(
            [row["Ttilde"] for row in rows],
            y_values,
            marker="o",
            linewidth=1.6,
            label=model,
        )
    ax.set_xlabel("Ttilde")
    ax.set_ylabel("test relL2")
    ax.set_title("Model 1/2/3: Ttilde vs error")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Ttilde for model123_burgers_1d.py and save error plots."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--models", default="model1,model2,model3")
    parser.add_argument("--ttilde-values", default="")
    parser.add_argument("--ttilde-start", type=float, default=0.5)
    parser.add_argument("--ttilde-stop", type=float, default=1.5)
    parser.add_argument("--ttilde-step", type=float, default=0.05)
    parser.add_argument("--data-file", default="data/burgers_T10_nu001.mat")
    parser.add_argument("--out-root", default="outputs/model123_burgers_ttilde_sweep")
    parser.add_argument("--train-split", type=float, default=1000.0 / 1200.0)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--feature-times", type=str, default="")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--reservoir", choices=("burgers", "reaction_diffusion", "ks"), default="burgers")
    parser.add_argument("--obs", choices=("full", "points", "fourier", "proj"), default="full")
    parser.add_argument("--J", type=int, default=1028)
    parser.add_argument("--sensor-mode", choices=("equispaced", "random"), default="equispaced")
    parser.add_argument("--sensor-seed", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--input-shift", type=float, default=0.0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-4)
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
    parser.add_argument("--burgers-fine-dt", type=float, default=1e-5)
    parser.add_argument("--burgers-dealias", type=int, choices=(0, 1), default=1)
    parser.add_argument("--elm-h", type=int, default=1024)
    parser.add_argument("--elm-activation", choices=("tanh", "relu", "identity"), default="tanh")
    parser.add_argument("--elm-seed", type=int, default=0)
    parser.add_argument("--elm-weight-scale", type=float, default=0.0)
    parser.add_argument("--elm-bias-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> tuple[list[str], list[float]]:
    if not os.path.exists(args.data_file):
        raise FileNotFoundError("Data file not found: %s" % args.data_file)
    if not (0.0 < args.train_split < 1.0):
        raise ValueError("--train-split must be in (0, 1)")
    if args.ntrain <= 0 or args.ntest <= 0 or args.batch_size <= 0 or args.sub <= 0:
        raise ValueError("ntrain, ntest, batch-size, and sub must be positive")
    if args.T <= 0.0 or args.dt <= 0.0 or args.burgers_fine_dt <= 0.0:
        raise ValueError("T, dt, and burgers-fine-dt must be positive")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    models = parse_models(args.models)
    ttilde_values = parse_ttilde_values(
        args.ttilde_values,
        args.ttilde_start,
        args.ttilde_stop,
        args.ttilde_step,
    )
    return models, ttilde_values


def run_model_jobs(
    *,
    args: argparse.Namespace,
    model: str,
    ttilde_values: list[float],
    model_dir: Path,
    env: dict[str, str],
    eps: float,
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    pending_jobs: list[dict[str, Any]] = []
    had_failure = False

    for ttilde in ttilde_values:
        run_dir = model_dir / ("ttilde_" + safe_tag(ttilde))
        config_path = run_dir / "run_config.json"
        row: dict[str, Any] = {
            "model": model,
            "Ttilde": float(ttilde),
            "status": "pending",
            "return_code": None,
            "train_relL2": None,
            "test_relL2": None,
            "test_relL2_plot": None,
            "resolved_obs": None,
            "resolved_J": None,
            "run_dir": str(run_dir),
        }

        if args.skip_existing and config_path.exists():
            metrics = load_run_metrics(run_dir)
            row.update(metrics)
            row["status"] = "ok"
            row["return_code"] = 0
            row["test_relL2_plot"] = clip_for_log(float(row["test_relL2"]), eps)
            rows.append(row)
            print("[%s] Ttilde=%g -> reuse test=%.6e" % (model, ttilde, row["test_relL2"]), flush=True)
            continue

        if args.dry_run:
            cmd = build_run_command(args, model, ttilde, run_dir)
            run_one(cmd, run_dir / "stdout_stderr.log", env=env, dry_run=True)
            row["status"] = "dry_run"
            row["return_code"] = 0
            rows.append(row)
            print("[%s] Ttilde=%g -> dry-run" % (model, ttilde), flush=True)
            continue

        pending_jobs.append(
            {
                "Ttilde": ttilde,
                "run_dir": run_dir,
                "row": row,
                "cmd": build_run_command(args, model, ttilde, run_dir),
            }
        )

    if args.dry_run or not pending_jobs:
        rows.sort(key=lambda item: float(item["Ttilde"]))
        return rows, had_failure

    total_jobs = len(pending_jobs)
    max_workers = min(args.max_workers, total_jobs)
    print("[%s] Launching %d worker(s) for %d job(s)" % (model, max_workers, total_jobs), flush=True)

    running_jobs: list[dict[str, Any]] = []
    completed = 0
    pending_idx = 0

    while pending_idx < total_jobs or running_jobs:
        while pending_idx < total_jobs and len(running_jobs) < max_workers:
            job = pending_jobs[pending_idx]
            pending_idx += 1
            proc, log_file = start_job(job["cmd"], job["run_dir"] / "stdout_stderr.log", env)
            job["proc"] = proc
            job["log_file"] = log_file
            running_jobs.append(job)

        still_running: list[dict[str, Any]] = []
        for job in running_jobs:
            return_code = job["proc"].poll()
            if return_code is None:
                still_running.append(job)
                continue

            job["log_file"].close()
            completed += 1
            row = job["row"]
            row["return_code"] = return_code
            config_path = Path(row["run_dir"]) / "run_config.json"
            if return_code == 0 and config_path.exists():
                metrics = load_run_metrics(Path(row["run_dir"]))
                row.update(metrics)
                row["status"] = "ok"
                row["test_relL2_plot"] = clip_for_log(float(row["test_relL2"]), eps)
                print(
                    "[done %d/%d] [%s] Ttilde=%g -> test=%.6e"
                    % (completed, total_jobs, model, job["Ttilde"], row["test_relL2"]),
                    flush=True,
                )
            else:
                row["status"] = "fail"
                had_failure = True
                print(
                    "[fail %d/%d] [%s] Ttilde=%g -> rc=%s"
                    % (completed, total_jobs, model, job["Ttilde"], return_code),
                    flush=True,
                )
            rows.append(row)
        running_jobs = still_running
        if running_jobs:
            time.sleep(0.1)

    rows.sort(key=lambda item: float(item["Ttilde"]))
    return rows, had_failure


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    models, ttilde_values = validate_args(args)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    eps = float(np.finfo(np.float64).eps)
    env = build_job_env(os.environ.copy())
    env.setdefault("MPLCONFIGDIR", str(out_root / ".mplconfig"))

    model_rows: dict[str, list[dict[str, Any]]] = {}
    had_failure = False

    for model in models:
        model_dir = out_root / model
        model_dir.mkdir(parents=True, exist_ok=True)
        rows, model_failed = run_model_jobs(
            args=args,
            model=model,
            ttilde_values=ttilde_values,
            model_dir=model_dir,
            env=env,
            eps=eps,
        )
        had_failure = had_failure or model_failed

        fieldnames = [
            "model",
            "Ttilde",
            "status",
            "return_code",
            "train_relL2",
            "test_relL2",
            "test_relL2_plot",
            "resolved_obs",
            "resolved_J",
            "run_dir",
        ]
        write_csv(model_dir / "summary.csv", rows, fieldnames)
        (model_dir / "summary.json").write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "model": model,
                    "ttilde_values": ttilde_values,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        ok_rows = [row for row in rows if row["status"] == "ok"]
        if ok_rows:
            save_model_plot(model, ok_rows, model_dir / "ttilde_vs_error")
            model_rows[model] = ok_rows

    if model_rows:
        combined_rows = {model: rows for model, rows in model_rows.items() if rows}
        if combined_rows:
            save_combined_plot(combined_rows, out_root / "ttilde_vs_error_all_models")
            save_combined_plot(
                combined_rows,
                out_root / "ttilde_vs_error_all_models_logy",
                log_y=True,
            )

    top_level = {
        "config": vars(args),
        "models": models,
        "ttilde_values": ttilde_values,
        "available_model_summaries": {
            model: str((out_root / model / "summary.csv")) for model in models
        },
    }
    (out_root / "summary.json").write_text(json.dumps(top_level, indent=2), encoding="utf-8")

    if args.dry_run:
        return 0
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
