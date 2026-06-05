#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


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


@dataclass(frozen=True)
class SweepParameter:
    name: str
    cli_flag: str
    kind: str
    label: str
    positive: bool = False
    nonnegative: bool = False
    prefer_log_axis: bool = False
    reservoirs: tuple[str, ...] = ()


PARAMETERS: dict[str, SweepParameter] = {
    "alpha": SweepParameter(
        name="alpha",
        cli_flag="",
        kind="float",
        label="alpha = Ttilde/T",
        positive=True,
    ),
    "Ttilde": SweepParameter(
        name="Ttilde",
        cli_flag="--Ttilde",
        kind="float",
        label="Ttilde",
        positive=True,
    ),
    "dt": SweepParameter(
        name="dt",
        cli_flag="--dt",
        kind="float",
        label="dt",
        positive=True,
        prefer_log_axis=True,
    ),
    "K": SweepParameter(
        name="K",
        cli_flag="--K",
        kind="int",
        label="K",
        positive=True,
    ),
    "J": SweepParameter(
        name="J",
        cli_flag="--J",
        kind="int",
        label="J",
        positive=True,
    ),
    "rd_nu": SweepParameter(
        name="rd_nu",
        cli_flag="--rd-nu",
        kind="float",
        label="rd_nu",
        positive=True,
        prefer_log_axis=True,
        reservoirs=("reaction_diffusion",),
    ),
    "rd_alpha": SweepParameter(
        name="rd_alpha",
        cli_flag="--rd-alpha",
        kind="float",
        label="rd_alpha",
        reservoirs=("reaction_diffusion",),
    ),
    "rd_beta": SweepParameter(
        name="rd_beta",
        cli_flag="--rd-beta",
        kind="float",
        label="rd_beta",
        reservoirs=("reaction_diffusion",),
    ),
    "res_burgers_nu": SweepParameter(
        name="res_burgers_nu",
        cli_flag="--res-burgers-nu",
        kind="float",
        label="res_burgers_nu",
        positive=True,
        prefer_log_axis=True,
        reservoirs=("burgers",),
    ),
    "res_burgers_b": SweepParameter(
        name="res_burgers_b",
        cli_flag="--res-burgers-b",
        kind="float",
        label="res_burgers_b",
        reservoirs=("burgers",),
    ),
    "ks_b": SweepParameter(
        name="ks_b",
        cli_flag="--ks-b",
        kind="float",
        label="ks_b",
        reservoirs=("ks",),
    ),
    "ks_eta": SweepParameter(
        name="ks_eta",
        cli_flag="--ks-eta",
        kind="float",
        label="ks_eta",
        positive=True,
        prefer_log_axis=True,
        reservoirs=("ks",),
    ),
    "ks_kappa": SweepParameter(
        name="ks_kappa",
        cli_flag="--ks-kappa",
        kind="float",
        label="ks_kappa",
        nonnegative=True,
        prefer_log_axis=True,
        reservoirs=("ks",),
    ),
}

ALIASES = {
    "alpha": "alpha",
    "time_alpha": "alpha",
    "time-alpha": "alpha",
    "alpha_scale": "alpha",
    "alpha-scale": "alpha",
    "ttilde": "Ttilde",
    "t_tilde": "Ttilde",
    "res_burgers_nu": "res_burgers_nu",
    "res-burgers-nu": "res_burgers_nu",
    "res_burgers_b": "res_burgers_b",
    "res-burgers-b": "res_burgers_b",
    "rd_nu": "rd_nu",
    "rd-nu": "rd_nu",
    "rd_alpha": "rd_alpha",
    "rd-alpha": "rd_alpha",
    "rd_beta": "rd_beta",
    "rd-beta": "rd_beta",
    "ks_b": "ks_b",
    "ks-b": "ks_b",
    "ks_eta": "ks_eta",
    "ks-eta": "ks_eta",
    "ks_kappa": "ks_kappa",
    "ks-kappa": "ks_kappa",
    "dt": "dt",
    "k": "K",
    "j": "J",
}


@dataclass(frozen=True)
class SweepSpec:
    parameter: SweepParameter
    values: tuple[Any, ...]


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


def canonical_parameter_name(raw: str) -> str:
    key = raw.strip()
    if not key:
        raise ValueError("Empty parameter name")
    normalized = key.replace("-", "_")
    if normalized in PARAMETERS:
        return normalized
    lower_key = key.lower()
    lower_norm = normalized.lower()
    if lower_key in ALIASES:
        return ALIASES[lower_key]
    if lower_norm in ALIASES:
        return ALIASES[lower_norm]
    if key in PARAMETERS:
        return key
    raise ValueError(
        "Unsupported sweep parameter: %s. Available: %s"
        % (raw, ", ".join(sorted(PARAMETERS.keys())))
    )


def safe_tag(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".12g").replace(".", "p").replace("-", "m")


def clip_for_log(value: float, eps: float) -> float:
    return value if value > 0.0 else eps


def append_optional_flag(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def cast_value(raw: str, spec: SweepParameter) -> Any:
    if spec.kind == "int":
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"Parameter {spec.name} requires int values") from exc
    elif spec.kind == "float":
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"Parameter {spec.name} requires float values") from exc
    else:
        lowered = raw.lower()
        if lowered in {"1", "true", "yes", "on"}:
            value = True
        elif lowered in {"0", "false", "no", "off"}:
            value = False
        else:
            raise ValueError(f"Parameter {spec.name} requires boolean values")
    if spec.positive and value <= 0:
        raise ValueError(f"Parameter {spec.name} must be positive")
    if spec.nonnegative and value < 0:
        raise ValueError(f"Parameter {spec.name} must be nonnegative")
    return value


def parse_sweep_assignment(raw: str) -> SweepSpec:
    if "=" not in raw:
        raise ValueError("Sweep assignment must be of the form param=v1,v2,...")
    lhs, rhs = raw.split("=", 1)
    name = canonical_parameter_name(lhs)
    parameter = PARAMETERS[name]
    raw_values = parse_csv_values(rhs)
    if not raw_values:
        raise ValueError(f"No values provided for sweep parameter {name}")
    values = tuple(cast_value(item, parameter) for item in raw_values)
    return SweepSpec(parameter=parameter, values=values)


def build_range_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("range step must be positive")
    if stop < start:
        raise ValueError("range stop must be >= start")
    count = int(math.floor((stop - start) / step + 1e-12))
    values = [round(start + idx * step, 12) for idx in range(count + 1)]
    if not np.isclose(values[-1], stop):
        values.append(round(stop, 12))
    return values


def parse_sweep_range(raw: str) -> SweepSpec:
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) != 4:
        raise ValueError("Sweep range must be of the form param:start:stop:step")
    name = canonical_parameter_name(parts[0])
    parameter = PARAMETERS[name]
    if parameter.kind not in {"int", "float"}:
        raise ValueError(f"Sweep ranges are only supported for numeric parameters: {name}")
    start = float(parts[1])
    stop = float(parts[2])
    step = float(parts[3])
    values = build_range_values(start, stop, step)
    casted = [cast_value(str(value), parameter) for value in values]
    if parameter.kind == "int":
        casted = [int(value) for value in casted]
    return SweepSpec(parameter=parameter, values=tuple(casted))


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


def run_one(cmd: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("[dry-run]\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return 0
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log_path.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")
    return proc.returncode


def dedupe_fieldnames(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        if name in seen:
            continue
        deduped.append(name)
        seen.add(name)
    return deduped


def require_time_grid_aligned_value(value: float, dt: float, name: str, *, alpha: float | None = None, T: float | None = None) -> None:
    steps = int(round(float(value) / float(dt)))
    if not math.isclose(steps * float(dt), float(value), rel_tol=1e-9, abs_tol=1e-12):
        context = ""
        if alpha is not None and T is not None:
            context = f" from alpha={alpha} and T={T}"
        raise ValueError(
            f"{name}={value}{context} is not aligned with dt={dt}; "
            "alpha-derived Ttilde must satisfy Ttilde = alpha*T = k*dt."
        )


def resolved_times(args: argparse.Namespace, overrides: dict[str, Any]) -> dict[str, float]:
    T = float(args.T)
    if "alpha" in overrides:
        alpha = float(overrides["alpha"])
        Ttilde = alpha * T
        dt = float(overrides.get("dt", args.dt))
        require_time_grid_aligned_value(Ttilde, dt, "Ttilde", alpha=alpha, T=T)
    else:
        Ttilde = float(overrides.get("Ttilde", args.Ttilde))
        alpha = Ttilde / T
    return {"T": T, "Ttilde": Ttilde, "alpha": alpha}


def load_run_metrics(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    args_payload = payload.get("args", {})
    T = float(payload.get("T", args_payload.get("T", 1.0)))
    Ttilde = float(payload.get("Ttilde", args_payload.get("Ttilde", T)))
    metrics = {
        "train_absL2h": float(payload["train_absL2h"]),
        "test_absL2h": float(payload["test_absL2h"]),
        "train_relL2": float(payload["train_relL2"]),
        "test_relL2": float(payload["test_relL2"]),
        "T": T,
        "Ttilde": Ttilde,
        "alpha": float(payload.get("alpha", Ttilde / T)),
        "resolved_obs": payload["resolved_obs"],
        "resolved_J": int(payload["resolved_J"]),
    }
    optional_keys = [
        "delta_scale_rms_abs_l2h",
        "delta_scale_mean_abs_l2h",
        "delta_scale_std_abs_l2h",
        "corr_error_delta_scale_pearson",
        "corr_error_delta_scale_spearman",
        "applies_directly_to_model1_bound",
        "time_scaled_defect_metric",
    ]
    defect_path = run_dir / "time_scaled_defect_metrics.json"
    defect_payload = json.loads(defect_path.read_text(encoding="utf-8")) if defect_path.exists() else {}
    for key in optional_keys:
        if key == "time_scaled_defect_metric" and "defect_metric" in defect_payload:
            metrics[key] = defect_payload["defect_metric"]
        elif key in defect_payload:
            metrics[key] = defect_payload[key]
        elif key in payload:
            metrics[key] = payload[key]
        else:
            metrics[key] = None
    return metrics


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-sweep model123_burgers_1d.py over one or more surrogate parameters and "
            "save summaries plus projected visualizations."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--models", default="model1,model2,model3")
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="Discrete sweep spec such as Ttilde=0.8,1.0,1.2 or rd_nu=1e-4,1e-3",
    )
    parser.add_argument(
        "--sweep-range",
        action="append",
        default=[],
        help="Range sweep spec such as Ttilde:0.5:1.5:0.05",
    )
    parser.add_argument("--data-file", default="data/burgers_model123.mat")
    parser.add_argument("--out-root", default="outputs/model123_param_sweep")
    parser.add_argument("--train-split", type=float, default=1000.0 / 1200.0)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--Ttilde", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-2)
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
    parser.add_argument("--burgers-fine-dt", type=float, default=1e-4)
    parser.add_argument("--burgers-dealias", type=int, choices=(0, 1), default=1)
    parser.add_argument("--elm-h", type=int, default=1024)
    parser.add_argument("--elm-activation", choices=("tanh", "relu", "identity"), default="tanh")
    parser.add_argument("--elm-seed", type=int, default=0)
    parser.add_argument("--elm-weight-scale", type=float, default=0.0)
    parser.add_argument("--elm-bias-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compute-time-scaled-defect", action="store_true")
    parser.add_argument("--defect-target-nu", type=float, default=None)
    parser.add_argument("--defect-time-quadrature", choices=("trapezoid", "left"), default="trapezoid")
    parser.add_argument("--defect-beta-mode", choices=("zero", "fixed"), default="zero")
    parser.add_argument("--defect-beta-fixed", type=float, default=0.0)
    parser.add_argument("--defect-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--best-k", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_sweeps(args: argparse.Namespace) -> list[SweepSpec]:
    specs: list[SweepSpec] = []
    seen: set[str] = set()
    for raw in args.sweep:
        spec = parse_sweep_assignment(raw)
        if spec.parameter.name in seen:
            raise ValueError(f"Duplicate sweep parameter: {spec.parameter.name}")
        specs.append(spec)
        seen.add(spec.parameter.name)
    for raw in args.sweep_range:
        spec = parse_sweep_range(raw)
        if spec.parameter.name in seen:
            raise ValueError(f"Duplicate sweep parameter: {spec.parameter.name}")
        specs.append(spec)
        seen.add(spec.parameter.name)
    if not specs:
        raise ValueError("At least one --sweep or --sweep-range must be provided")
    for spec in specs:
        allowed = spec.parameter.reservoirs
        if allowed and args.reservoir not in allowed:
            raise ValueError(
                "Parameter %s is not valid for reservoir=%s; valid reservoirs: %s"
                % (spec.parameter.name, args.reservoir, ", ".join(allowed))
            )
    names = {spec.parameter.name for spec in specs}
    if "alpha" in names and "Ttilde" in names:
        raise ValueError("Cannot sweep both alpha and Ttilde; alpha determines Ttilde = alpha*T.")
    return specs


def validate_args(args: argparse.Namespace) -> tuple[list[str], list[SweepSpec]]:
    if not (0.0 < args.train_split < 1.0):
        raise ValueError("--train-split must be in (0, 1)")
    if args.ntrain <= 0 or args.ntest <= 0 or args.batch_size <= 0 or args.sub <= 0:
        raise ValueError("ntrain, ntest, batch-size, and sub must be positive")
    if args.T <= 0.0 or args.Ttilde <= 0.0 or args.dt <= 0.0 or args.burgers_fine_dt <= 0.0:
        raise ValueError("T, Ttilde, dt, and burgers-fine-dt must be positive")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if args.best_k <= 0:
        raise ValueError("--best-k must be positive")
    if not args.dry_run and not os.path.exists(args.data_file):
        raise FileNotFoundError("Data file not found: %s" % args.data_file)
    models = parse_models(args.models)
    return models, validate_sweeps(args)


def build_param_grid(specs: list[SweepSpec]) -> list[dict[str, Any]]:
    names = [spec.parameter.name for spec in specs]
    grids: list[dict[str, Any]] = []
    for combo in itertools.product(*(spec.values for spec in specs)):
        grids.append({name: value for name, value in zip(names, combo)})
    return grids


def build_run_command(
    args: argparse.Namespace,
    model: str,
    overrides: dict[str, Any],
    out_dir: Path,
) -> list[str]:
    times = resolved_times(args, overrides)
    effective = {
        "Ttilde": times["Ttilde"],
        "dt": overrides.get("dt", args.dt),
        "K": overrides.get("K", args.K),
        "J": overrides.get("J", args.J),
        "rd_nu": overrides.get("rd_nu", args.rd_nu),
        "rd_alpha": overrides.get("rd_alpha", args.rd_alpha),
        "rd_beta": overrides.get("rd_beta", args.rd_beta),
        "res_burgers_nu": overrides.get("res_burgers_nu", args.res_burgers_nu),
        "res_burgers_b": overrides.get("res_burgers_b", args.res_burgers_b),
        "ks_b": overrides.get("ks_b", args.ks_b),
        "ks_eta": overrides.get("ks_eta", args.ks_eta),
        "ks_kappa": overrides.get("ks_kappa", args.ks_kappa),
    }
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
        str(effective["Ttilde"]),
        "--dt",
        str(effective["dt"]),
        "--K",
        str(effective["K"]),
        "--obs",
        args.obs,
        "--J",
        str(effective["J"]),
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
        str(effective["res_burgers_nu"]),
        "--res-burgers-b",
        str(effective["res_burgers_b"]),
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
        append_optional_flag(cmd, "--rd-nu", effective["rd_nu"])
        append_optional_flag(cmd, "--rd-alpha", effective["rd_alpha"])
        append_optional_flag(cmd, "--rd-beta", effective["rd_beta"])
    elif args.reservoir == "ks":
        append_optional_flag(cmd, "--ks-b", effective["ks_b"])
        append_optional_flag(cmd, "--ks-eta", effective["ks_eta"])
        append_optional_flag(cmd, "--ks-kappa", effective["ks_kappa"])
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

    if getattr(args, "compute_time_scaled_defect", False):
        cmd.append("--compute-time-scaled-defect")
        append_optional_flag(cmd, "--defect-target-nu", getattr(args, "defect_target_nu", None))
        append_optional_flag(cmd, "--defect-time-quadrature", getattr(args, "defect_time_quadrature", "trapezoid"))
        append_optional_flag(cmd, "--defect-beta-mode", getattr(args, "defect_beta_mode", "zero"))
        append_optional_flag(cmd, "--defect-beta-fixed", getattr(args, "defect_beta_fixed", 0.0))
        append_optional_flag(cmd, "--defect-dtype", getattr(args, "defect_dtype", "float64"))

    return cmd


def build_run_dir(model_dir: Path, overrides: dict[str, Any], sweep_names: list[str]) -> Path:
    tag = "__".join(f"{name}_{safe_tag(overrides[name])}" for name in sweep_names)
    return model_dir / tag


def make_base_row(
    *,
    model: str,
    run_dir: Path,
    sweep_names: list[str],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "status": "pending",
        "return_code": None,
        "train_absL2h": None,
        "test_absL2h": None,
        "test_absL2h_plot": None,
        "train_relL2": None,
        "test_relL2": None,
        "test_relL2_plot": None,
        "T": None,
        "Ttilde": None,
        "alpha": None,
        "delta_scale_rms_abs_l2h": None,
        "delta_scale_mean_abs_l2h": None,
        "delta_scale_std_abs_l2h": None,
        "corr_error_delta_scale_pearson": None,
        "corr_error_delta_scale_spearman": None,
        "applies_directly_to_model1_bound": None,
        "time_scaled_defect_metric": None,
        "resolved_obs": None,
        "resolved_J": None,
        "run_dir": str(run_dir),
        "sweep_id": "__".join(f"{name}={overrides[name]}" for name in sweep_names),
    }
    for name in sweep_names:
        row[name] = overrides[name]
    return row


def run_model_jobs(
    *,
    args: argparse.Namespace,
    model: str,
    sweep_specs: list[SweepSpec],
    model_dir: Path,
    env: dict[str, str],
    eps: float,
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    pending_jobs: list[dict[str, Any]] = []
    had_failure = False
    sweep_names = [spec.parameter.name for spec in sweep_specs]

    for overrides in build_param_grid(sweep_specs):
        run_dir = build_run_dir(model_dir, overrides, sweep_names)
        row = make_base_row(model=model, run_dir=run_dir, sweep_names=sweep_names, overrides=overrides)
        row.update(resolved_times(args, overrides))
        config_path = run_dir / "run_config.json"

        if args.skip_existing and config_path.exists():
            metrics = load_run_metrics(run_dir)
            row.update(metrics)
            row["status"] = "ok"
            row["return_code"] = 0
            row["test_absL2h_plot"] = clip_for_log(float(row["test_absL2h"]), eps)
            row["test_relL2_plot"] = clip_for_log(float(row["test_relL2"]), eps)
            rows.append(row)
            print(
                "[%s] reuse %s -> test_absL2h=%.6e"
                % (model, row["sweep_id"], row["test_absL2h"]),
                flush=True,
            )
            continue

        if args.dry_run:
            cmd = build_run_command(args, model, overrides, run_dir)
            run_one(cmd, run_dir / "stdout_stderr.log", env=env, dry_run=True)
            row["status"] = "dry_run"
            row["return_code"] = 0
            rows.append(row)
            print("[%s] dry-run %s" % (model, row["sweep_id"]), flush=True)
            continue

        pending_jobs.append(
            {
                "run_dir": run_dir,
                "row": row,
                "cmd": build_run_command(args, model, overrides, run_dir),
            }
        )

    if args.dry_run or not pending_jobs:
        rows.sort(key=lambda item: item["sweep_id"])
        return rows, had_failure

    total_jobs = len(pending_jobs)
    max_workers = min(args.max_workers, total_jobs)
    print(
        "[%s] Launching %d worker(s) for %d job(s)" % (model, max_workers, total_jobs),
        flush=True,
    )

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
                row["test_absL2h_plot"] = clip_for_log(float(row["test_absL2h"]), eps)
                row["test_relL2_plot"] = clip_for_log(float(row["test_relL2"]), eps)
                print(
                    "[done %d/%d] [%s] %s -> test_absL2h=%.6e"
                    % (completed, total_jobs, model, row["sweep_id"], row["test_absL2h"]),
                    flush=True,
                )
            else:
                row["status"] = "fail"
                had_failure = True
                print(
                    "[fail %d/%d] [%s] %s -> rc=%s"
                    % (completed, total_jobs, model, row["sweep_id"], return_code),
                    flush=True,
                )
            rows.append(row)
        running_jobs = still_running
        if running_jobs:
            time.sleep(0.1)

    rows.sort(key=lambda item: item["sweep_id"])
    return rows, had_failure


def set_axis_scale(ax, axis: str, spec: SweepParameter, values: list[Any]) -> None:
    if not spec.prefer_log_axis:
        return
    numeric = [float(value) for value in values]
    if numeric and min(numeric) > 0.0:
        if axis == "x":
            ax.set_xscale("log")
        elif axis == "y":
            ax.set_yscale("log")


def save_1d_profile_plot(
    *,
    model: str,
    parameter: SweepParameter,
    rows: list[dict[str, Any]],
    out_path: Path,
    eps: float,
) -> None:
    groups: dict[Any, list[float]] = {}
    for row in rows:
        key = row[parameter.name]
        groups.setdefault(key, []).append(float(row["test_absL2h"]))
    x_values = sorted(groups.keys(), key=float)
    best_values = [min(groups[key]) for key in x_values]
    mean_values = [float(np.mean(groups[key])) for key in x_values]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    y_best = [clip_for_log(value, eps) for value in best_values]
    y_mean = [clip_for_log(value, eps) for value in mean_values]
    ax.plot(x_values, y_best, marker="o", linewidth=1.8, label="best")
    ax.plot(x_values, y_mean, marker="s", linewidth=1.2, linestyle="--", label="mean")
    set_axis_scale(ax, "x", parameter, x_values)
    ax.set_yscale("log")
    ax.set_xlabel(parameter.label)
    ax.set_ylabel("test absL2h")
    ax.set_title(f"{model}: {parameter.label} profile")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def build_pair_projection(
    rows: list[dict[str, Any]],
    x_name: str,
    y_name: str,
) -> tuple[list[Any], list[Any], np.ndarray]:
    x_values = sorted({row[x_name] for row in rows}, key=float)
    y_values = sorted({row[y_name] for row in rows}, key=float)
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    x_to_idx = {value: idx for idx, value in enumerate(x_values)}
    y_to_idx = {value: idx for idx, value in enumerate(y_values)}
    for row in rows:
        xi = x_to_idx[row[x_name]]
        yi = y_to_idx[row[y_name]]
        score = float(row["test_absL2h"])
        current = matrix[yi, xi]
        if np.isnan(current) or score < current:
            matrix[yi, xi] = score
    return x_values, y_values, matrix


def save_pair_heatmap(
    *,
    model: str,
    x_spec: SweepParameter,
    y_spec: SweepParameter,
    rows: list[dict[str, Any]],
    out_path: Path,
    eps: float,
) -> None:
    x_values, y_values, matrix = build_pair_projection(rows, x_spec.name, y_spec.name)
    if matrix.size == 0:
        return
    filled = np.where(np.isnan(matrix), np.nan, np.maximum(matrix, eps))
    finite = filled[np.isfinite(filled)]
    if finite.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    image = ax.imshow(
        filled,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=float(finite.min()), vmax=float(finite.max())),
    )
    ax.set_xticks(range(len(x_values)), labels=[format(float(v), ".4g") for v in x_values], rotation=45, ha="right")
    ax.set_yticks(range(len(y_values)), labels=[format(float(v), ".4g") for v in y_values])
    ax.set_xlabel(x_spec.label)
    ax.set_ylabel(y_spec.label)
    ax.set_title(f"{model}: min projected error")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("test absL2h")

    for yi in range(matrix.shape[0]):
        for xi in range(matrix.shape[1]):
            value = matrix[yi, xi]
            if np.isnan(value):
                continue
            ax.text(xi, yi, f"{value:.2e}", ha="center", va="center", fontsize=7, color="white")

    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def save_combined_single_param_plot(
    *,
    model_rows: dict[str, list[dict[str, Any]]],
    parameter: SweepParameter,
    out_path: Path,
    eps: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for model, rows in model_rows.items():
        groups: dict[Any, list[float]] = {}
        for row in rows:
            groups.setdefault(row[parameter.name], []).append(float(row["test_absL2h"]))
        x_values = sorted(groups.keys(), key=float)
        y_values = [clip_for_log(min(groups[key]), eps) for key in x_values]
        ax.plot(x_values, y_values, marker="o", linewidth=1.6, label=model)
    set_axis_scale(ax, "x", parameter, [row[parameter.name] for rows in model_rows.values() for row in rows])
    ax.set_yscale("log")
    ax.set_xlabel(parameter.label)
    ax.set_ylabel("test absL2h")
    ax.set_title(f"Model 1/2/3: {parameter.label} profile")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def numeric_values_equal(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def best_error_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: float(row["test_absL2h"]))


def all_model_profile_series(
    rows: list[dict[str, Any]],
    parameter_name: str,
    eps: float,
) -> tuple[list[Any], list[float]]:
    groups: dict[Any, list[float]] = {}
    for row in rows:
        groups.setdefault(row[parameter_name], []).append(float(row["test_absL2h"]))
    x_values = sorted(groups.keys(), key=float)
    y_values = [clip_for_log(min(groups[x_value]), eps) for x_value in x_values]
    return x_values, y_values


def all_model_fixed_slice_series(
    rows: list[dict[str, Any]],
    parameter_name: str,
    fixed_values: dict[str, Any],
    eps: float,
) -> tuple[list[Any], list[float]]:
    groups: dict[Any, list[float]] = {}
    for row in rows:
        if all(
            numeric_values_equal(row[name], value)
            for name, value in fixed_values.items()
            if name != parameter_name
        ):
            groups.setdefault(row[parameter_name], []).append(float(row["test_absL2h"]))
    x_values = sorted(groups.keys(), key=float)
    y_values = [clip_for_log(min(groups[x_value]), eps) for x_value in x_values]
    return x_values, y_values


def save_all_model_param_vs_error_plot(
    *,
    series_by_model: dict[str, tuple[list[Any], list[float]]],
    parameter: SweepParameter,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plotted = False
    for model in VALID_MODELS:
        if model not in series_by_model:
            continue
        x_values, y_values = series_by_model[model]
        if not x_values:
            continue
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model),
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return

    set_axis_scale(
        ax,
        "x",
        parameter,
        [x for x_values, _ in series_by_model.values() for x in x_values],
    )
    ax.set_yscale("log")
    ax.set_xlabel(param_vs_error_xlabel(parameter), fontsize=18)
    ax.set_ylabel("Error", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=14)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_path.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def param_vs_error_xlabel(parameter: SweepParameter) -> str:
    if parameter.name == "alpha":
        return r"$\alpha$"
    if parameter.name == "res_burgers_nu":
        return r"$\tilde{\nu}$"
    return parameter.label


def save_all_model_param_comparison_plots(
    *,
    model_rows: dict[str, list[dict[str, Any]]],
    sweep_specs: list[SweepSpec],
    out_root: Path,
    eps: float,
) -> None:
    ok_by_model = {
        model: [row for row in rows if row["status"] == "ok"]
        for model, rows in model_rows.items()
        if any(row["status"] == "ok" for row in rows)
    }
    if not ok_by_model:
        return

    sweep_names = [spec.parameter.name for spec in sweep_specs]
    model_best_rows = {model: best_error_row(rows) for model, rows in ok_by_model.items()}
    common_best_model, common_best_row = min(
        model_best_rows.items(),
        key=lambda item: float(item[1]["test_absL2h"]),
    )
    common_fixed_values = {name: common_best_row[name] for name in sweep_names}

    settings = {
        "profile_optimized": {
            "description": (
                "For each plotted parameter value and model, all other swept parameters "
                "are optimized by taking the minimum test_absL2h over their grid values."
            )
        },
        "slice_model_best_fixed": {
            "description": (
                "For each model, non-plotted swept parameters are fixed to that model's "
                "global-best row; the plotted parameter is varied."
            ),
            "fixed_values_by_model": {
                model: {name: row[name] for name in sweep_names}
                for model, row in model_best_rows.items()
            },
        },
        "slice_common_best_fixed": {
            "description": (
                "For all models, non-plotted swept parameters are fixed to the "
                "dataset-wide global-best row; the plotted parameter is varied."
            ),
            "source_model": common_best_model,
            "source_test_absL2h": common_best_row["test_absL2h"],
            "fixed_values": common_fixed_values,
        },
    }
    (out_root / "all_models_param_vs_error_plot_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    for spec in sweep_specs:
        parameter = spec.parameter
        save_all_model_param_vs_error_plot(
            series_by_model={
                model: all_model_profile_series(rows, parameter.name, eps)
                for model, rows in ok_by_model.items()
            },
            parameter=parameter,
            out_path=out_root / f"{parameter.name}_vs_error_profile_optimized_all_models",
        )
        save_all_model_param_vs_error_plot(
            series_by_model={
                model: all_model_fixed_slice_series(
                    rows=rows,
                    parameter_name=parameter.name,
                    fixed_values={name: model_best_rows[model][name] for name in sweep_names},
                    eps=eps,
                )
                for model, rows in ok_by_model.items()
            },
            parameter=parameter,
            out_path=out_root / f"{parameter.name}_vs_error_slice_model_best_fixed_all_models",
        )
        save_all_model_param_vs_error_plot(
            series_by_model={
                model: all_model_fixed_slice_series(
                    rows=rows,
                    parameter_name=parameter.name,
                    fixed_values=common_fixed_values,
                    eps=eps,
                )
                for model, rows in ok_by_model.items()
            },
            parameter=parameter,
            out_path=out_root / f"{parameter.name}_vs_error_slice_common_best_fixed_all_models",
        )


def save_best_runs(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    sweep_names: list[str],
    best_k: int,
) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    ok_rows.sort(key=lambda row: float(row["test_absL2h"]))
    trimmed = ok_rows[:best_k]
    fieldnames = [
        "model",
        "test_absL2h",
        "train_absL2h",
        "test_relL2",
        "train_relL2",
        *sweep_names,
        "run_dir",
        "sweep_id",
    ]
    write_csv(out_path.with_suffix(".csv"), trimmed, dedupe_fieldnames(fieldnames))
    out_path.with_suffix(".json").write_text(json.dumps(trimmed, indent=2), encoding="utf-8")


def save_visualizations(
    *,
    model: str,
    rows: list[dict[str, Any]],
    sweep_specs: list[SweepSpec],
    model_dir: Path,
    eps: float,
    best_k: int,
) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return

    sweep_names = [spec.parameter.name for spec in sweep_specs]
    save_best_runs(
        rows=ok_rows,
        out_path=model_dir / "best_runs",
        sweep_names=sweep_names,
        best_k=best_k,
    )

    for spec in sweep_specs:
        save_1d_profile_plot(
            model=model,
            parameter=spec.parameter,
            rows=ok_rows,
            out_path=model_dir / f"profile_{spec.parameter.name}",
            eps=eps,
        )

    if len(sweep_specs) >= 2:
        for x_spec, y_spec in itertools.combinations(sweep_specs, 2):
            save_pair_heatmap(
                model=model,
                x_spec=x_spec.parameter,
                y_spec=y_spec.parameter,
                rows=ok_rows,
                out_path=model_dir / f"pair_{x_spec.parameter.name}__{y_spec.parameter.name}",
                eps=eps,
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    models, sweep_specs = validate_args(args)

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
            sweep_specs=sweep_specs,
            model_dir=model_dir,
            env=env,
            eps=eps,
        )
        had_failure = had_failure or model_failed
        model_rows[model] = rows

        fieldnames = [
            "model",
            *[spec.parameter.name for spec in sweep_specs],
            "sweep_id",
            "status",
            "return_code",
            "train_absL2h",
            "test_absL2h",
            "test_absL2h_plot",
            "train_relL2",
            "test_relL2",
            "test_relL2_plot",
            "T",
            "Ttilde",
            "alpha",
            "delta_scale_rms_abs_l2h",
            "delta_scale_mean_abs_l2h",
            "delta_scale_std_abs_l2h",
            "corr_error_delta_scale_pearson",
            "corr_error_delta_scale_spearman",
            "resolved_obs",
            "resolved_J",
            "run_dir",
        ]
        write_csv(model_dir / "summary.csv", rows, dedupe_fieldnames(fieldnames))
        (model_dir / "summary.json").write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "model": model,
                    "sweeps": [
                        {
                            "name": spec.parameter.name,
                            "values": list(spec.values),
                        }
                        for spec in sweep_specs
                    ],
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        save_visualizations(
            model=model,
            rows=rows,
            sweep_specs=sweep_specs,
            model_dir=model_dir,
            eps=eps,
            best_k=args.best_k,
        )

    if len(sweep_specs) == 1:
        ok_by_model = {
            model: [row for row in rows if row["status"] == "ok"]
            for model, rows in model_rows.items()
            if any(row["status"] == "ok" for row in rows)
        }
        if ok_by_model:
            save_combined_single_param_plot(
                model_rows=ok_by_model,
                parameter=sweep_specs[0].parameter,
                out_path=out_root / f"combined_profile_{sweep_specs[0].parameter.name}",
                eps=eps,
            )
            single_name = sweep_specs[0].parameter.name
            if single_name == "alpha":
                save_combined_single_param_plot(
                    model_rows=ok_by_model,
                    parameter=sweep_specs[0].parameter,
                    out_path=out_root / "alpha_vs_error_all_models",
                    eps=eps,
                )
            else:
                save_combined_single_param_plot(
                    model_rows=ok_by_model,
                    parameter=sweep_specs[0].parameter,
                    out_path=out_root / f"parameter_{single_name}_vs_error_all_models",
                    eps=eps,
                )
                if single_name == "Ttilde":
                    save_combined_single_param_plot(
                        model_rows=ok_by_model,
                        parameter=PARAMETERS["alpha"],
                        out_path=out_root / "alpha_vs_error_all_models",
                        eps=eps,
                    )

    save_all_model_param_comparison_plots(
        model_rows=model_rows,
        sweep_specs=sweep_specs,
        out_root=out_root,
        eps=eps,
    )

    top_level = {
        "config": vars(args),
        "models": models,
        "sweeps": [
            {"name": spec.parameter.name, "values": list(spec.values)}
            for spec in sweep_specs
        ],
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
