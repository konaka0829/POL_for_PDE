from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_floats(raw: str) -> list[float]:
    values = [float(item) for item in parse_csv(raw)]
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def parse_models(raw: str, *, allowed: set[str] | None = None) -> list[str]:
    values = parse_csv(raw)
    if not values:
        raise ValueError("expected at least one model")
    allowed = allowed or {"model1", "model2", "model3"}
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"unsupported model(s): {', '.join(invalid)}")
    return values


def parse_reservoirs(raw: str) -> list[str]:
    allowed = {"static", "heat", "advection", "burgers", "reaction_diffusion", "ks"}
    values = parse_csv(raw)
    if not values:
        raise ValueError("expected at least one reservoir")
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"unsupported reservoir(s): {', '.join(invalid)}")
    return values


def safe_tag(value: Any) -> str:
    text = str(value)
    try:
        text = format(float(value), ".12g")
    except (TypeError, ValueError):
        pass
    return text.replace(".", "p").replace("-", "m").replace("+", "p").replace("/", "_")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def dedupe_fieldnames(rows: Iterable[dict[str, Any]], preferred: list[str] | None = None) -> list[str]:
    seen: set[str] = set()
    fields: list[str] = []
    for name in preferred or []:
        if name not in seen:
            fields.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                fields.append(name)
                seen.add(name)
    return fields


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or dedupe_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def command_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("TORCH_NUM_THREADS", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    return env


@dataclass
class CommandRecord:
    name: str
    command: list[str]
    cwd: str
    return_code: int | None
    status: str
    log_path: str | None = None


@dataclass(frozen=True)
class SuiteJob:
    index: int
    label: str
    run: Callable[[], "SuiteJobResult"]


@dataclass
class SuiteJobResult:
    index: int
    row: dict[str, Any]
    commands: list[dict[str, Any]]
    failures: list[dict[str, Any]]


def validate_max_workers(max_workers: int) -> int:
    value = int(max_workers)
    if value <= 0:
        raise ValueError("--max-workers must be positive")
    return value


def run_suite_jobs(*, jobs: list[SuiteJob], max_workers: int, progress_label: str) -> list[SuiteJobResult]:
    if not jobs:
        validate_max_workers(max_workers)
        return []
    workers = min(validate_max_workers(max_workers), len(jobs))
    if workers == 1:
        return [job.run() for job in jobs]

    print(f"[{progress_label}] Launching {workers} worker(s) for {len(jobs)} job(s)", flush=True)
    results_by_index: dict[int, SuiteJobResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(job.run): job for job in jobs}
        completed = 0
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                row = {"status": "fail", "reason": str(exc), "job_label": job.label}
                result = SuiteJobResult(index=job.index, row=row, commands=[], failures=[row])
            results_by_index[job.index] = result
            status = result.row.get("status")
            print(f"[{progress_label} {completed}/{len(jobs)}] {job.label} -> {status}", flush=True)

    return [results_by_index[job.index] for job in sorted(jobs, key=lambda item: item.index)]


def run_recorded_command(
    *,
    name: str,
    command: list[str],
    cwd: Path,
    commands: list[dict[str, Any]],
    dry_run: bool,
    log_path: Path | None = None,
) -> None:
    record = CommandRecord(
        name=name,
        command=list(command),
        cwd=str(cwd),
        return_code=None,
        status="dry_run" if dry_run else "pending",
        log_path=str(log_path) if log_path is not None else None,
    )
    if dry_run:
        commands.append(record.__dict__)
        return
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=command_env())
    record.return_code = proc.returncode
    record.status = "ok" if proc.returncode == 0 else "fail"
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")
    commands.append(record.__dict__)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command, output=proc.stdout, stderr=proc.stderr)


def read_config_defaults(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    return load_json(path)


def config_value(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    return config.get(section, {}).get(key, default)


def resolve_T(config: dict[str, Any], explicit: float | None = None) -> float:
    if explicit is not None:
        return float(explicit)
    return float(config_value(config, "target", "T", 1.0))


def alpha_ttilde_pairs(*, alpha_values: str, ttilde_values: str | None, T: float) -> list[tuple[float, float]]:
    if ttilde_values:
        values = []
        for ttilde in parse_floats(ttilde_values):
            values.append((float(ttilde) / float(T), float(ttilde)))
        return values
    return [(alpha, alpha * float(T)) for alpha in parse_floats(alpha_values)]


def maybe_add_flag(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def bool_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def base_data_args(args: Any) -> list[str]:
    command = ["--config", str(args.config), "--data-file", str(args.data_file)]
    for name, flag in [
        ("sub", "--sub"),
        ("sim_dtype", "--sim-dtype"),
        ("ridge_dtype", "--ridge-dtype"),
        ("standardize_features", "--standardize-features"),
        ("elm_h", "--elm-h"),
        ("elm_activation", "--elm-activation"),
        ("elm_seed", "--elm-seed"),
        ("elm_weight_scale", "--elm-weight-scale"),
        ("elm_bias_scale", "--elm-bias-scale"),
    ]:
        if hasattr(args, name):
            maybe_add_flag(command, flag, getattr(args, name))
    if getattr(args, "allow_metadata_mismatch", False):
        command.append("--allow-metadata-mismatch")
    if getattr(args, "require_complete_metadata", False):
        command.append("--require-complete-metadata")
    return command


def reservoir_grid(reservoir: str, args: Any) -> list[dict[str, Any]]:
    if reservoir == "static":
        return [{}]
    if reservoir == "heat":
        return [{"heat_nu": value} for value in parse_floats(args.heat_nu_values)]
    if reservoir == "advection":
        return [{"advection_c": value} for value in parse_floats(args.advection_c_values)]
    if reservoir == "burgers":
        rows = []
        for nu in parse_floats(args.res_burgers_nu_values):
            for b in parse_floats(args.res_burgers_b_values):
                rows.append({"res_burgers_nu": nu, "res_burgers_b": b})
        return rows
    if reservoir == "reaction_diffusion":
        rows = []
        for nu in parse_floats(args.rd_nu_values):
            for alpha in parse_floats(args.rd_alpha_values):
                for beta in parse_floats(args.rd_beta_values):
                    rows.append({"rd_nu": nu, "rd_alpha": alpha, "rd_beta": beta})
        return rows
    if reservoir == "ks":
        rows = []
        for b in parse_floats(args.ks_b_values):
            for eta in parse_floats(args.ks_eta_values):
                for kappa in parse_floats(args.ks_kappa_values):
                    rows.append({"ks_b": b, "ks_eta": eta, "ks_kappa": kappa})
        return rows
    raise ValueError(f"unsupported reservoir: {reservoir}")


def add_reservoir_params(command: list[str], params: dict[str, Any]) -> None:
    mapping = {
        "heat_nu": "--heat-nu",
        "advection_c": "--advection-c",
        "res_burgers_nu": "--res-burgers-nu",
        "res_burgers_b": "--res-burgers-b",
        "rd_nu": "--rd-nu",
        "rd_alpha": "--rd-alpha",
        "rd_beta": "--rd-beta",
        "ks_b": "--ks-b",
        "ks_eta": "--ks-eta",
        "ks_kappa": "--ks-kappa",
    }
    for key, flag in mapping.items():
        if key in params:
            command.extend([flag, str(params[key])])


def zeta_command(args: Any, *, model: str, reservoir: str, Ttilde: float, params: dict[str, Any], out_dir: Path) -> list[str]:
    command = [
        str(args.python),
        "scripts/run_zeta_path.py",
        *base_data_args(args),
        "--model",
        model,
        "--reservoir",
        reservoir,
        "--Ttilde",
        str(Ttilde),
        "--zeta-grid",
        str(args.zeta_grid),
        "--output-dir",
        str(out_dir),
    ]
    add_reservoir_params(command, params)
    bool_flag(command, getattr(args, "use_feature_cache", False), "--use-feature-cache")
    bool_flag(command, getattr(args, "refresh_feature_cache", False), "--refresh-feature-cache")
    if hasattr(args, "burgers_scheme"):
        maybe_add_flag(command, "--burgers-scheme", args.burgers_scheme)
    if hasattr(args, "burgers_dealias"):
        maybe_add_flag(command, "--burgers-dealias", int(args.burgers_dealias))
    if getattr(args, "ks_dealias", False):
        command.append("--ks-dealias")
    return command


def model123_command(
    args: Any,
    *,
    model: str,
    reservoir: str,
    Ttilde: float,
    params: dict[str, Any],
    out_dir: Path,
    zeta: float | None = None,
    compute_defect: bool = False,
) -> list[str]:
    command = [
        str(args.python),
        "model123_burgers_1d.py",
        *base_data_args(args),
        "--model",
        model,
        "--reservoir",
        reservoir,
        "--Ttilde",
        str(Ttilde),
        "--out-dir",
        str(out_dir),
    ]
    if zeta is not None:
        command.extend(["--ridge-zeta", str(zeta)])
    add_reservoir_params(command, params)
    if hasattr(args, "burgers_scheme"):
        maybe_add_flag(command, "--burgers-scheme", args.burgers_scheme)
    if hasattr(args, "burgers_dealias"):
        maybe_add_flag(command, "--burgers-dealias", int(args.burgers_dealias))
    if getattr(args, "ks_dealias", False):
        command.append("--ks-dealias")
    if compute_defect:
        command.append("--compute-time-scaled-defect")
        for name, flag in [
            ("defect_target_nu", "--defect-target-nu"),
            ("defect_time_quadrature", "--defect-time-quadrature"),
            ("defect_beta_mode", "--defect-beta-mode"),
        ]:
            value = getattr(args, name, None)
            if value is not None:
                maybe_add_flag(command, flag, value)
    return command


def row_from_zeta_run(run_dir: Path) -> dict[str, Any]:
    payload = load_json(run_dir / "best_by_val.json")
    best = payload["best_by_val"]
    run_cfg = load_json(run_dir / "run_config.json")
    metrics = run_cfg.get("metrics", {})
    feature_cache = payload.get("feature_cache", {})
    return {
        "zeta_selected": best.get("zeta"),
        "selection_metric_name": "val_absL2h",
        "selection_metric_value": best.get("val_absL2h"),
        "train_absL2h": best.get("train_absL2h"),
        "val_absL2h": best.get("val_absL2h"),
        "test_absL2h": best.get("test_absL2h"),
        "train_relL2_mean": best.get("train_relL2_mean"),
        "val_relL2_mean": best.get("val_relL2_mean"),
        "test_relL2_mean": best.get("test_relL2_mean"),
        "test_relL2_agg": best.get("test_relL2_agg"),
        "W_fro_norm": best.get("W_fro_norm"),
        "W_l2h_hs_norm": best.get("W_hs_norm_l2h"),
        "d_eff": best.get("d_eff"),
        "cond_zeta": best.get("cond_zeta"),
        "domain_length": payload.get("domain_length"),
        "effective_nx": payload.get("effective_nx"),
        "dx": payload.get("dx"),
        "data_hash": run_cfg.get("data_sha256") or feature_cache.get("dataset_hash"),
        "split_hash": feature_cache.get("split_hash"),
        "output_dir": str(run_dir),
        "run_metrics": metrics,
    }


def row_from_model123_run(run_dir: Path) -> dict[str, Any]:
    run_cfg = load_json(run_dir / "run_config.json")
    row = {
        "zeta_selected": run_cfg.get("ridge_zeta"),
        "selection_metric_name": run_cfg.get("selection", {}).get("selection_metric"),
        "selection_metric_value": run_cfg.get("val_absL2h") if run_cfg.get("val_absL2h") is not None else run_cfg.get("test_absL2h"),
        "train_absL2h": run_cfg.get("train_absL2h"),
        "val_absL2h": run_cfg.get("val_absL2h"),
        "test_absL2h": run_cfg.get("test_absL2h"),
        "train_relL2_mean": run_cfg.get("train_relL2_mean"),
        "val_relL2_mean": run_cfg.get("val_relL2_mean"),
        "test_relL2_mean": run_cfg.get("test_relL2_mean"),
        "test_relL2_agg": run_cfg.get("test_relL2_agg"),
        "domain_length": run_cfg.get("domain_length"),
        "effective_nx": run_cfg.get("effective_nx"),
        "dx": run_cfg.get("dx"),
        "data_hash": run_cfg.get("data_sha256"),
        "split_hash": run_cfg.get("split", {}).get("test_indices_hash"),
        "output_dir": str(run_dir),
    }
    defect_path = run_dir / "time_scaled_defect_metrics.json"
    if defect_path.exists():
        defect = load_json(defect_path)
        for key in [
            "delta_scale_rms_abs_l2h",
            "delta_scale_mean_abs_l2h",
            "delta_scale_std_abs_l2h",
            "corr_error_delta_scale_pearson",
            "corr_error_delta_scale_spearman",
        ]:
            row[key] = defect.get(key)
    return row


def best_by_validation(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in group_keys), []).append(row)
    best_rows = []
    for key in sorted(groups):
        candidates = [row for row in groups[key] if row.get("val_absL2h") is not None]
        if not candidates:
            raise ValueError(f"validation metric val_absL2h is missing for group {key}")
        best_rows.append(min(candidates, key=lambda row: float(row["val_absL2h"])))
    return best_rows


def save_all_formats(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path_no_ext.with_suffix(f".{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)
