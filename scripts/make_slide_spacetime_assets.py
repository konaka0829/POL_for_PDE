#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.burgers_spectral_1d import simulate_burgers_split_step
from pol.metadata import to_jsonable
from pol.plotting import plot_spacetime
from pol.reservoir_1d import Reservoir1DSolver, ReservoirConfig
from pol.spectral_etdrk4_1d import simulate_burgers_etdrk4_trajectory
from pol.time_grid import require_time_aligned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create individual target/surrogate space-time slide assets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-file", default="")
    parser.add_argument("--predictions", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-mode", choices=("first", "median-error", "best-error", "worst-error"), default="first")
    parser.add_argument("--trajectory", choices=("target", "surrogate", "both"), default="both")
    parser.add_argument("--reservoir", choices=("reaction_diffusion", "burgers", "ks", "static", "heat", "advection"), default="reaction_diffusion")
    parser.add_argument("--num-frames", type=int, default=101)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--symmetric-colorlim", action="store_true")
    parser.add_argument("--Ttilde", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--rd-nu", type=float, default=1e-3)
    parser.add_argument("--rd-alpha", type=float, default=1.0)
    parser.add_argument("--rd-beta", type=float, default=1.0)
    parser.add_argument("--res-burgers-nu", type=float, default=0.05)
    parser.add_argument("--res-burgers-b", type=float, default=1.0)
    parser.add_argument("--burgers-scheme", choices=("semi_implicit", "split_step", "etdrk4"), default="split_step")
    parser.add_argument("--burgers-fine-dt", type=float, default=1e-4)
    parser.add_argument("--burgers-dealias", type=int, choices=(0, 1), default=1)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--input-shift", type=float, default=0.0)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested by --device cuda but CUDA is unavailable")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    return torch.float32 if name == "float32" else torch.float64


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def target_config(config_path: Path, run_config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = load_json(config_path)
    target = cfg.get("target", {})
    domain = cfg.get("domain", {})
    run_target = (run_config or {}).get("resolved_config", {}).get("target", {})
    args_payload = (run_config or {}).get("args", {})
    nu_value = first_not_none(run_target.get("target_nu"), args_payload.get("target_nu"), target.get("nu"), target.get("target_nu"))
    if nu_value is None:
        raise ValueError(f"Could not resolve target Burgers viscosity from {config_path} or run_config.json")
    return {
        "T": float(first_not_none(run_target.get("T"), args_payload.get("T"), target.get("T"))),
        "dt": float(first_not_none(run_target.get("dt"), args_payload.get("dt"), target.get("dt"))),
        "nu": float(nu_value),
        "solver": str(first_not_none(run_target.get("solver"), args_payload.get("burgers_scheme"), target.get("solver"), target.get("time_integrator"), "split_step")),
        "dealias": bool(first_not_none(run_target.get("dealias"), args_payload.get("burgers_dealias"), target.get("dealias"), True)),
        "domain_length": float(first_not_none(run_target.get("domain_length"), args_payload.get("expected_domain_length"), domain.get("length"), 1.0)),
        "fine_dt": float(args_payload.get("burgers_fine_dt", 1e-4)),
    }


def load_predictions(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"predictions.pt does not exist: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def select_sample_from_predictions(payload: dict[str, Any], sample_index: int, sample_mode: str) -> tuple[int, torch.Tensor, dict[str, Any]]:
    if "x_test" not in payload:
        raise KeyError("predictions.pt is missing x_test")
    n = int(payload["x_test"].shape[0])
    if sample_mode == "first":
        idx = int(sample_index)
    else:
        if "per_sample_absL2h" not in payload:
            raise KeyError("sample-mode by error requires per_sample_absL2h in predictions.pt")
        errors = payload["per_sample_absL2h"].reshape(-1)
        order = torch.argsort(errors)
        idx = int(order[0] if sample_mode == "best-error" else order[-1] if sample_mode == "worst-error" else order[len(order) // 2])
    if idx < 0 or idx >= n:
        raise IndexError(f"sample index {idx} is out of range for {n} prediction samples")
    return idx, payload["x_test"][idx].detach().cpu(), {"source": "predictions", "sample_selection": sample_mode}


def select_sample_from_data_file(path: Path, sample_index: int) -> tuple[int, torch.Tensor, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"data file does not exist: {path}")
    if path.suffix.lower() != ".pt":
        raise ValueError(f"Only .pt datasets are supported without predictions.pt; got {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "u0_test" in payload:
        data = payload["u0_test"]
    elif "a" in payload:
        data = payload["a"]
    else:
        raise KeyError(f"{path} does not contain u0_test or a")
    n = int(data.shape[0])
    idx = int(sample_index)
    if idx < 0 or idx >= n:
        raise IndexError(f"sample index {idx} is out of range for {n} dataset samples")
    return idx, data[idx].detach().cpu(), {"source": "data_file", "sample_selection": "first"}


def linspace_obs_steps(total_steps: int, num_frames: int) -> list[int]:
    if num_frames <= 1:
        raise ValueError("--num-frames must be greater than 1")
    if total_steps < 1:
        raise ValueError("total integration steps must be positive")
    steps = np.unique(np.linspace(0, total_steps, num=min(num_frames, total_steps + 1), dtype=int))
    if steps[0] != 0:
        steps = np.insert(steps, 0, 0)
    return [int(v) for v in steps.tolist()]


def simulate_target_trajectory(u0: torch.Tensor, cfg: dict[str, Any], device: torch.device, dtype: torch.dtype, num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    T = float(cfg["T"])
    dt = float(cfg["dt"])
    steps = require_time_aligned(T, dt, "target T")
    obs_steps = linspace_obs_steps(steps, num_frames)
    positive = [step for step in obs_steps if step > 0]
    z0 = u0.reshape(1, -1).to(device=device, dtype=dtype)
    solver = str(cfg["solver"])
    if solver in {"etdrk4", "fourier_pseudospectral_etdrk4"}:
        states = simulate_burgers_etdrk4_trajectory(
            z0,
            nu=float(cfg["nu"]),
            b=1.0,
            T=T,
            dt=dt,
            obs_steps=positive,
            dealias=bool(cfg["dealias"]),
            domain_length=float(cfg["domain_length"]),
        )
    elif solver in {"split_step", "semi_implicit"}:
        states = simulate_burgers_split_step(
            z0,
            dt=dt,
            Tr=T,
            obs_steps=positive,
            nu=float(cfg["nu"]),
            fine_dt=float(cfg["fine_dt"]),
            b=1.0,
            dealias=bool(cfg["dealias"]),
            domain_length=float(cfg["domain_length"]),
        )
    else:
        raise ValueError(f"Unsupported target Burgers solver: {solver}")
    state_by_step = {step: state for step, state in zip(positive, states)}
    all_states = [z0.detach().cpu()]
    for step in obs_steps[1:]:
        all_states.append(state_by_step[step].detach().cpu())
    t = torch.tensor([step * dt for step in obs_steps], dtype=torch.float64)
    return t, torch.cat(all_states, dim=0)


def surrogate_config(args: argparse.Namespace, run_config: dict[str, Any] | None, target: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    args_payload = (run_config or {}).get("args", {})
    raw = (run_config or {}).get("resolved_config", {}).get("surrogate", {})
    reservoir = str(raw.get("reservoir", args_payload.get("reservoir", args.reservoir)))
    model = str(args_payload.get("model", ""))
    dt = float(raw.get("dt", args_payload.get("dt", args.dt if args.dt is not None else target["dt"])))
    Ttilde = float(raw.get("Ttilde", args_payload.get("Ttilde", args.Ttilde if args.Ttilde is not None else target["T"])))
    cfg = {
        "Ttilde": Ttilde,
        "dt": dt,
        "input_scale": float(raw.get("input_scale", args_payload.get("input_scale", args.input_scale))),
        "input_shift": float(raw.get("input_shift", args_payload.get("input_shift", args.input_shift))),
        "reservoir_config": ReservoirConfig(
            reservoir=reservoir,
            rd_nu=float(raw.get("rd_nu", args_payload.get("rd_nu", args.rd_nu))),
            rd_alpha=float(raw.get("rd_alpha", args_payload.get("rd_alpha", args.rd_alpha))),
            rd_beta=float(raw.get("rd_beta", args_payload.get("rd_beta", args.rd_beta))),
            res_burgers_nu=float(raw.get("res_burgers_nu", args_payload.get("res_burgers_nu", args.res_burgers_nu))),
            res_burgers_b=float(raw.get("res_burgers_b", args_payload.get("res_burgers_b", args.res_burgers_b))),
            burgers_scheme=str(raw.get("burgers_scheme", args_payload.get("burgers_scheme", args.burgers_scheme))),
            burgers_fine_dt=float(raw.get("burgers_fine_dt", args_payload.get("burgers_fine_dt", args.burgers_fine_dt))),
            burgers_dealias=bool(raw.get("burgers_dealias", args_payload.get("burgers_dealias", bool(args.burgers_dealias)))),
            domain_length=float(raw.get("domain_length", target["domain_length"])),
        ),
    }
    return model, reservoir, cfg


def simulate_surrogate_trajectory(u0: torch.Tensor, cfg: dict[str, Any], device: torch.device, dtype: torch.dtype, num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    Ttilde = float(cfg["Ttilde"])
    dt = float(cfg["dt"])
    steps = require_time_aligned(Ttilde, dt, "surrogate Ttilde")
    obs_steps = linspace_obs_steps(steps, num_frames)
    positive = [step for step in obs_steps if step > 0]
    z0 = (float(cfg["input_scale"]) * u0.reshape(1, -1).to(device=device, dtype=dtype)) + float(cfg["input_shift"])
    solver = Reservoir1DSolver(cfg["reservoir_config"])
    states = solver.simulate(z0, dt=dt, Tr=Ttilde, obs_steps=positive)
    state_by_step = {step: state for step, state in zip(positive, states)}
    all_states = [z0.detach().cpu()]
    for step in obs_steps[1:]:
        all_states.append(state_by_step[step].detach().cpu())
    t = torch.tensor([step * dt for step in obs_steps], dtype=torch.float64)
    return t, torch.cat(all_states, dim=0)


def paths_to_dict(paths: tuple[str, str, str]) -> dict[str, str]:
    return {"png": paths[0], "pdf": paths[1], "svg": paths[2]}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = load_json(Path(args.run_dir) / "run_config.json") if args.run_dir else None
    target = target_config(Path(args.config), run_config)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    if args.predictions:
        predictions = load_predictions(Path(args.predictions))
        idx, u0, selection_meta = select_sample_from_predictions(predictions, args.sample_index, args.sample_mode)
    elif args.data_file:
        idx, u0, selection_meta = select_sample_from_data_file(Path(args.data_file), args.sample_index)
    else:
        raise ValueError("Provide --predictions or --data-file so the script can load u0")

    model, reservoir, surrogate = surrogate_config(args, run_config, target)
    prefix = args.prefix.strip() if args.prefix.strip() else (model if model else reservoir)
    x = torch.linspace(0.0, float(target["domain_length"]), int(u0.numel()) + 1, dtype=torch.float64)[:-1]
    stem = f"{prefix}_sample{idx:03d}"
    files: dict[str, dict[str, str]] = {}

    if args.trajectory in {"target", "both"}:
        t, u_xt = simulate_target_trajectory(u0, target, device, dtype, args.num_frames)
        files["target_burgers_spacetime"] = paths_to_dict(
            plot_spacetime(
                x,
                t,
                u_xt,
                str(out_dir / f"{stem}_target_burgers_spacetime"),
                title="target Burgers trajectory",
                vmin=args.vmin,
                vmax=args.vmax,
                symmetric_colorlim=args.symmetric_colorlim,
            )
        )

    if args.trajectory in {"surrogate", "both"}:
        t, u_xt = simulate_surrogate_trajectory(u0, surrogate, device, dtype, args.num_frames)
        files["surrogate_spacetime"] = paths_to_dict(
            plot_spacetime(
                x,
                t,
                u_xt,
                str(out_dir / f"{stem}_{reservoir}_surrogate_spacetime"),
                title=f"{reservoir} surrogate trajectory",
                vmin=args.vmin,
                vmax=args.vmax,
                symmetric_colorlim=args.symmetric_colorlim,
            )
        )

    manifest = {
        "config": str(args.config),
        "data_file": args.data_file or None,
        "predictions_path": args.predictions or None,
        "run_dir": args.run_dir or None,
        "model": model or None,
        "reservoir": reservoir,
        "sample_index": int(idx),
        **selection_meta,
        "T": float(target["T"]),
        "Ttilde": float(surrogate["Ttilde"]),
        "alpha": float(surrogate["Ttilde"]) / float(target["T"]),
        "dt": {"target": float(target["dt"]), "surrogate": float(surrogate["dt"])},
        "num_frames": int(args.num_frames),
        "target": target,
        "surrogate": {
            "Ttilde": float(surrogate["Ttilde"]),
            "dt": float(surrogate["dt"]),
            "input_scale": float(surrogate["input_scale"]),
            "input_shift": float(surrogate["input_shift"]),
            "reservoir_config": surrogate["reservoir_config"].__dict__,
        },
        "files": files,
    }
    (out_dir / "asset_manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
