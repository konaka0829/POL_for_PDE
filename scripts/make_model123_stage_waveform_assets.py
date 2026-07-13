#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.elm import FixedRandomELM
from pol.metadata import to_jsonable
from pol.model123_1d.predictors import Model123Config, ObservedTrajectoryFeature1D
from pol.plotting import plot_feature_vector, plot_single_waveform, plot_waveform_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create individual slide waveform assets from a saved Model123 run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--predictions", default="")
    parser.add_argument("--model-state", default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--sample-mode",
        choices=("first", "median-error", "best-error", "worst-error"),
        default="first",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--shared-ylim", action="store_true")
    parser.add_argument("--formats", default="png,pdf,svg", help="Accepted for manifest filtering; figures are saved in all formats.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested by --device cuda but torch.cuda.is_available() is false")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    return torch.float32 if name == "float32" else torch.float64


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"predictions.pt does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ["x_test", "y_test", "pred_test", "per_sample_absL2h", "per_sample_relL2"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")
    return payload


def select_sample(payload: dict[str, Any], sample_index: int, sample_mode: str) -> tuple[int, str]:
    n = int(payload["x_test"].shape[0])
    if n <= 0:
        raise ValueError("predictions.pt contains no test samples")
    if sample_mode == "first":
        idx = int(sample_index)
    else:
        errors = payload["per_sample_absL2h"].detach().cpu().reshape(-1)
        order = torch.argsort(errors)
        if sample_mode == "best-error":
            idx = int(order[0])
        elif sample_mode == "worst-error":
            idx = int(order[-1])
        else:
            idx = int(order[len(order) // 2])
    if idx < 0 or idx >= n:
        raise IndexError(f"sample index {idx} is out of range for {n} test samples")
    return idx, sample_mode


def model_config_from_payload(predictions: dict[str, Any], run_config: dict[str, Any], device: torch.device, dtype: torch.dtype) -> Model123Config:
    raw = predictions.get("model_config") or run_config.get("resolved_config", {}).get("surrogate", {})
    if not raw:
        raise KeyError("Could not find model_config in predictions.pt or resolved_config.surrogate in run_config.json")
    allowed = {field.name for field in fields(Model123Config)}
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    kwargs["device"] = device
    kwargs["dtype"] = dtype
    if "ridge_dtype" in kwargs and isinstance(kwargs["ridge_dtype"], str):
        kwargs["ridge_dtype"] = torch.float32 if kwargs["ridge_dtype"].endswith("float32") else torch.float64
    if isinstance(kwargs.get("dtype"), str):
        kwargs["dtype"] = torch.float32 if kwargs["dtype"].endswith("float32") else torch.float64
    return Model123Config(**kwargs)


def paths_to_dict(paths: tuple[str, str, str], formats: set[str]) -> dict[str, str]:
    by_ext = {"png": paths[0], "pdf": paths[1], "svg": paths[2]}
    return {ext: by_ext[ext] for ext in ("png", "pdf", "svg") if ext in formats}


def add_file(files: dict[str, dict[str, str]], key: str, paths: tuple[str, str, str], formats: set[str]) -> None:
    files[key] = paths_to_dict(paths, formats)


def prefix_for(args: argparse.Namespace, model: str) -> str:
    return args.prefix.strip() if args.prefix.strip() else model


def shared_ylim_for(curves: list[torch.Tensor]) -> tuple[float, float]:
    values = torch.cat([curve.detach().cpu().reshape(-1).to(torch.float64) for curve in curves])
    ymin = float(torch.min(values))
    ymax = float(torch.max(values))
    if ymin == ymax:
        margin = max(1e-6, abs(ymin) * 0.05)
    else:
        margin = 0.05 * (ymax - ymin)
    return ymin - margin, ymax + margin


def load_model_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"model state must be a dict, got {type(state)} from {path}")
    return state


def compute_hidden_feature(phi: torch.Tensor, cfg: Model123Config, state: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    phi_device = phi.to(device=device, dtype=dtype)
    if "elm_weight" in state and "elm_bias" in state:
        weight = state["elm_weight"].to(device=device, dtype=dtype)
        bias = state["elm_bias"].to(device=device, dtype=dtype)
        activation = str(state.get("elm_activation", cfg.elm_activation))
        h = phi_device @ weight.t() + bias
        if activation == "tanh":
            return torch.tanh(h).detach().cpu()
        if activation == "relu":
            return torch.relu(h).detach().cpu()
        if activation == "identity":
            return h.detach().cpu()
        raise ValueError(f"Unsupported ELM activation in model state: {activation}")
    elm = FixedRandomELM(
        in_dim=phi.shape[1],
        hidden_dim=cfg.elm_hidden_dim,
        activation=cfg.elm_activation,
        seed=cfg.elm_seed,
        weight_scale=cfg.elm_weight_scale,
        bias_scale=cfg.elm_bias_scale,
        device=device,
        dtype=dtype,
    )
    return elm(phi_device).detach().cpu()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    predictions_path = Path(args.predictions) if args.predictions else run_dir / "predictions.pt"
    model_state_path = Path(args.model_state) if args.model_state else run_dir / "model.pt"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = {item.strip().lower() for item in args.formats.split(",") if item.strip()}
    invalid_formats = formats.difference({"png", "pdf", "svg"})
    if invalid_formats:
        raise ValueError(f"Unsupported --formats entries: {sorted(invalid_formats)}")

    predictions = load_predictions(predictions_path)
    run_config = load_json(run_dir / "run_config.json")
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    model = str(predictions.get("model") or run_config.get("args", {}).get("model"))
    reservoir = str(predictions.get("reservoir") or run_config.get("args", {}).get("reservoir"))
    if model not in {"model1", "model2", "model3"}:
        raise ValueError(f"Unsupported model in run metadata: {model}")
    cfg = model_config_from_payload(predictions, run_config, device=device, dtype=dtype)
    idx, selection = select_sample(predictions, args.sample_index, args.sample_mode)
    prefix = prefix_for(args, model)

    x_grid = predictions.get("x_grid")
    if x_grid is None:
        nx = int(predictions["x_test"].shape[-1])
        domain_length = float(predictions.get("domain_length", cfg.domain_length))
        x_grid = torch.linspace(0.0, domain_length, nx + 1)[:-1]
    else:
        x_grid = torch.as_tensor(x_grid)

    u0 = predictions["x_test"][idx].detach().cpu()
    y_true = predictions["y_test"][idx].detach().cpu()
    y_pred = predictions["pred_test"][idx].detach().cpu()
    feature_map = ObservedTrajectoryFeature1D(s=int(u0.shape[-1]), config=cfg)
    u0_batch = u0.reshape(1, -1)
    surrogate_terminal = feature_map.simulate_state_at_time(u0_batch, cfg.Ttilde).detach().cpu().reshape(-1)
    phi = feature_map(u0_batch).detach().cpu()

    ylim = shared_ylim_for([u0, y_true, y_pred, surrogate_terminal]) if args.shared_ylim else None
    stem = f"{prefix}_sample{idx:03d}"
    files: dict[str, dict[str, str]] = {}
    add_file(files, "input_u0", plot_single_waveform(x_grid, u0, str(out_dir / f"{stem}_input_u0"), label="u(x,0)", title="input u(x,0)", ylim=ylim), formats)
    add_file(files, "target_yT", plot_single_waveform(x_grid, y_true, str(out_dir / f"{stem}_target_yT"), label="target y(x,T)", title="target y(x,T)", ylim=ylim), formats)
    add_file(files, "prediction_yhat", plot_single_waveform(x_grid, y_pred, str(out_dir / f"{stem}_prediction_yhat"), label="prediction", title="prediction yhat(x,T)", ylim=ylim), formats)
    add_file(files, "surrogate_terminal", plot_single_waveform(x_grid, surrogate_terminal, str(out_dir / f"{stem}_surrogate_terminal"), label="surrogate terminal", title="surrogate terminal state", ylim=ylim), formats)
    add_file(
        files,
        "target_vs_prediction",
        plot_waveform_overlay(
            x_grid,
            [
                {"y": y_true, "label": "target y(x,T)", "linestyle": "-"},
                {"y": y_pred, "label": "prediction", "linestyle": "--"},
            ],
            str(out_dir / f"{stem}_target_vs_prediction"),
            title="target vs prediction",
            ylim=ylim,
        ),
        formats,
    )

    feature_metadata: dict[str, Any] = {
        "phi_dim": int(phi.shape[-1]),
        "obs": cfg.obs,
        "K": int(cfg.K),
        "feature_times": cfg.feature_times,
    }
    if model in {"model2", "model3"}:
        phi_key = f"{model}_phi_waveform" if cfg.obs == "full" and int(cfg.K) == 1 and phi.shape[-1] == u0.numel() else f"{model}_phi_vector"
        phi_name = f"{stem}_{phi_key}"
        if phi_key.endswith("_waveform"):
            add_file(files, phi_key, plot_single_waveform(x_grid, phi.reshape(-1), str(out_dir / phi_name), label="Phi", title=f"{model} Phi waveform", ylim=ylim), formats)
            feature_metadata["phi_rendering"] = "waveform"
        else:
            add_file(files, phi_key, plot_feature_vector(phi.reshape(-1), str(out_dir / phi_name), title=f"{model} Phi feature vector"), formats)
            feature_metadata["phi_rendering"] = "feature_vector"

    if model == "model3":
        state = load_model_state(model_state_path)
        hidden = compute_hidden_feature(phi, cfg, state, device=device, dtype=dtype).reshape(-1)
        add_file(
            files,
            "model3_hidden_feature_vector",
            plot_feature_vector(hidden, str(out_dir / f"{stem}_model3_hidden_feature_vector"), title="Model 3 hidden feature vector"),
            formats,
        )
        feature_metadata["hidden_dim"] = int(hidden.numel())
        feature_metadata["hidden_source"] = "model_state" if "elm_weight" in state and "elm_bias" in state else "reconstructed_from_config"

    manifest = {
        "source_run_dir": str(run_dir),
        "predictions_path": str(predictions_path),
        "model_state_path": str(model_state_path) if model_state_path.exists() else None,
        "model": model,
        "reservoir": reservoir,
        "sample_index": int(idx),
        "sample_selection": selection,
        "T": float(run_config.get("args", {}).get("T", predictions.get("args", {}).get("T", 0.0))),
        "Ttilde": float(cfg.Ttilde),
        "alpha": float(cfg.Ttilde) / float(run_config.get("args", {}).get("T", predictions.get("args", {}).get("T", cfg.Ttilde))),
        "domain_length": float(predictions.get("domain_length", cfg.domain_length)),
        "effective_nx": int(predictions.get("effective_nx", u0.numel())),
        "shared_ylim": list(ylim) if ylim is not None else None,
        "model1_identity_decoder_like": bool(model == "model1" and torch.allclose(y_pred, surrogate_terminal, atol=1e-6, rtol=1e-5)),
        "features": feature_metadata,
        "files": files,
    }
    (out_dir / "asset_manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
