#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model123_burgers_1d import load_data, ridge_dtype_from_name
from pol.cache import file_sha256, stable_hash, to_jsonable
from pol.metadata import get_command_line, get_git_info, get_runtime_info, normalize_dataset_metadata
from pol.model123_1d import Model123Config, Model2Regressor1D, Model3Regressor1D
from pol.model123_1d.feature_cache import (
    feature_cache_dir,
    load_feature_cache,
    make_feature_cache_key,
    save_feature_cache,
)
from pol.model123_1d.metrics import (
    dataset_abs_l2h_rmse,
    dataset_rel_l2h_aggregate,
    dataset_rel_l2h_mean,
)
from pol.ridge import fit_ridge_from_tensors, predict_linear


def parse_zeta_grid(raw: str) -> list[float]:
    values = [float(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--zeta-grid must contain at least one value")
    return values


def _set_if_default(parser: argparse.ArgumentParser, args: argparse.Namespace, name: str, value: Any) -> None:
    if value is None or not hasattr(args, name):
        return
    if getattr(args, name) == parser.get_default(name):
        setattr(args, name, value)
        args._config_applied[name] = value


def apply_config_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.config:
        return
    args._config_applied = {}
    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    target = cfg.get("target", {})
    data = cfg.get("data", {})
    readout = cfg.get("readout", {})
    _set_if_default(parser, args, "T", target.get("T"))
    _set_if_default(parser, args, "target_nu", target.get("nu", target.get("target_nu")))
    _set_if_default(parser, args, "dt", target.get("dt"))
    _set_if_default(parser, args, "ntrain", data.get("ntrain"))
    _set_if_default(parser, args, "nval", data.get("nval"))
    _set_if_default(parser, args, "ntest", data.get("ntest"))
    _set_if_default(parser, args, "data_seed", data.get("data_seed"))
    _set_if_default(parser, args, "split_seed", data.get("split_seed"))
    _set_if_default(parser, args, "data_dtype", data.get("data_dtype"))
    _set_if_default(parser, args, "sim_dtype", data.get("sim_dtype"))
    _set_if_default(parser, args, "ridge_convention", readout.get("ridge_convention"))
    _set_if_default(parser, args, "ridge_dtype", readout.get("ridge_dtype"))
    if args.zeta_grid == parser.get_default("zeta_grid") and readout.get("zeta_grid"):
        args.zeta_grid = ",".join(str(v) for v in readout["zeta_grid"])
        args._config_applied["zeta_grid"] = readout["zeta_grid"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a validation-selected zeta path using cached Model 2/3 features.")
    parser.add_argument("--config", default="")
    parser.add_argument("--model", choices=("model2", "model3"), default="model2")
    parser.add_argument("--reservoir", choices=("burgers", "reaction_diffusion", "ks", "static", "heat", "advection"), default="reaction_diffusion")
    parser.add_argument("--data-file", default="data/burgers_model123.mat")
    parser.add_argument("--data-mode", choices=("single_split", "separate_files"), default="single_split")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--output-dir", default="outputs/zeta_path")
    parser.add_argument("--feature-cache-dir", default="outputs/cache/features")
    parser.add_argument("--use-feature-cache", action="store_true")
    parser.add_argument("--refresh-feature-cache", action="store_true")
    parser.add_argument("--zeta-grid", default="1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0")
    parser.add_argument("--ntrain", type=int, default=800)
    parser.add_argument("--nval", type=int, default=200)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--train-split", type=float, default=0.75)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dtype", choices=("preserve", "float32", "float64"), default="float32")
    parser.add_argument("--sim-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--target-nu", type=float, default=None)
    parser.add_argument("--Ttilde", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--feature-times", default="")
    parser.add_argument("--obs", choices=("full", "points", "fourier", "proj"), default="full")
    parser.add_argument("--J", type=int, default=128)
    parser.add_argument("--sensor-mode", choices=("equispaced", "random"), default="equispaced")
    parser.add_argument("--sensor-seed", type=int, default=0)
    parser.add_argument("--elm-h", type=int, default=1024)
    parser.add_argument("--elm-activation", choices=("tanh", "relu", "identity"), default="tanh")
    parser.add_argument("--elm-seed", type=int, default=0)
    parser.add_argument("--elm-weight-scale", type=float, default=0.0)
    parser.add_argument("--elm-bias-scale", type=float, default=1.0)
    parser.add_argument("--ridge-convention", default="normalized_empirical_l2h_unweighted_frobenius")
    parser.add_argument("--ridge-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--standardize-features", type=int, choices=(0, 1), default=0)
    parser.add_argument("--feature-std-eps", type=float, default=1e-6)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--rd-nu", type=float, default=1e-3)
    parser.add_argument("--rd-alpha", type=float, default=1.0)
    parser.add_argument("--rd-beta", type=float, default=1.0)
    parser.add_argument("--res-burgers-nu", type=float, default=0.05)
    parser.add_argument("--res-burgers-b", type=float, default=1.0)
    parser.add_argument("--burgers-scheme", choices=("semi_implicit", "split_step", "etdrk4"), default="split_step")
    parser.add_argument("--burgers-fine-dt", type=float, default=1e-4)
    parser.add_argument("--burgers-dealias", type=int, choices=(0, 1), default=1)
    parser.add_argument("--ks-dealias", action="store_true")
    parser.add_argument("--ks-b", type=float, default=1.0)
    parser.add_argument("--ks-eta", type=float, default=1.0)
    parser.add_argument("--ks-kappa", type=float, default=1.0)
    parser.add_argument("--heat-nu", type=float, default=1e-2)
    parser.add_argument("--advection-c", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-metadata-mismatch", action="store_true")
    return parser


def model_config_from_args(args: argparse.Namespace, *, zeta: float = 1e-8) -> Model123Config:
    if args.Ttilde <= 0.0:
        args.Ttilde = args.T
    return Model123Config(
        reservoir=args.reservoir,
        Ttilde=args.Ttilde,
        dt=args.dt,
        K=args.K,
        feature_times=args.feature_times,
        obs=args.obs,
        J=args.J,
        sensor_mode=args.sensor_mode,
        sensor_seed=args.sensor_seed,
        ridge_lambda=zeta,
        ridge_zeta=zeta,
        ridge_convention=args.ridge_convention,
        ridge_dtype=ridge_dtype_from_name(args.ridge_dtype),
        standardize_features=bool(args.standardize_features),
        elm_hidden_dim=args.elm_h,
        elm_activation=args.elm_activation,
        elm_seed=args.elm_seed,
        elm_weight_scale=args.elm_weight_scale,
        elm_bias_scale=args.elm_bias_scale,
        rd_nu=args.rd_nu,
        rd_alpha=args.rd_alpha,
        rd_beta=args.rd_beta,
        res_burgers_nu=args.res_burgers_nu,
        res_burgers_b=args.res_burgers_b,
        ks_dealias=args.ks_dealias,
        ks_b=args.ks_b,
        ks_eta=args.ks_eta,
        ks_kappa=args.ks_kappa,
        burgers_scheme=args.burgers_scheme,
        burgers_fine_dt=args.burgers_fine_dt,
        burgers_dealias=bool(args.burgers_dealias),
        heat_nu=args.heat_nu,
        advection_c=args.advection_c,
        device=args.device,
        dtype=torch.float32,
    )


@torch.no_grad()
def collect_features(model, x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, x.shape[0], batch_size):
        chunks.append(model.features(x[start : start + batch_size]).detach().cpu())
    return torch.cat(chunks, dim=0)


def feature_tensors(args, x_train, x_val, x_test, split_meta) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    cfg = model_config_from_args(args)
    s = int(x_train.shape[1])
    model = Model2Regressor1D(s=s, config=cfg) if args.model == "model2" else Model3Regressor1D(s=s, config=cfg)
    dataset_hash = file_sha256(args.data_file)
    split_hash = stable_hash(split_meta)
    surrogate_config = {
        "model": args.model,
        "reservoir": args.reservoir,
        "Ttilde": args.Ttilde,
        "dt": args.dt,
        "K": args.K,
        "rd_nu": args.rd_nu,
        "rd_alpha": args.rd_alpha,
        "rd_beta": args.rd_beta,
        "res_burgers_nu": args.res_burgers_nu,
        "res_burgers_b": args.res_burgers_b,
        "burgers_scheme": args.burgers_scheme,
        "ks_b": args.ks_b,
        "ks_eta": args.ks_eta,
        "ks_kappa": args.ks_kappa,
        "heat_nu": args.heat_nu,
        "advection_c": args.advection_c,
        "elm_seed": args.elm_seed if args.model == "model3" else None,
        "elm_h": args.elm_h if args.model == "model3" else None,
    }
    observation_config = {
        "obs": args.obs,
        "J": args.J,
        "sensor_mode": args.sensor_mode,
        "sensor_seed": args.sensor_seed,
        "feature_times": args.feature_times,
    }
    key = make_feature_cache_key(
        dataset_hash=dataset_hash,
        split_hash=split_hash,
        surrogate_config=surrogate_config,
        observation_config=observation_config,
    )
    cache_dir = feature_cache_dir(args.feature_cache_dir, key)
    expected_meta = {"dataset_hash": dataset_hash, "split_hash": split_hash, **key}
    if args.use_feature_cache and not args.refresh_feature_cache:
        try:
            tensors = load_feature_cache(cache_dir, expected_metadata=expected_meta)
            if {"train", "val", "test"}.issubset(tensors):
                return tensors, {**expected_meta, "cache_hit": True, "cache_dir": str(cache_dir)}
        except FileNotFoundError:
            pass

    tensors = {
        "train": collect_features(model, x_train, batch_size=args.batch_size),
        "val": collect_features(model, x_val, batch_size=args.batch_size),
        "test": collect_features(model, x_test, batch_size=args.batch_size),
    }
    metadata = {
        **expected_meta,
        "cache_hit": False,
        "cache_dir": str(cache_dir),
        "feature_shape": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "feature_hash": stable_hash({name: {"shape": list(tensor.shape), "sum": float(tensor.double().sum().item())} for name, tensor in tensors.items()}),
        "dtype": str(tensors["train"].dtype).replace("torch.", ""),
        "ntrain": int(x_train.shape[0]),
        "nval": int(x_val.shape[0]),
        "ntest": int(x_test.shape[0]),
        "nx": s,
        "surrogate_config": surrogate_config,
        "observation_config": observation_config,
    }
    if args.use_feature_cache:
        save_feature_cache(cache_dir, tensors=tensors, metadata=metadata)
    return tensors, metadata


def metrics_for(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "absL2h": dataset_abs_l2h_rmse(pred, target),
        "relL2_mean": dataset_rel_l2h_mean(pred, target),
        "relL2_agg": dataset_rel_l2h_aggregate(pred, target),
    }


def standardize_from_train(tensors: dict[str, torch.Tensor], *, eps: float) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    train = tensors["train"]
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True)
    scale = std + float(eps)
    transformed = {name: (tensor - mean) / scale for name, tensor in tensors.items()}
    return transformed, {
        "enabled": True,
        "eps": float(eps),
        "mean_shape": list(mean.shape),
        "std_min": float(std.min().item()) if std.numel() else None,
        "std_max": float(std.max().item()) if std.numel() else None,
    }


def objective(pred: torch.Tensor, target: torch.Tensor, W: torch.Tensor, zeta: float) -> float:
    mse = dataset_abs_l2h_rmse(pred, target) ** 2
    return float(mse + float(zeta) * torch.linalg.matrix_norm(W[:-1, :], ord="fro").item() ** 2)


def run_zeta_path(args: argparse.Namespace, *, train_limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    args.data_seed = args.seed if args.data_seed is None else args.data_seed
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.nval <= 0:
        raise ValueError("zeta-path requires --nval > 0 so validation, not test, selects zeta")
    x_train, y_train, x_val, y_val, x_test, y_test, split_meta, dataset_meta, metadata_validation = load_data(args)
    tensors, cache_meta = feature_tensors(args, x_train, x_val, x_test, split_meta)
    standardization_meta = {"enabled": False}
    if bool(args.standardize_features):
        tensors, standardization_meta = standardize_from_train(tensors, eps=args.feature_std_eps)
    train_features = tensors["train"]
    if train_limit is not None:
        train_features = train_features[:train_limit]
        y_train = y_train[:train_limit]
    dx = 1.0 / float(y_train.shape[1])
    rows = []
    for zeta in parse_zeta_grid(args.zeta_grid):
        state = fit_ridge_from_tensors(
            train_features,
            y_train,
            ridge_zeta=zeta,
            dx=dx,
            dtype=ridge_dtype_from_name(args.ridge_dtype),
            regularize_bias=False,
            convention=args.ridge_convention,
        )
        W = state["W"]
        pred_train = predict_linear(train_features.to(dtype=W.dtype), W).cpu()
        pred_val = predict_linear(tensors["val"].to(dtype=W.dtype), W).cpu()
        pred_test = predict_linear(tensors["test"].to(dtype=W.dtype), W).cpu()
        m_train = metrics_for(pred_train, y_train)
        m_val = metrics_for(pred_val, y_val)
        m_test = metrics_for(pred_test, y_test)
        rows.append(
            {
                "zeta": zeta,
                "train_absL2h": m_train["absL2h"],
                "val_absL2h": m_val["absL2h"],
                "test_absL2h": m_test["absL2h"],
                "train_relL2_mean": m_train["relL2_mean"],
                "val_relL2_mean": m_val["relL2_mean"],
                "test_relL2_mean": m_test["relL2_mean"],
                "train_relL2_agg": m_train["relL2_agg"],
                "val_relL2_agg": m_val["relL2_agg"],
                "test_relL2_agg": m_test["relL2_agg"],
                "W_fro_norm": float(state["W_fro_norm"].item()),
                "W_hs_norm_l2h": float(state["W_hs_norm_l2h"].item()),
                "d_eff": float(state["d_eff"].item()),
                "cond_zeta": float(state["cond_zeta"].item()),
                "objective_train": objective(pred_train, y_train, W, zeta),
                "objective_val": objective(pred_val, y_val, W, zeta),
                "objective_test": objective(pred_test, y_test, W, zeta),
                "selected_by_val": False,
            }
        )
    best = min(rows, key=lambda row: float(row["val_absL2h"]))
    best["selected_by_val"] = True
    summary = {
        "selection_metric": "val_absL2h",
        "selected_zeta": best["zeta"],
        "best_by_val": best,
        "feature_cache": cache_meta,
        "feature_standardization": standardization_meta,
        "split": split_meta,
        "dataset_metadata": normalize_dataset_metadata(dataset_meta),
        "metadata_validation": metadata_validation,
        "selection": {
            "selection_metric": "val_absL2h",
            "selected_by": "validation",
            "selected_zeta": best["zeta"],
            "legacy_fallback_warning": False,
        },
    }
    return rows, summary


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "zeta_path.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "zeta_path.json").write_text(json.dumps(to_jsonable(rows), indent=2), encoding="utf-8")
    (out_dir / "best_by_val.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    run_config = {
        "args": vars(args),
        "git": get_git_info(REPO_ROOT),
        **get_runtime_info(),
        "command_line": get_command_line(),
        "selection": summary.get("selection"),
        "feature_cache": summary.get("feature_cache"),
        "split": summary.get("split"),
        "dataset_metadata": summary.get("dataset_metadata"),
        "metadata_validation": summary.get("metadata_validation"),
        "dtype": {
            "data_dtype": args.data_dtype,
            "sim_dtype": args.sim_dtype,
            "ridge_dtype": args.ridge_dtype,
        },
        "metrics": summary.get("best_by_val"),
    }
    (out_dir / "run_config.json").write_text(json.dumps(to_jsonable(run_config), indent=2), encoding="utf-8")
    if args.plot:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        zeta = [row["zeta"] for row in rows]
        ax.plot(zeta, [row["train_absL2h"] for row in rows], marker="o", label="train")
        ax.plot(zeta, [row["val_absL2h"] for row in rows], marker="o", label="val")
        ax.plot(zeta, [row["test_absL2h"] for row in rows], marker="o", label="test")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("zeta")
        ax.set_ylabel("absL2h")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "zeta_path.png", dpi=200)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_config_defaults(parser, args)
    if not hasattr(args, "_config_applied"):
        args._config_applied = {}
    rows, summary = run_zeta_path(args)
    write_outputs(args, rows, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
